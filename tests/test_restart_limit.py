"""重启上限 / 毒丸保护测试（M10.2）。

覆盖：
- `AgentRuntime(..., restart_limit=...)` 与 CLI `--restart-limit` 的参数校验；
- `--runtime langgraph` 显式传 `--restart-limit` 的 fail-closed 拒绝与省略兼容；
- `blocked`（未 `resolve` / `ApprovalRevoked`）、`interrupted`（未 `approve`）、`failed`（模型 500）与真 `SIGKILL` 在 `harness agent` 下达上限熔断退出码 5；
- `restart_limit=1` 边界、上调/下调 `--restart-limit` 恢复、切点压缩提交清零计数、以及 `Durability.EXIT` 熔断落盘；
- 已 `faulted` 的 run 即使缓存被改坏也能先修复投影并继续返回 `FAULTED`（不新增 `run_resumed` / `run_stopped`）。
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bridle import cli
from bridle.agent.task import AgentConfig, build_agent_task
from bridle.effects import Replay
from bridle.fake_model.server import FakeModelServer, ScriptedModel
from bridle.model.gateway import ModelGateway
from bridle.policy import ApprovalPolicy, CommandRules, Decision, PolicyEngine, Rule
from bridle.runtime import AgentRuntime, Checkpointer, Durability, EffectLedger, RunStatus
from bridle.runtime.demo import EffectSink
from bridle.runtime.events import EventLog, restarts_since_commit
from bridle.runtime.types import NodeInterrupt, Step, StepContext, Task
from bridle.tools.builtin import flaky_write_tool, note_read


class _ProcessDied(BaseException):
    """模拟进程崩溃（未提交当前步）。"""


def test_agent_runtime_and_cli_reject_invalid_restart_limit(tmp_path: Path, capsys) -> None:
    task = Task("t", [Step("s0", lambda ctx: None)])
    for bad in (0, -1, True, False, 1.5, "5"):
        target = tmp_path / f"rt-{type(bad).__name__}-{bad}"
        with pytest.raises(ValueError):
            AgentRuntime(task, target, restart_limit=bad)  # type: ignore[arg-type]  # 原因：这条测的就是传错类型时被拒
        assert not target.exists()

    for cmd in (
        ["run", "--state-dir", str(tmp_path / "cli-run"), "--restart-limit", "0"],
        ["run", "--state-dir", str(tmp_path / "cli-run-neg"), "--restart-limit", "-2"],
        [
            "agent", "--state-dir", str(tmp_path / "cli-agent"),
            "--base-url", "http://127.0.0.1:9/v1", "--prompt", "x", "--restart-limit", "0",
        ],
    ):
        assert cli.main(cmd) == 2
        assert json.loads(capsys.readouterr().out)["code"] == "BAD_CONFIG"
        state_arg = Path(cmd[cmd.index("--state-dir") + 1])
        assert not state_arg.exists()


def test_restart_limit_one_and_lowering_limit_on_resume(tmp_path: Path) -> None:
    def crash_step(ctx: StepContext) -> None:
        raise _ProcessDied()

    # restart_limit=1：首次崩溃（0 次恢复）→ 第 1 次 resume（第 1 次恢复，允许）→ 第 2 次 resume 熔断
    dir_one = tmp_path / "one"
    task = Task("t", [Step("s0", crash_step)])
    with pytest.raises(_ProcessDied):
        AgentRuntime(task, dir_one, restart_limit=1).run()
    with pytest.raises(_ProcessDied):
        AgentRuntime(task, dir_one, restart_limit=1).run(resume=True)
    faulted = AgentRuntime(task, dir_one, restart_limit=1).run(resume=True)
    assert faulted.status is RunStatus.FAULTED
    assert faulted.steps_run == []
    assert "restart_limit=1" in (faulted.error or "")

    # 即使之后把 restart_limit 调高到 10 再恢复，已经 faulted 的 run 也不得复活
    still_faulted = AgentRuntime(task, dir_one, restart_limit=10).run(resume=True)
    assert still_faulted.status is RunStatus.FAULTED
    assert still_faulted.error == faulted.error

    # 在默认上限下已恢复 2 次，下一次恢复改用 --restart-limit 2 应立即熔断
    dir_lower = tmp_path / "lower"
    with pytest.raises(_ProcessDied):
        AgentRuntime(task, dir_lower).run()
    for _ in range(2):
        with pytest.raises(_ProcessDied):
            AgentRuntime(task, dir_lower).run(resume=True)
    tripped = AgentRuntime(task, dir_lower, restart_limit=2).run(resume=True)
    assert tripped.status is RunStatus.FAULTED
    events = EventLog(EffectLedger(dir_lower).db_path).read("run-1")
    assert [e.type for e in events].count("run_resumed") == 2


def test_blocked_run_without_resolve_counts_toward_restart_limit(tmp_path: Path) -> None:
    calls: list[int] = []

    def unsafe_step(ctx: StepContext) -> None:
        def action() -> dict:
            calls.append(1)
            if len(calls) == 1:
                raise _ProcessDied()
            return {"ok": True}

        ctx.ledger.attempt(ctx.effect_key("u"), {"op": "u"}, action, replay=Replay.UNSAFE)

    task = Task("t", [Step("u", unsafe_step)])
    with pytest.raises(_ProcessDied):
        AgentRuntime(task, tmp_path, restart_limit=2).run()

    # 两次未 resolve 的恢复均停在 blocked，但分别消耗 1 次恢复额度
    r1 = AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
    r2 = AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
    assert r1.status is RunStatus.BLOCKED and r2.status is RunStatus.BLOCKED
    assert len(calls) == 1

    # 第 3 次恢复命中上限，直接熔断为 faulted
    r3 = AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
    assert r3.status is RunStatus.FAULTED
    assert r3.steps_run == []
    events = EventLog(EffectLedger(tmp_path).db_path).read("run-1")
    assert restarts_since_commit(events) == 2
    assert events[-1].type == "run_stopped" and events[-1].payload["status"] == "faulted"

    # 第 4 次幂等返回，不新增事件
    r4 = AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
    assert r4.status is RunStatus.FAULTED
    assert r4.error == r3.error
    assert len(EventLog(EffectLedger(tmp_path).db_path).read("run-1")) == len(events)


def test_agent_cli_interrupted_without_approve_trips_restart_limit_exit_5(
    tmp_path: Path, capsys
) -> None:
    model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "flaky_write", "arguments": {"target": "notes/a.md"}}]},
        {"content": "完成"},
    ])
    with FakeModelServer(model) as server:
        sd = tmp_path / "agent"
        c0 = cli.main([
            "agent", "--state-dir", str(sd), "--base-url", server.base_url,
            "--prompt", "写笔记", "--ask-tool", "tool:flaky_write*", "--restart-limit", "2",
        ])
        assert c0 == 3
        assert json.loads(capsys.readouterr().out)["status"] == "interrupted"

        for _ in range(2):
            ci = cli.main([
                "agent", "--state-dir", str(sd), "--base-url", server.base_url,
                "--prompt", "写笔记", "--ask-tool", "tool:flaky_write*",
                "--resume", "--restart-limit", "2",
            ])
            assert ci == 3
            assert json.loads(capsys.readouterr().out)["status"] == "interrupted"

        c_fault = cli.main([
            "agent", "--state-dir", str(sd), "--base-url", server.base_url,
            "--prompt", "写笔记", "--ask-tool", "tool:flaky_write*",
            "--resume", "--restart-limit", "2",
        ])
        out_fault = json.loads(capsys.readouterr().out)
        assert c_fault == 5
        assert out_fault["status"] == "faulted"
        assert out_fault["steps_run"] == ["model"]
        assert out_fault["sink_rows"] == 0
        events_at_fault = EventLog(EffectLedger(sd).db_path).read("run-1")

        # 再次 --resume 仍返回 5，事件不增加
        c_again = cli.main([
            "agent", "--state-dir", str(sd), "--base-url", server.base_url,
            "--prompt", "写笔记", "--ask-tool", "tool:flaky_write*",
            "--resume", "--restart-limit", "2",
        ])
        out_again = json.loads(capsys.readouterr().out)
        assert c_again == 5
        assert out_again["status"] == "faulted"
        assert out_again["error"] == out_fault["error"]
        assert len(EventLog(EffectLedger(sd).db_path).read("run-1")) == len(events_at_fault)
        assert model.calls == 1


def test_agent_approve_commits_and_resets_restart_counter(tmp_path: Path) -> None:
    model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "flaky_write", "arguments": {"target": "notes/a.md"}}]},
        {"content": "完成"},
    ])
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        registry = build_agent_registry(sink)
        cfg = AgentConfig(
            gateway=ModelGateway(server.base_url, retries=0, sleep=lambda _s: None,
                                 first_byte_timeout=5.0, total_timeout=5.0),
            registry=registry,
            policy=PolicyEngine(
                rules=CommandRules([
                    Rule("ask", Decision.PROMPT, "tool:flaky_write*"),
                    Rule("allow", Decision.ALLOW, "tool:*"),
                ]),
                approval=ApprovalPolicy.ON_REQUEST,
            ),
        )
        task = build_agent_task("写", cfg)
        assert AgentRuntime(task, tmp_path, restart_limit=2).run().status is RunStatus.INTERRUPTED
        assert AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True).status is RunStatus.INTERRUPTED

        ledger = EffectLedger(tmp_path)
        aid = ledger.approvals(run_id="run-1")[0]["approval_id"]
        ledger.approve(aid, ttl_s=60.0, granted_by="why")

        # 第 2 次恢复（restarts_since_commit == 1 < 2）拿到批准并提交工具步，计数清零
        done = AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
        assert done.status is RunStatus.COMPLETED
        assert restarts_since_commit(EventLog(ledger.db_path).read("run-1")) == 0
        assert len(sink.rows()) == 1


def build_agent_registry(sink: EffectSink):
    from bridle.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(flaky_write_tool(sink, slow_first=0.0))
    return registry


def test_exit_durability_flushes_faulted_and_tampered_cache_repairs_without_new_resumes(
    tmp_path: Path, capsys
) -> None:
    def always_interrupt(ctx: StepContext) -> None:
        raise NodeInterrupt()

    task = Task("t", [Step("s0", always_interrupt)])
    assert AgentRuntime(task, tmp_path, durability=Durability.EXIT, restart_limit=1).run().status is RunStatus.INTERRUPTED
    assert AgentRuntime(task, tmp_path, durability=Durability.EXIT, restart_limit=1).run(resume=True).status is RunStatus.INTERRUPTED
    faulted = AgentRuntime(task, tmp_path, durability=Durability.EXIT, restart_limit=1).run(resume=True)
    assert faulted.status is RunStatus.FAULTED

    latest = Checkpointer(tmp_path).latest("run-1")
    assert latest is not None and latest.kind == "faulted" and latest.state["status"] == "faulted"
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 0
    capsys.readouterr()

    # 改坏缓存：把 status 改回 running，恢复应先写 projection_repaired，仍返回 FAULTED，且不新增 run_resumed/run_stopped
    with sqlite3.connect(tmp_path / "checkpoints.sqlite") as conn:
        conn.execute(
            "UPDATE checkpoints SET state_json = json_set(state_json, '$.status', 'running') "
            "WHERE thread_id = 'run-1'"
        )
        conn.commit()
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 1
    capsys.readouterr()

    events_before = EventLog(EffectLedger(tmp_path).db_path).read("run-1")
    repaired_report = AgentRuntime(task, tmp_path, restart_limit=1).run(resume=True)
    assert repaired_report.status is RunStatus.FAULTED
    assert repaired_report.error == faulted.error

    events_after = EventLog(EffectLedger(tmp_path).db_path).read("run-1")
    new_events = events_after[len(events_before):]
    assert [e.type for e in new_events] == ["projection_repaired"]
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 0


def test_cli_langgraph_rejects_explicit_restart_limit_while_default_still_works(
    tmp_path: Path, capsys
) -> None:
    assert cli.build_parser().parse_args(["run", "--state-dir", str(tmp_path)]).restart_limit == 5

    ok_dir = tmp_path / "lg-default"
    assert cli.main(["run", "--state-dir", str(ok_dir), "--runtime", "langgraph"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    for limit in ("2", "5"):
        rejected_dir = tmp_path / f"lg-explicit-{limit}"
        assert cli.main([
            "run", "--state-dir", str(rejected_dir), "--runtime", "langgraph", "--restart-limit", limit,
        ]) == 2
        assert json.loads(capsys.readouterr().out)["code"] == "UNSUPPORTED_ON_LANGGRAPH"
        assert not rejected_dir.exists()


def _peek_ro(db_path: Path, sql: str) -> int | None:
    try:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=0.2)
    except sqlite3.OperationalError:
        return None
    try:
        row = conn.execute(sql).fetchone()
        return int(row[0]) if row is not None else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def test_agent_cli_real_sigkill_trips_restart_limit_exit_5(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "flaky_write", "arguments": {"target": "notes/a.md"}}]},
        {"content": "完成"},
    ])
    with FakeModelServer(model) as server:
        sd = tmp_path / "agent-sigkill"

        def run_clean(args: list[str]) -> subprocess.CompletedProcess[str]:
            proc = subprocess.Popen(
                [sys.executable, "-m", "bridle.cli", *args],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                stdout, stderr = proc.communicate(timeout=20.0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10.0)
            return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

        def kill_at_hold(extra_args: list[str], expected_executions: int) -> int | None:
            proc = subprocess.Popen(
                [
                    sys.executable, "-m", "bridle.cli", "agent",
                    "--state-dir", str(sd), "--base-url", server.base_url,
                    "--prompt", "写笔记", "--hold-after-write", "flaky_write",
                    "--restart-limit", "2", *extra_args,
                ],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                expires = time.monotonic() + 10.0
                while time.monotonic() < expires:
                    if proc.poll() is not None:
                        return None
                    execs = _peek_ro(
                        sd / "ledger.sqlite",
                        "SELECT executions FROM effects WHERE key LIKE '%:flaky_write:%'",
                    )
                    sink_rows = _peek_ro(sd / "sink.sqlite", "SELECT COUNT(*) FROM effects_log")
                    if execs == expected_executions and sink_rows == 1:
                        time.sleep(0.05)  # 让工具写入下游后进入 hold_until_killed
                        os.kill(proc.pid, signal.SIGKILL)
                        return proc.wait(timeout=10.0)
                    time.sleep(0.02)
                return None
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10.0)

        assert kill_at_hold([], 1) == -signal.SIGKILL
        assert kill_at_hold(["--resume"], 2) == -signal.SIGKILL
        assert kill_at_hold(["--resume"], 3) == -signal.SIGKILL

        # 第 3 次 --resume（仍带 --hold-after-write）：未再执行工具，直接熔断为 faulted
        faulted = run_clean([
            "agent", "--state-dir", str(sd), "--base-url", server.base_url,
            "--prompt", "写笔记", "--hold-after-write", "flaky_write",
            "--resume", "--restart-limit", "2",
        ])
        fault_payload = json.loads(faulted.stdout)
        assert faulted.returncode == 5
        assert fault_payload["status"] == "faulted"
        assert fault_payload["steps_run"] == ["model"]
        assert fault_payload["sink_rows"] == 1
        assert model.calls == 1
        events_at_fault = EventLog(EffectLedger(sd).db_path).read("run-1")
        assert restarts_since_commit(events_at_fault) == 2

        again = run_clean([
            "agent", "--state-dir", str(sd), "--base-url", server.base_url,
            "--prompt", "写笔记", "--resume", "--restart-limit", "2",
        ])
        again_payload = json.loads(again.stdout)
        assert again.returncode == 5
        assert again_payload["status"] == "faulted"
        assert again_payload["error"] == fault_payload["error"]
        assert len(EventLog(EffectLedger(sd).db_path).read("run-1")) == len(events_at_fault)
        assert model.calls == 1


def test_agent_cli_approval_revoked_and_run_failed_trip_restart_limit_exit_5(
    tmp_path: Path, capsys
) -> None:
    # (a) ApprovalRevoked -> blocked -> 反复 --resume 达上限熔断为 faulted
    revoked_model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "flaky_write", "arguments": {"target": "notes/a.md"}}]},
        {"content": "不应走到"},
    ])
    with FakeModelServer(revoked_model) as server:
        sd_a = tmp_path / "revoked"
        assert cli.main([
            "agent", "--state-dir", str(sd_a), "--base-url", server.base_url,
            "--prompt", "写", "--ask-tool", "tool:flaky_write*", "--restart-limit", "2",
        ]) == 3
        capsys.readouterr()
        assert cli.main(["approve", "--state-dir", str(sd_a), "--run-id", "run-1"]) == 0
        capsys.readouterr()

        for _ in range(2):
            assert cli.main([
                "agent", "--state-dir", str(sd_a), "--base-url", server.base_url,
                "--prompt", "写", "--deny-tool", "tool:flaky_write*", "--resume", "--restart-limit", "2",
            ]) == 4
            assert json.loads(capsys.readouterr().out)["status"] == "blocked"

        assert cli.main([
            "agent", "--state-dir", str(sd_a), "--base-url", server.base_url,
            "--prompt", "写", "--deny-tool", "tool:flaky_write*", "--resume", "--restart-limit", "2",
        ]) == 5
        out_a = json.loads(capsys.readouterr().out)
        assert out_a["status"] == "faulted" and out_a["sink_rows"] == 0
        assert revoked_model.calls == 1

    # (b) RunFailed（模型 500）-> failed -> 反复 --resume 达上限熔断为 faulted
    failed_model = ScriptedModel([{"status": 500}, {"status": 500}, {"status": 500}, {"content": "不该调用"}])
    with FakeModelServer(failed_model) as server:
        sd_b = tmp_path / "run-failed"
        assert cli.main([
            "agent", "--state-dir", str(sd_b), "--base-url", server.base_url,
            "--prompt", "写", "--restart-limit", "2",
        ]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "failed"

        for _ in range(2):
            assert cli.main([
                "agent", "--state-dir", str(sd_b), "--base-url", server.base_url,
                "--prompt", "写", "--resume", "--restart-limit", "2",
            ]) == 1
            assert json.loads(capsys.readouterr().out)["status"] == "failed"

        assert cli.main([
            "agent", "--state-dir", str(sd_b), "--base-url", server.base_url,
            "--prompt", "写", "--resume", "--restart-limit", "2",
        ]) == 5
        out_b = json.loads(capsys.readouterr().out)
        assert out_b["status"] == "faulted"
        assert failed_model.calls == 3


def test_agent_compaction_step_commit_resets_restart_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bridle.tools.registry import ToolRegistry

    def note_call(call_id: str, path: str) -> dict:
        return {"tool_calls": [{"id": call_id, "name": "note_read", "arguments": {"path": path}}]}

    model = ScriptedModel([
        note_call("c1", "seg1-" + "甲" * 60),
        note_call("c2", "seg2-" + "乙" * 60),
        {"content": "摘要：已读 seg1"},
        {"content": "完成"},
    ])
    with FakeModelServer(model) as server:
        registry = ToolRegistry()
        registry.register(note_read())
        cfg = AgentConfig(
            gateway=ModelGateway(
                server.base_url, retries=0, sleep=lambda _s: None,
                first_byte_timeout=5.0, total_timeout=5.0,
            ),
            registry=registry,
            policy=PolicyEngine(rules=CommandRules([Rule("allow", Decision.ALLOW, "tool:*")])),
            max_steps=12,
            context_budget_tokens=350,
            keep_recent_tokens=160,
        )
        task = build_agent_task("连续读取资料并汇总", cfg)
        original_emit = AgentRuntime._emit
        original_append_many = EventLog.append_many

        # 在第 4 步（compact）提交前崩一次 → resume 1 次 → compact 提交并写 summary（计数清零）
        def die_before_compact_commit(self, run_id, items):
            if any(t == "step_committed" and idx == 4 for t, idx, _ in items):
                raise _ProcessDied()
            return original_append_many(self, run_id, items)

        monkeypatch.setattr(EventLog, "append_many", die_before_compact_commit)
        with pytest.raises(_ProcessDied):
            AgentRuntime(task, tmp_path, restart_limit=2).run()
        with pytest.raises(_ProcessDied):
            AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
        monkeypatch.setattr(EventLog, "append_many", original_append_many)

        # 第 2 次 resume：compact 成功提交（与 summary 同事务），但在第 5 步（final model）提交前再崩 1 次
        def die_before_final_commit(self, type_, *, step_index=None, payload=None):
            if type_ == "step_committed" and step_index == 5:
                raise _ProcessDied()
            return original_emit(self, type_, step_index=step_index, payload=payload)

        monkeypatch.setattr(AgentRuntime, "_emit", die_before_final_commit)
        with pytest.raises(_ProcessDied):
            AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
        monkeypatch.setattr(AgentRuntime, "_emit", original_emit)

        events_mid = EventLog(EffectLedger(tmp_path).db_path).read("run-1")
        assert any(e.type == "summary" for e in events_mid)
        assert restarts_since_commit(events_mid) == 0

        # 再崩 1 次（消耗 compact 后的第 1 次恢复额度）后恢复完成：全程共恢复 4 次 > restart_limit=2
        with pytest.raises(_ProcessDied):
            monkeypatch.setattr(AgentRuntime, "_emit", die_before_final_commit)
            AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
        monkeypatch.setattr(AgentRuntime, "_emit", original_emit)

        done = AgentRuntime(task, tmp_path, restart_limit=2).run(resume=True)
        assert done.status is RunStatus.COMPLETED
        final_events = EventLog(EffectLedger(tmp_path).db_path).read("run-1")
        assert sum(1 for e in final_events if e.type == "run_resumed") == 4
        assert restarts_since_commit(final_events) == 0
