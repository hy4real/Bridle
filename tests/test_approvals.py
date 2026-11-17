"""审批记录状态机与事件事务一致性测试（M11.2）。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from bridle import cli
from bridle.agent.task import AgentConfig, build_agent_task
from bridle.effects import ApprovalRevoked, IdempotencyKeyConflict, Replay, RunLocked
from bridle.fake_model.server import FakeModelServer, ScriptedModel
from bridle.model.gateway import ModelGateway
from bridle.policy import ApprovalPolicy, CommandRules, Decision, PolicyEngine, Rule, SandboxMode
from bridle.runtime import AgentRuntime, EffectLedger, RunStatus
from bridle.runtime import ledger as ledger_module
from bridle.runtime.demo import EffectSink
from bridle.runtime.events import EventLog
from bridle.runtime.lock import run_lock
from bridle.runtime.ledger import SCHEMA_VERSION, hash_args
from bridle.tools.base import Tool
from bridle.tools.builtin import flaky_write_tool, note_read
from bridle.tools.registry import ToolRegistry


def test_ledger_schema_version_is_4_and_creates_approvals_table(tmp_path: Path) -> None:
    assert SCHEMA_VERSION == 4
    ledger = EffectLedger(tmp_path)
    with sqlite3.connect(ledger.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert {"effects", "effect_attempts", "events", "approvals"} <= tables
    assert version == 4


def test_hash_args_ignores_key_order() -> None:
    assert hash_args({"a": 1, "b": {"x": 2, "y": 3}}) == hash_args({"b": {"y": 3, "x": 2}, "a": 1})


def test_request_approval_is_idempotent_on_unique_tuple(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    args_sha = hash_args({"target": "notes/a.md"})
    first_id = ledger.request_approval(
        "run-1", 1, "call_1", "flaky_write", args_sha, "ver-1", reason="需要审批"
    )
    second_id = ledger.request_approval(
        "run-1", 1, "call_1", "flaky_write", args_sha, "ver-1", reason="需要审批"
    )

    assert first_id == second_id
    rows = ledger.approvals(run_id="run-1")
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    events = EventLog(ledger.db_path).read("run-1")
    assert [e.type for e in events] == ["approval_requested"]
    assert events[0].payload["approval_id"] == first_id


def test_request_approval_rejects_mismatched_call_for_same_tuple(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({"target": "a"}), "v1")
    with pytest.raises(IdempotencyKeyConflict):
        ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({"target": "b"}), "v1")


def test_approve_and_revoke_lifecycle_and_filtering(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    aid = ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({"target": "a"}), "v1")

    got = ledger.get_approval("run-1", 1, "call_1")
    assert got is not None and got["approval_id"] == aid
    assert ledger.get_approval_by_id(aid) == got

    approved = ledger.approve(aid, ttl_s=60.0, granted_by="why")
    assert approved["status"] == "approved"
    assert approved["granted_by"] == "why"
    assert approved["expires_at"] > approved["granted_at"]

    assert ledger.approvals(status="pending") == []
    assert len(ledger.approvals(run_id="run-1", status="approved")) == 1

    revoked = ledger.revoke(aid, reason="策略变更")
    assert revoked["status"] == "revoked"
    assert revoked["revoked_reason"] == "策略变更"
    assert revoked["revoked_at"] is not None

    events = EventLog(ledger.db_path).read("run-1")
    assert [e.type for e in events] == ["approval_requested", "approval_granted", "approval_revoked"]


def test_approve_writes_row_and_event_in_same_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = EffectLedger(tmp_path)
    aid = ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({}), "v1")

    def boom(conn, run_id, type_, **kwargs):
        raise sqlite3.OperationalError("磁盘写满（模拟）")

    monkeypatch.setattr(ledger_module, "append_event", boom)
    with pytest.raises(sqlite3.OperationalError):
        ledger.approve(aid, ttl_s=60.0, granted_by="why")

    row = ledger.get_approval_by_id(aid)
    assert row is not None and row["status"] == "pending" and row["granted_by"] is None
    assert [e.type for e in EventLog(ledger.db_path).read("run-1")] == ["approval_requested"]


def test_request_and_revoke_also_rollback_on_event_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = EffectLedger(tmp_path)
    original_append = ledger_module.append_event

    def fail_on_request(conn, run_id, type_, **kwargs):
        raise sqlite3.OperationalError("请求事件失败")

    monkeypatch.setattr(ledger_module, "append_event", fail_on_request)
    with pytest.raises(sqlite3.OperationalError):
        ledger.request_approval("run-1", 0, "c0", "flaky_write", hash_args({}), "v1")
    assert ledger.approvals() == []
    assert EventLog(ledger.db_path).read("run-1") == []

    monkeypatch.setattr(ledger_module, "append_event", original_append)
    aid = ledger.request_approval("run-1", 0, "c0", "flaky_write", hash_args({}), "v1")

    def fail_on_revoke(conn, run_id, type_, **kwargs):
        raise sqlite3.OperationalError("吊销事件失败")

    monkeypatch.setattr(ledger_module, "append_event", fail_on_revoke)
    with pytest.raises(sqlite3.OperationalError):
        ledger.revoke(aid, reason="test")
    assert ledger.get_approval_by_id(aid)["status"] == "pending"
    assert [e.type for e in EventLog(ledger.db_path).read("run-1")] == ["approval_requested"]


def test_approval_validation_errors_and_exception(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    with pytest.raises(KeyError):
        ledger.approve("apr_missing", ttl_s=10.0, granted_by="why")
    with pytest.raises(KeyError):
        ledger.revoke("apr_missing", reason="x")

    aid = ledger.request_approval("run-1", 1, "c1", "flaky_write", hash_args({}), "v1")
    for bad_str in ("", "   "):
        with pytest.raises(ValueError):
            ledger.request_approval("run-1", 1, bad_str, "flaky_write", hash_args({}), "v1")
        with pytest.raises(ValueError):
            ledger.request_approval("run-1", 1, "c2", bad_str, hash_args({}), "v1")
        with pytest.raises(ValueError):
            ledger.approve(aid, ttl_s=10.0, granted_by=bad_str)
        with pytest.raises(ValueError):
            ledger.revoke(aid, reason=bad_str)
    for bad_ttl in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            ledger.approve(aid, ttl_s=bad_ttl, granted_by="why")

    ledger.approve(aid, ttl_s=10.0, granted_by="why")
    with pytest.raises(ValueError):
        ledger.approve(aid, ttl_s=10.0, granted_by="why")  # 已不是 pending
    ledger.revoke(aid, reason="expired")
    with pytest.raises(ValueError):
        ledger.revoke(aid, reason="again")  # 已经是 revoked

    err = ApprovalRevoked(aid, "策略变更")
    assert err.key == aid and err.reason == "策略变更"


# --- 运行时校验与 CLI approve（M11.3）--------------------------------------
def _agent_config(
    server: FakeModelServer,
    sink: EffectSink,
    *,
    rules: list[Rule] | None = None,
    approval: ApprovalPolicy = ApprovalPolicy.ON_REQUEST,
    sandbox_mode: SandboxMode = SandboxMode.READ_ONLY,
    extra_tools: list[Tool] | None = None,
    tool_retries: int = 0,
) -> AgentConfig:
    registry = ToolRegistry()
    registry.register(note_read())
    registry.register(flaky_write_tool(sink, slow_first=0.0))
    for tool in extra_tools or []:
        registry.register(tool)
    policy_rules = CommandRules(list(rules or []) + [Rule("allow", Decision.ALLOW, "tool:*")])
    return AgentConfig(
        gateway=ModelGateway(
            server.base_url, retries=0, sleep=lambda _s: None,
            first_byte_timeout=5.0, total_timeout=5.0,
        ),
        registry=registry,
        policy=PolicyEngine(rules=policy_rules, approval=approval, sandbox_mode=sandbox_mode),
        tool_retries=tool_retries,
    )


def _call_write(call_id: str = "call_w1", target: str = "notes/a.md") -> dict:
    return {"tool_calls": [{"id": call_id, "name": "flaky_write", "arguments": {"target": target}}]}


def test_resume_without_approval_keeps_interrupted_with_same_id(tmp_path: Path) -> None:
    model = ScriptedModel([_call_write(), {"content": "done"}])
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        cfg = _agent_config(server, sink, rules=[Rule("ask", Decision.PROMPT, "tool:flaky_write*")])
        first = AgentRuntime(build_agent_task("写", cfg), tmp_path).run()
        second = AgentRuntime(build_agent_task("写", cfg), tmp_path).run(resume=True)

    assert first.status is RunStatus.INTERRUPTED and second.status is RunStatus.INTERRUPTED
    ledger = EffectLedger(tmp_path)
    rows = ledger.approvals(run_id="run-1")
    assert len(rows) == 1 and rows[0]["status"] == "pending"
    assert [e.type for e in EventLog(ledger.db_path).read("run-1") if e.type.startswith("approval_")] == [
        "approval_requested"
    ]
    assert model.calls == 1 and sink.rows() == []


@pytest.mark.parametrize("mutate_name", ["drop_ask", "change_sandbox", "change_approval"])
def test_removing_ask_or_changing_sandbox_or_approval_revokes_and_blocks(
    tmp_path: Path, mutate_name: str
) -> None:
    model = ScriptedModel([_call_write(), {"content": "不应走到"}])
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        ask_rules = [Rule("ask", Decision.PROMPT, "tool:flaky_write*")]
        cfg1 = _agent_config(server, sink, rules=ask_rules)
        r1 = AgentRuntime(build_agent_task("写", cfg1), tmp_path).run()
        assert r1.status is RunStatus.INTERRUPTED

        ledger = EffectLedger(tmp_path)
        aid = ledger.approvals(run_id="run-1")[0]["approval_id"]
        ledger.approve(aid, ttl_s=60.0, granted_by="why")

        if mutate_name == "drop_ask":
            cfg2 = _agent_config(server, sink, rules=[])
        elif mutate_name == "change_sandbox":
            cfg2 = _agent_config(server, sink, rules=ask_rules, sandbox_mode=SandboxMode.WORKSPACE_WRITE)
        else:
            cfg2 = _agent_config(server, sink, rules=ask_rules, approval=ApprovalPolicy.UNLESS_TRUSTED)

        r2 = AgentRuntime(build_agent_task("写", cfg2), tmp_path).run(resume=True)
        # 已经是 revoked 之后再恢复，直接阻塞，不会重复写 approval_revoked
        r3 = AgentRuntime(build_agent_task("写", cfg2), tmp_path).run(resume=True)

    assert r2.status is RunStatus.BLOCKED and r2.blocked_on == aid
    assert r3.status is RunStatus.BLOCKED and r3.blocked_on == aid
    assert sink.rows() == [] and model.calls == 1
    row = EffectLedger(tmp_path).get_approval_by_id(aid)
    assert row is not None and row["status"] == "revoked"
    revoked_events = [
        e for e in EventLog(EffectLedger(tmp_path).db_path).read("run-1")
        if e.type == "approval_revoked"
    ]
    assert len(revoked_events) == 1


@pytest.mark.parametrize("field", ["args_hash", "tool_name"])
def test_tampered_args_or_tool_revokes_approval(tmp_path: Path, field: str) -> None:
    model = ScriptedModel([_call_write(), {"content": "不应走到"}])
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        cfg = _agent_config(server, sink, rules=[Rule("ask", Decision.PROMPT, "tool:flaky_write*")])
        assert AgentRuntime(build_agent_task("写", cfg), tmp_path).run().status is RunStatus.INTERRUPTED

        ledger = EffectLedger(tmp_path)
        aid = ledger.approvals(run_id="run-1")[0]["approval_id"]
        ledger.approve(aid, ttl_s=60.0, granted_by="why")

        with sqlite3.connect(ledger.db_path) as conn:
            conn.execute(f"UPDATE approvals SET {field} = 'tampered' WHERE approval_id = ?", (aid,))
            conn.commit()

        blocked = AgentRuntime(build_agent_task("写", cfg), tmp_path).run(resume=True)

    assert blocked.status is RunStatus.BLOCKED and blocked.blocked_on == aid
    assert EffectLedger(tmp_path).get_approval_by_id(aid)["status"] == "revoked"
    assert sink.rows() == []


def test_missing_expires_at_fails_closed_and_revokes(tmp_path: Path) -> None:
    model = ScriptedModel([_call_write(), {"content": "不应走到"}])
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        cfg = _agent_config(server, sink, rules=[Rule("ask", Decision.PROMPT, "tool:flaky_write*")])
        assert AgentRuntime(build_agent_task("写", cfg), tmp_path).run().status is RunStatus.INTERRUPTED

        ledger = EffectLedger(tmp_path)
        aid = ledger.approvals(run_id="run-1")[0]["approval_id"]
        ledger.approve(aid, ttl_s=60.0, granted_by="why")
        with sqlite3.connect(ledger.db_path) as conn:
            conn.execute("UPDATE approvals SET expires_at = NULL WHERE approval_id = ?", (aid,))
            conn.commit()

        blocked = AgentRuntime(build_agent_task("写", cfg), tmp_path).run(resume=True)

    assert blocked.status is RunStatus.BLOCKED and blocked.blocked_on == aid
    assert EffectLedger(tmp_path).get_approval_by_id(aid)["status"] == "revoked"
    assert sink.rows() == []


def test_tool_crash_after_approval_reruns_without_reasking_human(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def crash_first(target: str) -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("模拟工具进程崩了")
        return {"ok": True, "written": target}

    flaky = Tool(
        name="flaky_once",
        description="第一次失败、第二次成功的工具",
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
        handler=crash_first,
        side_effect=True,
        idempotency_key=lambda args: f"flaky:{args['target']}",
        replay=Replay.SAFE,
    )
    model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "flaky_once", "arguments": {"target": "a.md"}}]},
        {"content": "写好了"},
    ])
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        cfg = _agent_config(
            server, sink,
            rules=[Rule("ask", Decision.PROMPT, "tool:flaky_once*")],
            extra_tools=[flaky],
        )
        r1 = AgentRuntime(build_agent_task("写", cfg), tmp_path).run()
        assert r1.status is RunStatus.INTERRUPTED

        ledger = EffectLedger(tmp_path)
        aid = ledger.approvals(run_id="run-1")[0]["approval_id"]
        ledger.approve(aid, ttl_s=60.0, granted_by="why")

        # 批准后执行工具崩溃 → failed；再次 resume 不再问人，直接重跑工具并完成
        r2 = AgentRuntime(build_agent_task("写", cfg), tmp_path).run(resume=True)
        assert r2.status is RunStatus.FAILED

        r3 = AgentRuntime(build_agent_task("写", cfg), tmp_path).run(resume=True)

    assert r3.status is RunStatus.COMPLETED
    assert attempts["n"] == 2 and model.calls == 2


def test_cli_approve_paths_lock_and_agent_sandbox_mode(tmp_path: Path, capsys) -> None:
    typo = tmp_path / "missing"
    assert cli.main(["approve", "--state-dir", str(typo)]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "NO_LEDGER"
    assert not typo.exists()

    model = ScriptedModel([_call_write(), {"content": "完成"}])
    with FakeModelServer(model) as server:
        code1 = cli.main([
            "agent", "--state-dir", str(tmp_path), "--base-url", server.base_url,
            "--prompt", "写", "--run-id", "r9", "--ask-tool", "tool:flaky_write*",
            "--sandbox-mode", "workspace-write",
        ])
        out1 = json.loads(capsys.readouterr().out)
        assert code1 == 3 and out1["status"] == "interrupted"
        aid = out1["approvals"][0]["approval_id"]

        # 记录下来的 policy_version 对应 workspace-write
        stored = EffectLedger(tmp_path).get_approval_by_id(aid)
        expected_engine = PolicyEngine(
            rules=CommandRules([
                Rule("ask-tool-1", Decision.PROMPT, "tool:flaky_write*"),
                Rule("allow-tools", Decision.ALLOW, "tool:*"),
            ]),
            approval=ApprovalPolicy.ON_REQUEST,
            sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        )
        assert stored is not None and stored["policy_version"] == expected_engine.version()

        # run 锁被持有时 approve 返回退出码 7（分别测 --run-id 与 --id）
        with run_lock(tmp_path, "r9"):
            assert cli.main(["approve", "--state-dir", str(tmp_path), "--run-id", "r9"]) == 7
            assert json.loads(capsys.readouterr().out)["code"] == "RUN_LOCKED"
            assert cli.main(["approve", "--state-dir", str(tmp_path), "--id", aid]) == 7
            assert json.loads(capsys.readouterr().out)["code"] == "RUN_LOCKED"

        # 错误入参校验
        assert cli.main(["approve", "--state-dir", str(tmp_path), "--id", "apr_nope"]) == 1
        assert json.loads(capsys.readouterr().out)["code"] == "NO_SUCH_APPROVAL"
        assert cli.main(["approve", "--state-dir", str(tmp_path), "--run-id", "r9", "--ttl", "0"]) == 1
        assert json.loads(capsys.readouterr().out)["code"] == "NOT_APPROVABLE"

        # 按 --id 批准并恢复
        assert cli.main(["approve", "--state-dir", str(tmp_path), "--id", aid, "--by", "why"]) == 0
        approved_out = json.loads(capsys.readouterr().out)
        assert approved_out["approval_id"] == aid and approved_out["status"] == "approved"

        # 已无 pending 再 approve 报错
        assert cli.main(["approve", "--state-dir", str(tmp_path), "--run-id", "r9"]) == 1
        assert json.loads(capsys.readouterr().out)["code"] == "NO_PENDING_APPROVAL"

        # 如果恢复时未传相同的 --sandbox-mode workspace-write（退回默认 read-only），必须吊销并 blocked
        code2 = cli.main([
            "agent", "--state-dir", str(tmp_path), "--base-url", server.base_url,
            "--prompt", "写", "--run-id", "r9", "--ask-tool", "tool:flaky_write*", "--resume",
        ])
        out2 = json.loads(capsys.readouterr().out)
        assert code2 == 4 and out2["status"] == "blocked" and out2["blocked_on"] == aid


def test_cli_approve_rejects_empty_run_id_without_falling_back(tmp_path: Path, capsys) -> None:
    ledger = EffectLedger(tmp_path)
    aid = ledger.request_approval("run-1", 1, "c1", "flaky_write", hash_args({}), "v1")
    assert cli.main(["approve", "--state-dir", str(tmp_path), "--run-id", ""]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "NOT_APPROVABLE"
    assert ledger.get_approval_by_id(aid)["status"] == "pending"


def test_multi_tool_calls_require_per_step_approval(tmp_path: Path) -> None:
    model = ScriptedModel([
        {"tool_calls": [
            {"id": "c1", "name": "flaky_write", "arguments": {"target": "notes/1.md"}},
            {"id": "c2", "name": "flaky_write", "arguments": {"target": "notes/2.md"}},
        ]},
        {"content": "两条都好了"},
    ])
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        cfg = _agent_config(server, sink, rules=[Rule("ask", Decision.PROMPT, "tool:flaky_write*")])
        r1 = AgentRuntime(build_agent_task("写两份", cfg), tmp_path).run()
        assert r1.status is RunStatus.INTERRUPTED
        ledger = EffectLedger(tmp_path)
        first_pending = ledger.approvals(run_id="run-1", status="pending")
        assert len(first_pending) == 1 and first_pending[0]["step_index"] == 1
        ledger.approve(first_pending[0]["approval_id"], ttl_s=60.0, granted_by="why")

        r2 = AgentRuntime(build_agent_task("写两份", cfg), tmp_path).run(resume=True)
        assert r2.status is RunStatus.INTERRUPTED
        second_pending = ledger.approvals(run_id="run-1", status="pending")
        assert len(second_pending) == 1 and second_pending[0]["step_index"] == 2
        ledger.approve(second_pending[0]["approval_id"], ttl_s=60.0, granted_by="why")

        r3 = AgentRuntime(build_agent_task("写两份", cfg), tmp_path).run(resume=True)

    assert r3.status is RunStatus.COMPLETED
    assert len(sink.rows()) == 2 and model.calls == 2


def test_sigkill_after_approval_resumes_without_reasking_human_or_model(tmp_path: Path) -> None:
    from bridle.verify.checks_m7 import _KILLED, _effects, _kill_when, _payload, _run, _written
    from bridle.verify.checks_m8 import _agent

    model = ScriptedModel([_call_write(), {"content": "killed then resumed"}])
    with FakeModelServer(model) as server:
        sd = tmp_path / "state"
        r1 = _run(_agent(sd, server, "--ask-tool", "tool:flaky_write*"))
        assert r1.returncode == 3

        assert _run(["approve", "--state-dir", str(sd), "--run-id", "run-1", "--by", "why"]).returncode == 0

        killed = _kill_when(
            _agent(sd, server, "--ask-tool", "tool:flaky_write*", "--hold-after-write", "flaky_write", "--resume"),
            _written(sd, "effects_log"),
        )
        assert killed == _KILLED

        r2 = _run(_agent(sd, server, "--ask-tool", "tool:flaky_write*", "--resume"))
        payload = _payload(r2)
        tool_rows = [r for r in _effects(sd) if ":flaky_write:" in r["key"]]

    assert r2.returncode == 0 and payload["status"] == "completed"
    assert payload["sink_rows"] == 1 and tool_rows[0]["executions"] == 2
    assert model.calls == 2
