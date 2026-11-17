"""blocked、步入口快照、run 锁、effect 命令（M7.5）。全部在进程内；真 SIGKILL 在 verify/checks_m7.py。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridle import cli
from bridle.effects import Replay, RunLocked, run_id_of
from bridle.runtime import AgentRuntime, Checkpointer, Durability, EffectLedger, RunStatus
from bridle.runtime.lock import run_lock
from bridle.runtime.types import Cancelled, NodeInterrupt, Step, StepContext, Task


class _ProcessDied(BaseException):
    """进程死亡的替身：runtime 接不到它，账本行停在 pending，这一步也没有 checkpoint。"""


def _task(log: list[str], *, die: bool) -> Task:
    """三步：prepare → 一条不可重放的副作用 → finish。`die` 让副作用在执行中途「进程死亡」。"""

    def prepare(ctx: StepContext) -> None:
        ctx.state["prepared"] = True

    def charge(ctx: StepContext) -> None:
        ctx.state["scratch"] = "写了一半的状态"  # 步骤在副作用之前就改了 state

        def action() -> dict:
            log.append("charged")
            if die:
                raise _ProcessDied()
            return {"charged": True}

        ctx.ledger.attempt(ctx.effect_key("charge"), {"op": "charge", "amount": 1}, action,
                           replay=Replay.UNSAFE)
        ctx.state["charged"] = True

    def finish(ctx: StepContext) -> None:
        ctx.state["finished"] = True

    return Task(name="t", steps=[Step("prepare", prepare), Step("charge", charge),
                                 Step("finish", finish)])


def _crash_mid_effect(state_dir: Path, log: list[str]) -> None:
    with pytest.raises(_ProcessDied):
        AgentRuntime(_task(log, die=True), state_dir).run()
    assert EffectLedger(state_dir).rows()[0]["status"] == "pending"


def test_unknown_outcome_blocks_the_run_and_saves_the_step_entry_snapshot(tmp_path: Path) -> None:
    log: list[str] = []
    _crash_mid_effect(tmp_path, log)

    report = AgentRuntime(_task(log, die=False), tmp_path).run(resume=True)

    assert report.status is RunStatus.BLOCKED
    assert report.blocked_on == "run-1:1:charge"
    assert report.steps_run == ["prepare"]
    assert log == ["charged"]  # 恢复时没有再执行
    latest = Checkpointer(tmp_path).latest("run-1")
    assert latest is not None and latest.kind == "blocked"
    assert latest.state["status"] == "blocked" and latest.state["step_index"] == 1
    assert "scratch" not in latest.state  # 半步修改没有被存下来
    assert latest.state["prepared"] is True


def test_blocked_run_stays_blocked_until_resolved_then_completes(tmp_path: Path) -> None:
    log: list[str] = []
    _crash_mid_effect(tmp_path, log)

    for _ in range(2):
        assert AgentRuntime(_task(log, die=False), tmp_path).run(resume=True).status is RunStatus.BLOCKED
    ledger = EffectLedger(tmp_path)
    assert ledger.rows()[0]["status"] == "unknown" and ledger.rows()[0]["executions"] == 1

    ledger.resolve("run-1:1:charge", "applied", resolved_by="why")
    report = AgentRuntime(_task(log, die=False), tmp_path).run(resume=True)

    assert report.status is RunStatus.COMPLETED
    assert report.steps_run == ["charge", "finish"]  # steps_run 只报本次执行的步骤
    assert log == ["charged"]  # 确认已执行 → 不再执行
    final = Checkpointer(tmp_path).latest("run-1").state
    assert final["steps_done"] == ["prepare", "charge", "finish"] and final["charged"] is True


def test_resolve_not_applied_runs_the_effect_exactly_once_more(tmp_path: Path) -> None:
    log: list[str] = []
    _crash_mid_effect(tmp_path, log)
    assert AgentRuntime(_task(log, die=False), tmp_path).run(resume=True).status is RunStatus.BLOCKED

    EffectLedger(tmp_path).resolve("run-1:1:charge", "not_applied", resolved_by="why")
    report = AgentRuntime(_task(log, die=False), tmp_path).run(resume=True)

    assert report.status is RunStatus.COMPLETED
    assert log == ["charged", "charged"]
    assert EffectLedger(tmp_path).rows()[0]["executions"] == 2


def test_blocked_checkpoint_is_flushed_under_exit_durability(tmp_path: Path) -> None:
    log: list[str] = []
    # exit 档下崩溃什么 checkpoint 都没留；同一个 run id 重新跑，走到同一个键，照样 blocked
    with pytest.raises(_ProcessDied):
        AgentRuntime(_task(log, die=True), tmp_path, durability=Durability.EXIT).run()
    report = AgentRuntime(_task(log, die=False), tmp_path, durability=Durability.EXIT).run()

    assert report.status is RunStatus.BLOCKED and log == ["charged"]
    assert Checkpointer(tmp_path).latest("run-1").kind == "blocked"  # 另开一个连接也读得到


@pytest.mark.parametrize("stop, status", [(NodeInterrupt, "interrupted"), (Cancelled, "cancelled")])
def test_interrupt_and_cancel_also_save_the_step_entry_snapshot(
    tmp_path: Path, stop: type[Exception], status: str
) -> None:
    def half(ctx: StepContext) -> None:
        ctx.state["half"] = True
        raise stop()

    report = AgentRuntime(Task("t", [Step("half", half)]), tmp_path).run()

    latest = Checkpointer(tmp_path).latest("run-1")
    assert report.status.value == status and latest.kind == status
    assert "half" not in latest.state


def test_resume_after_last_step_checkpoint_but_before_final_does_not_index_error(tmp_path: Path) -> None:
    from bridle.runtime.events import EventLog

    task = Task("t", [Step("only", lambda ctx: None)])
    # 最后一步已提交、run_finished 还没写就被杀：step_index 等于步数
    log = EventLog(EffectLedger(tmp_path).db_path)
    log.append("run-1", "run_started", payload={"state": {
        "run_id": "run-1", "step_index": 0, "steps_done": [], "status": "running"}})
    log.append("run-1", "step_committed", step_index=0, payload={
        "step": "only", "delta": {"set": {"step_index": 1}, "extend": {"steps_done": ["only"]}}})

    report = AgentRuntime(task, tmp_path).run(resume=True)

    assert report.status is RunStatus.COMPLETED and report.replayed_steps == []


# --- run 锁 ---------------------------------------------------------------
def test_run_lock_excludes_a_second_holder_and_releases_on_exit(tmp_path: Path) -> None:
    with run_lock(tmp_path, "run-1"):
        with pytest.raises(RunLocked):
            with run_lock(tmp_path, "run-1"):
                pass
        with run_lock(tmp_path, "other-run"):  # 不同的 run 互不影响
            pass
    with run_lock(tmp_path, "run-1"):  # 释放后可以再拿
        pass


def test_cli_run_exits_7_when_the_run_is_held_elsewhere(tmp_path: Path, capsys) -> None:
    with run_lock(tmp_path, "run-1"):
        code = cli.main(["run", "--state-dir", str(tmp_path)])
    assert code == cli.EXIT_RUN_LOCKED
    assert json.loads(capsys.readouterr().out)["code"] == "RUN_LOCKED"
    assert Checkpointer(tmp_path).latest("run-1") is None  # 没有跑任何一步


# --- effect 命令 -----------------------------------------------------------
def test_cli_effect_list_on_a_mistyped_dir_errors_without_creating_a_ledger(
    tmp_path: Path, capsys
) -> None:
    typo = tmp_path / "typo"
    assert cli.main(["effect", "list", "--state-dir", str(typo)]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "NO_LEDGER"
    assert not typo.exists()

    assert cli.main(["effect", "resolve", "--state-dir", str(typo), "--key", "run-1:1:k",
                     "--outcome", "applied"]) == 1
    assert not typo.exists()


def test_cli_effect_list_and_resolve_roundtrip(tmp_path: Path, capsys) -> None:
    log: list[str] = []
    _crash_mid_effect(tmp_path, log)
    AgentRuntime(_task(log, die=False), tmp_path).run(resume=True)

    assert cli.main(["effect", "list", "--state-dir", str(tmp_path), "--status", "unknown"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["count"] == 1
    entry = listing["effects"][0]
    assert entry["key"] == "run-1:1:charge" and entry["replay"] == "unsafe"
    assert entry["intent"] == {"amount": 1, "op": "charge"}

    # run 还被持有时不许解除
    with run_lock(tmp_path, "run-1"):
        assert cli.main(["effect", "resolve", "--state-dir", str(tmp_path), "--key", entry["key"],
                         "--outcome", "applied", "--by", "why"]) == cli.EXIT_RUN_LOCKED
    capsys.readouterr()
    assert EffectLedger(tmp_path).rows()[0]["status"] == "unknown"

    assert cli.main(["effect", "resolve", "--state-dir", str(tmp_path), "--key", entry["key"],
                     "--outcome", "applied", "--by", "why"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved == {"key": entry["key"], "outcome": "applied", "status": "completed",
                        "resolved_by": "why"}

    # 已经不是 unknown 了，再解除一次是错误
    assert cli.main(["effect", "resolve", "--state-dir", str(tmp_path), "--key", entry["key"],
                     "--outcome", "not-applied"]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "NOT_RESOLVABLE"
    assert cli.main(["effect", "resolve", "--state-dir", str(tmp_path), "--key", "run-1:9:nope",
                     "--outcome", "applied"]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "NO_SUCH_EFFECT"


def test_run_id_of_reads_the_scope_back() -> None:
    assert run_id_of("run-1:1:demo-effect-1") == "run-1"
    assert run_id_of("r.2:10:a:b:c") == "r.2"
    with pytest.raises(ValueError):
        run_id_of("no-scope")
