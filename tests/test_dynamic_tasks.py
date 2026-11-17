"""动态任务：下一步由 state 决定（M8.2）。模型循环建在这上面。"""

from __future__ import annotations

from pathlib import Path

import pytest

from bridle.runtime import AgentRuntime, Checkpointer, RunFailed, RunStatus
from bridle.runtime.types import Step, StepContext, Task


def _counting_task(limit: int, *, log: list[str] | None = None, crash_on: int | None = None) -> Task:
    """每步把计数加一，计数到 limit 就结束。下一步只看 state，不看外部变量。"""

    def bump(ctx: StepContext) -> None:
        if log is not None:
            log.append(f"bump@{ctx.step_index}")
        if crash_on is not None and ctx.step_index == crash_on:
            raise _ProcessDied()
        ctx.state["count"] = ctx.state.get("count", 0) + 1

    def next_step(state: dict) -> Step | None:
        return None if state.get("count", 0) >= limit else Step(f"bump-{state.get('count', 0)}", bump)

    return Task(name="count", next_step=next_step)


class _ProcessDied(BaseException):
    pass


def test_next_step_drives_the_run_until_it_returns_none(tmp_path: Path) -> None:
    report = AgentRuntime(_counting_task(3), tmp_path).run()

    assert report.status is RunStatus.COMPLETED
    assert report.steps_run == ["bump-0", "bump-1", "bump-2"]
    final = Checkpointer(tmp_path).latest("run-1").state
    assert final["count"] == 3 and final["step_index"] == 3


def test_next_step_returning_none_immediately_completes_with_no_steps(tmp_path: Path) -> None:
    report = AgentRuntime(_counting_task(0), tmp_path).run()
    assert report.status is RunStatus.COMPLETED and report.steps_run == []


def test_resume_asks_next_step_again_from_the_checkpointed_state(tmp_path: Path) -> None:
    log: list[str] = []
    with pytest.raises(_ProcessDied):
        AgentRuntime(_counting_task(4, log=log, crash_on=2), tmp_path).run()

    log.clear()
    report = AgentRuntime(_counting_task(4, log=log), tmp_path).run(resume=True)

    assert report.status is RunStatus.COMPLETED
    assert report.replayed_steps == ["bump-2"]  # 从第 2 步（下标）重跑，前两步不再执行
    assert log == ["bump@2", "bump@3"]
    assert Checkpointer(tmp_path).latest("run-1").state["count"] == 4


def test_run_failed_from_next_step_marks_failed_without_running_a_step(tmp_path: Path) -> None:
    def next_step(state: dict) -> Step | None:
        if state["step_index"] >= 2:
            raise RunFailed("max_steps=2 已用尽")
        return Step("noop", lambda ctx: None)

    report = AgentRuntime(Task("t", next_step=next_step), tmp_path).run()

    latest = Checkpointer(tmp_path).latest("run-1")
    assert report.status is RunStatus.FAILED and "max_steps" in report.error
    assert latest.kind == "failed" and latest.state["step_index"] == 2


def test_run_failed_inside_a_step_saves_the_entry_snapshot(tmp_path: Path) -> None:
    def broken(ctx: StepContext) -> None:
        ctx.state["half"] = True
        raise RunFailed("模型不可用")

    report = AgentRuntime(Task("t", steps=[Step("broken", broken)]), tmp_path).run()

    latest = Checkpointer(tmp_path).latest("run-1")
    assert report.status is RunStatus.FAILED and "模型不可用" in report.error
    assert latest.kind == "failed" and "half" not in latest.state


def test_static_task_still_reports_the_replayed_step(tmp_path: Path) -> None:
    from bridle.runtime import EffectLedger
    from bridle.runtime.events import EventLog

    task = Task("t", steps=[Step("a", lambda ctx: None), Step("b", lambda ctx: None)])
    log = EventLog(EffectLedger(tmp_path).db_path)
    log.append("run-1", "run_started", payload={"state": {
        "run_id": "run-1", "step_index": 0, "steps_done": [], "status": "running"}})
    log.append("run-1", "step_committed", step_index=0, payload={
        "step": "a", "delta": {"set": {"step_index": 1}, "extend": {"steps_done": ["a"]}}})

    report = AgentRuntime(task, tmp_path).run(resume=True)

    assert report.status is RunStatus.COMPLETED and report.replayed_steps == ["b"]


def test_langgraph_backend_refuses_dynamic_tasks(tmp_path: Path) -> None:
    from bridle.runtime.langgraph_backend import LangGraphRuntime

    with pytest.raises(ValueError):
        LangGraphRuntime(_counting_task(1), tmp_path)
