"""回归集、并发上限、双 runtime 互操作的单元测试。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from bridle.concurrency import BoundedRunner
from bridle.eval.suite import FailureMode, run_suite, summarize


def test_suite_has_twenty_cases_and_all_pass() -> None:
    results = run_suite()
    assert len(results) == 20
    summary = summarize(results)
    assert summary["failed"] == 0, summary["failures"]
    assert summary["pass_rate"] == 1.0


def test_suite_is_deterministic() -> None:
    assert summarize(run_suite()) == summarize(run_suite())


def test_failure_modes_are_enumerated() -> None:
    assert FailureMode.QUOTA_OVERSOLD.value == "quota_oversold"
    assert len({mode.value for mode in FailureMode}) == len(FailureMode)


def test_bounded_runner_respects_limit() -> None:
    runner = BoundedRunner(limit=2)
    runner.map([lambda: time.sleep(0.03) for _ in range(6)])
    assert runner.stats.max_in_flight <= 2
    assert runner.stats.completed == 6
    assert runner.stats.queue_waits == 6


def test_bounded_runner_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        BoundedRunner(0)


@pytest.mark.parametrize("runtime_name", ["native", "langgraph"])
def test_demo_task_runs_on_both_runtimes(runtime_name: str, tmp_path: Path) -> None:
    from bridle.runtime import AgentRuntime, Durability
    from bridle.runtime.demo import EffectSink, build_demo_task

    task = build_demo_task(tmp_path)
    runtime = (AgentRuntime(task, tmp_path, durability=Durability.SYNC)
               if runtime_name == "native"
               else __import__("bridle.runtime.langgraph_backend",
                               fromlist=["LangGraphRuntime"]).LangGraphRuntime(task, tmp_path))
    report = runtime.run()
    assert report.status.value == "completed"
    assert report.steps_run == ["plan", "apply_effect", "report"]
    assert len(EffectSink(tmp_path).rows()) == 1


def test_langgraph_export_roundtrip(tmp_path: Path) -> None:
    from bridle.runtime.langgraph_export import export_checkpoint, import_checkpoint, tables
    from bridle.runtime.types import Checkpoint

    checkpoint = Checkpoint(thread_id="t9", checkpoint_id="c9",
                            state={"step_index": 2, "steps_done": ["plan", "apply_effect"]})
    db = export_checkpoint(checkpoint, tmp_path / "lg")
    assert tables(db) == ["checkpoints", "writes"]  # SQLite saver 是两表
    back = import_checkpoint(db, "t9")
    assert back.state == checkpoint.state
