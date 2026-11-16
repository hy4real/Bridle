"""键作用域与新 run 守卫（M7.3）。

作用域解决「不同 run 撞同一个键」；守卫解决「同一个 run id 被当成新 run 再跑一遍」。
两者缺一个，跨 run 误去重都还在。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bridle.effects import Replay, RunAlreadyExists, scoped_key, validate_run_id
from bridle.runtime import AgentRuntime, EffectLedger
from bridle.runtime.demo import EffectSink, build_demo_task, demo_effect_key
from bridle.tools.base import Tool
from bridle.tools.builtin import flaky_write_tool, note_read
from bridle.tools.runner import ToolRunner


def test_scoped_key_format_and_no_cross_scope_collision() -> None:
    assert scoped_key("run-1", 1, "demo-effect-1") == "run-1:1:demo-effect-1"
    # 业务键里带冒号也不会和别的 (run, step) 撞：前两段不含冒号
    assert scoped_key("a", 1, "2:x") != scoped_key("a", 12, "x")
    assert demo_effect_key("r9") == "r9:1:demo-effect-1"


@pytest.mark.parametrize("bad", ["a:1", "", "has space", "斜/杠", None])
def test_run_id_charset_is_enforced(bad: object) -> None:
    with pytest.raises(ValueError):
        validate_run_id(bad)  # type: ignore[arg-type]  # 原因：这条测的就是传错类型时被拒
    with pytest.raises(ValueError):
        scoped_key(bad, 0, "k")  # type: ignore[arg-type]  # 原因：同上


@pytest.mark.parametrize("step", [-1, True, "1", 1.0])
def test_scoped_key_rejects_non_index_steps(step: object) -> None:
    with pytest.raises(ValueError):
        scoped_key("run-1", step, "k")  # type: ignore[arg-type]  # 原因：测的就是非下标值被拒


def test_scoped_key_rejects_empty_business_key() -> None:
    with pytest.raises(ValueError):
        scoped_key("run-1", 0, "")


def test_two_runs_in_one_state_dir_do_not_dedup_each_other(tmp_path: Path) -> None:
    for run_id in ("r1", "r2"):
        report = AgentRuntime(build_demo_task(tmp_path), tmp_path, run_id=run_id).run()
        assert report.status.value == "completed"

    rows = EffectLedger(tmp_path).rows()
    assert [row["key"] for row in rows] == ["r1:1:demo-effect-1", "r2:1:demo-effect-1"]
    assert [row["executions"] for row in rows] == [1, 1]
    assert len(EffectSink(tmp_path).rows()) == 2  # 下游也是各写各的一行


def test_fresh_run_with_an_existing_run_id_is_refused_native(tmp_path: Path) -> None:
    AgentRuntime(build_demo_task(tmp_path), tmp_path).run()
    with pytest.raises(RunAlreadyExists):
        AgentRuntime(build_demo_task(tmp_path), tmp_path).run()
    # 被拒的那次什么都没动
    assert EffectLedger(tmp_path).rows()[0]["attempts"] == 1
    # resume 仍然可以：已完成的 run 直接返回
    assert AgentRuntime(build_demo_task(tmp_path), tmp_path).run(resume=True).status.value == "completed"


def test_fresh_run_with_an_existing_run_id_is_refused_langgraph(tmp_path: Path) -> None:
    from bridle.runtime.langgraph_backend import LangGraphRuntime

    first = LangGraphRuntime(build_demo_task(tmp_path), tmp_path)
    first.run()
    first.close()
    second = LangGraphRuntime(build_demo_task(tmp_path), tmp_path)
    try:
        with pytest.raises(RunAlreadyExists):
            second.run()
    finally:
        second.close()
    assert EffectLedger(tmp_path).rows()[0]["attempts"] == 1


def test_runtimes_reject_malformed_run_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AgentRuntime(build_demo_task(tmp_path), tmp_path, run_id="a:b")


def test_tool_runner_scopes_keys_by_run_and_step(tmp_path: Path) -> None:
    sink = EffectSink(tmp_path)
    ledger = EffectLedger(tmp_path)
    tool = flaky_write_tool(sink, slow_first=0.0)

    ToolRunner(ledger, run_id="r1").call(tool, {"target": "a"}, step_index=3)
    ToolRunner(ledger, run_id="r2").call(tool, {"target": "a"}, step_index=3)
    ToolRunner(ledger, run_id="r1").call(tool, {"target": "a"}, step_index=4)

    assert [row["key"] for row in ledger.rows()] == [
        "r1:3:flaky_write:a", "r1:4:flaky_write:a", "r2:3:flaky_write:a"]
    assert all(row["executions"] == 1 for row in ledger.rows())


def test_tool_runner_refuses_side_effects_without_ledger_or_scope(tmp_path: Path) -> None:
    sink = EffectSink(tmp_path)
    tool = flaky_write_tool(sink, slow_first=0.0)

    with pytest.raises(RuntimeError):
        ToolRunner().call(tool, {"target": "a"})
    with pytest.raises(RuntimeError):
        ToolRunner(EffectLedger(tmp_path)).call(tool, {"target": "a"})  # 有账本、没 run id
    assert sink.rows() == []  # 两次都没有写下游

    # 只读工具不需要账本
    assert ToolRunner().call(note_read(), {"path": "notes/a.md"}).tool == "note_read"


def test_undeclared_side_effect_tool_defaults_to_unsafe() -> None:
    write = Tool(name="w", description="", parameters={"type": "object"},
                 handler=lambda: {}, side_effect=True)
    read = Tool(name="r", description="", parameters={"type": "object"}, handler=lambda: {})
    assert write.replay_mode() is Replay.UNSAFE
    assert read.replay_mode() is Replay.SAFE
    assert flaky_write_tool(sink=None).replay_mode() is Replay.SAFE
