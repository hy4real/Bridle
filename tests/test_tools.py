"""工具层：预算、渲染、失败分层。"""

from __future__ import annotations

import pytest

from bridle.tools.base import Tool, ToolProtocolError, estimate_tokens
from bridle.tools.builtin import crashing_tool, note_read
from bridle.tools.registry import build_fixture_registry


def test_fixture_has_sixty_tools_and_they_cost_tokens() -> None:
    registry = build_fixture_registry(60)
    assert len(registry) == 60
    assert registry.total_definition_tokens() > 1000


def test_selection_respects_budget_and_relevance() -> None:
    registry = build_fixture_registry(60)
    selection = registry.select("读取文件 read file", budget_tokens=300)
    assert selection.count < 60
    assert selection.tokens <= 300
    assert all(tool.name.startswith("file_") for tool in selection.tools)


def test_bigger_budget_selects_more() -> None:
    registry = build_fixture_registry(60)
    small = registry.select("file", budget_tokens=120)
    large = registry.select("file", budget_tokens=600)
    assert large.count > small.count


def test_token_estimate_is_ordered_by_length() -> None:
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("hello world this is longer") > estimate_tokens("hi")


def test_empty_output_is_explicit() -> None:
    tool = Tool(name="noop", description="无输出", parameters={"type": "object"}, handler=lambda: {})
    outcome = tool.run({})
    assert outcome.payload.get("note", "").startswith("命令成功执行")


def test_long_output_is_truncated() -> None:
    tool = Tool(
        name="long", description="长输出", parameters={"type": "object"},
        handler=lambda: {"text": "x" * 5000}, max_output_chars=100,
    )
    outcome = tool.run({})
    assert outcome.payload.get("truncated") is True
    assert outcome.payload.get("total_chars") == len('{"text": "' + "x" * 5000 + '"}')


def test_business_error_is_not_exception() -> None:
    outcome = note_read().run({"path": "missing"})
    assert outcome.is_error is True
    assert outcome.kind.value == "business_error"


def test_tool_crash_becomes_protocol_error() -> None:
    with pytest.raises(ToolProtocolError):
        crashing_tool().run({"path": "x"})


def test_missing_required_arg_is_protocol_error() -> None:
    with pytest.raises(ToolProtocolError):
        note_read().run({})


def test_wrong_type_is_protocol_error() -> None:
    with pytest.raises(ToolProtocolError):
        note_read().run({"path": 123})


# --- 超时与重试（M7.4）-----------------------------------------------------
def _runner_fixture(tmp_path):
    from bridle.runtime import EffectLedger
    from bridle.runtime.demo import EffectSink

    return EffectSink(tmp_path), EffectLedger(tmp_path)


def test_timeout_bounds_wall_time_even_when_the_tool_keeps_running() -> None:
    import time

    from bridle.tools.runner import ToolRunner

    slow = Tool(name="slow", description="", parameters={"type": "object"},
                handler=lambda: time.sleep(1.5) or {"ok": True})
    started = time.monotonic()
    with pytest.raises(ToolProtocolError):
        ToolRunner(retries=0, timeout_s=0.05).call(slow, {})
    assert time.monotonic() - started < 0.5  # 不等那 1.5 秒


def test_unsafe_tool_timeout_is_not_retried_and_ledger_says_unknown(tmp_path) -> None:
    from bridle.effects import EffectOutcomeUnknown
    from bridle.tools.builtin import flaky_append_tool
    from bridle.tools.runner import ToolRunner

    sink, ledger = _runner_fixture(tmp_path)
    tool = flaky_append_tool(sink, slow=0.6)

    with pytest.raises(EffectOutcomeUnknown) as info:
        ToolRunner(ledger, run_id="r1", retries=2, timeout_s=0.05).call(tool, {"target": "a"})

    row = ledger.rows()[0]
    assert isinstance(info.value.__cause__, ToolProtocolError)  # 起因是超时
    assert row["status"] == "unknown" and row["executions"] == 1
    assert len(sink.append_rows()) == 1  # 重试一次就会是 2 行

    # 再调一次也不执行：unknown 要先解除
    with pytest.raises(EffectOutcomeUnknown):
        ToolRunner(ledger, run_id="r1", retries=2, timeout_s=0.05).call(tool, {"target": "a"})
    assert len(sink.append_rows()) == 1


def test_safe_tool_timeout_is_retried_and_converges(tmp_path) -> None:
    from bridle.tools.builtin import flaky_write_tool
    from bridle.tools.runner import ToolRunner

    sink, ledger = _runner_fixture(tmp_path)
    outcome = ToolRunner(ledger, run_id="r1", retries=2, timeout_s=0.05).call(
        flaky_write_tool(sink, slow_first=0.3), {"target": "a"})

    row = ledger.rows()[0]
    assert outcome.kind.value == "ok"
    assert row["status"] == "completed" and row["executions"] == 2
    assert len(sink.rows()) == 1  # 下游按键 upsert，执行两次仍是一行


def test_resolved_applied_effect_yields_a_synthetic_outcome(tmp_path) -> None:
    from bridle.effects import EffectOutcomeUnknown
    from bridle.tools.builtin import flaky_append_tool
    from bridle.tools.runner import ToolRunner

    sink, ledger = _runner_fixture(tmp_path)
    tool = flaky_append_tool(sink, slow=0.3)
    runner = ToolRunner(ledger, run_id="r1", retries=2, timeout_s=0.05)
    with pytest.raises(EffectOutcomeUnknown):
        runner.call(tool, {"target": "a"})

    ledger.resolve(ledger.rows()[0]["key"], "applied", resolved_by="why")
    outcome = runner.call(tool, {"target": "a"})

    assert outcome.kind.value == "ok" and outcome.tool == "flaky_append"
    assert "人工确认" in outcome.payload["note"]
    assert len(sink.append_rows()) == 1  # 没有再执行


def test_tool_verify_turns_a_timed_out_unsafe_call_into_a_result_on_the_next_call(tmp_path) -> None:
    """工具自带反查：第一次超时 → unknown 之前先留 pending；这里模拟进程重启后的第二次调用。"""
    import dataclasses

    from bridle.effects import EffectOutcomeUnknown, Replay, VerifyResult
    from bridle.tools.builtin import flaky_append_tool
    from bridle.tools.runner import ToolRunner

    sink, ledger = _runner_fixture(tmp_path)

    def verify(args: dict, key: str) -> VerifyResult:
        found = sink.append_rows(f"append:{args['target']}")
        return VerifyResult("applied", {"ok": True, "appended": found[0]["id"]}) if found \
            else VerifyResult("not_applied")

    tool = dataclasses.replace(flaky_append_tool(sink, slow=0.3), verify=verify)
    key = "r1:0:flaky_append:a"

    # 造出「意图已写、下游已写、结果未写」：直接在账本里留一条 pending，下游补一行
    class _Died(BaseException):
        pass

    def die() -> dict:
        sink.append("append:a", {"target": "a"})
        raise _Died()

    with pytest.raises(_Died):
        ledger.attempt(key, {"tool": "flaky_append", "args": {"target": "a"}}, die,
                       replay=Replay.UNSAFE)

    outcome = ToolRunner(ledger, run_id="r1", retries=2, timeout_s=0.05).call(tool, {"target": "a"})

    assert outcome.kind.value == "ok" and outcome.payload == {"ok": True, "appended": 1}
    assert len(sink.append_rows()) == 1  # 没有再执行
    assert ledger.rows()[0]["status"] == "completed"

    # 没写过的情况：反查不到 → 未知，不执行
    other = "r1:0:flaky_append:b"
    with pytest.raises(_Died):
        ledger.attempt(other, {"tool": "flaky_append", "args": {"target": "b"}},
                       lambda: (_ for _ in ()).throw(_Died()), replay=Replay.UNSAFE)
    with pytest.raises(EffectOutcomeUnknown):
        ToolRunner(ledger, run_id="r1", retries=2, timeout_s=0.05).call(tool, {"target": "b"})
    assert sink.append_rows("append:b") == []
