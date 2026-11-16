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
