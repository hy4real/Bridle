"""消息构造与工具调用解析（M8.3）。模型给的东西一律当不可信输入。"""

from __future__ import annotations

import json

import pytest

from bridle.agent.messages import (
    assistant_message,
    parse_tool_call,
    tool_error,
    tool_result,
)


def _raw(arguments, *, name="flaky_write", call_id="c1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_parse_json_string_arguments() -> None:
    call = parse_tool_call(_raw('{"target": "a"}'))
    assert call.error is None and call.id == "c1" and call.name == "flaky_write"
    assert call.arguments == {"target": "a"}


def test_parse_accepts_already_decoded_arguments_and_empty_string() -> None:
    assert parse_tool_call(_raw({"target": "a"})).arguments == {"target": "a"}
    assert parse_tool_call(_raw("")).arguments == {}


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", "42", '"text"', None])
def test_parse_bad_arguments_never_raises(arguments) -> None:
    call = parse_tool_call(_raw(arguments))
    if arguments is None:
        assert call.error is None and call.arguments == {}  # 缺省等于没有参数
    else:
        assert call.error == "BAD_ARGUMENTS" and call.arguments == {}


@pytest.mark.parametrize("raw", [{}, {"id": "c1"}, {"function": {"name": "x"}}, {"id": "", "function": {"name": "x"}}])
def test_parse_malformed_call_never_raises(raw) -> None:
    assert parse_tool_call(raw).error == "MALFORMED_TOOL_CALL"


def test_assistant_message_keeps_tool_calls_verbatim() -> None:
    calls = [_raw('{"target": "a"}')]
    message = assistant_message("", calls)
    assert message == {"role": "assistant", "content": None, "tool_calls": calls}
    assert assistant_message("完成", []) == {"role": "assistant", "content": "完成"}


def test_tool_result_is_openai_shaped_with_string_content() -> None:
    message = tool_result("c1", {"ok": True}, is_error=False)
    assert message["role"] == "tool" and message["tool_call_id"] == "c1"
    assert isinstance(message["content"], str)
    assert json.loads(message["content"]) == {"is_error": False, "ok": True}


def test_tool_error_carries_code_and_is_error() -> None:
    body = json.loads(tool_error("c9", "POLICY_DENIED", "命中 deny 规则")["content"])
    assert body == {"is_error": True, "code": "POLICY_DENIED", "error": "命中 deny 规则"}
