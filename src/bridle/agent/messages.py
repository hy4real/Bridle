"""对话消息的形状：OpenAI chat completions 的那一套。

模型给的工具调用是**不可信输入**：参数可能不是 JSON、可能不是对象、工具名可能不存在。
这些都不该让 run 崩掉——它们是模型的错，应该作为带 `is_error` 的工具结果回灌给模型，
让它自己改。只有 harness 自己的问题（工具协议失败、模型不可用）才终止 run。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """一个解析过的工具调用。`error` 非空时 `arguments` 不可用，不要执行。"""

    id: str
    name: str
    arguments: dict[str, Any]
    error: str | None = None


def parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    """把模型返回的一个 `tool_calls` 元素解析成 ToolCall。永不抛异常。"""
    call_id = str(raw.get("id") or "")
    function = raw.get("function") or {}
    name = str(function.get("name") or "")
    if not call_id or not name:
        return ToolCall(id=call_id, name=name, arguments={}, error="MALFORMED_TOOL_CALL")
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return ToolCall(id=call_id, name=name, arguments=arguments)
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return ToolCall(id=call_id, name=name, arguments={}, error="BAD_ARGUMENTS")
    if not isinstance(parsed, dict):
        return ToolCall(id=call_id, name=name, arguments={}, error="BAD_ARGUMENTS")
    return ToolCall(id=call_id, name=name, arguments=parsed)


def assistant_message(text: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """模型这一轮的回复，原样留在 transcript 里——包括 `tool_calls`，下一轮的工具结果要对得上。"""
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def tool_result(call_id: str, payload: dict[str, Any], *, is_error: bool) -> dict[str, Any]:
    """回灌给模型的工具结果。OpenAI 的 tool 消息没有 is_error 字段，所以放进 content 里。"""
    body = {"is_error": is_error, **payload}
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps(body, ensure_ascii=False, sort_keys=True)}


def tool_error(call_id: str, code: str, detail: str) -> dict[str, Any]:
    """模型这次调用没法执行：参数坏了、工具不存在、被策略拒绝。"""
    return tool_result(call_id, {"code": code, "error": detail}, is_error=True)
