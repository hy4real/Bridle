"""工具：不是 API 包装，是给模型用的界面（ACI）。

三条设计约束（来自 SWE-agent 的 ACI 与面经 A4）：
1. **工具定义吃 token**：定义注入 system message 并按输入计费，所以要有预算；
2. **失败分两类**：业务失败（工具跑完但结果不理想）与协议失败（调用本身坏了），
   两者必须走不同路径——前者回给模型自己处理，后者是 harness 的问题；
3. **返回要短**：长输出截断、空输出显式化（"没有输出"不等于"失败"）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class OutcomeKind(str, Enum):
    OK = "ok"
    BUSINESS_ERROR = "business_error"


class ToolProtocolError(RuntimeError):
    """协议失败：参数不合 schema、工具崩了、超时。这类不走模型循环，由 harness 处理。"""


def estimate_tokens(text: str) -> int:
    """粗略 token 估算，只用于**预算**，不用于计费。

    口径：CJK 约 1 字 1 token，ASCII 约 4 字符 1 token。
    误差 ±20%，所以预算要留余量；真要计费仍以模型返回的 usage 为准。
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, math.ceil(cjk + other / 4))


@dataclass(frozen=True)
class ToolOutcome:
    kind: OutcomeKind
    payload: dict[str, Any]
    tool: str
    attempts: int = 1

    @property
    def is_error(self) -> bool:
        return self.kind is OutcomeKind.BUSINESS_ERROR

    def to_tool_message(self) -> dict[str, Any]:
        """回灌给模型的消息。业务失败用 is_error 标记，**传输层仍是正常 payload**。"""
        return {"role": "tool", "tool": self.tool, "content": self.payload,
                "is_error": self.is_error}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema 的最小子集：type/properties/required
    handler: Callable[..., dict[str, Any]]
    side_effect: bool = False
    idempotency_key: Callable[[dict[str, Any]], str] | None = None
    max_output_chars: int = 2000
    tags: tuple[str, ...] = field(default_factory=tuple)

    # --- 定义（会被计入 token 预算）---
    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def definition_tokens(self) -> int:
        import json

        return estimate_tokens(json.dumps(self.definition(), ensure_ascii=False))

    # --- 校验与执行 ---
    def validate(self, args: dict[str, Any]) -> None:
        required = self.parameters.get("required", [])
        missing = [key for key in required if key not in args]
        if missing:
            raise ToolProtocolError(f"{self.name}: 缺少必填参数 {missing}")
        properties = self.parameters.get("properties", {})
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected == "string" and not isinstance(value, str):
                raise ToolProtocolError(f"{self.name}: 参数 {key} 应为 string，实际 {type(value).__name__}")

    def render(self, payload: dict[str, Any]) -> dict[str, Any]:
        """返回给模型的内容：超长截断，空输出显式化。"""
        import json

        text = json.dumps(payload, ensure_ascii=False)
        if not text.strip() or payload == {}:
            return {"ok": True, "note": "命令成功执行，没有任何输出"}
        if len(text) > self.max_output_chars:
            return {"truncated": True, "head": text[: self.max_output_chars], "total_chars": len(text)}
        return payload

    def run(self, args: dict[str, Any]) -> ToolOutcome:
        self.validate(args)
        try:
            payload = self.handler(**args)
        except ToolProtocolError:
            raise
        except Exception as error:  # 工具崩了 = 协议失败，不是业务失败
            raise ToolProtocolError(f"{self.name} 执行异常：{type(error).__name__}: {error}") from error

        is_error = bool(payload.get("is_error"))
        rendered = self.render(payload)
        return ToolOutcome(
            kind=OutcomeKind.BUSINESS_ERROR if is_error else OutcomeKind.OK,
            payload=rendered,
            tool=self.name,
        )
