"""内置工具：只用于验收与演示，不是业务。

两种失败刻意做成可对比的形态：
- 业务失败：工具**跑完了**，返回 `{is_error: true}` —— 回给模型自己处理；
- 协议失败：参数不合 schema / 工具崩了 / 超时 —— harness 处理，不进模型循环。
"""

from __future__ import annotations

import time
from typing import Any

from ..effects import Replay
from .base import Tool


def note_read() -> Tool:
    def handler(path: str) -> dict[str, Any]:
        if path == "missing":
            return {"is_error": True, "error": "note not found", "path": path}
        return {"path": path, "content": "hello"}

    return Tool(
        name="note_read",
        description="读取一条笔记，文件不存在时返回业务错误",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        handler=handler,
        tags=("file", "read"),
    )


def crashing_tool() -> Tool:
    def handler(path: str) -> dict[str, Any]:
        raise ValueError(f"internal explode while reading {path}")

    return Tool(
        name="explode",
        description="总是抛出异常的工具，用来区分协议失败",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        handler=handler,
        tags=("debug",),
    )


def flaky_write_tool(sink: Any, *, slow_first: float = 0.5) -> Tool:
    """第一次：先写下游、再装死（模拟"响应丢失但已执行"）；第二次：正常返回。"""
    state = {"calls": 0}

    def handler(target: str) -> dict[str, Any]:
        state["calls"] += 1
        payload = {"target": target, "call": state["calls"]}
        sink.upsert(f"write:{target}", payload)  # 下游按键 upsert
        if state["calls"] == 1:
            time.sleep(slow_first)
        return {"ok": True, "written": f"write:{target}"}

    return Tool(
        name="flaky_write",
        description="第一次调用会超时但副作用已生效的写工具",
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
        handler=handler,
        side_effect=True,
        idempotency_key=lambda args: f"flaky_write:{args['target']}",
        replay=Replay.SAFE,  # 下游按键 upsert：重跑、并发重跑都只留一条
        tags=("write",),
    )
