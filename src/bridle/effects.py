"""副作用的公共词汇：可重放性声明与相关异常。

放在包根而不是 `runtime/` 或 `tools/` 下：两边都要用，而现有依赖方向是
`tools → runtime`，不该为了一个枚举反过来。本模块不导入包内任何其他模块。
"""

from __future__ import annotations

from enum import Enum


class Replay(str, Enum):
    """工具作者要回答的那个问题：这个操作重跑一次会怎样。

    safe    重复执行下游结果不变，**并发重复也不变**：只读，或下游按键 upsert。
    unsafe  重复执行会产生第二份结果：追加、扣减、发消息。

    没有声明的带副作用操作一律按 unsafe 处理（fail-closed）。
    """

    SAFE = "safe"
    UNSAFE = "unsafe"


def stricter(first: Replay, second: Replay) -> Replay:
    """两处声明不一致时取严的那个。

    账本行里存着第一次执行时的声明；恢复时调用方可能漏传或传错，
    只要任何一边说过 unsafe，就按 unsafe 处理。
    """
    if Replay.UNSAFE in (first, second):
        return Replay.UNSAFE
    return Replay.SAFE


class EffectOutcomeUnknown(RuntimeError):
    """副作用的结果无法判定：意图已落盘，结果没有落盘，且操作不可重放。

    **故意不继承 `ToolProtocolError`**：协议失败会被 `ToolRunner` 重试，
    而结果未知恰恰是不许重试的那一类。
    """

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"{key}: {reason}")
        self.key = key
        self.reason = reason


class IdempotencyKeyConflict(RuntimeError):
    """同一个幂等键配了不同的意图。返回旧结果会答非所问，所以直接报错。"""

    def __init__(self, key: str) -> None:
        super().__init__(f"{key}: 同一个幂等键对应了不同的意图")
        self.key = key


class StateVersionMismatch(RuntimeError):
    """state 目录是旧结构写的。不做迁移：换一个新目录。"""
