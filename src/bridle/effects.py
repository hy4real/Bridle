"""副作用的公共词汇：可重放性声明与相关异常。

放在包根而不是 `runtime/` 或 `tools/` 下：两边都要用，而现有依赖方向是
`tools → runtime`，不该为了一个枚举反过来。本模块不导入包内任何其他模块。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Replay(str, Enum):
    """工具作者要回答的那个问题：这个操作重跑一次会怎样。

    safe    重复执行下游结果不变，**并发重复也不变**：只读，或下游按键 upsert。
    unsafe  重复执行会产生第二份结果：追加、扣减、发消息。

    没有声明的带副作用操作一律按 unsafe 处理（fail-closed）。
    """

    SAFE = "safe"
    UNSAFE = "unsafe"


VERIFY_VERDICTS = ("applied", "not_applied", "unknown")


@dataclass(frozen=True)
class VerifyResult:
    """反查下游得到的结论。

    applied      下游确实有这次操作的记录。账本据此落 completed，不再执行。
    not_applied  下游没查到。**不等于没执行**：请求可能还在路上，或下游有延迟。
    unknown      查不了，或查到的东西无法判断。

    后两种账本一律按结果未知处理，不会因此重放。反查只能证明「已执行」。
    `result` 是 applied 时可用的结果，拿不到可以留空。
    """

    verdict: str
    result: Any = None

    def __post_init__(self) -> None:
        if self.verdict not in VERIFY_VERDICTS:
            raise ValueError(f"verdict 只能是 {VERIFY_VERDICTS}，实际 {self.verdict!r}")


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


class ApprovalRevoked(RuntimeError):
    """已有审批记录失效（策略变化、过期或已吊销）。不再自动重问或放行，停在 blocked。"""

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


class RunAlreadyExists(RuntimeError):
    """这个 run id 在这个 state 目录里已经跑过。要接着跑用 resume，要重新跑换一个 run id。

    没有这道守卫，键作用域就是摆设：两次都用默认 run id，键仍然一样，第二次照样命中第一次的结果。
    """

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run_id={run_id} 已存在：继续执行请用 resume，重新执行请换一个 run id")
        self.run_id = run_id


class RunLocked(RuntimeError):
    """这个 run 正被另一个进程持有。"""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run_id={run_id} 正被另一个进程持有")
        self.run_id = run_id


_RUN_ID = re.compile(r"[A-Za-z0-9._-]+")


def validate_run_id(run_id: str) -> str:
    """run id 要拼进键里，所以不许含分隔符 `:`，否则不同的 (run, step) 能拼出同一个键。"""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"run_id 只能由字母、数字、点、下划线、连字符组成，实际 {run_id!r}")
    return run_id


def scoped_key(run_id: str, step_index: int, business_key: str) -> str:
    """账本键的唯一生成处：`{run_id}:{step_index}:{business_key}`。

    前两段不含 `:`（run id 有字符集限制，step_index 是非负整数），所以从左切两刀一定还原得回来，
    业务键里带 `:` 也不会和别的作用域撞。同一步内相同的业务键会合并成一个副作用，这是有意的。
    """
    validate_run_id(run_id)
    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
        raise ValueError(f"step_index 必须是非负整数，实际 {step_index!r}")
    if not business_key:
        raise ValueError("business_key 不能为空")
    return f"{run_id}:{step_index}:{business_key}"


def run_id_of(key: str) -> str:
    """从带作用域的键里取回 run id（`scoped_key` 的逆运算的第一段）。"""
    run_id, separator, _ = key.partition(":")
    if not separator:
        raise ValueError(f"不是带作用域的键：{key!r}")
    return validate_run_id(run_id)
