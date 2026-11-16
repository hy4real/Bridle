"""验收框架的核心数据结构。

设计约束（来自 specs/a1-a10-acceptance-specs.md）：
- 每条规格的验收必须是**命令**，不是描述；
- 规格只有三种状态：未实现 / 通过 / 失败；
- 任何一条都不得在"没跑过"的情况下被标成通过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Status(str, Enum):
    NOT_IMPLEMENTED = "未实现"
    PASSED = "通过"
    FAILED = "失败"


@dataclass(frozen=True)
class CheckResult:
    """一次验收的结果。

    evidence 是**证据行**：命令输出摘要、断言值、文件路径。面试时要能被指认，
    所以它可以为空，但为空时 status 不可能是 PASSED。
    """

    status: Status
    reason: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is Status.PASSED and not self.evidence:
            raise ValueError("PASSED 必须带证据行：没有证据的通过等于没跑")


def not_implemented(why: str) -> CheckResult:
    """M0 基线用：规格已定义、实现未开始。"""
    return CheckResult(Status.NOT_IMPLEMENTED, why)


def passed(*evidence: str) -> CheckResult:
    return CheckResult(Status.PASSED, "断言全部成立", evidence)


def failed(reason: str, *evidence: str) -> CheckResult:
    return CheckResult(Status.FAILED, reason, evidence)


@dataclass(frozen=True)
class Spec:
    """一条可验收规格。

    id        S2.1 这样的稳定编号，文档与代码共用
    area      面经维度（A1–A10）
    title     要证明的能力（一句话）
    command   当场怎么跑（写进 README 与面试稿的那条命令）
    milestone 由哪个里程碑交付
    check     真正跑的断言函数
    """

    id: str
    area: str
    title: str
    command: str
    milestone: str
    check: Callable[[], CheckResult] = field(default=None)  # type: ignore[assignment]
