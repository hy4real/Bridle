"""策略层：沙箱档位 × 审批策略 × 命令规则。

三层正交（来自 codex 的 `SandboxMode` × `SandboxPolicy` × `AskForApproval` × execpolicy，
见 `sources/github-roadmap-analysis.md` §1.1）：

    sandbox   最多能做到什么（文件/网络边界，由 OS 强制）
    rules     这条命令允不允许（allow / prompt / forbid，确定性匹配）
    approval  需要人点头时怎么办（且 `never` 会把 prompt 降级成 forbid）

关键语义：**deny 永远赢**，allow 不能为 deny 开例外；
`approval=never` 时 `prompt` 变成拒绝，而不是放行——这是「无人值守」的正确含义。
"""

from .types import (
    ApprovalPolicy,
    Decision,
    PolicyVerdict,
    Rule,
    SandboxMode,
)
from .rules import CommandRules
from .engine import PolicyEngine

__all__ = [
    "ApprovalPolicy",
    "CommandRules",
    "Decision",
    "PolicyEngine",
    "PolicyVerdict",
    "Rule",
    "SandboxMode",
]
