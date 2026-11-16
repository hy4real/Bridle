"""策略层的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SandboxMode(str, Enum):
    """沙箱能做什么。默认 read-only——最严的一档。"""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"


class ApprovalPolicy(str, Enum):
    """谁来点头。`never` 把 prompt 降级为拒绝，不是放行。"""

    UNLESS_TRUSTED = "unless-trusted"
    ON_REQUEST = "on-request"
    NEVER = "never"


class Decision(str, Enum):
    ALLOW = "allow"
    PROMPT = "prompt"
    FORBID = "forbid"


@dataclass(frozen=True)
class Rule:
    """一条命令规则。pattern 支持 fnmatch 通配（`rm -rf *`）。"""

    id: str
    decision: Decision
    pattern: str
    note: str = ""


@dataclass(frozen=True)
class PolicyVerdict:
    """判定结果。必须说清**是哪一层、哪条规则**做的决定——排障要用。"""

    decision: Decision
    layer: str          # rules | approval | sandbox | default
    rule_id: str | None
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW
