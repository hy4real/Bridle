"""沙箱后端的契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class SandboxUnavailable(RuntimeError):
    """后端不可用。上层必须据此 fail-closed，而不是降级裸跑。"""


@dataclass(frozen=True)
class SandboxCaps:
    fs_read: bool = True
    fs_write: bool = False
    net: bool = False

    def missing(self, required: "SandboxCaps") -> list[str]:
        gaps = []
        for field_name in ("fs_read", "fs_write", "net"):
            if getattr(required, field_name) and not getattr(self, field_name):
                gaps.append(field_name)
        return gaps


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    denied: bool = False
    backend: str = "unknown"
    extra: dict[str, str] = field(default_factory=dict)


class SandboxBackend(Protocol):
    name: str

    def capabilities(self) -> SandboxCaps: ...

    def probe(self) -> None: ...

    def execute(self, command: str, *, cwd: Path, timeout_s: float = 30.0,
                deny_read: tuple[str, ...] = (), deny_write: tuple[str, ...] = ()) -> ExecResult: ...
