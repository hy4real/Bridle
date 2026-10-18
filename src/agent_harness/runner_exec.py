"""exec 的执行编排：会话工作区 + 策略 + 沙箱。"""

from __future__ import annotations

from pathlib import Path

from .policy import SandboxMode
from .sandbox import ExecResult, build_backend
from .workspace import SessionWorkspace


def run_exec(
    *,
    session: str,
    command: str,
    state_dir: Path,
    sandbox_mode: SandboxMode,
    deny_read: tuple[str, ...] = (),
    deny_write: tuple[str, ...] = (),
    allow_unsandboxed: bool = False,
    timeout_s: float = 30.0,
) -> tuple[ExecResult, Path]:
    workspace = SessionWorkspace(state_dir, session).ensure()

    writable_root = None if sandbox_mode is SandboxMode.READ_ONLY else workspace.workspace
    backend = build_backend(writable_root=writable_root, allow_unsandboxed=allow_unsandboxed)

    result = backend.execute(
        command,
        cwd=workspace.workspace,
        timeout_s=timeout_s,
        deny_read=deny_read,
        deny_write=deny_write,
    )
    return result, workspace.workspace
