"""macOS Seatbelt 后端：`/usr/bin/sandbox-exec` + 动态生成的 SBPL profile。

一个必须踩过才知道的坑：**路径要先 realpath**。
`/tmp` 其实是 `/private/tmp`，`/var/folders/...` 其实是 `/private/var/folders/...`，
SBPL 按真实路径匹配——不做 realpath 的话，deny 规则**永远不生效，而且不报错**，
只表现为"看起来没拦住"。这是排障最难的一类问题，所以本文件把它写在最上面。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import ExecResult, SandboxCaps, SandboxUnavailable

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_DENY_MARKERS = (
    "Operation not permitted",
    "Permission denied",
    "PermissionError",
    "EPERM",
    "EACCES",
)


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


class SeatbeltBackend:
    name = "seatbelt"

    def __init__(self, *, writable_root: Path | None = None, allow_network: bool = False) -> None:
        self.writable_root = Path(writable_root) if writable_root else None
        self.allow_network = allow_network

    def capabilities(self) -> SandboxCaps:
        return SandboxCaps(
            fs_read=True,
            fs_write=self.writable_root is not None,
            net=self.allow_network,
        )

    def probe(self) -> None:
        if not Path(_SANDBOX_EXEC).exists():
            raise SandboxUnavailable("找不到 /usr/bin/sandbox-exec（非 macOS 或已被移除）")
        try:
            result = subprocess.run(
                [_SANDBOX_EXEC, "-p", self._profile((), ()), "/bin/echo", "ok"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SandboxUnavailable(f"sandbox-exec 无法执行：{error}") from error
        if result.returncode != 0 or "ok" not in result.stdout:
            raise SandboxUnavailable(
                f"sandbox-exec 自检失败：rc={result.returncode} stderr={result.stderr.strip()[:200]}"
            )

    def _profile(self, deny_read: tuple[str, ...], deny_write: tuple[str, ...]) -> str:
        parts = ["(version 1)", "(allow default)"]
        for path in deny_read:
            parts.append(f'(deny file-read* (subpath "{_real(path)}"))')
        for path in deny_write:
            parts.append(f'(deny file-write* (subpath "{_real(path)}"))')
        if self.writable_root is None:
            parts.append("(deny file-write*)")  # read-only 档：全局禁写
        if not self.allow_network:
            parts.append("(deny network*)")
        return "\n".join(parts)

    def execute(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_s: float = 30.0,
        deny_read: tuple[str, ...] = (),
        deny_write: tuple[str, ...] = (),
    ) -> ExecResult:
        profile = self._profile(deny_read, deny_write)
        try:
            result = subprocess.run(
                [_SANDBOX_EXEC, "-p", profile, "/bin/sh", "-c", command],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"timeout after {timeout_s}s", False, self.name)
        blob = result.stdout + result.stderr
        return ExecResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            denied=any(marker in blob for marker in _DENY_MARKERS),
            backend=self.name,
        )
