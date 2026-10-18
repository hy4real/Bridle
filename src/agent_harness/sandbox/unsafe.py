"""显式放弃沙箱的后端。

只在 `--allow-unsandboxed` 时启用，且每次执行都打印警告——
它的存在是为了让「没有沙箱」这件事**可见**，而不是让程序悄悄裸跑。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import ExecResult, SandboxCaps


class UnsafeBackend:
    name = "unsafe-local"

    def capabilities(self) -> SandboxCaps:
        return SandboxCaps(fs_read=True, fs_write=True, net=True)

    def probe(self) -> None:
        return None

    def execute(self, command: str, *, cwd: Path, timeout_s: float = 30.0,
                deny_read: tuple[str, ...] = (), deny_write: tuple[str, ...] = ()) -> ExecResult:
        print("[harness][WARN] 未启用沙箱：命令直接在宿主执行", file=sys.stderr)
        try:
            result = subprocess.run(
                ["/bin/sh", "-c", command], cwd=str(cwd),
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", f"timeout after {timeout_s}s", False, self.name)
        return ExecResult(result.returncode, result.stdout, result.stderr, False, self.name)
