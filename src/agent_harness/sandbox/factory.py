"""后端选择：不可用就抛 SandboxUnavailable——这是 fail-closed 的唯一入口。"""

from __future__ import annotations

import os
from pathlib import Path

from .base import SandboxBackend, SandboxUnavailable
from .seatbelt import SeatbeltBackend
from .unsafe import UnsafeBackend


def build_backend(
    *,
    writable_root: Path | None = None,
    allow_unsandboxed: bool = False,
    allow_network: bool = False,
    probe: bool = True,
) -> SandboxBackend:
    # 测试注入点：模拟「沙箱起不来」，S5.1 要的就是这个分支
    if os.environ.get("HARNESS_SANDBOX_DISABLE") == "1":
        if allow_unsandboxed:
            return UnsafeBackend()
        raise SandboxUnavailable("HARNESS_SANDBOX_DISABLE=1：沙箱被显式关闭")

    backend: SandboxBackend = SeatbeltBackend(writable_root=writable_root, allow_network=allow_network)
    if probe:
        try:
            backend.probe()
        except SandboxUnavailable:
            if allow_unsandboxed:
                return UnsafeBackend()
            raise
    return backend
