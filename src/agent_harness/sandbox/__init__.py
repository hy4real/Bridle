"""沙箱后端：可替换的执行边界。

纪律（对应 design/harness-blocks.md §5 D3）：
1. 起不来就 fail-closed——由上层拒绝启动，不许悄悄裸跑；
2. 能力缺失就摘掉对应工具，不假装能跑；
3. 换后端（Docker / 远端）只实现同一个 Protocol，不改上层。
"""

from .base import ExecResult, SandboxBackend, SandboxCaps, SandboxUnavailable
from .factory import build_backend

__all__ = [
    "ExecResult",
    "SandboxBackend",
    "SandboxCaps",
    "SandboxUnavailable",
    "build_backend",
]
