"""会话工作区：隔离单元的落地。

隔离单元是 **session**，不是 user，也不是进程（A1-①）。
每个 session 有自己的 workspace 与 state 目录，路径里带 session_id，
互不可见靠文件系统边界，不靠"约定"。
"""

from __future__ import annotations

from pathlib import Path


class SessionWorkspace:
    def __init__(self, root: Path, session_id: str) -> None:
        self.root = Path(root)
        self.session_id = session_id
        self.base = self.root / "sessions" / session_id
        self.workspace = self.base / "workspace"
        self.state_dir = self.base / "state"

    def ensure(self) -> "SessionWorkspace":
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self

    def visible_entries(self) -> list[str]:
        return sorted(p.name for p in self.workspace.iterdir()) if self.workspace.exists() else []
