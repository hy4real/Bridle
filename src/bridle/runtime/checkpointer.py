"""checkpoint 缓存（SQLite）。

M9 起事件日志（`ledger.sqlite` 的 `events`）是真相，这里只存由日志算出来的投影：
    checkpoints   主键 (thread_id, checkpoint_ns, checkpoint_id)，`last_seq` 记它对应日志的哪一条

主键形状沿用 LangGraph 的 checkpointer（`thread_id / checkpoint_ns / checkpoint_id`），
`langgraph_export.py` 导出时才写成 LangGraph 自己的结构。M1–M8 曾仿 PostgresSaver 多建
blobs 与 writes 两张表，它们只写不读，M9.5 删除。
恢复时缓存与日志对不上，以日志为准并重写缓存（见 `loop.py`）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..effects import StateVersionMismatch
from .types import Checkpoint, Durability

SCHEMA_VERSION = 3  # 2：checkpoints 加 last_seq；3：删掉只写不读的 blobs / writes 两张表

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id            TEXT NOT NULL,
    checkpoint_ns        TEXT NOT NULL DEFAULT '',
    checkpoint_id        TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    ts                   REAL NOT NULL,
    kind                 TEXT NOT NULL DEFAULT 'step',
    state_json           TEXT NOT NULL,
    metadata_json        TEXT NOT NULL DEFAULT '{}',
    last_seq             INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints (thread_id, checkpoint_ns, ts);
"""


class Checkpointer:
    """按 durability 决定"什么时候写"，而不是"写不写"。"""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "checkpoints.sqlite"
        self._lock = threading.Lock()
        self._exit_queue: list[Checkpoint] = []
        self._stopped = False
        with self._connect() as conn:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints'").fetchone()
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if has_table and version != SCHEMA_VERSION:
                conn.close()
                raise StateVersionMismatch(
                    f"{self.db_path} 的结构版本是 {version}，需要 {SCHEMA_VERSION}。"
                    "不做迁移：请换一个新的 state 目录。")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("PRAGMA journal_mode=WAL")

    # --- 连接 ---
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # --- 写入 ---
    def save(self, checkpoint: Checkpoint, durability: Durability) -> bool:
        """返回是否**已经落盘**。exit 档返回 False，表示只进内存队列。"""
        if durability is Durability.EXIT:
            self._exit_queue.append(checkpoint)
            return False
        self._write(checkpoint)
        return True

    def flush(self) -> None:
        """运行结束时把 exit 档攒下的所有 checkpoint 一次写掉。"""
        with self._lock:
            pending, self._exit_queue = self._exit_queue, []
        for checkpoint in pending:
            self._write(checkpoint)

    def _write(self, checkpoint: Checkpoint) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO checkpoints
                   (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    ts, kind, state_json, metadata_json, last_seq)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    checkpoint.thread_id,
                    checkpoint.checkpoint_ns,
                    checkpoint.checkpoint_id,
                    checkpoint.parent_checkpoint_id,
                    checkpoint.ts or time.time(),
                    checkpoint.kind,
                    json.dumps(checkpoint.state, ensure_ascii=False),
                    json.dumps({"writer": "bridle"}, ensure_ascii=False),
                    checkpoint.last_seq,
                ),
            )

    # --- 读取 ---
    def latest(self, thread_id: str) -> Checkpoint | None:
        self.flush()  # exit 档也要能读回来
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM checkpoints WHERE thread_id = ?
                   ORDER BY ts DESC, checkpoint_id DESC LIMIT 1""",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(
            thread_id=row["thread_id"],
            checkpoint_id=row["checkpoint_id"],
            state=json.loads(row["state_json"]),
            parent_checkpoint_id=row["parent_checkpoint_id"],
            checkpoint_ns=row["checkpoint_ns"],
            ts=row["ts"],
            kind=row["kind"],
            last_seq=row["last_seq"],
        )

    def history(self, thread_id: str) -> list[Checkpoint]:
        self.flush()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM checkpoints WHERE thread_id = ?
                   ORDER BY ts ASC, checkpoint_id ASC""",
                (thread_id,),
            ).fetchall()
        return [
            Checkpoint(
                thread_id=row["thread_id"],
                checkpoint_id=row["checkpoint_id"],
                state=json.loads(row["state_json"]),
                parent_checkpoint_id=row["parent_checkpoint_id"],
                checkpoint_ns=row["checkpoint_ns"],
                ts=row["ts"],
                kind=row["kind"],
                last_seq=row["last_seq"],
            )
            for row in rows
        ]

    def tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [row["name"] for row in rows]

    @staticmethod
    def new_checkpoint_id() -> str:
        return uuid.uuid4().hex[:12]
