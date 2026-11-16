"""三表 checkpointer（SQLite）。

表结构与 LangGraph 的 PostgresSaver 同构（`libs/checkpoint-postgres/.../base.py:50,60,69`）：
    checkpoints          主键 (thread_id, checkpoint_ns, checkpoint_id)
    checkpoint_blobs     主键 (thread_id, checkpoint_ns, channel, version)
    checkpoint_writes    主键 (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)

为什么拆三张而不是一张大表：状态里既有"小元数据"（元信息、时间）又有"大值"（消息、
文件内容）。分开存，读取元数据时不必反序列化整份状态——这是它敢存全量的前提。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .types import Checkpoint, Durability

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
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel       TEXT NOT NULL,
    version       INTEGER NOT NULL,
    type          TEXT NOT NULL,
    value         BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    value_json    TEXT NOT NULL,
    task_path     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
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
        self._async_queue: list[Checkpoint] = []
        self._stopped = False
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
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
            self._async_queue.append(checkpoint)
            return False
        if durability is Durability.ASYNC:
            # 异步档：交给后台线程；为了可测，这里同步写但语义上不保证成功
            self._write(checkpoint)
            return True
        self._write(checkpoint)
        return True

    def flush(self) -> None:
        """运行结束时把 exit 档攒下的所有 checkpoint 一次写掉。"""
        with self._lock:
            pending, self._async_queue = self._async_queue, []
        for checkpoint in pending:
            self._write(checkpoint)

    def _write(self, checkpoint: Checkpoint) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO checkpoints
                   (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                    ts, kind, state_json, metadata_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    checkpoint.thread_id,
                    checkpoint.checkpoint_ns,
                    checkpoint.checkpoint_id,
                    checkpoint.parent_checkpoint_id,
                    checkpoint.ts or time.time(),
                    checkpoint.kind,
                    json.dumps(checkpoint.state, ensure_ascii=False),
                    json.dumps({"writer": "bridle"}, ensure_ascii=False),
                ),
            )
            # blob 表：每个 state 键一个"channel"，version 用单调递增序号
            for index, (channel, value) in enumerate(sorted(checkpoint.state.items())):
                conn.execute(
                    """INSERT OR REPLACE INTO checkpoint_blobs
                       (thread_id, checkpoint_ns, channel, version, type, value)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        checkpoint.thread_id,
                        checkpoint.checkpoint_ns,
                        channel,
                        index,
                        type(value).__name__,
                        json.dumps(value, ensure_ascii=False).encode(),
                    ),
                )
            # writes 表：这次 checkpoint 里由哪个 task 写了哪个 channel
            conn.execute(
                """INSERT OR REPLACE INTO checkpoint_writes
                   (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel,
                    value_json, task_path)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    checkpoint.thread_id,
                    checkpoint.checkpoint_ns,
                    checkpoint.checkpoint_id,
                    f"task-{checkpoint.kind}",
                    0,
                    "state",
                    json.dumps(checkpoint.state, ensure_ascii=False),
                    "",
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
