"""checkpoint 的跨 runtime 互操作。

一个必须澄清的事实（本机实测）：**LangGraph 的 SQLite saver 是两张表**
（`checkpoints` + `writes`），只有 Postgres saver 才是三表（多一张 blobs 表）。
所以「导出成三表」这个说法只在 Postgres 后端成立；这里导出到官方 SQLite saver，
并直接用它自己的 `put()` / `get_tuple()` 做读回——**互操作经过的是对方代码，不是我们照抄的 DDL**。
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .types import Checkpoint

LANGGRAPH_SQLITE_TABLES = ("checkpoints", "writes")


def export_checkpoint(checkpoint: Checkpoint, out_dir: Path) -> Path:
    """把一个 native checkpoint 写进 langgraph 的 SQLite saver。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "langgraph.sqlite"

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    channel_versions = {channel: "1" for channel in checkpoint.state}
    saver.put(
        {"configurable": {"thread_id": checkpoint.thread_id, "checkpoint_ns": ""}},
        {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(checkpoint.ts or time.time())),
            "channel_values": dict(checkpoint.state),
            "channel_versions": channel_versions,
            "versions_seen": {},
            "updated_channels": None,
        },
        {"source": "bridle", "step": checkpoint.state.get("step_index", 0)},
        channel_versions,
    )
    conn.close()
    return db_path


def import_checkpoint(db_path: Path, thread_id: str) -> Checkpoint:
    """用 langgraph 自己的读取路径把 checkpoint 读回来。"""
    conn = sqlite3.connect(Path(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    bundle = saver.get_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
    conn.close()
    if bundle is None:
        raise LookupError(f"langgraph 里没有这个 thread：{thread_id}")

    payload = bundle.checkpoint
    return Checkpoint(
        thread_id=thread_id,
        checkpoint_id=payload["id"],
        state=dict(payload["channel_values"]),
        checkpoint_ns=payload.get("channel_ns", ""),
        kind="imported",
    )


def tables(db_path: Path) -> list[str]:
    conn = sqlite3.connect(Path(db_path))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    conn.close()
    return [row[0] for row in rows]
