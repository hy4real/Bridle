"""副作用账本：意图 + 标识 + 结果，三件套。

这是 A2 那条反直觉结论的落地：
**端到端做不到 exactly-once，实际是 at-least-once + 幂等。**
所以账本不做"阻止重复执行"，它做三件事：
1. 执行前先落**意图**（outbox 语义）——崩溃后能判断"这一步本来要干什么"；
2. 每次执行记 **attempt_id**——恢复重跑时能数清跑了几次；
3. 执行后落**结果**——重放时可以直接返回，不必再调下游。

搭配的纪律：**下游写操作必须是按键 upsert**（`EffectSink`），
账本负责记录，sink 负责去重，两者缺一不可。
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .types import Cancelled

_SCHEMA = """
CREATE TABLE IF NOT EXISTS effects (
    key         TEXT PRIMARY KEY,
    intent_json TEXT NOT NULL,
    attempt_id  TEXT NOT NULL,
    status      TEXT NOT NULL,            -- pending | completed | cancelled
    result_json TEXT,
    attempts    INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS effect_attempts (
    attempt_id  TEXT PRIMARY KEY,
    key         TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    outcome     TEXT
);
"""


@dataclass(frozen=True)
class ApplyResult:
    key: str
    value: Any
    deduplicated: bool
    attempts: int


class EffectLedger:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "ledger.sqlite"
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def attempt(self, key: str, intent: dict[str, Any], action: Callable[[], Any]) -> ApplyResult:
        """执行一次带幂等键的副作用。

        - 已完成 → 不执行 action，直接返回上次结果（deduplicated=True）
        - 待定（上次崩在执行中）→ 重放 action，attempts+1（at-least-once）
        - 新键 → 先落意图，再执行
        """
        attempt_id = uuid.uuid4().hex[:12]
        now = time.time()

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM effects WHERE key = ?", (key,)).fetchone()
            if row is not None and row["status"] == "completed":
                conn.execute(
                    "UPDATE effects SET attempts = attempts + 1, updated_at = ? WHERE key = ?",
                    (now, key),
                )
                conn.execute(
                    """INSERT INTO effect_attempts
                       (attempt_id, key, started_at, finished_at, outcome)
                       VALUES (?,?,?,?,?)""",
                    (attempt_id, key, now, now, "deduplicated"),
                )
                return ApplyResult(
                    key=key,
                    value=json.loads(row["result_json"]),
                    deduplicated=True,
                    attempts=row["attempts"] + 1,
                )

            if row is None:
                conn.execute(
                    """INSERT INTO effects
                       (key, intent_json, attempt_id, status, attempts, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (key, json.dumps(intent, ensure_ascii=False), attempt_id,
                     "pending", 1, now, now),
                )
                attempts = 1
            else:  # pending：上次执行到一半崩了，这次是重放
                attempts = row["attempts"] + 1
                conn.execute(
                    "UPDATE effects SET attempts = ?, attempt_id = ?, updated_at = ? WHERE key = ?",
                    (attempts, attempt_id, now, key),
                )

            conn.execute(
                """INSERT INTO effect_attempts
                   (attempt_id, key, started_at, outcome) VALUES (?,?,?,?)""",
                (attempt_id, key, now, "started"),
            )

        # 意图已经落盘，这里才真正调下游
        try:
            value = action()
        except Cancelled:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE effects SET status = 'cancelled', updated_at = ? WHERE key = ?",
                    (time.time(), key),
                )
                conn.execute(
                    "UPDATE effect_attempts SET finished_at = ?, outcome = 'cancelled' WHERE attempt_id = ?",
                    (time.time(), attempt_id),
                )
            raise

        with self._connect() as conn:
            conn.execute(
                "UPDATE effects SET status = 'completed', result_json = ?, updated_at = ? WHERE key = ?",
                (json.dumps(value, ensure_ascii=False), time.time(), key),
            )
            conn.execute(
                "UPDATE effect_attempts SET finished_at = ?, outcome = 'applied' WHERE attempt_id = ?",
                (time.time(), attempt_id),
            )
        return ApplyResult(key=key, value=value, deduplicated=False, attempts=attempts)

    # --- 查询（验收与排障用）---
    def rows(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, status, attempts, intent_json, attempt_id, result_json FROM effects ORDER BY key"
            ).fetchall()
        return [dict(row) for row in rows]

    def attempts(self, key: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT attempts FROM effects WHERE key = ?", (key,)).fetchone()
        return int(row["attempts"]) if row else 0

    def attempts_log(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT attempt_id, key, outcome FROM effect_attempts ORDER BY started_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def has_executed(self, key: str) -> bool:
        """S2.4 要的那个判断：这个副作用到底做过没有。"""
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM effects WHERE key = ?", (key,)).fetchone()
        return bool(row and row["status"] == "completed")
