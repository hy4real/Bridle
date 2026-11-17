"""事件日志：append-only，按 run 单调编号。它是真相，checkpoint 只是由它算出来的缓存。

三类事件写在同一张表里（`ledger.sqlite` 的 `events`），共用一个 `seq` 序列：

run 级（runtime 写）    run_started / step_committed / run_stopped / run_resumed /
                        run_finished / projection_repaired / summary
副作用级（账本写）      effect_intent / effect_outcome / effect_resolved
审批级（账本写）        approval_requested / approval_granted / approval_revoked

副作用与审批事件均与对应账本行在**同一个事务**里写：要么都在，要么都不在。
`step_committed` 是一步的提交点——它写进去了，这一步就算做完了，恢复时不再重跑；
checkpoint 缓存写没写成都不影响这一点。

state 的变化按增量记，不记整份快照：
    set     顶层键改成新值
    extend  顶层列表只在尾部追加（模型循环的 messages 每步都是这样）
    del     顶层键被删掉
`messages` 这类只增长的列表如果每步整份重写，日志大小会随步数平方增长。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

RUN_EVENTS = frozenset({"run_started", "step_committed", "run_stopped", "run_resumed",
                        "run_finished", "projection_repaired", "summary"})
EFFECT_EVENTS = frozenset({"effect_intent", "effect_outcome", "effect_resolved"})
APPROVAL_EVENTS = frozenset({"approval_requested", "approval_granted", "approval_revoked"})

EVENTS_SCHEMA = """
    CREATE TABLE events (
        run_id       TEXT NOT NULL,
        seq          INTEGER NOT NULL,       -- 每个 run 从 1 开始，单调、无空洞
        type         TEXT NOT NULL,
        step_index   INTEGER,
        payload_json TEXT NOT NULL,
        ts           REAL NOT NULL,
        PRIMARY KEY (run_id, seq)
    )
"""


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    type: str
    step_index: int | None
    payload: dict[str, Any]
    ts: float


# --- 增量 -----------------------------------------------------------------
def compute_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """只看顶层键。列表如果是「旧列表 + 尾部新元素」，记 extend；其余改动记 set。"""
    delta: dict[str, Any] = {}
    sets: dict[str, Any] = {}
    extends: dict[str, list[Any]] = {}
    for key, value in after.items():
        if key in before and before[key] == value:
            continue
        old = before.get(key)
        if (isinstance(old, list) and isinstance(value, list) and len(value) > len(old)
                and value[: len(old)] == old):
            extends[key] = value[len(old):]
        else:
            sets[key] = value
    removed = sorted(key for key in before if key not in after)
    if sets:
        delta["set"] = sets
    if extends:
        delta["extend"] = extends
    if removed:
        delta["del"] = removed
    return delta


def apply_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """返回新 dict，不改传入的 state。"""
    result = json.loads(json.dumps(state))  # 深拷贝，且只留 JSON 能表达的东西
    for key, value in delta.get("set", {}).items():
        result[key] = value
    for key, tail in delta.get("extend", {}).items():
        current = result.get(key)
        if not isinstance(current, list):
            raise ValueError(f"extend 的目标 {key!r} 不是列表：日志与投影对不上")
        result[key] = current + list(tail)
    for key in delta.get("del", []):
        result.pop(key, None)
    return result


# --- 投影 -----------------------------------------------------------------
def project(events: list[Event]) -> tuple[dict[str, Any] | None, int]:
    """由事件重算 run 的状态。返回 (state, 最后一条事件的 seq)。没有 run_started 返回 (None, 0)。"""
    state: dict[str, Any] | None = None
    last_seq = 0
    for event in events:
        last_seq = event.seq
        if event.type == "run_started":
            state = json.loads(json.dumps(event.payload["state"]))
        elif state is None:
            continue
        elif event.type == "step_committed":
            state = apply_delta(state, event.payload["delta"])
        elif event.type == "summary":
            state["summary"] = event.payload["summary"]
            state["first_kept_seq"] = event.payload["first_kept_seq"]
            state["kept_from_index"] = event.payload["kept_from_index"]
        elif event.type == "run_stopped":
            state["status"] = event.payload["status"]
        elif event.type == "run_resumed":
            state["status"] = "running"
        elif event.type == "run_finished":
            state["status"] = "completed"
        # projection_repaired、effect_* 与 approval_* 不改变 run 的状态
    return state, last_seq


# --- 写 -------------------------------------------------------------------
def append_event(
    conn: sqlite3.Connection,
    run_id: str,
    type_: str,
    *,
    step_index: int | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    """在调用方已经打开的写事务里追加一条事件，返回它的 seq。

    seq 在 `BEGIN IMMEDIATE` 的写锁之内分配，所以多个写者（runtime、账本）不会撞号。
    """
    if type_ not in RUN_EVENTS | EFFECT_EVENTS | APPROVAL_EVENTS:
        raise ValueError(f"未知的事件类型：{type_}")
    seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?",
                       (run_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO events (run_id, seq, type, step_index, payload_json, ts) VALUES (?,?,?,?,?,?)",
        (run_id, seq, type_, step_index,
         json.dumps(payload or {}, ensure_ascii=False, sort_keys=True), time.time()),
    )
    return int(seq)


class EventLog:
    """runtime 用的读写入口。表由 `EffectLedger` 建（同一个库、同一个结构版本）。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def append(self, run_id: str, type_: str, *, step_index: int | None = None,
               payload: dict[str, Any] | None = None) -> int:
        return self.append_many(run_id, [(type_, step_index, payload or {})])[-1]

    def append_many(self, run_id: str, items: list[tuple[str, int | None, dict[str, Any]]]) -> list[int]:
        """一个事务里追加多条（exit 档在 run 结束时一次写掉攒下的事件）。"""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                seqs = [append_event(conn, run_id, type_, step_index=step, payload=payload)
                        for type_, step, payload in items]
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        return seqs

    def read(self, run_id: str) -> list[Event]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
        return [Event(run_id=row["run_id"], seq=row["seq"], type=row["type"],
                      step_index=row["step_index"], payload=json.loads(row["payload_json"]),
                      ts=row["ts"]) for row in rows]

    def exists(self, run_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM events WHERE run_id = ? AND type = 'run_started' LIMIT 1",
                                (run_id,)).fetchone() is not None
