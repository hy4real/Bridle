"""副作用账本：意图 + 标识 + 结果，外加「结果未知」这一态。

端到端做不到 exactly-once，实际是 at-least-once + 幂等。账本做四件事：
1. 执行前先落**意图**（outbox 语义）——崩溃后能判断「这一步本来要干什么」；
2. 每次执行记 **attempt_id**——恢复重跑时能数清跑了几次；
3. 执行后落**结果**——重放时直接返回，不必再调下游；
4. 意图已落、结果未落时，按操作声明的 `replay` 决定：
   可重放的重放，**不可重放的记 `unknown` 并拒绝再执行**，直到有人 `resolve`。

状态：
    pending    意图已写，结果未写。恢复时看到它，就是「不知道做没做」。
    completed  结果已写。再来一次直接返回上次结果。
    cancelled  确认没有执行。下一次可以执行。
    unknown    结果未知且不可重放。只有 `resolve()` 能把它带走。

一次 `attempt` 分两个事务：**领取**（读、判、写在一个 `BEGIN IMMEDIATE` 里）和**收尾**
（对 `key + attempt_id` 做比较并交换）。`action` 在两个事务之间执行，不占事务。
所有「置状态然后抛异常」的路径都是先提交再抛——在事务里抛会把刚写的状态回滚掉。

`replay=safe` 的操作仍然要求**下游按键 upsert**（`EffectSink`）：账本负责记录，
sink 负责去重，两者缺一不可。持久性口径：已提交的意图能扛进程被杀，不承诺断电。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from ..effects import (
    EffectOutcomeUnknown,
    IdempotencyKeyConflict,
    Replay,
    StateVersionMismatch,
    VerifyResult,
    stricter,
    validate_run_id,
)
from .events import EVENTS_SCHEMA, append_event
from .types import Cancelled

SCHEMA_VERSION = 3  # 3：加了 events 表（M9）

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE effects (
        key         TEXT PRIMARY KEY,
        intent_json TEXT NOT NULL,
        intent_hash TEXT NOT NULL,
        replay      TEXT NOT NULL,                -- safe | unsafe
        attempt_id  TEXT NOT NULL,                -- 当前持有这一行的那次执行
        status      TEXT NOT NULL,                -- pending | completed | cancelled | unknown
        result_json TEXT,
        attempts    INTEGER NOT NULL DEFAULT 1,   -- 进入账本的次数（去重返回也计）
        executions  INTEGER NOT NULL DEFAULT 0,   -- action 真正被调用的次数
        resolved_by TEXT,
        resolved_at REAL,
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL
    )
    """,
    """
    CREATE TABLE effect_attempts (
        attempt_id  TEXT PRIMARY KEY,
        key         TEXT NOT NULL,
        started_at  REAL NOT NULL,
        finished_at REAL,
        outcome     TEXT
    )
    """,
    EVENTS_SCHEMA,
)


def _effect_event(conn: sqlite3.Connection, key: str, type_: str, **payload: Any) -> None:
    """和账本行在同一个事务里记一条副作用事件。不带作用域的键（只在单测里出现）不记。"""
    run_id, _, rest = key.partition(":")
    step, _, _ = rest.partition(":")
    try:
        validate_run_id(run_id)
        step_index = int(step)
    except ValueError:
        return
    append_event(conn, run_id, type_, step_index=step_index, payload={"key": key, **payload})


@dataclass(frozen=True)
class ApplyResult:
    key: str
    value: Any
    deduplicated: bool
    attempts: int
    executions: int = 0
    resolved: bool = False  # 这一行被人工 resolve 过；resolve applied 且没给结果时 value 是 None
    verified: bool = False  # 这次没有执行：反查下游确认已执行过


@dataclass(frozen=True)
class _Claim:
    """领取事务的结论。先提交，再由调用方决定执行还是抛异常。"""

    kind: str  # run | dedup | unknown | conflict | verify
    attempt_id: str = ""
    replay: Replay = Replay.UNSAFE
    attempts: int = 0
    executions: int = 0
    value: Any = None
    resolved: bool = False
    reason: str = ""


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class EffectLedger:
    def __init__(self, state_dir: Path, *, create: bool = True) -> None:
        """`create=False` 只打开已存在的账本——查询类命令敲错目录时不该凭空建一个空库。"""
        self.state_dir = Path(state_dir)
        self.db_path = self.state_dir / "ledger.sqlite"
        if not create and not self.db_path.exists():
            raise FileNotFoundError(f"没有账本：{self.db_path}")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        conn = self._open()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                has_table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='effects'"
                ).fetchone() is not None
                if has_table and version != SCHEMA_VERSION:
                    # 旧库和全新的库 user_version 都是 0，所以要连「表是否已存在」一起看
                    raise StateVersionMismatch(
                        f"{self.db_path} 的结构版本是 {version}，需要 {SCHEMA_VERSION}。"
                        "不做迁移：请换一个新的 state 目录。"
                    )
                if not has_table:
                    for statement in _SCHEMA_STATEMENTS:
                        conn.execute(statement)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")
        finally:
            conn.close()

    # --- 连接与事务 ---
    def _open(self) -> sqlite3.Connection:
        # isolation_level=None：不要 sqlite3 模块替我们隐式 BEGIN。
        # 隐式 BEGIN 只在第一条写语句前发出，SELECT 会落在事务之外，读和写之间就有缝。
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """一个写事务。IMMEDIATE：进来就拿写锁，读到的行在提交前不会被别人改。"""
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = self._open()
        try:
            yield conn
        finally:
            conn.close()

    # --- 对外入口 ---
    def attempt(
        self,
        key: str,
        intent: dict[str, Any],
        action: Callable[[], Any],
        *,
        replay: Replay,
        verify: Callable[[], VerifyResult] | None = None,
    ) -> ApplyResult:
        """执行一次带幂等键的副作用。`replay` 必填：调用方必须说清这个操作能不能重跑。

        `verify` 是可选的反查：只在「pending + unsafe」时被调用。查到已执行 → 落 completed、
        不再执行；查不到、查不了、反查自己抛错 → 一律 unknown。它不会让任何操作被重放。

        领取时该键的状态 → 结果：
        - 无记录 → 写 pending，执行
        - 意图不同 → `IdempotencyKeyConflict`，不改行
        - completed → 返回上次结果，不执行
        - cancelled → 执行
        - pending + safe → 重放
        - pending + unsafe → 有 `verify` 就先反查，否则置 unknown，`EffectOutcomeUnknown`
        - unknown → `EffectOutcomeUnknown`（有没有 `verify` 都一样，只有 `resolve` 能带走）
        """
        replay = Replay(replay)
        intent_json = json.dumps(intent, ensure_ascii=False, sort_keys=True)
        intent_hash = hashlib.sha256(intent_json.encode("utf-8")).hexdigest()

        claim = self._claim(key, intent_json, intent_hash, replay, can_verify=verify is not None)
        # 领取事务已经提交，下面才允许抛异常
        if claim.kind == "conflict":
            raise IdempotencyKeyConflict(key)
        if claim.kind == "unknown":
            raise EffectOutcomeUnknown(key, claim.reason)
        if claim.kind == "dedup":
            return ApplyResult(
                key=key, value=claim.value, deduplicated=True,
                attempts=claim.attempts, executions=claim.executions, resolved=claim.resolved,
            )
        if claim.kind == "verify":
            assert verify is not None
            return self._verify(key, claim, verify)
        return self._execute(key, claim, action)

    def resolve(
        self,
        key: str,
        outcome: str,
        *,
        resolved_by: str,
        result: Any = None,
    ) -> None:
        """`unknown` 的唯一出口。applied → completed（不再执行）；not_applied → cancelled（下次执行）。

        调用方要保证此刻没有别的进程还在跑这个 run（CLI 用 run 锁保证）。
        """
        if outcome not in ("applied", "not_applied"):
            raise ValueError(f"outcome 只能是 applied / not_applied，实际 {outcome!r}")
        if not resolved_by:
            raise ValueError("resolved_by 不能为空：解除必须留下是谁解除的")
        now = time.time()
        with self._tx() as conn:
            row = conn.execute("SELECT status FROM effects WHERE key = ?", (key,)).fetchone()
            if row is None:
                raise KeyError(f"账本里没有这个键：{key}")
            if row["status"] != "unknown":
                raise ValueError(f"只有 unknown 可以解除，{key} 当前是 {row['status']}")
            if outcome == "applied":
                conn.execute(
                    """UPDATE effects SET status = 'completed', result_json = ?,
                              resolved_by = ?, resolved_at = ?, updated_at = ?
                       WHERE key = ? AND status = 'unknown'""",
                    (json.dumps(result, ensure_ascii=False), resolved_by, now, now, key),
                )
            else:
                conn.execute(
                    """UPDATE effects SET status = 'cancelled',
                              resolved_by = ?, resolved_at = ?, updated_at = ?
                       WHERE key = ? AND status = 'unknown'""",
                    (resolved_by, now, now, key),
                )
            conn.execute(
                """INSERT INTO effect_attempts (attempt_id, key, started_at, finished_at, outcome)
                   VALUES (?,?,?,?,?)""",
                (_new_id(), key, now, now, f"resolved_{outcome}"),
            )
            _effect_event(conn, key, "effect_resolved", outcome=outcome, resolved_by=resolved_by)

    # --- 领取 ---
    def _claim(self, key: str, intent_json: str, intent_hash: str, replay: Replay,
               *, can_verify: bool = False) -> _Claim:
        attempt_id = _new_id()
        now = time.time()

        def audit(conn: sqlite3.Connection, outcome: str) -> None:
            conn.execute(
                """INSERT INTO effect_attempts (attempt_id, key, started_at, finished_at, outcome)
                   VALUES (?,?,?,?,?)""",
                (attempt_id, key, now, now, outcome),
            )
            _effect_event(conn, key, "effect_outcome", attempt_id=attempt_id, outcome=outcome)

        def start(conn: sqlite3.Connection, effective: Replay, executions: int) -> None:
            conn.execute(
                "INSERT INTO effect_attempts (attempt_id, key, started_at, outcome) VALUES (?,?,?,?)",
                (attempt_id, key, now, "started"),
            )
            _effect_event(conn, key, "effect_intent", attempt_id=attempt_id, replay=effective.value,
                          executions=executions, intent_hash=intent_hash)

        with self._tx() as conn:
            row = conn.execute("SELECT * FROM effects WHERE key = ?", (key,)).fetchone()

            if row is None:
                conn.execute(
                    """INSERT INTO effects
                       (key, intent_json, intent_hash, replay, attempt_id, status,
                        attempts, executions, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (key, intent_json, intent_hash, replay.value, attempt_id, "pending",
                     1, 1, now, now),
                )
                start(conn, replay, 1)
                return _Claim("run", attempt_id, replay, attempts=1, executions=1)

            if row["intent_hash"] != intent_hash:
                return _Claim("conflict")

            attempts = row["attempts"] + 1
            executions = row["executions"]
            effective = stricter(Replay(row["replay"]), replay)
            status = row["status"]

            if status == "completed":
                conn.execute(
                    "UPDATE effects SET attempts = ?, updated_at = ? WHERE key = ?",
                    (attempts, now, key),
                )
                audit(conn, "deduplicated")
                value = json.loads(row["result_json"]) if row["result_json"] is not None else None
                return _Claim("dedup", attempts=attempts, executions=executions, value=value,
                              resolved=row["resolved_by"] is not None)

            if status == "unknown":
                conn.execute(
                    "UPDATE effects SET attempts = ?, replay = ?, updated_at = ? WHERE key = ?",
                    (attempts, effective.value, now, key),
                )
                audit(conn, "blocked_unknown")
                return _Claim("unknown", reason="结果未知且尚未解除，拒绝再次执行")

            if status == "pending" and effective is Replay.UNSAFE and can_verify:
                # 先不下结论：出了事务去问下游。行保持 pending——此刻它本来就是「不知道」
                conn.execute(
                    "UPDATE effects SET attempts = ?, replay = ?, updated_at = ? WHERE key = ?",
                    (attempts, effective.value, now, key),
                )
                return _Claim("verify", row["attempt_id"], effective,
                              attempts=attempts, executions=executions)

            if status == "pending" and effective is Replay.UNSAFE:
                # 意图在、结果不在、又不能重跑。attempt_id 保持不变：
                # 第一次执行如果其实还活着，它稍后仍能凭自己的 attempt_id 把这一行落成 completed。
                conn.execute(
                    """UPDATE effects SET status = 'unknown', attempts = ?, replay = ?, updated_at = ?
                       WHERE key = ?""",
                    (attempts, effective.value, now, key),
                )
                audit(conn, "outcome_unknown")
                return _Claim("unknown", reason="意图已落盘、结果未落盘，且操作不可重放")

            # 剩下两种都要执行：cancelled（确认没做过），或 pending + safe（重放）
            if status == "pending":
                conn.execute(
                    """UPDATE effect_attempts SET outcome = 'orphaned', finished_at = ?
                       WHERE attempt_id = ? AND outcome = 'started'""",
                    (now, row["attempt_id"]),
                )
                _effect_event(conn, key, "effect_outcome", attempt_id=row["attempt_id"],
                              outcome="orphaned")
            executions += 1
            conn.execute(
                """UPDATE effects SET status = 'pending', attempt_id = ?, attempts = ?,
                          executions = ?, replay = ?, updated_at = ?
                   WHERE key = ?""",
                (attempt_id, attempts, executions, effective.value, now, key),
            )
            start(conn, effective, executions)
            return _Claim("run", attempt_id, effective, attempts=attempts, executions=executions)

    # --- 反查 ---
    def _verify(self, key: str, claim: _Claim, verify: Callable[[], VerifyResult]) -> ApplyResult:
        """在事务之外问下游。只有 applied 能让这一行离开「不知道」；其余都落 unknown。"""
        try:
            outcome = verify()
            verdict = outcome.verdict if isinstance(outcome, VerifyResult) else "unknown"
            result = outcome.result if isinstance(outcome, VerifyResult) else None
        except Exception as error:  # noqa: BLE001 —— 反查自己坏了，也是「不知道」
            verdict, result = "error", None
            detail = f"反查抛出 {type(error).__name__}"
        else:
            detail = {"not_applied": "反查没有在下游找到这次操作（不等于没执行）",
                      "unknown": "反查无法判断"}.get(verdict, "")

        now = time.time()
        applied = False
        with self._tx() as conn:
            if verdict == "applied":
                cursor = conn.execute(
                    """UPDATE effects SET status = 'completed', result_json = ?, updated_at = ?
                       WHERE key = ? AND attempt_id = ? AND status = 'pending'""",
                    (json.dumps(result, ensure_ascii=False), now, key, claim.attempt_id),
                )
                applied = cursor.rowcount == 1
                if not applied:
                    detail = "反查期间这一行被别的执行改动了"
            else:
                conn.execute(
                    """UPDATE effects SET status = 'unknown', updated_at = ?
                       WHERE key = ? AND attempt_id = ? AND status = 'pending'""",
                    (now, key, claim.attempt_id),
                )
            outcome = ("verified_applied" if applied
                       else f"verify_{'raced' if verdict == 'applied' else verdict}")
            conn.execute(
                """INSERT INTO effect_attempts (attempt_id, key, started_at, finished_at, outcome)
                   VALUES (?,?,?,?,?)""",
                (_new_id(), key, now, now, outcome),
            )
            _effect_event(conn, key, "effect_outcome", attempt_id=claim.attempt_id, outcome=outcome)
        if applied:
            return ApplyResult(key=key, value=result, deduplicated=True, attempts=claim.attempts,
                               executions=claim.executions, verified=True)
        raise EffectOutcomeUnknown(key, f"意图已落盘、结果未落盘，且操作不可重放；{detail}")

    # --- 执行与收尾 ---
    def _execute(self, key: str, claim: _Claim, action: Callable[[], Any]) -> ApplyResult:
        unsafe = claim.replay is Replay.UNSAFE
        try:
            value = action()
        except Cancelled:
            # 取消只有发生在写之前才等于「没做」。不可重放的操作不赌这一点。
            self._finish(key, claim.attempt_id,
                         status="unknown" if unsafe else "cancelled",
                         outcome="cancelled_unknown" if unsafe else "cancelled")
            raise
        except Exception as error:
            # 进程级的终止（KeyboardInterrupt / SystemExit / 被杀）不在这里：那种情况行保持 pending。
            self._finish(key, claim.attempt_id,
                         status="unknown" if unsafe else None,
                         outcome="error_unknown" if unsafe else "error")
            if unsafe:
                raise EffectOutcomeUnknown(
                    key, f"执行中抛出 {type(error).__name__}，无法判断下游是否已生效") from error
            raise

        result_json = json.dumps(value, ensure_ascii=False)
        now = time.time()
        with self._tx() as conn:
            # 比较并交换：只有这一行还是「我这次执行」持有的，才落 completed。
            # unknown 也接受——那是别人观察到 pending 后做的保守标记，而我们手里是第一手结果。
            cursor = conn.execute(
                """UPDATE effects SET status = 'completed', result_json = ?, updated_at = ?
                   WHERE key = ? AND attempt_id = ? AND status IN ('pending', 'unknown')""",
                (result_json, now, key, claim.attempt_id),
            )
            outcome = "applied" if cursor.rowcount == 1 else "stale_applied"
            conn.execute(
                "UPDATE effect_attempts SET finished_at = ?, outcome = ? WHERE attempt_id = ?",
                (now, outcome, claim.attempt_id),
            )
            _effect_event(conn, key, "effect_outcome", attempt_id=claim.attempt_id, outcome=outcome)
        return ApplyResult(key=key, value=value, deduplicated=False,
                           attempts=claim.attempts, executions=claim.executions)

    def _finish(self, key: str, attempt_id: str, *, status: str | None, outcome: str) -> None:
        now = time.time()
        with self._tx() as conn:
            if status is not None:
                conn.execute(
                    """UPDATE effects SET status = ?, updated_at = ?
                       WHERE key = ? AND attempt_id = ? AND status = 'pending'""",
                    (status, now, key, attempt_id),
                )
            conn.execute(
                "UPDATE effect_attempts SET finished_at = ?, outcome = ? WHERE attempt_id = ?",
                (now, outcome, attempt_id),
            )
            _effect_event(conn, key, "effect_outcome", attempt_id=attempt_id, outcome=outcome)

    # --- 查询（验收与排障用）---
    def rows(self, *, status: str | None = None) -> list[dict[str, Any]]:
        query = ("SELECT key, status, attempts, executions, replay, intent_json, attempt_id, "
                 "result_json, resolved_by, resolved_at FROM effects")
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        with self._read() as conn:
            rows = conn.execute(query + " ORDER BY key", params).fetchall()
        return [dict(row) for row in rows]

    def attempts(self, key: str) -> int:
        with self._read() as conn:
            row = conn.execute("SELECT attempts FROM effects WHERE key = ?", (key,)).fetchone()
        return int(row["attempts"]) if row else 0

    def attempts_log(self) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                "SELECT attempt_id, key, outcome FROM effect_attempts ORDER BY started_at, rowid"
            ).fetchall()
        return [dict(row) for row in rows]

    def has_executed(self, key: str) -> bool:
        """S2.4 要的那个判断：这个副作用到底做过没有。"""
        with self._read() as conn:
            row = conn.execute("SELECT status FROM effects WHERE key = ?", (key,)).fetchone()
        return bool(row and row["status"] == "completed")
