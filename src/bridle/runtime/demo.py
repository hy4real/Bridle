"""演示任务：一个会写下游的步骤。

它不是业务，是**故障模型的载体**：三步里第二步带副作用，且下游是按键 upsert。
这样才能问出真问题——"副作用执行了、checkpoint 没写就崩了，会发生什么"。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..effects import Replay, VerifyResult, scoped_key
from .types import Step, StepContext, Task

DEMO_EFFECT_STEP = 1  # apply_effect 在 demo 任务里的下标


def demo_effect_key(run_id: str, effect_key: str = "demo-effect-1") -> str:
    """demo 任务那条副作用在账本里的键。CLI 与验收用它查，不各自拼字符串。"""
    return scoped_key(run_id, DEMO_EFFECT_STEP, effect_key)


class EffectSink:
    """下游系统的最小替身，两种写法：

    upsert  按键 upsert（真实系统的幂等写）。重跑、并发重跑都只留一条 → `Replay.SAFE`。
    append  追加（发消息、扣库存那一类）。每调一次多一行 → `Replay.UNSAFE`。
    """

    def __init__(self, state_dir: Path) -> None:
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = state_dir / "sink.sqlite"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS effects_log (
                       key        TEXT PRIMARY KEY,
                       payload    TEXT NOT NULL,
                       created_at REAL NOT NULL,
                       updated_at REAL NOT NULL
                   )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS append_log (
                       id         INTEGER PRIMARY KEY AUTOINCREMENT,
                       key        TEXT NOT NULL,
                       payload    TEXT NOT NULL,
                       created_at REAL NOT NULL
                   )"""
            )

    def upsert(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO effects_log (key, payload, created_at, updated_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET payload = excluded.payload,
                                                  updated_at = excluded.updated_at""",
                (key, json.dumps(payload, ensure_ascii=False), now, now),
            )
        return {"key": key, "written_at": now}

    def append(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        """不幂等的写：同一个 key 调两次就是两行。"""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO append_log (key, payload, created_at) VALUES (?,?,?)",
                (key, json.dumps(payload, ensure_ascii=False), now),
            )
            row_id = cursor.lastrowid
        return {"key": key, "id": row_id, "appended_at": now}

    def rows(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM effects_log ORDER BY key")]

    def append_rows(self, key: str | None = None) -> list[dict[str, Any]]:
        """给了 key 就是反查：下游有没有这个键的记录。"""
        query, params = "SELECT * FROM append_log", ()
        if key is not None:
            query, params = query + " WHERE key = ?", (key,)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(query + " ORDER BY id", params)]


HOLD_POINTS = ("before-write", "after-write")
_HOLD_LIMIT_S = 30.0
EXIT_HOLD_EXPIRED = 98


def hold_until_killed() -> None:
    """验收用的停顿点：停在这里等着被 SIGKILL，永不返回。

    故意**不可取消**（不走 `sleep_or_cancel`）：要复现的是「进程在这个位置被杀」，
    不是「在这个位置被协作式取消」。两道保险防止变成孤儿进程：
    超过 30 秒自己退出；父进程没了（被 init 收养）也自己退出。
    """
    deadline = time.monotonic() + _HOLD_LIMIT_S
    while time.monotonic() < deadline and os.getppid() != 1:
        time.sleep(0.05)
    os._exit(EXIT_HOLD_EXPIRED)


def build_demo_task(
    state_dir: Path,
    *,
    effect_key: str = "demo-effect-1",
    replay: Replay = Replay.SAFE,
    hold: str | None = None,
    verify: bool = False,
) -> Task:
    """`replay` 决定第二步怎么写下游，也决定它在账本里怎么声明：

    safe    按键 upsert 到 `effects_log`（默认，M1 起的行为）
    unsafe  追加到 `append_log`，重跑一次就多一行

    `hold` 只给验收用：让进程停在写下游之前或之后，等着被杀。
    `verify` 给不可重放的写配一个反查：按键去 `append_log` 里找。
    """
    replay = Replay(replay)
    if hold is not None and hold not in HOLD_POINTS:
        raise ValueError(f"hold 只能是 {HOLD_POINTS}，实际 {hold!r}")
    sink = EffectSink(state_dir)
    unsafe = replay is Replay.UNSAFE

    def plan(ctx: StepContext) -> None:
        ctx.state["plan"] = {"effect_key": effect_key}

    def apply_effect(ctx: StepContext) -> None:
        # 账本键与下游键是同一个带作用域的键：两个 run 各写各的一行，不会互相覆盖
        key = ctx.effect_key(effect_key)

        # 慢下游：延迟放在 action 里，取消才会落在"意图已落、结果未落"的窗口内（S2.3）
        def check_downstream() -> VerifyResult:
            found = sink.append_rows(key)
            if found:
                return VerifyResult("applied", {"key": key, "id": found[0]["id"], "verified": True})
            return VerifyResult("not_applied")

        def action() -> dict[str, Any]:
            ctx.sleep_or_cancel(ctx.tool_delay)
            if hold == "before-write":
                hold_until_killed()
            written = (sink.append if unsafe else sink.upsert)(key, ctx.state["plan"])
            if hold == "after-write":
                hold_until_killed()
            return written

        result = ctx.ledger.attempt(
            key,
            intent={
                "op": "append" if unsafe else "upsert",
                "table": "append_log" if unsafe else "effects_log",
                "payload": ctx.state["plan"],
            },
            action=action,
            replay=replay,
            verify=check_downstream if (verify and unsafe) else None,
        )
        ctx.state["effect"] = {
            "key": result.key,
            "deduplicated": result.deduplicated,
            "attempts": result.attempts,
        }

    def report(ctx: StepContext) -> None:
        ctx.state["report"] = {
            "effect_key": effect_key,
            "done": ctx.state["effect"]["key"],
        }

    return Task(
        name="demo",
        steps=[
            Step("plan", plan),
            Step("apply_effect", apply_effect),
            Step("report", report),
        ],
    )
