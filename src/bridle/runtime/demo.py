"""演示任务：一个会写下游的步骤。

它不是业务，是**故障模型的载体**：三步里第二步带副作用，且下游是按键 upsert。
这样才能问出真问题——"副作用执行了、checkpoint 没写就崩了，会发生什么"。
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..effects import Replay
from .types import Step, StepContext, Task


class EffectSink:
    """下游系统的最小替身：按键 upsert（真实系统的幂等写）。"""

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

    def rows(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM effects_log ORDER BY key")]


def build_demo_task(state_dir: Path, *, effect_key: str = "demo-effect-1") -> Task:
    sink = EffectSink(state_dir)

    def plan(ctx: StepContext) -> None:
        ctx.state["plan"] = {"effect_key": effect_key}

    def apply_effect(ctx: StepContext) -> None:
        # 慢下游：延迟放在 action 里，取消才会落在"意图已落、结果未落"的窗口内（S2.3）
        def action() -> dict[str, Any]:
            ctx.sleep_or_cancel(ctx.tool_delay)
            return sink.upsert(effect_key, ctx.state["plan"])

        result = ctx.ledger.attempt(
            effect_key,
            intent={"op": "upsert", "table": "effects_log", "payload": ctx.state["plan"]},
            action=action,
            replay=Replay.SAFE,  # 下游是按键 upsert
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
