"""runtime 的单元测试（不跑子进程，只测组件契约）。

跨进程的崩溃/取消行为在 `verify/checks_m1.py` 里（S2.1–S2.4），
这里测的是组件本身：三表结构、durability 语义、幂等账本。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bridle.effects import Replay
from bridle.runtime import Checkpointer, Durability, EffectLedger
from bridle.runtime.checkpointer import _SCHEMA
from bridle.runtime.types import Checkpoint


def test_checkpointer_is_a_single_cache_table(tmp_path: Path) -> None:
    cp = Checkpointer(tmp_path)
    assert cp.tables() == ["checkpoints"]


def test_checkpoint_roundtrip_and_history(tmp_path: Path) -> None:
    cp = Checkpointer(tmp_path)
    first = Checkpoint(thread_id="t1", checkpoint_id="c1", state={"step_index": 0}, ts=1.0)
    second = Checkpoint(thread_id="t1", checkpoint_id="c2", state={"step_index": 1},
                        parent_checkpoint_id="c1", ts=2.0)
    cp.save(first, Durability.SYNC)
    cp.save(second, Durability.SYNC)

    latest = cp.latest("t1")
    assert latest is not None and latest.state["step_index"] == 1
    assert latest.parent_checkpoint_id == "c1"
    assert [c.checkpoint_id for c in cp.history("t1")] == ["c1", "c2"]


def test_exit_durability_does_not_persist_until_flush(tmp_path: Path) -> None:
    cp = Checkpointer(tmp_path)
    checkpoint = Checkpoint(thread_id="t2", checkpoint_id="c1", state={"x": 1}, ts=1.0)

    assert cp.save(checkpoint, Durability.EXIT) is False
    raw = sqlite3.connect(tmp_path / "checkpoints.sqlite")
    before = raw.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id='t2'").fetchone()[0]
    raw.close()
    assert before == 0  # exit 档：没 flush 就不该落盘

    cp.flush()
    assert cp.latest("t2") is not None


def test_ledger_dedups_completed_effect(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    calls = []

    def action() -> dict:
        calls.append(1)
        return {"written": len(calls)}

    first = ledger.attempt("k1", {"op": "write"}, action, replay=Replay.SAFE)
    second = ledger.attempt("k1", {"op": "write"}, action, replay=Replay.SAFE)

    assert first.deduplicated is False and second.deduplicated is True
    assert len(calls) == 1  # 第二次没有再调下游
    assert ledger.attempts("k1") == 2
    assert ledger.has_executed("k1") is True


def test_ledger_records_intent_before_action(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)

    def action() -> dict:
        # 执行期间，意图必须已经落盘——崩溃后才知道"本来要干什么"
        rows = ledger.rows()
        assert rows and rows[0]["status"] == "pending"
        return {"ok": True}

    ledger.attempt("k2", {"op": "write", "payload": "x"}, action, replay=Replay.SAFE)
    rows = ledger.rows()
    assert rows[0]["status"] == "completed"
    assert rows[0]["intent_json"]
    assert rows[0]["attempt_id"]


def test_ledger_marks_cancelled(tmp_path: Path) -> None:
    from bridle.runtime.types import Cancelled

    ledger = EffectLedger(tmp_path)

    def action() -> dict:
        raise Cancelled("stop")

    with pytest.raises(Cancelled):
        ledger.attempt("k3", {"op": "write"}, action, replay=Replay.SAFE)

    rows = ledger.rows()
    assert rows[0]["status"] == "cancelled"
    assert ledger.has_executed("k3") is False
