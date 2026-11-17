"""事件日志（M9.2）：增量、投影、编号、与账本行同事务。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bridle.effects import Replay, StateVersionMismatch, scoped_key
from bridle.runtime import EffectLedger
from bridle.runtime import ledger as ledger_module
from bridle.runtime.events import (
    APPROVAL_EVENTS,
    Event,
    EventLog,
    apply_delta,
    compute_delta,
    project,
)


# --- 增量 -----------------------------------------------------------------
@pytest.mark.parametrize("before, after, expected", [
    ({"a": 1}, {"a": 1}, {}),
    ({"a": 1}, {"a": 2}, {"set": {"a": 2}}),
    ({}, {"a": [1]}, {"set": {"a": [1]}}),
    ({"m": [1, 2]}, {"m": [1, 2, 3, 4]}, {"extend": {"m": [3, 4]}}),          # 只追加
    ({"m": [1, 2]}, {"m": [9, 2, 3]}, {"set": {"m": [9, 2, 3]}}),             # 前缀变了
    ({"m": [1, 2]}, {"m": [1]}, {"set": {"m": [1]}}),                          # 变短了
    ({"a": 1, "b": 2}, {"a": 1}, {"del": ["b"]}),
])
def test_delta_shapes_and_roundtrip(before: dict, after: dict, expected: dict) -> None:
    delta = compute_delta(before, after)
    assert delta == expected
    assert apply_delta(before, delta) == after


def test_apply_delta_does_not_mutate_its_input() -> None:
    before = {"m": [1]}
    apply_delta(before, {"extend": {"m": [2]}, "set": {"x": 1}})
    assert before == {"m": [1]}


def test_extend_onto_a_non_list_is_an_inconsistency_not_a_silent_fix() -> None:
    with pytest.raises(ValueError):
        apply_delta({"m": "text"}, {"extend": {"m": [1]}})


def test_growing_transcript_is_logged_linearly() -> None:
    """messages 每步只追加：每条增量只带新消息，不带整份历史。"""
    state: dict = {"messages": []}
    sizes = []
    for i in range(20):
        after = {"messages": state["messages"] + [{"role": "user", "content": f"m{i}"}]}
        delta = compute_delta(state, after)
        sizes.append(len(delta["extend"]["messages"]))
        state = apply_delta(state, delta)
    assert sizes == [1] * 20 and len(state["messages"]) == 20


# --- 投影 -----------------------------------------------------------------
def _ev(seq: int, type_: str, payload: dict | None = None) -> Event:
    return Event(run_id="r", seq=seq, type=type_, step_index=None, payload=payload or {}, ts=0.0)


def test_project_folds_run_events_and_ignores_effect_events() -> None:
    events = [
        _ev(1, "run_started", {"state": {"step_index": 0, "steps_done": [], "status": "running"}}),
        _ev(2, "effect_intent", {"key": "r:0:k"}),
        _ev(3, "step_committed", {"delta": {"set": {"step_index": 1}, "extend": {"steps_done": ["a"]}}}),
        _ev(4, "run_stopped", {"status": "blocked"}),
        _ev(5, "run_resumed"),
        _ev(6, "projection_repaired", {"reason": "x"}),
        _ev(7, "step_committed", {"delta": {"set": {"step_index": 2}, "extend": {"steps_done": ["b"]}}}),
        _ev(8, "run_finished"),
    ]
    state, last_seq = project(events)
    assert state == {"step_index": 2, "steps_done": ["a", "b"], "status": "completed"}
    assert last_seq == 8
    assert project(events[:4])[0]["status"] == "blocked"
    assert project(events[:5])[0]["status"] == "running"


def test_project_without_run_started_is_none() -> None:
    assert project([]) == (None, 0)
    assert project([_ev(1, "effect_intent", {"key": "r:0:k"})]) == (None, 1)


# --- 写入 -----------------------------------------------------------------
def test_seq_is_shared_by_both_writers_and_has_no_gaps(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    log = EventLog(ledger.db_path)
    log.append("r1", "run_started", payload={"state": {}})
    ledger.attempt(scoped_key("r1", 0, "a"), {"op": 1}, lambda: {}, replay=Replay.SAFE)
    log.append("r1", "step_committed", step_index=0, payload={"delta": {}})
    log.append("r2", "run_started", payload={"state": {}})  # 别的 run 有自己的序列
    ledger.attempt(scoped_key("r1", 1, "b"), {"op": 2}, lambda: {}, replay=Replay.SAFE)

    r1 = log.read("r1")
    assert [e.seq for e in r1] == list(range(1, len(r1) + 1))
    assert [e.type for e in r1] == ["run_started", "effect_intent", "effect_outcome",
                                    "step_committed", "effect_intent", "effect_outcome"]
    assert [e.seq for e in log.read("r2")] == [1]
    assert r1[1].step_index == 0 and r1[1].payload["key"] == "r1:0:a"
    assert log.exists("r1") and not log.exists("r3")


def test_effect_event_and_ledger_row_commit_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """事件写失败，账本行也不能留下：两者在同一个事务里。"""
    ledger = EffectLedger(tmp_path)
    calls: list[int] = []

    def broken(conn, run_id, type_, **kwargs):
        raise sqlite3.OperationalError("磁盘写满（模拟）")

    monkeypatch.setattr(ledger_module, "append_event", broken)
    with pytest.raises(sqlite3.OperationalError):
        ledger.attempt(scoped_key("r1", 0, "a"), {"op": 1}, lambda: calls.append(1) or {},
                       replay=Replay.UNSAFE)

    assert calls == []  # 意图没落盘，就不执行
    assert ledger.rows() == []
    assert EventLog(ledger.db_path).read("r1") == []


def test_unknown_event_types_are_rejected(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    with pytest.raises(ValueError):
        EventLog(ledger.db_path).append("r1", "something_else")


def test_version_2_ledger_is_rejected(tmp_path: Path) -> None:
    raw = sqlite3.connect(tmp_path / "ledger.sqlite")
    raw.execute("CREATE TABLE effects (key TEXT PRIMARY KEY)")
    raw.execute("PRAGMA user_version = 2")
    raw.commit()
    raw.close()
    with pytest.raises(StateVersionMismatch):
        EffectLedger(tmp_path)


def test_version_3_ledger_is_rejected(tmp_path: Path) -> None:
    raw = sqlite3.connect(tmp_path / "ledger.sqlite")
    raw.execute("CREATE TABLE effects (key TEXT PRIMARY KEY)")
    raw.execute("CREATE TABLE effect_attempts (attempt_id TEXT PRIMARY KEY)")
    raw.execute("CREATE TABLE events (run_id TEXT NOT NULL, seq INTEGER NOT NULL)")
    raw.execute("PRAGMA user_version = 3")
    raw.commit()
    raw.close()
    with pytest.raises(StateVersionMismatch):
        EffectLedger(tmp_path)


def test_approval_events_are_accepted_and_do_not_mutate_projected_state(tmp_path: Path) -> None:
    assert APPROVAL_EVENTS == frozenset({"approval_requested", "approval_granted", "approval_revoked"})
    ledger = EffectLedger(tmp_path)
    log = EventLog(ledger.db_path)
    log.append("r1", "run_started", payload={"state": {"step_index": 0, "status": "running"}})
    for ev_type in ("approval_requested", "approval_granted", "approval_revoked"):
        log.append("r1", ev_type, step_index=1, payload={"approval_id": "apr_1"})

    events = log.read("r1")
    assert [e.type for e in events] == [
        "run_started", "approval_requested", "approval_granted", "approval_revoked"
    ]
    state, last_seq = project(events)
    assert state == {"step_index": 0, "status": "running"}
    assert last_seq == 4
