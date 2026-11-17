"""审批记录状态机与事件事务一致性测试（M11.2）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bridle.effects import ApprovalRevoked, IdempotencyKeyConflict
from bridle.runtime import EffectLedger
from bridle.runtime import ledger as ledger_module
from bridle.runtime.events import EventLog
from bridle.runtime.ledger import SCHEMA_VERSION, hash_args


def test_ledger_schema_version_is_4_and_creates_approvals_table(tmp_path: Path) -> None:
    assert SCHEMA_VERSION == 4
    ledger = EffectLedger(tmp_path)
    with sqlite3.connect(ledger.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert {"effects", "effect_attempts", "events", "approvals"} <= tables
    assert version == 4


def test_hash_args_ignores_key_order() -> None:
    assert hash_args({"a": 1, "b": {"x": 2, "y": 3}}) == hash_args({"b": {"y": 3, "x": 2}, "a": 1})


def test_request_approval_is_idempotent_on_unique_tuple(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    args_sha = hash_args({"target": "notes/a.md"})
    first_id = ledger.request_approval(
        "run-1", 1, "call_1", "flaky_write", args_sha, "ver-1", reason="需要审批"
    )
    second_id = ledger.request_approval(
        "run-1", 1, "call_1", "flaky_write", args_sha, "ver-1", reason="需要审批"
    )

    assert first_id == second_id
    rows = ledger.approvals(run_id="run-1")
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    events = EventLog(ledger.db_path).read("run-1")
    assert [e.type for e in events] == ["approval_requested"]
    assert events[0].payload["approval_id"] == first_id


def test_request_approval_rejects_mismatched_call_for_same_tuple(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({"target": "a"}), "v1")
    with pytest.raises(IdempotencyKeyConflict):
        ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({"target": "b"}), "v1")


def test_approve_and_revoke_lifecycle_and_filtering(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    aid = ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({"target": "a"}), "v1")

    got = ledger.get_approval("run-1", 1, "call_1")
    assert got is not None and got["approval_id"] == aid
    assert ledger.get_approval_by_id(aid) == got

    approved = ledger.approve(aid, ttl_s=60.0, granted_by="why")
    assert approved["status"] == "approved"
    assert approved["granted_by"] == "why"
    assert approved["expires_at"] > approved["granted_at"]

    assert ledger.approvals(status="pending") == []
    assert len(ledger.approvals(run_id="run-1", status="approved")) == 1

    revoked = ledger.revoke(aid, reason="策略变更")
    assert revoked["status"] == "revoked"
    assert revoked["revoked_reason"] == "策略变更"
    assert revoked["revoked_at"] is not None

    events = EventLog(ledger.db_path).read("run-1")
    assert [e.type for e in events] == ["approval_requested", "approval_granted", "approval_revoked"]


def test_approve_writes_row_and_event_in_same_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = EffectLedger(tmp_path)
    aid = ledger.request_approval("run-1", 1, "call_1", "flaky_write", hash_args({}), "v1")

    def boom(conn, run_id, type_, **kwargs):
        raise sqlite3.OperationalError("磁盘写满（模拟）")

    monkeypatch.setattr(ledger_module, "append_event", boom)
    with pytest.raises(sqlite3.OperationalError):
        ledger.approve(aid, ttl_s=60.0, granted_by="why")

    row = ledger.get_approval_by_id(aid)
    assert row is not None and row["status"] == "pending" and row["granted_by"] is None
    assert [e.type for e in EventLog(ledger.db_path).read("run-1")] == ["approval_requested"]


def test_request_and_revoke_also_rollback_on_event_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = EffectLedger(tmp_path)
    original_append = ledger_module.append_event

    def fail_on_request(conn, run_id, type_, **kwargs):
        raise sqlite3.OperationalError("请求事件失败")

    monkeypatch.setattr(ledger_module, "append_event", fail_on_request)
    with pytest.raises(sqlite3.OperationalError):
        ledger.request_approval("run-1", 0, "c0", "flaky_write", hash_args({}), "v1")
    assert ledger.approvals() == []
    assert EventLog(ledger.db_path).read("run-1") == []

    monkeypatch.setattr(ledger_module, "append_event", original_append)
    aid = ledger.request_approval("run-1", 0, "c0", "flaky_write", hash_args({}), "v1")

    def fail_on_revoke(conn, run_id, type_, **kwargs):
        raise sqlite3.OperationalError("吊销事件失败")

    monkeypatch.setattr(ledger_module, "append_event", fail_on_revoke)
    with pytest.raises(sqlite3.OperationalError):
        ledger.revoke(aid, reason="test")
    assert ledger.get_approval_by_id(aid)["status"] == "pending"
    assert [e.type for e in EventLog(ledger.db_path).read("run-1")] == ["approval_requested"]


def test_approval_validation_errors_and_exception(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    with pytest.raises(KeyError):
        ledger.approve("apr_missing", ttl_s=10.0, granted_by="why")
    with pytest.raises(KeyError):
        ledger.revoke("apr_missing", reason="x")

    aid = ledger.request_approval("run-1", 1, "c1", "flaky_write", hash_args({}), "v1")
    with pytest.raises(ValueError):
        ledger.approve(aid, ttl_s=10.0, granted_by="")
    for bad_ttl in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ledger.approve(aid, ttl_s=bad_ttl, granted_by="why")
    with pytest.raises(ValueError):
        ledger.revoke(aid, reason="")

    ledger.approve(aid, ttl_s=10.0, granted_by="why")
    with pytest.raises(ValueError):
        ledger.approve(aid, ttl_s=10.0, granted_by="why")  # 已不是 pending
    ledger.revoke(aid, reason="expired")
    with pytest.raises(ValueError):
        ledger.revoke(aid, reason="again")  # 已经是 revoked

    err = ApprovalRevoked(aid, "策略变更")
    assert err.key == aid and err.reason == "策略变更"
