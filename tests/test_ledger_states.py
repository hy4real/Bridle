"""账本状态机的逐格测试（M7）。

对应 `runtime/ledger.py` 文件头的两张表：
- 领取表 7 行：无记录 / 意图不同 / completed / cancelled / pending+safe / pending+unsafe / unknown
- 收尾表 5 行：正常返回 / Cancelled / 其他异常 / 进程死亡 / 比较并交换没命中

外加几条专项：unknown 确实落盘、并发抢新键、resolve 之后的去重、旧库拒绝、从严取 replay。
跨进程的真 SIGKILL 行为在 `verify/checks_m7.py`。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from bridle.effects import (
    EffectOutcomeUnknown,
    IdempotencyKeyConflict,
    Replay,
    StateVersionMismatch,
)
from bridle.runtime import EffectLedger
from bridle.runtime.types import Cancelled

INTENT = {"op": "append", "payload": "x"}


class _ProcessDied(BaseException):
    """进程死亡的替身：不是 Exception，账本的收尾逻辑接不到它，行就停在 pending。"""


def _die() -> dict:
    raise _ProcessDied()


def _leave_pending(ledger: EffectLedger, key: str, replay: Replay) -> None:
    with pytest.raises(_ProcessDied):
        ledger.attempt(key, INTENT, _die, replay=replay)
    assert _row(ledger, key)["status"] == "pending"


def _row(ledger: EffectLedger, key: str) -> dict:
    return next(row for row in ledger.rows() if row["key"] == key)


def _outcomes(ledger: EffectLedger) -> list[str]:
    return [entry["outcome"] for entry in ledger.attempts_log()]


def _counter() -> tuple[list[int], callable]:
    calls: list[int] = []

    def action() -> dict:
        calls.append(1)
        return {"n": len(calls)}

    return calls, action


# --- 领取表 ---------------------------------------------------------------
@pytest.mark.parametrize("replay", [Replay.SAFE, Replay.UNSAFE])
def test_claim_new_key_writes_pending_then_runs(tmp_path: Path, replay: Replay) -> None:
    ledger = EffectLedger(tmp_path)
    seen: dict = {}

    def action() -> dict:
        seen.update(_row(ledger, "k"))
        return {"ok": True}

    result = ledger.attempt("k", INTENT, action, replay=replay)

    assert seen["status"] == "pending" and seen["replay"] == replay.value
    assert seen["executions"] == 1
    assert result.deduplicated is False and result.executions == 1
    assert _row(ledger, "k")["status"] == "completed"
    assert _outcomes(ledger) == ["applied"]


def test_claim_different_intent_conflicts_without_touching_row(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    calls, action = _counter()
    ledger.attempt("k", INTENT, action, replay=Replay.SAFE)
    before = _row(ledger, "k")

    with pytest.raises(IdempotencyKeyConflict):
        ledger.attempt("k", {"op": "append", "payload": "DIFFERENT"}, action, replay=Replay.SAFE)

    assert len(calls) == 1  # 第二个意图没有被执行，也没有拿到第一个意图的结果
    assert _row(ledger, "k") == before


def test_claim_intent_hash_ignores_key_order(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    calls, action = _counter()
    ledger.attempt("k", {"a": 1, "b": 2}, action, replay=Replay.SAFE)
    again = ledger.attempt("k", {"b": 2, "a": 1}, action, replay=Replay.SAFE)
    assert again.deduplicated is True and len(calls) == 1


def test_claim_completed_dedups(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    calls, action = _counter()
    first = ledger.attempt("k", INTENT, action, replay=Replay.UNSAFE)
    second = ledger.attempt("k", INTENT, action, replay=Replay.UNSAFE)

    assert len(calls) == 1
    assert second.deduplicated is True and second.value == first.value
    assert second.attempts == 2 and second.executions == 1  # 进账本 2 次，真正执行 1 次
    assert _outcomes(ledger) == ["applied", "deduplicated"]


@pytest.mark.parametrize("replay", [Replay.SAFE, Replay.UNSAFE])
def test_claim_cancelled_runs_again_and_goes_back_to_pending(tmp_path: Path, replay: Replay) -> None:
    ledger = EffectLedger(tmp_path)
    # 造一行 cancelled：safe 的取消直接得到；unsafe 的要经过 unknown + resolve(not_applied)
    with pytest.raises(Cancelled):
        ledger.attempt("k", INTENT, lambda: (_ for _ in ()).throw(Cancelled("stop")), replay=replay)
    if replay is Replay.UNSAFE:
        assert _row(ledger, "k")["status"] == "unknown"
        ledger.resolve("k", "not_applied", resolved_by="test")
    assert _row(ledger, "k")["status"] == "cancelled"

    seen: dict = {}

    def action() -> dict:
        seen.update(_row(ledger, "k"))
        return {"ok": True}

    result = ledger.attempt("k", INTENT, action, replay=replay)

    # 执行期间必须回到 pending：否则这次执行中途死掉，下一次会把 cancelled 当成「没做过」再跑一遍
    assert seen["status"] == "pending"
    assert result.executions == 2
    assert _row(ledger, "k")["status"] == "completed"


def test_claim_pending_safe_replays_and_orphans_previous_attempt(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", Replay.SAFE)
    calls, action = _counter()

    result = ledger.attempt("k", INTENT, action, replay=Replay.SAFE)

    assert len(calls) == 1 and result.deduplicated is False
    assert result.attempts == 2 and result.executions == 2
    assert _outcomes(ledger) == ["orphaned", "applied"]


def test_claim_pending_unsafe_goes_unknown_and_does_not_run(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", Replay.UNSAFE)
    first_attempt_id = _row(ledger, "k")["attempt_id"]
    calls, action = _counter()

    with pytest.raises(EffectOutcomeUnknown) as info:
        ledger.attempt("k", INTENT, action, replay=Replay.UNSAFE)

    row = _row(ledger, "k")
    assert info.value.key == "k"
    assert calls == []
    assert row["status"] == "unknown" and row["executions"] == 1
    assert row["attempt_id"] == first_attempt_id  # 不换持有者：第一次执行若还活着，仍能凭它落 completed


def test_claim_unknown_never_runs_until_resolved(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", Replay.UNSAFE)
    calls, action = _counter()

    for _ in range(3):
        with pytest.raises(EffectOutcomeUnknown):
            ledger.attempt("k", INTENT, action, replay=Replay.UNSAFE)

    row = _row(ledger, "k")
    assert calls == [] and row["status"] == "unknown" and row["executions"] == 1
    assert _outcomes(ledger).count("blocked_unknown") == 2


# --- 收尾表 ---------------------------------------------------------------
def test_finish_cancelled_safe_is_cancelled(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    with pytest.raises(Cancelled):
        ledger.attempt("k", INTENT, lambda: (_ for _ in ()).throw(Cancelled("stop")),
                       replay=Replay.SAFE)
    assert _row(ledger, "k")["status"] == "cancelled"
    assert _outcomes(ledger) == ["cancelled"]


def test_finish_cancelled_unsafe_is_unknown_and_still_raises_cancelled(tmp_path: Path) -> None:
    """取消只有发生在写之前才等于「没做」。不可重放的操作不赌这一点。"""
    ledger = EffectLedger(tmp_path)
    with pytest.raises(Cancelled):
        ledger.attempt("k", INTENT, lambda: (_ for _ in ()).throw(Cancelled("stop")),
                       replay=Replay.UNSAFE)
    assert _row(ledger, "k")["status"] == "unknown"
    assert _outcomes(ledger) == ["cancelled_unknown"]


def test_finish_error_safe_stays_pending_and_reraises(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)

    def boom() -> dict:
        raise TimeoutError("下游没回")

    with pytest.raises(TimeoutError):
        ledger.attempt("k", INTENT, boom, replay=Replay.SAFE)

    assert _row(ledger, "k")["status"] == "pending"
    assert _outcomes(ledger) == ["error"]


def test_finish_error_unsafe_is_unknown_and_wraps_cause(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)

    def boom() -> dict:
        raise TimeoutError("下游没回")

    with pytest.raises(EffectOutcomeUnknown) as info:
        ledger.attempt("k", INTENT, boom, replay=Replay.UNSAFE)

    assert isinstance(info.value.__cause__, TimeoutError)
    assert _row(ledger, "k")["status"] == "unknown"
    assert _outcomes(ledger) == ["error_unknown"]


@pytest.mark.parametrize("replay", [Replay.SAFE, Replay.UNSAFE])
def test_finish_process_death_leaves_pending(tmp_path: Path, replay: Replay) -> None:
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", replay)
    assert _outcomes(ledger) == ["started"]


def test_finish_late_first_attempt_may_complete_a_row_marked_unknown(tmp_path: Path) -> None:
    """第一次执行其实还活着：别人把行标成 unknown 之后，它仍能凭 attempt_id 落 completed。"""
    ledger = EffectLedger(tmp_path)

    def slow_action() -> dict:
        # 执行途中，另一个观察者看到 pending，保守地标成 unknown
        with pytest.raises(EffectOutcomeUnknown):
            ledger.attempt("k", INTENT, lambda: {"never": True}, replay=Replay.UNSAFE)
        assert _row(ledger, "k")["status"] == "unknown"
        return {"first": True}

    result = ledger.attempt("k", INTENT, slow_action, replay=Replay.UNSAFE)

    row = _row(ledger, "k")
    assert result.value == {"first": True}
    assert row["status"] == "completed" and row["executions"] == 1
    assert _outcomes(ledger) == ["applied", "outcome_unknown"]


def test_finish_stale_attempt_does_not_overwrite_a_reclaimed_row(tmp_path: Path) -> None:
    """比较并交换没命中：行已被解除并由另一次执行持有，迟到的那次只留一条 stale 记录。"""
    ledger = EffectLedger(tmp_path)

    def slow_action() -> dict:
        with pytest.raises(EffectOutcomeUnknown):
            ledger.attempt("k", INTENT, lambda: {"never": True}, replay=Replay.UNSAFE)
        ledger.resolve("k", "not_applied", resolved_by="test")
        ledger.attempt("k", INTENT, lambda: {"second": True}, replay=Replay.UNSAFE)
        return {"first": True}

    ledger.attempt("k", INTENT, slow_action, replay=Replay.UNSAFE)

    row = _row(ledger, "k")
    assert row["status"] == "completed" and row["result_json"] == '{"second": true}'
    assert "stale_applied" in _outcomes(ledger)


# --- 专项 -----------------------------------------------------------------
def test_unknown_is_committed_before_the_exception_leaves(tmp_path: Path) -> None:
    """在事务里抛异常会把刚写的状态回滚掉。这里用一条全新的连接确认 unknown 真的在盘上。"""
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", Replay.UNSAFE)
    with pytest.raises(EffectOutcomeUnknown):
        ledger.attempt("k", INTENT, lambda: {}, replay=Replay.UNSAFE)

    raw = sqlite3.connect(tmp_path / "ledger.sqlite")
    try:
        status = raw.execute("SELECT status FROM effects WHERE key = 'k'").fetchone()[0]
    finally:
        raw.close()
    assert status == "unknown"


def test_concurrent_claims_on_a_new_key_never_raise_integrity_error(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    gate = threading.Barrier(8)
    errors: list[BaseException] = []
    results: list = []

    def worker() -> None:
        try:
            gate.wait(timeout=5)
            results.append(ledger.attempt("k", INTENT, lambda: {"ok": True}, replay=Replay.SAFE))
        except BaseException as error:  # noqa: BLE001 —— 要看的就是有没有任何异常
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert errors == []
    assert len(results) == 8
    row = _row(ledger, "k")
    assert row["status"] == "completed" and row["attempts"] == 8


def test_resolve_applied_then_dedup_returns_none_value_without_type_error(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", Replay.UNSAFE)
    with pytest.raises(EffectOutcomeUnknown):
        ledger.attempt("k", INTENT, lambda: {}, replay=Replay.UNSAFE)

    ledger.resolve("k", "applied", resolved_by="why")
    calls, action = _counter()
    result = ledger.attempt("k", INTENT, action, replay=Replay.UNSAFE)

    row = _row(ledger, "k")
    assert calls == []  # 已确认执行过，不再执行
    assert result.deduplicated is True and result.resolved is True and result.value is None
    assert row["status"] == "completed" and row["resolved_by"] == "why" and row["resolved_at"]
    assert row["executions"] == 1
    assert "resolved_applied" in _outcomes(ledger)


def test_resolve_not_applied_allows_exactly_one_more_execution(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", Replay.UNSAFE)
    with pytest.raises(EffectOutcomeUnknown):
        ledger.attempt("k", INTENT, lambda: {}, replay=Replay.UNSAFE)

    ledger.resolve("k", "not_applied", resolved_by="why")
    calls, action = _counter()
    ledger.attempt("k", INTENT, action, replay=Replay.UNSAFE)
    ledger.attempt("k", INTENT, action, replay=Replay.UNSAFE)

    assert len(calls) == 1
    assert _row(ledger, "k")["executions"] == 2


def test_resolve_only_applies_to_unknown(tmp_path: Path) -> None:
    ledger = EffectLedger(tmp_path)
    ledger.attempt("done", INTENT, lambda: {}, replay=Replay.SAFE)

    with pytest.raises(ValueError):
        ledger.resolve("done", "applied", resolved_by="why")
    with pytest.raises(KeyError):
        ledger.resolve("missing", "applied", resolved_by="why")
    with pytest.raises(ValueError):
        ledger.resolve("done", "maybe", resolved_by="why")
    assert _row(ledger, "done")["resolved_by"] is None


@pytest.mark.parametrize("stored, passed", [
    (Replay.SAFE, Replay.UNSAFE),   # 第一次声明 safe，这次说 unsafe
    (Replay.UNSAFE, Replay.SAFE),   # 恢复时漏传 / 传错：账本里存的声明仍然生效
])
def test_replay_declarations_resolve_to_the_stricter_one(
    tmp_path: Path, stored: Replay, passed: Replay
) -> None:
    ledger = EffectLedger(tmp_path)
    _leave_pending(ledger, "k", stored)
    calls, action = _counter()

    with pytest.raises(EffectOutcomeUnknown):
        ledger.attempt("k", INTENT, action, replay=passed)

    assert calls == []
    assert _row(ledger, "k")["replay"] == "unsafe"


def test_legacy_ledger_is_rejected_not_silently_reused(tmp_path: Path) -> None:
    raw = sqlite3.connect(tmp_path / "ledger.sqlite")
    raw.execute("CREATE TABLE effects (key TEXT PRIMARY KEY, status TEXT NOT NULL)")
    raw.commit()
    raw.close()

    with pytest.raises(StateVersionMismatch):
        EffectLedger(tmp_path)


def test_open_existing_only_does_not_create_a_ledger(tmp_path: Path) -> None:
    missing = tmp_path / "typo"
    with pytest.raises(FileNotFoundError):
        EffectLedger(missing, create=False)
    assert not missing.exists()

    EffectLedger(tmp_path)  # 建好之后可以只读打开
    assert EffectLedger(tmp_path, create=False).rows() == []
