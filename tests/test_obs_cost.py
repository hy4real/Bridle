"""配额、成本账本、观测的单元测试。"""

from __future__ import annotations

import threading

import pytest

from agent_harness.cost.ledger import CallRecord, CostLedger
from agent_harness.obs.metrics import HighCardinalityLabel, Metrics
from agent_harness.obs.redact import Redactor
from agent_harness.obs.tracer import Tracer
from agent_harness.quota import QuotaLedger


def test_quota_reserve_blocks_second_request() -> None:
    quota = QuotaLedger(limit=100)
    assert quota.reserve(80, key="a") is not None
    assert quota.reserve(80, key="b") is None
    assert quota.available == 20


def test_quota_settle_releases_reservation_and_counts_actual() -> None:
    quota = QuotaLedger(limit=100)
    reservation = quota.reserve(80, key="a")
    assert reservation is not None
    quota.settle(reservation, actual=30)
    assert quota.settled == 30
    assert quota.reserved == 0
    assert quota.available == 70


def test_quota_settle_cannot_exceed_reservation() -> None:
    quota = QuotaLedger(limit=100)
    reservation = quota.reserve(50, key="a")
    assert reservation is not None
    quota.settle(reservation, actual=999)
    assert quota.settled == 50


def test_quota_is_thread_safe() -> None:
    quota = QuotaLedger(limit=100)
    barrier = threading.Barrier(8)
    granted: list[int] = []

    def worker(index: int) -> None:
        barrier.wait()
        if quota.reserve(80, key=f"j{index}"):
            granted.append(index)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 1


def test_cost_ledger_rejects_overlapping_buckets() -> None:
    with pytest.raises(ValueError):
        CallRecord(task_id="t", session_id="s", turn_id="u", call_id="c", attempt_id="a",
                   model="m", input_tokens=10, output_tokens=1, cache_read=8, cache_write=8)


def test_cost_ledger_five_levels() -> None:
    ledger = CostLedger()
    ledger.record(CallRecord("t", "s", "u1", "c1", "a1", "m", 10, 2))
    ledger.record(CallRecord("t", "s", "u1", "c1", "a2", "m", 10, 2))
    report = ledger.report()
    assert len(report["by_turn"]) == 1
    assert len(report["by_call"]) == 1
    assert len(report["by_attempt"]) == 2
    assert report["total"]["calls"] == 2


def test_metrics_rejects_high_cardinality() -> None:
    metrics = Metrics()
    with pytest.raises(HighCardinalityLabel):
        metrics.add("gen_ai.client.token.usage", 1, session_id="s")
    metrics.add("gen_ai.client.token.usage", 1, model="m")
    assert len(metrics.snapshot()) == 1


def test_tracer_emits_otel_shape() -> None:
    tracer = Tracer()
    with tracer.span("invoke_agent m", **{"gen_ai.request.model": "m"}):
        pass
    span = tracer.export()["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["name"] == "invoke_agent m"
    assert span["attributes"][0]["key"] == "gen_ai.request.model"


def test_redaction_scrubs_keys() -> None:
    redactor = Redactor()
    tracer = Tracer(redactor)
    with tracer.span("invoke_agent m", note="key sk-abcdefghijklmnop1234"):
        pass
    assert "sk-abcdefghijklmnop1234" not in tracer.to_json()
    assert redactor.hits >= 1


def test_redaction_failure_does_not_leak_or_drop() -> None:
    tracer = Tracer(Redactor(fail_on="execute_tool"))
    with tracer.span("invoke_agent m", note="sk-abcdefghijklmnop1234"):
        pass
    with tracer.span("execute_tool x", note="sk-abcdefghijklmnop1234"):
        pass
    exported = tracer.export()
    blob = str(exported)
    assert "sk-abcdefghijklmnop1234" not in blob
    assert len(exported["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2
    assert exported["redaction_failures"]
