"""上下文与网关的单元测试。"""

from __future__ import annotations

import pytest

from bridle.context.budget import WeightedBudget
from bridle.context.manager import ContextManager
from bridle.context.prefix import has_dynamic_content, prefix_hash
from bridle.fake_model.server import FakeModelServer, ScriptedModel
from bridle.model.gateway import ModelGateway, ModelTimeout, QuotaExhausted, Usage


def _gateway(base_url: str, **kwargs) -> ModelGateway:
    return ModelGateway(base_url, sleep=lambda _s: None, **kwargs)


def test_usage_rejects_cache_larger_than_input() -> None:
    with pytest.raises(ValueError):
        Usage(input_tokens=10, output_tokens=5, cache_read_tokens=20)


def test_compaction_keeps_needle_and_shrinks() -> None:
    context = ContextManager(system_prompt="sys")
    context.add_user("记住 ORION")
    for _ in range(8):
        context.add_user("背景" * 600)
    before = context.tokens()
    assert context.compact(keep_last=1, max_tokens=before // 3) is True
    assert context.tokens() < before
    assert "ORION" in context.summary


def test_prefix_hash_is_stable_when_appending() -> None:
    context = ContextManager(system_prompt="sys")
    context.add_user("q1")
    first = prefix_hash(context.build_request())
    context.add_user("q2")
    assert prefix_hash(context.build_request()) == first


def test_dynamic_system_prompt_is_detected() -> None:
    assert has_dynamic_content("今天是 2026-10-18")
    assert has_dynamic_content("当前时间 09:30")
    assert has_dynamic_content("今天是 Monday")
    assert not has_dynamic_content("你是内部助理")


def test_gateway_returns_usage_and_model() -> None:
    model = ScriptedModel([{"content": "ok"}])
    with FakeModelServer(model) as server:
        completion = _gateway(server.base_url).complete([{"role": "user", "content": "hi"}])
    assert completion.text == "ok"
    assert completion.model == "deepseek-v4-pro"
    assert completion.usage.input_tokens == 100


def test_gateway_quota_without_retry_after_fails_fast() -> None:
    model = ScriptedModel([{"status": 429}])
    with FakeModelServer(model) as server:
        with pytest.raises(QuotaExhausted):
            _gateway(server.base_url).complete([{"role": "user", "content": "hi"}])
    assert model.calls == 1


def test_gateway_first_byte_timeout_is_not_retried() -> None:
    model = ScriptedModel([{"content": "慢", "delay_s": 0.5}])
    with FakeModelServer(model) as server:
        with pytest.raises(ModelTimeout) as info:
            _gateway(server.base_url, first_byte_timeout=0.1).complete(
                [{"role": "user", "content": "hi"}])
    assert info.value.kind == "first_byte"
    assert model.calls == 1


def test_gateway_downgrades_after_budget_exceeded() -> None:
    model = ScriptedModel([{"content": "a"}, {"content": "b"}])
    budget = WeightedBudget(limit=50)
    with FakeModelServer(model) as server:
        gateway = _gateway(server.base_url, budget=budget)
        gateway.complete([{"role": "user", "content": "hi"}])
        second = gateway.complete([{"role": "user", "content": "hi"}])
    assert second.model == "deepseek-flash"
    assert any("降级" in d for d in second.decisions)


def test_reasoning_tokens_must_be_subset_of_output() -> None:
    with pytest.raises(ValueError):
        Usage(input_tokens=10, output_tokens=5, reasoning_tokens=9)


def test_cache_read_falls_back_to_deepseek_field() -> None:
    """真模型冒烟发现的差异：有的供应商只给 prompt_cache_hit_tokens。"""
    usage_payload = {
        "prompt_tokens": 296,
        "completion_tokens": 54,
        "prompt_cache_hit_tokens": 128,   # 没有 prompt_tokens_details
        "completion_tokens_details": {"reasoning_tokens": 15},
    }
    with FakeModelServer(ScriptedModel([{"content": "ok", "usage": usage_payload}])) as server:
        completion = _gateway(server.base_url).complete([{"role": "user", "content": "hi"}])
    assert completion.usage.cache_read_tokens == 128
    assert completion.usage.reasoning_tokens == 15
