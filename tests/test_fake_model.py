"""假模型服务端的行为契约。

这些测试同时是"故障注入能力"的说明书：后面 S5.3（三条旁路）、S9.2（两类 429）、
S9.3（超时口径）都要靠这里注入，所以注入点必须先被证明真的可用。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from bridle.fake_model.server import FakeModelServer, ScriptedModel


def _post(base_url: str, payload: dict) -> tuple[int, dict, dict]:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read()), dict(response.headers)
    except urllib.error.HTTPError as error:  # 4xx/5xx 也要能读 body
        return error.code, json.loads(error.read()), dict(error.headers)


def test_returns_scripted_tool_call_then_text() -> None:
    model = ScriptedModel(
        [
            {"tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]},
            {"content": "完成"},
        ]
    )
    with FakeModelServer(model) as server:
        status, first, _ = _post(server.base_url, {"messages": [{"role": "user", "content": "hi"}]})
        assert status == 200
        choice = first["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}

        status, second, _ = _post(server.base_url, {"messages": []})
        assert second["choices"][0]["message"]["content"] == "完成"
        assert second["choices"][0]["finish_reason"] == "stop"


def test_can_inject_429_with_and_without_retry_after() -> None:
    model = ScriptedModel(
        [
            {"status": 429, "headers": {"retry-after": "1"}, "message": "slow down"},
            {"status": 429, "message": "quota exhausted"},
        ]
    )
    with FakeModelServer(model) as server:
        first_status, first_body, first_headers = _post(server.base_url, {})
        second_status, _, second_headers = _post(server.base_url, {})

    assert first_status == 429 and second_status == 429
    assert first_headers.get("retry-after") == "1"
    assert "retry-after" not in {k.lower() for k in second_headers}
    assert "error" in first_body


def test_can_inject_refusal_and_length_finish() -> None:
    model = ScriptedModel(
        [
            {"content": "", "refusal": "我拒绝"},
            {"content": "半截", "finish_reason": "length"},
        ]
    )
    with FakeModelServer(model) as server:
        _, first, _ = _post(server.base_url, {})
        _, second, _ = _post(server.base_url, {})

    assert first["choices"][0]["message"]["refusal"] == "我拒绝"
    assert second["choices"][0]["finish_reason"] == "length"


def test_delay_is_enforced() -> None:
    model = ScriptedModel([{"content": "慢", "delay_s": 0.2}])
    with FakeModelServer(model) as server:
        started = time.monotonic()
        _post(server.base_url, {})
        elapsed = time.monotonic() - started
    assert elapsed >= 0.2


def test_exhausted_script_fails_loudly() -> None:
    model = ScriptedModel([{"content": "只有一次"}])
    with FakeModelServer(model) as server:
        assert _post(server.base_url, {})[0] == 200
        status, body, _ = _post(server.base_url, {})
    assert status == 500
    assert "脚本已耗尽" in json.dumps(body, ensure_ascii=False)


def test_models_endpoint() -> None:
    model = ScriptedModel([{"content": "x"}])
    with FakeModelServer(model) as server:
        url = server.base_url.replace("/v1", "/v1/models")
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read())
    assert payload["data"][0]["id"] == "fake-scripted"
