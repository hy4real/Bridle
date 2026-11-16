"""模型网关：超时、429 分流、用量口径、预算降级。

四条踩过的坑（面经 A3 / A6 / A9）：
1. **超时要分口径**：首字节超时和总超时的失效模式不同，不能合成一个 timeout。
   首字节都没来，说明上游没开始干活——重放整个调用没有意义，应当快速失败。
2. **429 要分两类**：带 `retry-after` 是限速（可退避重试），
   不带的是配额耗尽（重试永远失败，必须立刻失败）。
3. **用量口径是包含关系**：`input_tokens` 是总量，`cache_read` / `cache_write` 是它的子集，
   相加会把账单算高。
4. **超预算要降级**：换便宜模型继续，而不是整单失败。
"""

from __future__ import annotations

import http.client
import json
import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from ..context.budget import WeightedBudget


class ModelTimeout(RuntimeError):
    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind} timeout: {detail}")
        self.kind = kind  # "first_byte" | "total"


class QuotaExhausted(RuntimeError):
    """配额耗尽（无 retry-after 的 429）：重试没有意义，必须快速失败。"""


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        if self.cache_read_tokens > self.input_tokens:
            raise ValueError("cache_read 不能大于 input（它是子集）")
        if self.cache_write_tokens > self.input_tokens:
            raise ValueError("cache_write 不能大于 input（它是子集）")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning 不能大于 output（它是子集）")

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cache_read": self.cache_read_tokens,
            "cache_write": self.cache_write_tokens,
            "reasoning": self.reasoning_tokens,
            "total": self.total,
        }


@dataclass
class Completion:
    text: str
    model: str
    usage: Usage
    attempts: int
    decisions: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    raw_usage: dict = field(default_factory=dict)


class ModelGateway:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = "deepseek-v4-pro",
        cheap_model: str = "deepseek-flash",
        retries: int = 2,
        connect_timeout: float = 2.0,
        first_byte_timeout: float = 5.0,
        total_timeout: float = 30.0,
        budget: WeightedBudget | None = None,
        sleep: object = time.sleep,
    ) -> None:
        parts = urlsplit(base_url)
        self.scheme = parts.scheme or "http"
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or (443 if self.scheme == "https" else 80)
        self.path = (parts.path or "") + "/chat/completions"
        self._api_key = None  # 由调用方注入，绝不落盘
        self.model = model
        self.cheap_model = cheap_model
        self.retries = retries
        self.connect_timeout = connect_timeout
        self.first_byte_timeout = first_byte_timeout
        self.total_timeout = total_timeout
        self.budget = budget
        self._sleep = sleep
        self.decisions: list[str] = []

    # --- 单次请求 ---
    def _once(self, model: str, messages: list[dict[str, str]],
              extra: dict | None = None) -> tuple[int, dict[str, str], dict]:
        connection_cls = (http.client.HTTPSConnection if self.scheme == "https"
                          else http.client.HTTPConnection)
        conn = connection_cls(self.host, self.port, timeout=self.connect_timeout)
        try:
            conn.connect()
            payload = {"model": model, "messages": messages, **(extra or {})}
            body = json.dumps(payload).encode()
            headers = {"content-type": "application/json"}
            if self._api_key:
                headers["authorization"] = f"Bearer {self._api_key}"
            conn.request("POST", self.path, body=body, headers=headers)
            conn.sock.settimeout(self.first_byte_timeout)
            try:
                response = conn.getresponse()
            except socket.timeout as error:
                raise ModelTimeout("first_byte", f"{self.first_byte_timeout}s 内没有收到响应头") from error
            headers = {k.lower(): v for k, v in response.getheaders()}
            conn.sock.settimeout(self.total_timeout)
            try:
                raw = response.read()
            except socket.timeout as error:
                raise ModelTimeout("total", f"{self.total_timeout}s 内没有读完响应体") from error
            return response.status, headers, json.loads(raw or b"{}")
        finally:
            conn.close()

    def with_api_key(self, api_key: str) -> "ModelGateway":
        """密钥只在进程内传递，不写日志、不落盘。"""
        self._api_key = api_key
        return self

    # --- 对外入口 ---
    def complete(self, messages: list[dict[str, str]],
                 *, response_format: dict | None = None,
                 tools: list[dict] | None = None,
                 max_tokens: int | None = None) -> Completion:
        decisions: list[str] = []
        model = self.model

        if self.budget is not None and self.budget.exceeded:
            model = self.cheap_model
            decisions.append(f"budget exceeded({self.budget.spent:.0f}/{self.budget.limit:.0f}) → 降级到 {model}")

        attempts = 0
        while attempts <= self.retries:
            attempts += 1
            try:
                extra = {}
                if response_format is not None:
                    extra["response_format"] = response_format
                if tools is not None:
                    extra["tools"] = tools
                if max_tokens is not None:
                    extra["max_tokens"] = max_tokens
                status, headers, payload = self._once(model, messages, extra or None)
            except ModelTimeout as error:
                if error.kind == "first_byte":
                    raise  # 首字节超时：不重放，快速失败
                if attempts > self.retries:
                    raise
                continue

            if status == 429:
                retry_after = headers.get("retry-after")
                if retry_after is None:
                    raise QuotaExhausted("429 不带 retry-after：配额耗尽，重试无效")
                if attempts > self.retries:
                    raise QuotaExhausted(f"429 重试 {attempts} 次仍被限速")
                wait = min(float(retry_after), 1.0)
                decisions.append(f"429 带 retry-after={retry_after}s → 退避 {wait}s 后重试")
                self._sleep(wait)  # type: ignore[operator]
                continue

            if status != 200:
                raise RuntimeError(f"上游返回 {status}: {payload.get('error')}")

            choice = payload["choices"][0]
            usage_raw = payload.get("usage", {})
            details = usage_raw.get("prompt_tokens_details", {}) or {}
            completion_details = usage_raw.get("completion_tokens_details", {}) or {}
            # 缓存命中字段有两套命名：OpenAI 形状的 details.cached_tokens，
            # 与 DeepSeek 自己的 prompt_cache_hit_tokens。只读一套会在某些供应商上静默为 0。
            cache_read = details.get("cached_tokens")
            if cache_read is None:
                cache_read = usage_raw.get("prompt_cache_hit_tokens", 0)
            usage = Usage(
                input_tokens=int(usage_raw.get("prompt_tokens", 0)),
                output_tokens=int(usage_raw.get("completion_tokens", 0)),
                cache_read_tokens=int(cache_read or 0),
                cache_write_tokens=int(usage_raw.get("cache_write_tokens", 0)),
                reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0),
            )
            if self.budget is not None:
                self.budget.charge(usage.input_tokens, usage.output_tokens)
            message = choice.get("message") or {}
            return Completion(
                text=message.get("content") or "",
                model=model,
                usage=usage,
                attempts=attempts,
                decisions=decisions,
                tool_calls=message.get("tool_calls") or [],
                finish_reason=choice.get("finish_reason"),
                raw_usage=usage_raw,
            )

        raise QuotaExhausted("重试次数用尽")
