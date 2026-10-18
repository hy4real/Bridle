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

    def __post_init__(self) -> None:
        if self.cache_read_tokens > self.input_tokens:
            raise ValueError("cache_read 不能大于 input（它是子集）")
        if self.cache_write_tokens > self.input_tokens:
            raise ValueError("cache_write 不能大于 input（它是子集）")

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cache_read": self.cache_read_tokens,
            "cache_write": self.cache_write_tokens,
            "total": self.total,
        }


@dataclass
class Completion:
    text: str
    model: str
    usage: Usage
    attempts: int
    decisions: list[str] = field(default_factory=list)


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
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or 80
        self.path = (parts.path or "") + "/chat/completions"
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
    def _once(self, model: str, messages: list[dict[str, str]]) -> tuple[int, dict[str, str], dict]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.connect_timeout)
        try:
            conn.connect()
            body = json.dumps({"model": model, "messages": messages}).encode()
            conn.request("POST", self.path, body=body, headers={"content-type": "application/json"})
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

    # --- 对外入口 ---
    def complete(self, messages: list[dict[str, str]]) -> Completion:
        decisions: list[str] = []
        model = self.model

        if self.budget is not None and self.budget.exceeded:
            model = self.cheap_model
            decisions.append(f"budget exceeded({self.budget.spent:.0f}/{self.budget.limit:.0f}) → 降级到 {model}")

        attempts = 0
        while attempts <= self.retries:
            attempts += 1
            try:
                status, headers, payload = self._once(model, messages)
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
            usage = Usage(
                input_tokens=int(usage_raw.get("prompt_tokens", 0)),
                output_tokens=int(usage_raw.get("completion_tokens", 0)),
                cache_read_tokens=int(details.get("cached_tokens", 0)),
                cache_write_tokens=int(usage_raw.get("cache_write_tokens", 0)),
            )
            if self.budget is not None:
                self.budget.charge(usage.input_tokens, usage.output_tokens)
            return Completion(
                text=(choice.get("message") or {}).get("content") or "",
                model=model,
                usage=usage,
                attempts=attempts,
                decisions=decisions,
            )

        raise QuotaExhausted("重试次数用尽")
