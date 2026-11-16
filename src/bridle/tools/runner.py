"""工具执行：超时、重试、幂等。

核心事实（面经 D2-6 + A2-⑧）：**「未收到响应」不等于「未执行」**。
所以重试不能假设"上次没做成"：副作用必须按键收敛，
账本负责记录每次 attempt，下游负责 upsert 去重。
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from ..runtime.ledger import EffectLedger
from .base import OutcomeKind, Tool, ToolOutcome, ToolProtocolError


class ToolRunner:
    def __init__(
        self,
        ledger: EffectLedger | None = None,
        *,
        retries: int = 2,
        timeout_s: float = 2.0,
    ) -> None:
        self.ledger = ledger
        self.retries = retries
        self.timeout_s = timeout_s

    # --- 单次调用（带超时）---
    def _once(self, tool: Tool, args: dict[str, Any]) -> ToolOutcome:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(tool.run, args)
            try:
                return future.result(timeout=self.timeout_s)
            except concurrent.futures.TimeoutError as error:
                # 注意：线程仍在跑。这正是真实世界的样子——请求超时了，上游可能已经执行。
                raise ToolProtocolError(
                    f"{tool.name}: 超时 {self.timeout_s}s（上游可能已经执行）") from error

    @staticmethod
    def _as_payload(outcome: ToolOutcome) -> dict[str, Any]:
        """账本只存 JSON，所以动作的返回值必须先降成普通 dict。"""
        return {"kind": outcome.kind.value, "payload": outcome.payload, "tool": outcome.tool}

    @staticmethod
    def _from_payload(payload: dict[str, Any], attempts: int) -> ToolOutcome:
        return ToolOutcome(
            kind=OutcomeKind(payload["kind"]),
            payload=payload["payload"],
            tool=payload["tool"],
            attempts=attempts,
        )

    # --- 对外入口 ---
    def call(self, tool: Tool, args: dict[str, Any]) -> ToolOutcome:
        tool.validate(args)

        if tool.side_effect and self.ledger is not None:
            key_fn = tool.idempotency_key or (lambda a: f"{tool.name}:{sorted(a.items())}")
            key = key_fn(args)
            last_error: Exception | None = None
            for _ in range(self.retries + 1):
                try:
                    result = self.ledger.attempt(
                        key,
                        intent={"tool": tool.name, "args": args},
                        action=lambda: self._as_payload(self._once(tool, args)),
                    )
                except ToolProtocolError as error:
                    last_error = error
                    continue  # 超时/协议失败：重试（幂等键保证不重复写）
                return self._from_payload(result.value, result.attempts)
            raise last_error or ToolProtocolError(f"{tool.name}: 重试耗尽")

        last_error = None
        for _ in range(self.retries + 1):
            try:
                return self._once(tool, args)
            except ToolProtocolError as error:
                last_error = error
        raise last_error or ToolProtocolError(f"{tool.name}: 重试耗尽")
