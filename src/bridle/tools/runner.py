"""工具执行：超时、重试、幂等。

核心事实（面经 D2-6 + A2-⑧）：**「未收到响应」不等于「未执行」**。
所以重试不能假设"上次没做成"。能不能重试由工具声明的 `replay` 决定：
- safe：副作用按键收敛，账本记录每次 attempt，下游 upsert 去重，可以重试；
- unsafe：超时或异常后账本记 `unknown`，抛 `EffectOutcomeUnknown`，**不重试**。
"""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any

from ..effects import VerifyResult, scoped_key, validate_run_id
from ..runtime.ledger import EffectLedger
from .base import OutcomeKind, Tool, ToolOutcome, ToolProtocolError


class ToolRunner:
    def __init__(
        self,
        ledger: EffectLedger | None = None,
        *,
        run_id: str | None = None,
        retries: int = 2,
        timeout_s: float = 2.0,
    ) -> None:
        self.ledger = ledger
        self.run_id = validate_run_id(run_id) if run_id is not None else None
        self.retries = retries
        self.timeout_s = timeout_s

    # --- 单次调用（带超时）---
    def _once(self, tool: Tool, args: dict[str, Any]) -> ToolOutcome:
        """在守护线程里跑工具，等 `timeout_s`。

        不用 `with ThreadPoolExecutor(...)`：它退出时会 join 工作线程，超时就不再限时了；
        `shutdown(wait=False)` 也不够，池里的线程不是守护线程，解释器退出时仍会等它。
        超时之后线程还在跑，它的返回值没人读——这正是真实世界的样子：请求超时了，
        上游可能已经执行。所以超时之后能不能重试，要看工具声明的 `replay`，不看这里。
        一个后果：重试会和还没结束的上一次调用重叠，`Replay.SAFE` 必须在并发重复下也成立。
        """
        box: concurrent.futures.Future[ToolOutcome] = concurrent.futures.Future()

        def work() -> None:
            try:
                box.set_result(tool.run(args))
            except BaseException as error:  # noqa: BLE001 —— 原样交给等待方
                box.set_exception(error)

        threading.Thread(target=work, name=f"tool-{tool.name}", daemon=True).start()
        try:
            return box.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError as error:
            raise ToolProtocolError(
                f"{tool.name}: 超时 {self.timeout_s}s（上游可能已经执行）") from error

    def _verifier(self, tool: Tool, args: dict[str, Any], key: str):
        """把工具的反查包成账本要的形状：applied 时带回来的 payload 存成工具结果的格式。"""
        def verify() -> VerifyResult:
            outcome = tool.verify(args, key)  # type: ignore[misc]
            if outcome.verdict == "applied" and outcome.result is not None:
                return VerifyResult("applied", self._as_payload(ToolOutcome(
                    kind=OutcomeKind.OK, payload=tool.render(outcome.result), tool=tool.name)))
            return outcome

        return verify

    @staticmethod
    def _as_payload(outcome: ToolOutcome) -> dict[str, Any]:
        """账本只存 JSON，所以动作的返回值必须先降成普通 dict。"""
        return {"kind": outcome.kind.value, "payload": outcome.payload, "tool": outcome.tool}

    @staticmethod
    def _from_payload(payload: dict[str, Any] | None, attempts: int, tool: Tool) -> ToolOutcome:
        if payload is None:
            # 这一行是人工 resolve 或反查确认为「已执行」的，原始返回值没有留下来
            return ToolOutcome(
                kind=OutcomeKind.OK,
                payload={"ok": True, "note": "已由人工确认或反查确认为已执行，原始返回值不可得"},
                tool=tool.name,
                attempts=attempts,
            )
        return ToolOutcome(
            kind=OutcomeKind(payload["kind"]),
            payload=payload["payload"],
            tool=payload["tool"],
            attempts=attempts,
        )

    # --- 对外入口 ---
    def call(self, tool: Tool, args: dict[str, Any], *, step_index: int = 0) -> ToolOutcome:
        tool.validate(args)

        if tool.side_effect:
            if self.ledger is None or self.run_id is None:
                # 没有账本或没有作用域的写，就是一条不受约束的重试路径。不提供。
                raise RuntimeError(
                    f"{tool.name} 带副作用：必须给 ToolRunner 提供 ledger 与 run_id 才能调用")
            key_fn = tool.idempotency_key or (lambda a: f"{tool.name}:{sorted(a.items())}")
            key = scoped_key(self.run_id, step_index, key_fn(args))
            verify = self._verifier(tool, args, key) if tool.verify is not None else None
            last_error: Exception | None = None
            for _ in range(self.retries + 1):
                try:
                    result = self.ledger.attempt(
                        key,
                        intent={"tool": tool.name, "args": args},
                        action=lambda: self._as_payload(self._once(tool, args)),
                        replay=tool.replay_mode(),
                        verify=verify,
                    )
                except ToolProtocolError as error:
                    # 走到这里的一定是 safe 工具：unsafe 工具的超时/异常在账本里
                    # 已经变成 EffectOutcomeUnknown，不是 ToolProtocolError，不会被重试。
                    last_error = error
                    continue  # 超时/协议失败：重试（幂等键保证不重复写）
                return self._from_payload(result.value, result.attempts, tool)
            raise last_error or ToolProtocolError(f"{tool.name}: 重试耗尽")

        last_error = None
        for _ in range(self.retries + 1):
            try:
                return self._once(tool, args)
            except ToolProtocolError as error:
                last_error = error
        raise last_error or ToolProtocolError(f"{tool.name}: 重试耗尽")
