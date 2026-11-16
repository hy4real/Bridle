"""成本账本：五层归因 + token 桶互斥。

层级来自面经 A6-③：`task → session → turn → call → attempt`。
为什么 session 不够：session 只能回答「这段交互花了多少」，
长任务要回答「从发起到结束花了多少、恢复了几次」——那需要 task 级账本。

桶的口径（A6-⑤ + OTel GenAI）：`input` 是总量，`cache_read` / `cache_write`
是它的**子集**，不能相加。本模块在写入时就拒绝重叠口径，而不是等报表出错。

注意：这里记的是**加权 token 单位**，不是钱。价格随厂商与时间变，
写死假价格会让数字变成编造；真实成本由部署方按自己的价目表换算。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CallRecord:
    task_id: str
    session_id: str
    turn_id: str
    call_id: str
    attempt_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read: int = 0
    cache_write: int = 0
    input_weight: float = 1.0
    output_weight: float = 2.0

    def __post_init__(self) -> None:
        if self.cache_read + self.cache_write > self.input_tokens:
            raise ValueError(
                f"cache 桶超出 input：cache_read={self.cache_read} "
                f"cache_write={self.cache_write} input={self.input_tokens}")
        if min(self.input_tokens, self.output_tokens, self.cache_read, self.cache_write) < 0:
            raise ValueError("token 不能为负")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def weighted_units(self) -> float:
        return self.input_tokens * self.input_weight + self.output_tokens * self.output_weight


class CostLedger:
    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def record(self, record: CallRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self._records)

    # --- 五个层级的聚合 ---
    def _agg(self, key) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for record in self._records:
            bucket = result.setdefault(key(record), {
                "calls": 0, "input": 0, "output": 0,
                "cache_read": 0, "cache_write": 0, "weighted": 0.0,
            })
            bucket["calls"] += 1
            bucket["input"] += record.input_tokens
            bucket["output"] += record.output_tokens
            bucket["cache_read"] += record.cache_read
            bucket["cache_write"] += record.cache_write
            bucket["weighted"] += record.weighted_units
        return result

    def by_task(self): return self._agg(lambda r: r.task_id)
    def by_session(self): return self._agg(lambda r: r.session_id)
    def by_turn(self): return self._agg(lambda r: r.turn_id)
    def by_call(self): return self._agg(lambda r: r.call_id)
    def by_attempt(self): return self._agg(lambda r: r.attempt_id)

    def total(self) -> dict[str, float]:
        return {
            "calls": len(self._records),
            "input": sum(r.input_tokens for r in self._records),
            "output": sum(r.output_tokens for r in self._records),
            "cache_read": sum(r.cache_read for r in self._records),
            "cache_write": sum(r.cache_write for r in self._records),
            "weighted": sum(r.weighted_units for r in self._records),
        }

    def report(self) -> dict:
        return {
            "total": self.total(),
            "by_task": self.by_task(),
            "by_session": self.by_session(),
            "by_turn": self.by_turn(),
            "by_call": self.by_call(),
            "by_attempt": self.by_attempt(),
        }
