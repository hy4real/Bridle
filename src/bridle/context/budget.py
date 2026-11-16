"""加权预算：输出比输入贵，所以权重不能都是 1。

AgentScope 的 `ReplyBudgetControlMiddleware` 用的就是这套：
`cost = input_weight * input_tokens + output_weight * output_tokens`（示例权重 1.0 / 2.0）。
到达预算时的正确处理是**注入提醒或降级**，不是静默继续烧。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeightedBudget:
    limit: float
    input_weight: float = 1.0
    output_weight: float = 2.0
    spent: float = 0.0

    def charge(self, input_tokens: int, output_tokens: int) -> float:
        cost = input_tokens * self.input_weight + output_tokens * self.output_weight
        self.spent += cost
        return cost

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)

    @property
    def exceeded(self) -> bool:
        return self.spent >= self.limit
