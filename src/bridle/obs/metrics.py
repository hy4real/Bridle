"""指标：高基数标签必须被拒绝，而不是"先记下来再说"。

面经 A8-⑤ 的原话：指标维度只用应用/Agent/操作类型/模型/工具类型/结果这类**低基数**字段；
session_id / trace_id / user_id 属于高基数，只能进 trace 与日志。
这条规则在代码里是白名单，不是文档约定。
"""

from __future__ import annotations

from dataclasses import dataclass, field

LOW_CARDINALITY_LABELS = frozenset({
    "agent", "operation", "model", "provider", "tool", "status", "env",
})
FORBIDDEN_LABELS = frozenset({"session_id", "trace_id", "user_id", "prompt"})


class HighCardinalityLabel(ValueError):
    pass


@dataclass
class Metrics:
    counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(default_factory=dict)

    def add(self, name: str, value: float = 1.0, **labels: str) -> None:
        forbidden = set(labels) & FORBIDDEN_LABELS
        if forbidden:
            raise HighCardinalityLabel(
                f"{sorted(forbidden)} 是高基数标签，不能进指标；请写进 trace")
        unknown = set(labels) - LOW_CARDINALITY_LABELS
        if unknown:
            raise ValueError(f"未在白名单里的标签：{sorted(unknown)}")
        key = (name, tuple(sorted(labels.items())))
        self.counters[key] = self.counters.get(key, 0.0) + value

    def snapshot(self) -> list[dict]:
        return [{ "name": name, "labels": dict(labels), "value": value}
                for (name, labels), value in self.counters.items()]
