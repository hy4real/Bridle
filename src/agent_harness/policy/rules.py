"""命令规则：确定性匹配，不猜。

顺序即语义：**先看 deny，再看 prompt，最后才看 allow**。
所以「allow *  + deny rm -rf *」的结果是 rm 被拒——
allow 不能为 deny 开例外（claude-code 的官方口径也是 deny > ask > allow）。
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from .types import Decision, Rule


class CommandRules:
    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules: list[Rule] = list(rules or [])

    def add(self, rule: Rule) -> None:
        self._rules.append(rule)

    @property
    def rules(self) -> tuple[Rule, ...]:
        return tuple(self._rules)

    @staticmethod
    def _matches(text: str, pattern: str) -> bool:
        # 归一化空白，避免 "rm  -rf   *" 绕过
        normalized = " ".join(text.split())
        return fnmatchcase(normalized, pattern) or normalized.startswith(pattern.rstrip("*").rstrip())

    def first_match(self, command: str, decision: Decision) -> Rule | None:
        for rule in self._rules:
            if rule.decision is decision and self._matches(command, rule.pattern):
                return rule
        return None
