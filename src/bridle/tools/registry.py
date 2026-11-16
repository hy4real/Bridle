"""工具注册表与按需加载。

面经 A4 的硬数字：Claude 官方说**超过 30–50 个工具后选择准确率下降**，
Anthropic 见过 **134K token** 的工具定义。所以「全量注入」是错的默认。

这里的做法不是向量检索，而是**确定性打分 + 预算截断**：
名字/描述与查询的词重合度、以及历史使用频次。可解释、可复现、不需要额外依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import Tool, estimate_tokens

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]")


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD.finditer(text)}


@dataclass(frozen=True)
class Selection:
    tools: tuple[Tool, ...]
    tokens: int
    skipped: int
    budget: int

    @property
    def count(self) -> int:
        return len(self.tools)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._usage: dict[str, int] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"没有这个工具：{name}")
        return self._tools[name]

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def note_usage(self, name: str) -> None:
        self._usage[name] = self._usage.get(name, 0) + 1

    def score(self, tool: Tool, query: str) -> float:
        query_tokens = _tokens(query)
        haystack = f"{tool.name} {tool.description} {' '.join(tool.tags)}"
        overlap = len(query_tokens & _tokens(haystack))
        return overlap * 2.0 + self._usage.get(tool.name, 0) * 0.5

    def select(self, query: str, *, budget_tokens: int) -> Selection:
        """按相关性排序后贪心装入预算。装不下的跳过，不截断定义。"""
        ranked = sorted(self._tools.values(), key=lambda t: self.score(t, query), reverse=True)
        chosen: list[Tool] = []
        used = 0
        skipped = 0
        for tool in ranked:
            cost = tool.definition_tokens()
            if used + cost <= budget_tokens:
                chosen.append(tool)
                used += cost
            else:
                skipped += 1
        return Selection(tools=tuple(chosen), tokens=used, skipped=skipped, budget=budget_tokens)

    def total_definition_tokens(self) -> int:
        return sum(tool.definition_tokens() for tool in self._tools.values())

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        if names is None:
            return [tool.definition() for tool in self._tools.values()]
        return [self.get(name).definition() for name in names]


def build_fixture_registry(count: int) -> ToolRegistry:
    """60 个工具的 fixture：名字与描述刻意分散在不同领域，检验按需加载。"""

    from .base import Tool

    domains = [
        ("file", "读取、写入、列出工作区文件"),
        ("git", "查看提交、分支、diff"),
        ("http", "发起 HTTP 请求并解析响应"),
        ("sql", "执行查询并返回结果集"),
        ("shell", "在沙箱内执行命令"),
        ("search", "在代码库中检索符号与文本"),
    ]
    registry = ToolRegistry()
    for index in range(count):
        domain, desc = domains[index % len(domains)]
        name = f"{domain}_op_{index:02d}"
        registry.register(
            Tool(
                name=name,
                description=f"{desc}（{domain} 领域第 {index} 个操作）",
                parameters={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": ["target"],
                },
                handler=lambda target, _n=name: {"ok": True, "tool": _n, "target": target},
                tags=(domain,),
            )
        )
    return registry
