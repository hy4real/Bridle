"""上下文管理：双层 + 预算。

两条来自面经的硬约束：
1. **双层**（A3-③ AgentScope 的做法）：`summary` 是压缩摘要，前置注入；
   `messages` 保留未压缩的尾部。压缩不是"截断"，是"把旧的压成摘要、保留近的原文"。
2. **压缩要保留可回答的事实**：摘要必须携带早期关键信息，否则压完就"忘了"。
   所以压缩函数是显式注入的（可替换），并且验收要检查"事实是否活着"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..tools.base import estimate_tokens

Summarizer = Callable[[list[dict[str, str]]], str]


@dataclass
class ContextManager:
    system_prompt: str
    summarizer: Summarizer | None = None
    summary: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    compactions: int = 0

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def tokens(self) -> int:
        blob = self.system_prompt + self.summary + "".join(m["content"] for m in self.messages)
        return estimate_tokens(blob)

    def build_request(self) -> list[dict[str, str]]:
        """summary 前置注入，后面跟未压缩的尾部。system prompt 单独放，且必须无动态内容。"""
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"【历史摘要】{self.summary}"})
        messages.extend(self.messages)
        return messages

    def compact(self, *, keep_last: int = 4, max_tokens: int) -> bool:
        """超预算就压缩：把 `keep_last` 之前的消息折进 summary。返回是否发生了压缩。"""
        if self.tokens() <= max_tokens:
            return False
        if len(self.messages) <= keep_last:
            return False

        head, tail = self.messages[:-keep_last], self.messages[-keep_last:]
        if self.summarizer is not None:
            folded = self.summarizer(head)
        else:
            folded = self._default_summarize(head)
        self.summary = (self.summary + "\n" + folded).strip() if self.summary else folded
        self.messages = tail
        self.compactions += 1
        return True

    @staticmethod
    def _default_summarize(messages: list[dict[str, str]]) -> str:
        """默认摘要：保留每条的关键事实（前 80 字符），不丢主体。"""
        lines = []
        for message in messages:
            content = message["content"].strip().replace("\n", " ")
            lines.append(f"[{message['role']}] {content[:80]}")
        return "；".join(lines)
