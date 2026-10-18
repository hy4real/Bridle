"""结构化输出的三条旁路。

面经 A5-① 的原话：`strict: true` 只保证 schema adherence，**refusal 不遵守 schema**。
除此之外 `finish_reason=length`（被截断）与 `content_filter`（内容过滤）同样是
非正常终止——三种都不能当成合法业务数据写下游。

这条规则只在"解析器真的检查"时才成立，所以它是一个可执行的分类函数，
不是一句文档。
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Termination(str, Enum):
    OK = "ok"
    REFUSAL = "refusal"
    TRUNCATED = "truncated"
    FILTERED = "filtered"


def classify_choice(choice: dict[str, Any]) -> tuple[Termination, str]:
    """返回 (终止类型, 说明)。判定顺序：拒答 > 过滤 > 截断 > 正常。"""
    message = choice.get("message") or {}
    refusal = message.get("refusal")
    if refusal:
        return Termination.REFUSAL, f"模型拒答：{refusal}"

    finish_reason = choice.get("finish_reason")
    if finish_reason == "content_filter":
        return Termination.FILTERED, "finish_reason=content_filter：内容被过滤"
    if finish_reason == "length":
        return Termination.TRUNCATED, "finish_reason=length：输出被截断，JSON 可能不完整"
    return Termination.OK, "正常终止"
