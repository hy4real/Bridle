"""前缀稳定：缓存能不能命中，就看你有没有动过前面那段字节。

A3-①（OpenAI 缓存盈亏）+ A3-⑦（system prompt 写日期 → 过 0 点全部失效）的落地：
- `prefix_hash` 用来断言"连续两轮前缀字节一致"；
- `has_dynamic_content` 用来抓 system prompt 里的日期/时间/星期——这是最经典的缓存杀手。
"""

from __future__ import annotations

import hashlib
import re

from ..tools.base import estimate_tokens

_DYNAMIC_PATTERNS = [
    (re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}"), "日期"),
    (re.compile(r"\d{1,2}:\d{2}(:\d{2})?"), "时间"),
    (re.compile(r"今天是|当前时间|现在时间|current (date|time)|today is"), "时间表述"),
    (re.compile(r"星期一|星期二|星期三|星期四|星期五|星期六|星期日|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"), "星期"),
]


def prefix_hash(messages: list[dict[str, str]], *, min_tokens: int = 1) -> str:
    """对前 N 条（凑够 min_tokens 为止）做 sha256。"""
    buffer: list[str] = []
    total = 0
    for message in messages:
        buffer.append(f"{message['role']}:{message['content']}")
        total += estimate_tokens(message["content"])
        if total >= min_tokens:
            break
    return hashlib.sha256("\n".join(buffer).encode()).hexdigest()[:16]


def has_dynamic_content(text: str) -> list[str]:
    return [label for pattern, label in _DYNAMIC_PATTERNS if pattern.search(text)]
