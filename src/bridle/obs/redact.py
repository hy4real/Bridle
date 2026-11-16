"""导出前脱敏。

三条现实约束（面经 A8-④）：
1. 只能改**属性值**，改不了 span 名/ID/结构；
2. 必须发生在**导出阶段**，不能靠"写日志时小心"；
3. hook 抛异常不能丢掉整批导出——降级为「丢弃该字段」。
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{12,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{10,}"), "[REDACTED_TOKEN]"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
]


class Redactor:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.hits = 0

    def scrub(self, text: str) -> str:
        for pattern, replacement in _PATTERNS:
            text, count = pattern.subn(replacement, text)
            self.hits += count
        return text

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.fail_on and self.fail_on in str(payload.get("name")):
            raise RuntimeError("redactor injected failure")
        for attribute in payload.get("attributes", []):
            value = attribute.get("value", {})
            if "stringValue" in value:
                value["stringValue"] = self.scrub(value["stringValue"])
        return payload
