"""最小 tracer：span 名与属性对齐 OTel GenAI 语义约定。

约定（[semantic-conventions-genai]）：
- span 名：`{gen_ai.operation.name} {gen_ai.request.model}`（inference），工具是 `execute_tool {tool}`；
- 属性：`gen_ai.usage.input_tokens` / `output_tokens` / `cache_read.input_tokens`；
- `input_tokens` 是总量，cache 是它的子集，**不相加**。

导出的是 **OTLP 形状的 JSON**（resourceSpans/scopeSpans/spans），
不是完整 protobuf——这一点写在 README，避免夸大。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .redact import Redactor


@dataclass
class Span:
    name: str
    attributes: dict[str, Any]
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start_ns: int = field(default_factory=lambda: time.time_ns())
    end_ns: int | None = None
    status: str = "OK"

    def as_otlp(self) -> dict[str, Any]:
        attrs = [
            {"key": key, "value": _value(value)}
            for key, value in self.attributes.items()
        ]
        return {
            "traceId": self.span_id,  # 单 trace 场景下用 span_id 代替
            "spanId": self.span_id,
            "parentSpanId": self.parent_id or "",
            "name": self.name,
            "kind": 3,  # CLIENT
            "startTimeUnixNano": str(self.start_ns),
            "endTimeUnixNano": str(self.end_ns or time.time_ns()),
            "attributes": attrs,
            "status": {"code": 0 if self.status == "OK" else 2},
        }


def _value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


class Tracer:
    def __init__(self, redactor: Redactor | None = None) -> None:
        self._spans: list[Span] = []
        self.redactor = redactor or Redactor()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, attributes=dict(attributes))
        try:
            yield span
        except Exception:
            span.status = "ERROR"
            raise
        finally:
            span.end_ns = time.time_ns()
            self._spans.append(span)

    @property
    def spans(self) -> tuple[Span, ...]:
        return tuple(self._spans)

    def export(self) -> dict[str, Any]:
        """导出时脱敏；单条 span 脱敏失败只丢字段、不丢整批。"""
        exported = []
        dropped: list[str] = []
        for span in self._spans:
            payload = span.as_otlp()
            try:
                payload = self.redactor.apply(payload)
            except Exception as error:  # 脱敏 hook 挂了也不能丢整批……
                dropped.append(f"{span.name}:{type(error).__name__}")
                # ……但更不能把原文导出去：降级为「该 span 的字符串字段全部清空」
                for attribute in payload.get("attributes", []):
                    if "stringValue" in attribute.get("value", {}):
                        attribute["value"]["stringValue"] = "[REDACTION_FAILED]"
            exported.append(payload)
        return {
            "resourceSpans": [{
                "resource": {"attributes": [{"key": "service.name",
                                             "value": {"stringValue": "agent-harness"}}]},
                "scopeSpans": [{"scope": {"name": "agent_harness"},
                                "spans": exported}],
            }],
            "redaction_failures": dropped,
        }

    def to_json(self) -> str:
        return json.dumps(self.export(), ensure_ascii=False, indent=2)
