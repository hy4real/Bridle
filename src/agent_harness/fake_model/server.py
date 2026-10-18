"""脚本化假模型：OpenAI 兼容的最小服务端。

为什么需要它（不是"为了省 API 钱"而是工程理由）：
1. **可复现**：同一条规格跑两遍必须得到同一结果，真模型做不到。
2. **可注入故障**：超时、429、refusal、finish_reason=length、慢响应——这些是
   生产里真实存在、但真模型无法按需触发的分支。
3. **可离线**：CI 不依赖网络与密钥，验收才能在每次提交时跑。

脚本格式（一个 step 一个模型响应，按调用顺序消耗）::

    ScriptedModel([
        {"tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]},
        {"content": "完成"},
    ])

每个 step 支持：
    content          纯文本回复
    tool_calls       工具调用列表 [{name, arguments(dict|str), id?}]
    refusal          返回 refusal 字段（测 S5.3）
    finish_reason    默认由内容推断，可强制 "length" / "content_filter"
    status           直接返回该 HTTP 状态码（测 429 分流）
    headers          附加响应头（如 {"retry-after": "1"}）
    delay_s          响应前等待秒数（延迟响应头 → 测「首字节超时」）
    body_delay_s     先回响应头、再延迟发 body（测「总超时」）
    usage            覆盖 usage 字段（测 token 口径与缓存子集）
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class ScriptedModel:
    steps: list[dict[str, Any]]
    model_id: str = "fake-scripted"
    calls: int = 0
    requests: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def next_step(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.requests.append(request)
            index = self.calls
            self.calls += 1
        if index < len(self.steps):
            return self.steps[index]
        # 脚本耗尽后不要静默编造：显式报错，避免"测试莫名变绿"
        return {
            "status": 500,
            "body": {"error": f"脚本已耗尽：第 {index + 1} 次调用无可用 step"},
        }


def _to_openai(step: dict[str, Any], model_id: str) -> dict[str, Any]:
    content = step.get("content")
    tool_calls = step.get("tool_calls") or []
    message: dict[str, Any] = {"role": "assistant", "content": content}

    if tool_calls:
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"]
                    if isinstance(call["arguments"], str)
                    else json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for index, call in enumerate(tool_calls)
        ]
    if "refusal" in step:
        message["refusal"] = step["refusal"]

    finish_reason = step.get("finish_reason") or (
        "tool_calls" if tool_calls else "stop"
    )
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": step.get("usage") or {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    model: ScriptedModel  # 由工厂注入
    protocol_version = "HTTP/1.1"  # keep-alive：客户端需要同一个 socket 来设 body 超时

    def log_message(self, fmt: str, *args: Any) -> None:  # 静音，避免污染测试输出
        return

    def _send(
        self,
        status: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        body_delay_s: float = 0.0,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.flush()
        if body_delay_s:
            # 头已经回了，body 拖后：这正是「首字节正常、总时长超时」的形态
            time.sleep(body_delay_s)
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": self.model.model_id, "object": "model", "owned_by": "fake"}
                    ],
                },
            )
            return
        self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": f"unknown path {self.path}"})
            return

        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return

        step = self.model.next_step(request)

        if isinstance(step.get("delay_s"), (int, float)):
            time.sleep(float(step["delay_s"]))
        if "body" in step:
            self._send(int(step.get("status", 500)), step["body"], step.get("headers"))
            return
        if "status" in step:
            self._send(
                int(step["status"]),
                {"error": {"message": step.get("message", "injected failure")}},
                step.get("headers"),
            )
            return

        self._send(200, _to_openai(step, self.model.model_id), step.get("headers"),
                   float(step.get("body_delay_s") or 0.0))


class FakeModelServer:
    """可 with 用：`with FakeModelServer(ScriptedModel([...])) as srv: srv.base_url`"""

    def __init__(self, model: ScriptedModel, host: str = "127.0.0.1", port: int = 0) -> None:
        self.model = model
        handler = type("_BoundHandler", (_Handler,), {"model": model})
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "FakeModelServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
