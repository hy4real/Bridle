"""真模型冒烟：只做 2–3 个场景，目的是找**假模型测不出的差异**。

为什么要有这一步（止损清单第一条）：
假模型让验收可复现，但也让「协议兼容性」变成假设。
真模型一跑，最容易暴露的是**usage 字段命名**、**tool_calls 形状**、
**response_format 的实际行为**——这些在假模型里全是我们自己定义的。

用法（密钥只从环境变量读，不打印、不落盘）::

    set -a; source ~/.env.secrets; set +a
    uv run python scripts/smoke_real_model.py

输出：JSON 摘要（含原始 usage，便于对照我们的 Usage 字段映射）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_harness.model.gateway import ModelGateway, ModelTimeout, QuotaExhausted  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "deepseek-flash")
BASE_URL = "https://api.deepseek.com/v1"

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def scenario_tool_call(gateway: ModelGateway) -> dict:
    """A. 单轮工具调用：验证 tools 参数、tool_calls 形状、usage。"""
    started = time.monotonic()
    completion = gateway.complete(
        [{"role": "user", "content": "杭州今天天气怎么样？请调用工具查询。"}],
        tools=[TOOL],
        max_tokens=200,
    )
    return {
        "scenario": "tool_call",
        "latency_s": round(time.monotonic() - started, 2),
        "model": completion.model,
        "finish_reason": completion.finish_reason,
        "tool_calls": [
            {"name": (call.get("function") or {}).get("name"),
             "arguments": (call.get("function") or {}).get("arguments")}
            for call in completion.tool_calls
        ],
        "text_head": completion.text[:80],
        "usage_parsed": completion.usage.as_dict(),
        "usage_raw": completion.raw_usage,
    }


def scenario_structured(gateway: ModelGateway) -> dict:
    """B. 结构化输出：验证 response_format=json_object 真的返回合法 JSON。"""
    started = time.monotonic()
    completion = gateway.complete(
        [{"role": "system", "content": "只输出 JSON，不要任何解释。"},
         {"role": "user", "content": '把这句话的结构化成 {"intent": ..., "slots": {...}}："帮我订明天下午三点去北京的票"'}],
        response_format={"type": "json_object"},
        max_tokens=800,  # 推理模型的 reasoning_tokens 也算在 completion 里，200 会被吃光
    )
    parsed = None
    error = None
    try:
        parsed = json.loads(completion.text)
    except json.JSONDecodeError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "scenario": "structured_output",
        "latency_s": round(time.monotonic() - started, 2),
        "finish_reason": completion.finish_reason,
        "json_valid": parsed is not None,
        "json_error": error,
        "keys": sorted(parsed) if isinstance(parsed, dict) else None,
        "usage_parsed": completion.usage.as_dict(),
        "usage_raw": completion.raw_usage,
    }


def scenario_multi_turn(gateway: ModelGateway) -> dict:
    """C. 多轮：验证第二轮前缀是否复用（真模型才有缓存命中）。"""
    messages = [
        {"role": "system", "content": "你是内部助理，回答不超过 20 字。"},
        {"role": "user", "content": "记住代号 ORION。"},
    ]
    first = gateway.complete(messages, max_tokens=400)
    messages += [{"role": "assistant", "content": first.text},
                 {"role": "user", "content": "代号是什么？"}]
    second = gateway.complete(messages, max_tokens=400)
    return {
        "scenario": "multi_turn",
        "first_answer_head": first.text[:40],
        "second_answer_head": second.text[:40],
        "remembers_code": "ORION" in second.text.upper(),
        "usage_first": first.usage.as_dict(),
        "usage_second": second.usage.as_dict(),
        "usage_second_raw": second.raw_usage,
        "cache_fields_present": sorted(k for k in second.raw_usage if "cache" in k.lower()),
    }


def main() -> int:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    print(f"[smoke] DEEPSEEK_API_KEY present={bool(api_key)} len={len(api_key or '')}")
    if not api_key:
        print("[smoke] 缺少 DEEPSEEK_API_KEY：先 `set -a; source ~/.env.secrets; set +a`")
        return 2

    gateway = ModelGateway(
        BASE_URL, model=MODEL, cheap_model="deepseek-flash",
        retries=1, connect_timeout=10, first_byte_timeout=30, total_timeout=90,
    ).with_api_key(api_key)

    results = []
    for name, scenario in (
        ("tool_call", scenario_tool_call),
        ("structured_output", scenario_structured),
        ("multi_turn", scenario_multi_turn),
    ):
        try:
            results.append(scenario(gateway))
        except (ModelTimeout, QuotaExhausted) as error:
            results.append({"scenario": name, "error": f"{type(error).__name__}: {error}"})
        except Exception as error:  # 真模型什么都可能返回
            results.append({"scenario": name, "error": f"{type(error).__name__}: {error}"})

    print(json.dumps({"model": MODEL, "results": results}, ensure_ascii=False, indent=2))
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
