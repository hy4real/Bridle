"""切点摘要与原文保留（M12，决定 5）。

与 `context/manager.py` 的差异：
- `state["messages"]` 保持严格 append-only，绝不原地截断 `messages`；
- 切点由纯函数 `compute_cut_point` 从尾向前按 `keep_recent_tokens` 计算，
  遇到 `role == "tool"` 时向前对齐到对应的 `assistant(tool_calls)` 边界，且不切走 `messages[0]`；
- 压缩时写一条 `"summary"` 事件（含 `summary` / `first_kept_seq` / `kept_from_index`），
  可随时从 `events`（`seq < summary_event.seq`，跳过 `"summary"`）重算切点并对照 `first_kept_seq`。
"""

from __future__ import annotations

import json
from typing import Any

from ..runtime.events import Event
from ..tools.base import estimate_tokens

SUMMARY_PREFIX = "【历史摘要】"
COMPACTION_SYSTEM_PROMPT = (
    "请将历史对话折叠为简洁的中文摘要，保留关键事实、约束、文件路径、工具结果与待办事项。"
)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """估算单条 OpenAI 形状消息的 token 数（只用于预算，不用于计费）。

    支持 `content=None` 搭配 `tool_calls` 的 assistant 消息。
    """
    parts: list[str] = [str(message.get("role") or "")]
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False, sort_keys=True))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True))
    tool_call_id = message.get("tool_call_id")
    if tool_call_id:
        parts.append(str(tool_call_id))
    name = message.get("name")
    if name:
        parts.append(str(name))
    return estimate_tokens("\n".join(parts))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def compute_cut_point(messages: list[dict[str, Any]], *, keep_recent_tokens: int) -> int:
    """从尾部向前累计至 `keep_recent_tokens`，返回保留切片的起始下标 `kept_from_index`。

    约束：
    1. 保留 `messages[0]`（system prompt）不被切进摘要，返回值 >= 1（当 `messages` 非空时）；
    2. 至少保留最后一条消息（或最后一个完整 `assistant(tool_calls) + tool` 组），
       防止尾部为空或因单条超大消息触发无限压缩；
    3. 切点落在 `role == "tool"` 上时，向前对齐到对应的 `assistant(tool_calls)` 边界，
       避免切出孤儿 `tool_call_id` 导致上游 400 错误。
    """
    if keep_recent_tokens < 0:
        raise ValueError(f"keep_recent_tokens 不能为负：{keep_recent_tokens}")
    total_count = len(messages)
    if total_count == 0:
        return 0
    if total_count <= 2:
        return 1

    kept_tokens = 0
    cut = total_count
    for index in range(total_count - 1, 0, -1):
        msg_tokens = estimate_message_tokens(messages[index])
        if cut < total_count and kept_tokens + msg_tokens > keep_recent_tokens:
            break
        kept_tokens += msg_tokens
        cut = index
        if kept_tokens >= keep_recent_tokens:
            break

    while cut > 1 and messages[cut].get("role") == "tool":
        cut -= 1

    return cut


def summary_message(summary: str) -> dict[str, Any]:
    return {"role": "system", "content": f"{SUMMARY_PREFIX}{summary}"}


def build_model_messages(
    messages: list[dict[str, Any]],
    *,
    summary: str | None = None,
    kept_from_index: int = 1,
) -> list[dict[str, Any]]:
    """构造送往模型的消息列表，保持 `[messages[0]]` 前缀稳定。"""
    if not messages:
        return []
    head = [messages[0]]
    if summary:
        head.append(summary_message(summary))
    tail_start = max(1, min(kept_from_index, len(messages)))
    return head + list(messages[tail_start:])


def format_compaction_messages(
    messages_slice: list[dict[str, Any]],
    *,
    prev_summary: str | None = None,
) -> list[dict[str, Any]]:
    """把待折叠切片 `[prev_summary] + messages[prev_cut:new_cut]` 整理为摘要模型请求。"""
    sections: list[str] = []
    if prev_summary:
        sections.append(f"【已有摘要】\n{prev_summary}")
    lines: list[str] = []
    for msg in messages_slice:
        role = msg.get("role", "unknown")
        content = msg.get("content")
        chunks: list[str] = []
        if isinstance(content, str) and content:
            chunks.append(content)
        elif content is not None:
            chunks.append(json.dumps(content, ensure_ascii=False, sort_keys=True))
        if msg.get("tool_calls"):
            chunks.append("tool_calls=" + json.dumps(msg["tool_calls"], ensure_ascii=False, sort_keys=True))
        if msg.get("tool_call_id"):
            chunks.append(f"tool_call_id={msg['tool_call_id']}")
        lines.append(f"[{role}] " + " ".join(chunks))
    sections.append("【待折叠历史】\n" + "\n".join(lines))
    return [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def message_seqs_from_events(events: list[Event]) -> tuple[list[dict[str, Any]], list[int]]:
    """从事件日志重算 append-only 的 `messages` 与每条消息引入时的 `seq`。

    跳过 `"summary"` 事件，仅追踪 `run_started` 与 `step_committed` 对 `messages` 的写入。
    """
    messages: list[dict[str, Any]] = []
    seqs: list[int] = []
    for event in events:
        if event.type == "summary":
            continue
        if event.type == "run_started":
            initial = event.payload.get("state", {}).get("messages") or []
            messages = json.loads(json.dumps(initial))
            seqs = [event.seq] * len(messages)
        elif event.type == "step_committed":
            delta = event.payload.get("delta") or {}
            sets = delta.get("set") or {}
            extends = delta.get("extend") or {}
            dels = delta.get("del") or []
            if "messages" in dels:
                messages = []
                seqs = []
            if "messages" in sets:
                new_msgs = json.loads(json.dumps(sets["messages"]))
                if len(new_msgs) >= len(messages) and new_msgs[: len(messages)] == messages:
                    seqs = seqs + [event.seq] * (len(new_msgs) - len(messages))
                else:
                    seqs = [event.seq] * len(new_msgs)
                messages = new_msgs
            if "messages" in extends:
                tail = json.loads(json.dumps(extends["messages"]))
                messages.extend(tail)
                seqs.extend([event.seq] * len(tail))
    return messages, seqs


def recompute_cut_point_from_events(
    events: list[Event],
    *,
    before_seq: int,
    keep_recent_tokens: int,
) -> tuple[int, int]:
    """用 `seq < before_seq` 且跳过 `"summary"` 的事件重算 `(kept_from_index, first_kept_seq)`。"""
    prefix = [event for event in events if event.seq < before_seq and event.type != "summary"]
    messages, seqs = message_seqs_from_events(prefix)
    cut = compute_cut_point(messages, keep_recent_tokens=keep_recent_tokens)
    if not seqs or cut >= len(seqs):
        raise ValueError(f"无法从事件重算 first_kept_seq：len(seqs)={len(seqs)}, cut={cut}")
    return cut, seqs[cut]
