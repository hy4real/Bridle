"""把模型循环写成一个动态任务：下一步是调模型还是调工具，看 state。

state 里多三样东西，都是 JSON，跟着 checkpoint 一起落盘：
    messages            OpenAI 形状的 transcript（system / user / assistant / tool）
    pending_tool_calls  上一次模型回复里还没执行的工具调用，按顺序逐个成步
    final               模型不再要求调用工具时的回答；有了它，任务结束

每次模型调用是一步，每次工具调用是一步。关键性质：**模型步提交之后才执行工具步**，
所以恢复落在工具步时，工具调用列表是从 checkpoint 里取的，不再询问模型。

模型调用本身也经过账本（`replay=safe`）：它不是副作用，但同样的输入可能得到不同的回答。
回答一旦落进账本，即使 checkpoint 还没写进程就被杀，恢复时也拿回同一个回答，而不是重问。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from ..effects import ApprovalRevoked, Replay
from ..model.gateway import ModelGateway, ModelTimeout, QuotaExhausted
from ..policy import Decision, PolicyEngine
from ..runtime.events import EventLog
from ..runtime.ledger import hash_args
from ..runtime.types import NodeInterrupt, RunFailed, Step, StepContext, Task
from ..tools.base import Tool, ToolProtocolError
from ..tools.registry import ToolRegistry
from ..tools.runner import ToolRunner
from .compaction import (
    build_model_messages,
    compute_cut_point,
    estimate_messages_tokens,
    format_compaction_messages,
    message_seqs_from_events,
)
from .messages import assistant_message, parse_tool_call, tool_error, tool_result

SYSTEM_PROMPT = "你是一个会调用工具完成任务的助手。需要时调用工具；完成后用一句话说明结果。"
_NORMAL_FINISH = {"stop", "tool_calls"}


@dataclass
class AgentConfig:
    gateway: ModelGateway
    registry: ToolRegistry
    policy: PolicyEngine
    max_steps: int = 8
    tool_timeout_s: float = 5.0
    tool_retries: int = 2
    max_tokens: int | None = None
    system_prompt: str = SYSTEM_PROMPT
    context_budget_tokens: int | None = None
    keep_recent_tokens: int = 200


class _ModelUnavailable(Exception):
    """模型这次调用没有拿到可用的回答。交给账本记 error，再由模型步转成 RunFailed。"""


def _initial_messages(config: AgentConfig, prompt: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}]


def build_agent_task(prompt: str, config: AgentConfig) -> Task:
    if config.max_steps < 1:
        raise ValueError("max_steps 至少为 1")
    if config.context_budget_tokens is not None and config.context_budget_tokens < 1:
        raise ValueError("context_budget_tokens 至少为 1")
    if config.keep_recent_tokens < 0:
        raise ValueError("keep_recent_tokens 不能为负")

    def next_step(state: dict[str, Any]) -> Step | None:
        # 只读 state：恢复时同一份 state 必须给出同一个下一步
        if state.get("final") is not None:
            return None
        if int(state["step_index"]) >= config.max_steps:
            raise RunFailed(f"max_steps={config.max_steps} 已用尽，模型仍要求继续")
        pending = state.get("pending_tool_calls") or []
        if pending:
            name = (pending[0].get("function") or {}).get("name") or "?"
            return Step(f"tool:{name}", tool_step)
        if config.context_budget_tokens is not None:
            raw_messages = state.get("messages") or _initial_messages(config, prompt)
            kept_from_index = int(state.get("kept_from_index", 1))
            prompt_messages = build_model_messages(
                raw_messages,
                summary=state.get("summary"),
                kept_from_index=kept_from_index,
            )
            if estimate_messages_tokens(prompt_messages) > config.context_budget_tokens:
                new_cut = compute_cut_point(
                    raw_messages, keep_recent_tokens=config.keep_recent_tokens
                )
                if new_cut > kept_from_index:
                    return Step("compact", compact_step)
        return Step("model", model_step)

    def compact_step(ctx: StepContext) -> None:
        raw_messages = ctx.state.get("messages") or _initial_messages(config, prompt)
        prev_cut = int(ctx.state.get("kept_from_index", 1))
        prev_summary = ctx.state.get("summary")
        new_cut = compute_cut_point(raw_messages, keep_recent_tokens=config.keep_recent_tokens)
        if new_cut <= prev_cut or new_cut >= len(raw_messages):
            raise RunFailed(
                f"切点计算异常：prev_cut={prev_cut}, new_cut={new_cut}, len={len(raw_messages)}"
            )

        events = EventLog(ctx.ledger.db_path).read(ctx.run_id)
        event_messages, seqs = message_seqs_from_events(events)
        if len(seqs) != len(raw_messages) or event_messages != raw_messages:
            raise RunFailed("事件日志中的 messages 与当前 state 不一致，无法确定 first_kept_seq")
        first_kept_seq = seqs[new_cut]

        compact_messages = format_compaction_messages(
            raw_messages[prev_cut:new_cut], prev_summary=prev_summary
        )
        request = {
            "messages": compact_messages,
            "max_tokens": config.max_tokens,
            "prev_cut": prev_cut,
            "new_cut": new_cut,
        }
        request_sha = hashlib.sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        def summarize() -> dict[str, Any]:
            try:
                completion = config.gateway.complete(
                    compact_messages, max_tokens=config.max_tokens
                )
            except (ModelTimeout, QuotaExhausted, RuntimeError, OSError) as error:
                raise _ModelUnavailable(f"{type(error).__name__}: {error}") from error
            return {
                "text": completion.text,
                "finish_reason": completion.finish_reason,
                "model": completion.model,
                "usage": completion.usage.as_dict(),
            }

        try:
            answer = ctx.ledger.attempt(
                ctx.effect_key("compact"),
                {"op": "compact", "request_sha": request_sha},
                summarize,
                replay=Replay.SAFE,
            ).value
        except _ModelUnavailable as error:
            raise RunFailed(f"摘要模型不可用：{error}") from error

        if answer["finish_reason"] != "stop":
            raise RunFailed(f"摘要输出非正常终止：finish_reason={answer['finish_reason']}")
        summary_text = (answer.get("text") or "").strip()
        if not summary_text:
            raise RunFailed("摘要模型返回空摘要")

        ctx.emit_summary(
            summary_text,
            first_kept_seq=first_kept_seq,
            kept_from_index=new_cut,
        )

    def model_step(ctx: StepContext) -> None:
        raw_messages = ctx.state.get("messages") or _initial_messages(config, prompt)
        prompt_messages = build_model_messages(
            raw_messages,
            summary=ctx.state.get("summary"),
            kept_from_index=int(ctx.state.get("kept_from_index", 1)),
        )
        schemas = config.registry.schemas()
        request = {"messages": prompt_messages, "tools": schemas, "max_tokens": config.max_tokens}
        request_sha = hashlib.sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

        def ask() -> dict[str, Any]:
            try:
                completion = config.gateway.complete(prompt_messages, tools=schemas or None,
                                                     max_tokens=config.max_tokens)
            except (ModelTimeout, QuotaExhausted, RuntimeError, OSError) as error:
                raise _ModelUnavailable(f"{type(error).__name__}: {error}") from error
            return {"text": completion.text, "tool_calls": completion.tool_calls,
                    "finish_reason": completion.finish_reason, "model": completion.model,
                    "usage": completion.usage.as_dict()}

        try:
            answer = ctx.ledger.attempt(ctx.effect_key("model"),
                                        {"op": "model", "request_sha": request_sha},
                                        ask, replay=Replay.SAFE).value
        except _ModelUnavailable as error:
            raise RunFailed(f"模型不可用：{error}") from error

        if answer["finish_reason"] not in _NORMAL_FINISH:
            # 截断、内容过滤：这样的回答既不是完整答案，也不是可执行的工具调用
            raise RunFailed(f"模型输出非正常终止：finish_reason={answer['finish_reason']}")

        ctx.state["messages"] = [*raw_messages, assistant_message(answer["text"], answer["tool_calls"])]
        ctx.state["pending_tool_calls"] = list(answer["tool_calls"])
        ctx.state["model_steps"] = int(ctx.state.get("model_steps", 0)) + 1
        if not answer["tool_calls"]:
            ctx.state["final"] = answer["text"]

    def tool_step(ctx: StepContext) -> None:
        pending = list(ctx.state["pending_tool_calls"])
        message = _run_tool_call(ctx, pending[0], config)
        ctx.state["messages"] = [*ctx.state["messages"], message]
        ctx.state["pending_tool_calls"] = pending[1:]

    return Task(name="agent", next_step=next_step)


def _run_tool_call(ctx: StepContext, raw: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    """执行一个工具调用，返回要回灌给模型的消息。

    模型的错（参数坏了、工具不存在、被策略拒绝）→ 带 is_error 的工具结果，run 继续。
    harness 的问题：工具协议失败（重试耗尽）→ RunFailed；结果未知 → EffectOutcomeUnknown 上抛，
    run 停在 blocked；需要人审批 → 先落盘待批记录再 NodeInterrupt，run 停在 interrupted；
    已有审批失效/过期/吊销 → ApprovalRevoked 上抛，run 停在 blocked。
    """
    call = parse_tool_call(raw)
    if call.error is not None:
        return tool_error(call.id, call.error, "模型给出的工具调用无法解析")

    # 必须在 config.policy.decide() 之前检查审批记录：
    # 防止改成 deny 后被 decide() 吞成 POLICY_DENIED 继续跑完，也防止删掉 ask 规则后绕过吊销。
    args_sha = hash_args(call.arguments)
    approval = ctx.ledger.get_approval(ctx.run_id, ctx.step_index, call.id)
    approved_ok = False
    if approval is not None:
        if approval["status"] == "revoked":
            raise ApprovalRevoked(
                approval["approval_id"],
                approval["revoked_reason"] or "审批已吊销",
            )
        if approval["status"] == "approved":
            mismatch_reason: str | None = None
            if approval["tool_name"] != call.name or approval["args_hash"] != args_sha:
                mismatch_reason = "工具名或参数与批准记录不一致"
            elif approval["expires_at"] is None or time.time() > float(approval["expires_at"]):
                mismatch_reason = "审批已过期"
            elif approval["policy_version"] != config.policy.version():
                mismatch_reason = "策略版本已变更，原审批失效"
            if mismatch_reason is not None:
                ctx.ledger.revoke(approval["approval_id"], reason=mismatch_reason)
                raise ApprovalRevoked(approval["approval_id"], mismatch_reason)
            approved_ok = True

    tool: Tool | None = next((t for t in config.registry.tools if t.name == call.name), None)
    if tool is None:
        return tool_error(call.id, "UNKNOWN_TOOL", f"没有这个工具：{call.name}")

    if not approved_ok:
        verdict = config.policy.decide(f"tool:{call.name}")
        if verdict.decision is Decision.FORBID:
            return tool_error(call.id, "POLICY_DENIED", verdict.reason)
        if verdict.decision is Decision.PROMPT:
            approval_id = ctx.ledger.request_approval(
                ctx.run_id,
                ctx.step_index,
                call.id,
                call.name,
                args_sha,
                config.policy.version(),
                reason=verdict.reason,
            )
            raise NodeInterrupt({
                "approval_id": approval_id,
                "tool": call.name,
                "tool_call_id": call.id,
                "reason": verdict.reason,
            })

    try:
        tool.validate(call.arguments)
    except ToolProtocolError as error:
        return tool_error(call.id, "INVALID_ARGUMENTS", str(error))

    runner = ToolRunner(ctx.ledger, run_id=ctx.run_id, retries=config.tool_retries,
                        timeout_s=config.tool_timeout_s)
    try:
        outcome = runner.call(tool, call.arguments, step_index=ctx.step_index)
    except ToolProtocolError as error:
        raise RunFailed(f"工具 {call.name} 协议失败：{error}") from error
    return tool_result(call.id, outcome.payload, is_error=outcome.is_error)
