"""M12 的回归断言（S3.4–S3.5）：切点摘要与原文保留。

验收由父进程起脚本化假模型，子进程跑 `harness agent --context-budget ... --keep-recent-tokens ...`，
检查：
- S3.4：送模 token <= budget；事件日志与 `state["messages"]` 原文 append-only 不减；
  用 `events`（`seq < summary.seq`，跳过 `"summary"`）重算切点得到的 `first_kept_seq` 与现场一致；
  `harness log verify` 退出码 0；
- S3.5：二次压缩请求正文包含第一次摘要文本（增量折叠，不从全量原文重来），原文不删、`log verify` 返回 0。
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from ..agent.compaction import estimate_messages_tokens, recompute_cut_point_from_events
from ..context.prefix import prefix_hash
from ..fake_model.server import FakeModelServer, ScriptedModel
from ..runtime.events import Event, EventLog
from .checks_m7 import _ENV, _REPO_ROOT, _payload
from .registry import CheckResult, failed, passed

_PROMPT_S34 = "记住早期代号 ORION-7731，按步骤调用 note_read 汇总结果"
_PROMPT_S35 = "记住早期代号 ORION-8842，按步骤连续读取三份资料并汇总"


def _run_agent_clean(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """显式 Popen + try/finally 清理的子进程执行，保证任何异常路径都不留孤儿进程。"""
    process = subprocess.Popen(
        [sys.executable, "-m", "bridle.cli", *args],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_ENV,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _read_checkpoint_state(state_dir: Path, run_id: str = "run-1") -> dict:
    with sqlite3.connect(state_dir / "checkpoints.sqlite") as conn:
        row = conn.execute(
            "SELECT state_json FROM checkpoints WHERE thread_id = ? "
            "ORDER BY ts DESC, checkpoint_id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    return json.loads(row[0]) if row else {}


def _read_events(state_dir: Path, run_id: str = "run-1") -> list[Event]:
    return EventLog(state_dir / "ledger.sqlite").read(run_id)


def _note_call(call_id: str, path: str) -> dict:
    return {"tool_calls": [{"id": call_id, "name": "note_read", "arguments": {"path": path}}]}


# --- S3.4 ---------------------------------------------------------------
def check_s34() -> CheckResult:
    budget = 350
    keep_recent = 160
    summary_text = "摘要1：早期代号 ORION-7731，已读取 seg1。"
    model = ScriptedModel([
        _note_call("c1", "seg1-" + "甲" * 60),
        _note_call("c2", "seg2-" + "乙" * 60),
        {"content": summary_text},                 # compact_step 调用
        {"content": "已完成：ORION-7731"},         # 压缩后最终回答
    ])
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"
        process = _run_agent_clean([
            "agent", "--state-dir", str(state_dir), "--base-url", server.base_url,
            "--prompt", _PROMPT_S34, "--context-budget", str(budget),
            "--keep-recent-tokens", str(keep_recent), "--max-steps", "12",
        ])
        payload = _payload(process)
        if process.returncode != 0 or payload.get("status") != "completed":
            return failed(
                f"agent 应完成（退出码 0）：实际 {process.returncode} / {payload.get('status')}",
                (process.stderr or process.stdout)[-400:],
            )

        steps_done = payload.get("steps_done") or []
        if "compact" not in steps_done:
            return failed(f"未触发 compact 步骤：{steps_done}")
        if model.calls != 4:
            return failed(f"模型调用次数应为 4（含 1 次摘要），实际 {model.calls}")

        # 验证压缩后发给模型的请求 token <= budget 且前缀稳定、无孤儿 tool 消息
        last_req_msgs = model.requests[-1]["messages"]
        sent_tokens = estimate_messages_tokens(last_req_msgs)
        if sent_tokens > budget:
            return failed(f"压缩后送模 token 超出预算：{sent_tokens} > {budget}")
        if prefix_hash(model.requests[0]["messages"], min_tokens=1) != prefix_hash(last_req_msgs, min_tokens=1):
            return failed("压缩后 system 前缀 hash 发生变化，破坏了 S3.2 前缀结构")
        if last_req_msgs[1].get("role") != "system" or summary_text not in (last_req_msgs[1].get("content") or ""):
            return failed(f"摘要未紧随 system_prompt 前置注入：{last_req_msgs[:3]}")
        if len(last_req_msgs) > 2 and last_req_msgs[2].get("role") == "tool":
            return failed(f"切点后尾部以孤儿 tool 消息开头：{last_req_msgs[2]}")

        # 验证原文 append-only：state["messages"] 与 step_committed 均未删减
        state = _read_checkpoint_state(state_dir)
        raw_messages = state.get("messages") or []
        if len(raw_messages) != 7:
            return failed(f"state['messages'] 原文应保留全部 7 条，实际 {len(raw_messages)} 条")
        if len(last_req_msgs) >= len(raw_messages):
            return failed(f"送模消息数应小于全量原文：{len(last_req_msgs)} vs {len(raw_messages)}")

        events = _read_events(state_dir)
        msg_commits = [
            e for e in events
            if e.type == "step_committed" and "messages" in (e.payload.get("delta") or {}).get("extend", {})
        ]
        msg_shrinks = [
            e for e in events
            if e.type == "step_committed"
            and "messages" in (e.payload.get("delta") or {}).get("set", {})
            and e.step_index != 0
        ]
        if msg_shrinks or len(msg_commits) < 4:
            return failed(f"messages 增量未保持 append-only：shrinks={len(msg_shrinks)} extends={len(msg_commits)}")

        summary_events = [e for e in events if e.type == "summary"]
        if len(summary_events) != 1:
            return failed(f"应有 1 条 summary 事件，实际 {len(summary_events)} 条")
        sev = summary_events[0]
        re_cut, re_seq = recompute_cut_point_from_events(
            events, before_seq=sev.seq, keep_recent_tokens=keep_recent
        )
        if re_seq != sev.payload.get("first_kept_seq") or re_cut != sev.payload.get("kept_from_index"):
            return failed(
                f"从事件重算切点不一致：recomputed=({re_cut}, {re_seq}) vs event=({sev.payload.get('kept_from_index')}, {sev.payload.get('first_kept_seq')})"
            )

        verify_proc = _run_agent_clean(["log", "verify", "--state-dir", str(state_dir), "--run-id", "run-1"])
        verify_payload = _payload(verify_proc)
        if verify_proc.returncode != 0 or verify_payload.get("projection_equals_cache") is not True:
            return failed(
                f"log verify 失败：{verify_proc.returncode} {verify_payload}",
                verify_proc.stderr[-300:],
            )

        return passed(
            f"步骤序列 {steps_done}，退出码 {process.returncode}；送模 {sent_tokens} <= budget={budget} token",
            f"state['messages'] 原文保留 {len(raw_messages)} 条（送模 {len(last_req_msgs)} 条），append-only extend={len(msg_commits)} 次",
            f"summary 事件 seq={sev.seq}，first_kept_seq={sev.payload['first_kept_seq']} 与重算结果 {re_seq} 一致",
            f"log verify 退出码 {verify_proc.returncode}，projection_equals_cache={verify_payload.get('projection_equals_cache')}",
        )


# --- S3.5 ---------------------------------------------------------------
def check_s35() -> CheckResult:
    budget = 350
    keep_recent = 160
    summary_1 = "第一次摘要标记-ALPHA-91：已读 s1"
    summary_2 = "第二次摘要标记-BETA-02（继承 ALPHA-91）：已读 s1、s2"
    model = ScriptedModel([
        _note_call("c1", "s1-" + "甲" * 60),
        _note_call("c2", "s2-" + "乙" * 60),
        {"content": summary_1},                     # 第 1 次 compact
        _note_call("c3", "s3-" + "丙" * 60),
        {"content": summary_2},                     # 第 2 次 compact
        {"content": "两次压缩全部完成"},
    ])
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"
        process = _run_agent_clean([
            "agent", "--state-dir", str(state_dir), "--base-url", server.base_url,
            "--prompt", _PROMPT_S35, "--context-budget", str(budget),
            "--keep-recent-tokens", str(keep_recent), "--max-steps", "16",
        ])
        payload = _payload(process)
        if process.returncode != 0 or payload.get("status") != "completed":
            return failed(
                f"agent 应完成（退出码 0）：实际 {process.returncode} / {payload.get('status')}",
                (process.stderr or process.stdout)[-400:],
            )

        steps_done = payload.get("steps_done") or []
        if steps_done.count("compact") < 2:
            return failed(f"应至少触发 2 次 compact：{steps_done}")
        if model.calls != 6:
            return failed(f"模型调用次数应为 6（4 轮对话 + 2 次摘要），实际 {model.calls}")

        second_compact_body = json.dumps(model.requests[4], ensure_ascii=False)
        if summary_1 not in second_compact_body:
            return failed(f"第二次压缩请求未包含第一次摘要文本：{second_compact_body[:300]}")
        if "s2-" not in second_compact_body or "s1-" in second_compact_body:
            return failed("第二次压缩应增量折叠 [prev_summary] + messages[prev_cut:new_cut]，而非重算早期已折叠原文")

        last_tokens = estimate_messages_tokens(model.requests[-1]["messages"])
        if last_tokens > budget:
            return failed(f"二次压缩后送模 token 超预算：{last_tokens} > {budget}")

        events = _read_events(state_dir)
        summary_events = [e for e in events if e.type == "summary"]
        if len(summary_events) != 2:
            return failed(f"应有 2 条 summary 事件，实际 {len(summary_events)}")
        for sev in summary_events:
            re_cut, re_seq = recompute_cut_point_from_events(
                events, before_seq=sev.seq, keep_recent_tokens=keep_recent
            )
            if re_seq != sev.payload.get("first_kept_seq") or re_cut != sev.payload.get("kept_from_index"):
                return failed(f"summary seq={sev.seq} 重算切点不一致")

        raw_messages = _read_checkpoint_state(state_dir).get("messages") or []
        if len(raw_messages) != 9:
            return failed(f"原文应保留全部 9 条，实际 {len(raw_messages)} 条")

        verify_proc = _run_agent_clean(["log", "verify", "--state-dir", str(state_dir), "--run-id", "run-1"])
        verify_payload = _payload(verify_proc)
        if verify_proc.returncode != 0 or verify_payload.get("projection_equals_cache") is not True:
            return failed(f"log verify 失败：{verify_proc.returncode} {verify_payload}")

        return passed(
            f"触发 2 次 compact，步骤 {steps_done}，模型调用 {model.calls} 次",
            f"第二次摘要请求包含第一次摘要文本「{summary_1}」，且只增量折叠 s2 区间（不重发 s1）",
            f"二次压缩后送模 {last_tokens} <= budget={budget} token；原文保留 {len(raw_messages)} 条",
            f"两条 summary 事件（seq={[e.seq for e in summary_events]}）重算 first_kept_seq 一致，log verify 退出码 0",
        )
