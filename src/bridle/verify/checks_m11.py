"""M11 的回归断言（S5.4–S5.5）：审批记录跨进程存活，恢复前校验策略版本与过期时间。

出处：dsh-superagent-prototype 的 Mandate 校验机制（决定 4）。
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

from ..fake_model.server import FakeModelServer, ScriptedModel
from .checks_m7 import EXIT_OUTCOME_UNKNOWN, _count, _effects, _payload, _run
from .checks_m8 import _agent, _write_call
from .registry import CheckResult, failed, passed

EXIT_INTERRUPTED = 3


def _approvals(state_dir: Path) -> list[dict]:
    conn = sqlite3.connect(state_dir / "ledger.sqlite")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM approvals ORDER BY created_at, rowid")]
    finally:
        conn.close()


def _approval_events(state_dir: Path, run_id: str = "run-1") -> list[str]:
    conn = sqlite3.connect(state_dir / "ledger.sqlite")
    try:
        return [
            row[0]
            for row in conn.execute(
                "SELECT type FROM events WHERE run_id = ? AND type LIKE 'approval_%' ORDER BY seq",
                (run_id,),
            )
        ]
    finally:
        conn.close()


def _sink_rows(state_dir: Path) -> int:
    db = state_dir / "sink.sqlite"
    if not db.exists():
        return 0
    return _count(db, "effects_log")


# --- S5.4 ---------------------------------------------------------------
def check_s54() -> CheckResult:
    model = ScriptedModel([_write_call(), {"content": "已写入 notes/a.md"}])
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"

        interrupted = _run(_agent(state_dir, server, "--ask-tool", "tool:flaky_write*"))
        interrupted_payload = _payload(interrupted)
        if interrupted.returncode != EXIT_INTERRUPTED or interrupted_payload.get("status") != "interrupted":
            return failed(
                f"--ask-tool 应中断（退出码 {EXIT_INTERRUPTED}）："
                f"实际 {interrupted.returncode} / {interrupted_payload.get('status')}",
                interrupted.stderr[-400:],
            )
        if _sink_rows(state_dir) != 0:
            return failed("中断时工具不应执行，下游却有记录")

        pending = _approvals(state_dir)
        if len(pending) != 1 or pending[0]["status"] != "pending":
            return failed(f"中断后应有一条 pending 审批记录：{pending}")
        aid = pending[0]["approval_id"]

        approved = _run([
            "approve", "--state-dir", str(state_dir), "--run-id", "run-1", "--by", "verify-s54"
        ])
        approved_payload = _payload(approved)
        if approved.returncode != 0 or approved_payload.get("status") != "approved":
            return failed(
                f"harness approve 失败：退出码 {approved.returncode} {approved.stdout[-200:]}",
                approved.stderr[-400:],
            )

        resumed = _run(_agent(state_dir, server, "--ask-tool", "tool:flaky_write*", "--resume"))
        resumed_payload = _payload(resumed)
        if resumed.returncode != 0 or resumed_payload.get("status") != "completed":
            return failed(
                f"approve 后 --resume 应完成（退出码 0）："
                f"实际 {resumed.returncode} / {resumed_payload.get('status')}",
                resumed.stderr[-400:],
            )

        sink_rows = _sink_rows(state_dir)
        if sink_rows != 1:
            return failed(f"恢复后下游应有 1 行，实际 {sink_rows}")
        tool_rows = [row for row in _effects(state_dir) if ":flaky_write:" in row["key"]]
        if len(tool_rows) != 1 or tool_rows[0]["executions"] != 1:
            return failed(f"工具副作用 executions 应为 1：{tool_rows}")
        if model.calls != 2:
            return failed(f"模型应被调用 2 次，实际 {model.calls}")
        events = _approval_events(state_dir)
        if events != ["approval_requested", "approval_granted"]:
            return failed(f"审批事件序列不对：{events}")

        return passed(
            f"首次 --ask-tool 'tool:flaky_write*'：退出码 {interrupted.returncode}，status=interrupted，approval_id={aid}",
            f"harness approve：退出码 {approved.returncode}，status={approved_payload['status']}，by={approved_payload.get('granted_by')}",
            f"--resume：退出码 {resumed.returncode}，status={resumed_payload['status']}（不再中断）",
            f"工具 executions={tool_rows[0]['executions']}，下游 effects_log={sink_rows} 行，审批事件 {events}",
        )


# --- S5.5 ---------------------------------------------------------------
def check_s55() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        # (a) 批准后改策略（--ask-tool → --deny-tool）再 --resume：必须吊销并停在 blocked，不能吞成 POLICY_DENIED
        model_a = ScriptedModel([_write_call(), {"content": "不应继续"}])
        with FakeModelServer(model_a) as server_a:
            dir_a = Path(tmp) / "policy-changed"
            first_a = _run(_agent(dir_a, server_a, "--ask-tool", "tool:flaky_write*"))
            if first_a.returncode != EXIT_INTERRUPTED:
                return failed(f"(a) 首次应中断（退出码 {EXIT_INTERRUPTED}）：实际 {first_a.returncode}",
                              first_a.stderr[-300:])
            pending_a = _approvals(dir_a)
            if len(pending_a) != 1:
                return failed(f"(a) 中断后应有 1 条审批记录：{pending_a}")
            aid_a = pending_a[0]["approval_id"]
            if _run(["approve", "--state-dir", str(dir_a), "--run-id", "run-1", "--by", "verify-s55"]).returncode != 0:
                return failed("(a) approve 失败")

            resumed_a = _run(_agent(dir_a, server_a, "--deny-tool", "tool:flaky_write*", "--resume"))
            payload_a = _payload(resumed_a)
            if resumed_a.returncode != EXIT_OUTCOME_UNKNOWN or payload_a.get("status") != "blocked":
                return failed(
                    f"(a) 改策略后恢复应 blocked（退出码 {EXIT_OUTCOME_UNKNOWN}）："
                    f"实际 {resumed_a.returncode} / {payload_a.get('status')}",
                    resumed_a.stderr[-400:],
                )
            if payload_a.get("blocked_on") != aid_a:
                return failed(f"(a) blocked_on 应为 {aid_a}，实际 {payload_a.get('blocked_on')}")
            row_a = _approvals(dir_a)[0]
            events_a = _approval_events(dir_a)
            sink_a = _sink_rows(dir_a)
            if (row_a["status"] != "revoked"
                    or events_a != ["approval_requested", "approval_granted", "approval_revoked"]
                    or sink_a != 0):
                return failed(f"(a) 吊销状态/事件/下游不对：row={row_a} events={events_a} sink={sink_a}")
            if model_a.calls != 1:
                return failed(f"(a) 不应继续询问模型：calls={model_a.calls}")

        # (b) 批准时给极短 TTL（--ttl 0.01），过期后用同一策略 --resume：同样吊销并停在 blocked
        model_b = ScriptedModel([_write_call(), {"content": "不应继续"}])
        with FakeModelServer(model_b) as server_b:
            dir_b = Path(tmp) / "expired"
            first_b = _run(_agent(dir_b, server_b, "--ask-tool", "tool:flaky_write*"))
            if first_b.returncode != EXIT_INTERRUPTED:
                return failed(f"(b) 首次应中断（退出码 {EXIT_INTERRUPTED}）：实际 {first_b.returncode}",
                              first_b.stderr[-300:])
            pending_b = _approvals(dir_b)
            if len(pending_b) != 1:
                return failed(f"(b) 中断后应有 1 条审批记录：{pending_b}")
            aid_b = pending_b[0]["approval_id"]
            if _run(["approve", "--state-dir", str(dir_b), "--run-id", "run-1",
                     "--ttl", "0.01", "--by", "verify-s55"]).returncode != 0:
                return failed("(b) approve --ttl 0.01 失败")
            time.sleep(0.05)

            resumed_b = _run(_agent(dir_b, server_b, "--ask-tool", "tool:flaky_write*", "--resume"))
            payload_b = _payload(resumed_b)
            if resumed_b.returncode != EXIT_OUTCOME_UNKNOWN or payload_b.get("status") != "blocked":
                return failed(
                    f"(b) 过期后恢复应 blocked（退出码 {EXIT_OUTCOME_UNKNOWN}）："
                    f"实际 {resumed_b.returncode} / {payload_b.get('status')}",
                    resumed_b.stderr[-400:],
                )
            if payload_b.get("blocked_on") != aid_b:
                return failed(f"(b) blocked_on 应为 {aid_b}，实际 {payload_b.get('blocked_on')}")
            row_b = _approvals(dir_b)[0]
            events_b = _approval_events(dir_b)
            sink_b = _sink_rows(dir_b)
            if (row_b["status"] != "revoked"
                    or events_b != ["approval_requested", "approval_granted", "approval_revoked"]
                    or sink_b != 0):
                return failed(f"(b) 吊销状态/事件/下游不对：row={row_b} events={events_b} sink={sink_b}")
            if model_b.calls != 1:
                return failed(f"(b) 不应继续询问模型：calls={model_b.calls}")

        return passed(
            f"(a) 改 --deny-tool 后 --resume：退出码 {resumed_a.returncode}，status={payload_a['status']}，"
            f"blocked_on={payload_a['blocked_on']}，approvals.status={row_a['status']}，下游 {sink_a} 行",
            f"(a) 审批事件：{events_a}（原因：{row_a['revoked_reason']}）",
            f"(b) approve --ttl 0.01 过期后 --resume：退出码 {resumed_b.returncode}，status={payload_b['status']}，"
            f"blocked_on={payload_b['blocked_on']}，approvals.status={row_b['status']}，下游 {sink_b} 行",
            f"(b) 审批事件：{events_b}（原因：{row_b['revoked_reason']}）",
        )
