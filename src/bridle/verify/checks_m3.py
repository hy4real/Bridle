"""M3 的回归断言（S4.1–S4.3）。"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from .registry import CheckResult, failed, passed

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "bridle.cli", *args],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
    )


def _json(process: subprocess.CompletedProcess[str]) -> dict:
    try:
        return json.loads(process.stdout or "{}")
    except json.JSONDecodeError:
        return {}


# --- S4.1 ---------------------------------------------------------------
def check_s41() -> CheckResult:
    full = _json(_run(["tools", "--count", "60", "--query", "", "--budget", "100000"]))
    budgeted = _json(_run(["tools", "--count", "60", "--query", "读取文件 read file", "--budget", "300"]))

    if full.get("registered") != 60:
        return failed(f"fixture 注册数不是 60：{full.get('registered')}")
    if budgeted.get("selected", 0) >= 60:
        return failed("仍然全量注入：selected == 60")
    if budgeted.get("selected_tokens", 10**9) > budgeted.get("budget", 0):
        return failed(f"注入 token {budgeted.get('selected_tokens')} 超过预算 {budgeted.get('budget')}")
    if not all(name.startswith("file_") for name in budgeted.get("names", [])):
        return failed(f"选出来的工具与查询不相关：{budgeted.get('names')}")

    return passed(
        f"注册 60 个工具，全量定义合计 {full.get('total_definition_tokens')} token",
        f"查询「读取文件 read file」→ 只选 {budgeted.get('selected')} 个，"
        f"{budgeted.get('selected_tokens')} token（预算 {budgeted.get('budget')}），跳过 {budgeted.get('skipped')} 个",
        f"被选中的是同类工具：{budgeted.get('names')}",
        "结论：工具定义按预算装载，不是全量塞进 system message",
    )


# --- S4.2 ---------------------------------------------------------------
def check_s42() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        sink_dir = Path(tmp) / "sink"
        process = _run(["tool-retry", "--sink-dir", str(sink_dir), "--timeout", "0.1"])
        payload = _json(process)
        if process.returncode != 0:
            return failed(f"tool-retry 失败：{process.returncode}", process.stderr[-400:])

        ledger = sqlite3.connect(sink_dir / "ledger.sqlite")
        rows = ledger.execute("SELECT key, status, attempts FROM effects").fetchall()
        attempts_log = ledger.execute("SELECT outcome FROM effect_attempts").fetchall()
        ledger.close()
        sink_rows = sqlite3.connect(sink_dir / "sink.sqlite").execute(
            "SELECT COUNT(*) FROM effects_log").fetchone()[0]

        if payload.get("attempts", 0) < 2:
            return failed(f"没有发生重试：attempts={payload.get('attempts')}")
        if len(rows) != 1 or rows[0][1] != "completed":
            return failed(f"账本行数/状态不对：{rows}")
        if sink_rows != 1:
            return failed(f"下游写了 {sink_rows} 次：超时重试导致重复写")

        return passed(
            f"第一次调用超时（上游已执行），重试 attempts={payload['attempts']}",
            f"ledger 1 条 completed；attempt 记录 {[a[0] for a in attempts_log]}",
            f"下游 effects_log={sink_rows} 条（幂等键 + upsert 生效）",
            "结论：「未收到响应」不等于「未执行」，重试安全靠幂等键",
        )


# --- S4.3 ---------------------------------------------------------------
def check_s43() -> CheckResult:
    business = _run(["tool-call", "--name", "note_read", "--args", '{"path":"missing"}'])
    crashed = _run(["tool-call", "--name", "explode", "--args", '{"path":"x"}'])
    invalid = _run(["tool-call", "--name", "note_read", "--args", "{}"])

    b, c, i = _json(business), _json(crashed), _json(invalid)

    if business.returncode != 0 or b.get("kind") != "business_error":
        return failed(f"业务失败路径不对：rc={business.returncode} kind={b.get('kind')}")
    if b.get("message", {}).get("is_error") is not True:
        return failed("业务失败没有在 tool message 上标 is_error")
    if b.get("message", {}).get("role") != "tool":
        return failed("业务失败没有回灌成 tool 消息")

    if crashed.returncode != 21 or c.get("kind") != "protocol_error":
        return failed(f"工具崩溃应走协议失败：rc={crashed.returncode} kind={c.get('kind')}")
    if invalid.returncode != 21 or i.get("kind") != "protocol_error":
        return failed(f"参数不合 schema 应走协议失败：rc={invalid.returncode} kind={i.get('kind')}")

    return passed(
        f"业务失败：rc=0，kind=business_error，回灌消息带 is_error=true",
        f"工具崩溃：rc=21，kind=protocol_error（{c.get('error', '')[:40]}…）",
        f"参数缺失：rc=21，kind=protocol_error（{i.get('error', '')[:40]}…）",
        "结论：业务失败进模型循环，协议失败由 harness 处理——两条路径退出码与日志都可区分",
    )
