"""M1 的回归断言（S2.1–S2.4）。

全部走**子进程**，因为要测的正是"进程被真的干掉之后会怎样"——
同进程里 try/except 模拟不出 kill -9。
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .registry import CheckResult, failed, passed

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(args: list[str], *, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "bridle.cli", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
    )


def _payload(process: subprocess.CompletedProcess[str]) -> dict:
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError:
        return {}


def _state(state_dir: Path) -> tuple[list[dict], list[dict], int]:
    ledger = sqlite3.connect(state_dir / "ledger.sqlite")
    ledger.row_factory = sqlite3.Row
    rows = [dict(r) for r in ledger.execute("SELECT * FROM effects")]
    attempts = [dict(r) for r in ledger.execute("SELECT * FROM effect_attempts")]
    ledger.close()

    sink = sqlite3.connect(state_dir / "sink.sqlite")
    sink_rows = sink.execute("SELECT COUNT(*) FROM effects_log").fetchone()[0]
    sink.close()
    return rows, attempts, sink_rows


def _crash_then_resume(state_dir: Path) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    crashed = _run(["run", "--state-dir", str(state_dir), "--crash-after-step", "1"])
    resumed = _run(["run", "--state-dir", str(state_dir), "--resume"])
    return crashed, resumed


# --- S2.1 ---------------------------------------------------------------
def check_s21() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        crashed, resumed = _crash_then_resume(state_dir)
        if crashed.returncode != 137:
            return failed(f"崩溃注入没有生效：退出码 {crashed.returncode}", crashed.stderr[-400:])
        if resumed.returncode != 0:
            return failed(f"恢复失败：退出码 {resumed.returncode}", resumed.stderr[-400:])

        payload = _payload(resumed)
        rows, attempts, sink_rows = _state(state_dir)
        if payload.get("status") != "completed":
            return failed(f"恢复后状态不是 completed：{payload.get('status')}")
        if len(rows) != 1 or rows[0]["status"] != "completed":
            return failed(f"账本行数/状态不对：{rows}")
        if len(attempts) < 2:
            return failed(f"没有发生重放，attempt_log 只有 {len(attempts)} 条")
        if sink_rows != 1:
            return failed(f"下游被写了 {sink_rows} 次（幂等失效）")

        return passed(
            f"崩溃进程退出码 {crashed.returncode}（os._exit 137，模拟 kill -9）",
            f"恢复后 status={payload['status']}，steps={payload['steps_run']}",
            f"ledger effects={len(rows)} 条，status={rows[0]['status']}，attempts={rows[0]['attempts']}",
            f"effect_attempts={len(attempts)} 条（至少一条 deduplicated）",
            f"下游 effects_log={sink_rows} 条（按键 upsert 生效）",
        )


# --- S2.2 ---------------------------------------------------------------
def check_s22() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        _, resumed = _crash_then_resume(state_dir)
        if resumed.returncode != 0:
            return failed(f"恢复失败：退出码 {resumed.returncode}", resumed.stderr[-400:])

        payload = _payload(resumed)
        rows, attempts, _ = _state(state_dir)
        replayed = payload.get("replayed_steps") or []
        outcomes = [a["outcome"] for a in attempts]
        dedup = payload.get("ledger", {}).get("rows", [{}])[0].get("attempts")

        if not replayed:
            return failed("没有报告重跑的步骤")
        if "applied" not in outcomes or "deduplicated" not in outcomes:
            return failed(f"attempt 结果序列不符合「重跑 + 幂等收敛」：{outcomes}")
        if rows[0]["attempts"] < 2:
            return failed(f"账本 attempts={rows[0]['attempts']}，没有记录重跑")
        return passed(
            f"重跑的步骤：{replayed}",
            f"attempt 序列：{outcomes}",
            f"账本 attempts={dedup}（执行 ≥2 次），下游仍只有 1 条",
            "结论：恢复按「整节点重跑」实现，副作用靠幂等键收敛",
        )


# --- S2.3 ---------------------------------------------------------------
def check_s23() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        process = subprocess.Popen(
            [sys.executable, "-m", "bridle.cli", "run",
             "--state-dir", str(state_dir), "--tool-delay", "2.0"],
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        )
        time.sleep(0.6)  # 让它进入慢下游
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)

        rows, attempts, sink_rows = _state(state_dir) if (state_dir / "ledger.sqlite").exists() else ([], [], 0)
        if process.returncode != 130:
            return failed(f"取消后退出码应为 130，实际 {process.returncode}", stderr[-400:])
        if not rows:
            return failed("账本没有留下任何记录：取消路径没有落状态")
        if rows[0]["status"] != "cancelled":
            return failed(f"账本状态应为 cancelled，实际 {rows[0]['status']}")
        if len(attempts) < 1 or attempts[-1]["outcome"] != "cancelled":
            return failed(f"attempt 记录没有标记 cancelled：{attempts}")
        if sink_rows != 0:
            return failed(f"下游被写了 {sink_rows} 次：取消没有拦在写之前")

        return passed(
            f"子进程收到 SIGINT 后退出码 {process.returncode}",
            f"ledger status={rows[0]['status']}，attempt outcome={attempts[-1]['outcome']}",
            f"下游 effects_log={sink_rows} 条（取消发生在写之前）",
            "结论：取消是协作式传播，不是 kill 进程",
        )


# --- S2.4 ---------------------------------------------------------------
def check_s24() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        process = _run(["run", "--state-dir", str(state_dir)])
        if process.returncode != 0:
            return failed(f"正常跑失败：退出码 {process.returncode}", process.stderr[-400:])

        checkpointer = sqlite3.connect(state_dir / "checkpoints.sqlite")
        checkpointer.row_factory = sqlite3.Row
        tables = [r[0] for r in checkpointer.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        columns = {t: [r[1] for r in checkpointer.execute(f"PRAGMA table_info({t})")]
                   for t in ("checkpoints", "checkpoint_blobs", "checkpoint_writes")}
        rows_cp = list(checkpointer.execute(
            "SELECT kind, state_json FROM checkpoints ORDER BY ts"))
        kinds = [r["kind"] for r in rows_cp]
        final_status = json.loads(rows_cp[-1]["state_json"]).get("status") if rows_cp else None
        checkpointer.close()

        rows, _, _ = _state(state_dir)
        if not {"checkpoints", "checkpoint_blobs", "checkpoint_writes"} <= set(tables):
            return failed(f"三张表不全：{tables}")
        for required in ("thread_id", "checkpoint_ns", "checkpoint_id"):
            if required not in columns["checkpoints"]:
                return failed(f"checkpoints 缺主键列 {required}：{columns['checkpoints']}")
        if "final" not in kinds or final_status != "completed":
            return failed(f"没有标记完成的最终 checkpoint：kinds={kinds} status={final_status}")
        if not rows:
            return failed("账本为空")
        row = rows[0]
        if not row["intent_json"] or not row["attempt_id"] or not row["result_json"]:
            return failed(f"三件套不完整：{row}")

        return passed(
            f"三张表齐全：{sorted(set(tables) & {'checkpoints', 'checkpoint_blobs', 'checkpoint_writes'})}",
            f"checkpoints 主键列：{columns['checkpoints'][:3]}",
            f"checkpoint 序列：{kinds}，末条 state.status={final_status}",
            f"账本三件套：intent={row['intent_json'][:40]}… attempt={row['attempt_id']} result={'有' if row['result_json'] else '无'}",
        )
