"""M7 的回归断言（S2.5–S2.9、S4.4）：副作用判据。

和 M1 一样全部走**子进程**。区别是 M1 用 `os._exit(137)` 在账本已 `completed` 之后退出，
这里用真 `SIGKILL` 打在「意图已写、结果未写」的窗口里——那条路径 M1 没有覆盖。

出处：长任务恢复的两个追问——幂等键怎么构造；工具调用发出后进程挂了，
恢复时知不知道成功、会不会写重复。
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..effects import IdempotencyKeyConflict, Replay, scoped_key
from ..runtime.ledger import EffectLedger
from .registry import CheckResult, failed, passed

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}

EXIT_OUTCOME_UNKNOWN = 4
EXIT_RUN_EXISTS = 6


def _cli(args: list[str]) -> list[str]:
    # 直接起解释器，不经 `uv run`：要拿到的是 harness 进程自己的 pid 和返回码
    return [sys.executable, "-m", "bridle.cli", *args]


def _run(args: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(_cli(args), cwd=_REPO_ROOT, capture_output=True, text=True,
                          timeout=timeout, env=_ENV)


def _payload(process: subprocess.CompletedProcess[str]) -> dict:
    try:
        return json.loads(process.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def _effects(state_dir: Path) -> list[dict]:
    conn = sqlite3.connect(state_dir / "ledger.sqlite")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM effects ORDER BY key")]
    finally:
        conn.close()


def _attempt_outcomes(state_dir: Path) -> list[str]:
    conn = sqlite3.connect(state_dir / "ledger.sqlite")
    try:
        return [row[0] for row in conn.execute(
            "SELECT outcome FROM effect_attempts ORDER BY started_at, rowid")]
    finally:
        conn.close()


def _count(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


# --- S2.8 ---------------------------------------------------------------
def check_s28() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        # 1) 同一个 state 目录，两个 run id：各自执行，互不去重
        state_dir = Path(tmp) / "state"
        for run_id in ("r1", "r2"):
            process = _run(["run", "--state-dir", str(state_dir), "--run-id", run_id])
            if process.returncode != 0:
                return failed(f"run {run_id} 失败：退出码 {process.returncode}", process.stderr[-400:])

        rows = _effects(state_dir)
        keys = [row["key"] for row in rows]
        outcomes = _attempt_outcomes(state_dir)
        sink_rows = _count(state_dir / "sink.sqlite", "effects_log")
        if len(rows) != 2:
            return failed(f"两个 run 应各有一条账本记录，实际 {len(rows)} 条：{keys}")
        if not (keys[0].startswith("r1:1:") and keys[1].startswith("r2:1:")):
            return failed(f"键没有带 run 与 step 作用域：{keys}")
        if any(row["status"] != "completed" or row["executions"] != 1 for row in rows):
            return failed(f"两条都应各执行一次并完成：{[(r['key'], r['status'], r['executions']) for r in rows]}")
        if outcomes != ["applied", "applied"]:
            return failed(f"第二个 run 不应走去重：attempt 记录 {outcomes}")
        if sink_rows != 2:
            return failed(f"下游应有 2 行（每个 run 一行），实际 {sink_rows}")

        # 2) 不带 --resume 重跑同一个 run id：拒绝，而不是悄悄命中上一次的结果
        again = _run(["run", "--state-dir", str(state_dir), "--run-id", "r1"])
        if again.returncode != EXIT_RUN_EXISTS:
            return failed(f"重跑 r1 应被拒绝（退出码 {EXIT_RUN_EXISTS}），实际 {again.returncode}",
                          again.stderr[-400:])

        # 3) 两次都用默认 run id：只加作用域修不掉这种情况，要靠新 run 守卫
        default_dir = Path(tmp) / "default"
        first = _run(["run", "--state-dir", str(default_dir)])
        second = _run(["run", "--state-dir", str(default_dir)])
        if first.returncode != 0 or second.returncode != EXIT_RUN_EXISTS:
            return failed(
                f"默认 run id 连跑两次：应为 0 然后 {EXIT_RUN_EXISTS}，"
                f"实际 {first.returncode} 然后 {second.returncode}")
        if _count(default_dir / "sink.sqlite", "effects_log") != 1:
            return failed("默认 run id 第二次被拒后，下游行数变了")

        # 4) 同一个键、不同的意图：报冲突，不返回旧结果
        ledger = EffectLedger(Path(tmp) / "conflict")
        calls: list[int] = []

        def action() -> dict:
            calls.append(1)
            return {"ok": True}

        key = scoped_key("r1", 1, "transfer")
        ledger.attempt(key, {"op": "transfer", "amount": 1}, action, replay=Replay.UNSAFE)
        try:
            ledger.attempt(key, {"op": "transfer", "amount": 2}, action, replay=Replay.UNSAFE)
        except IdempotencyKeyConflict as error:
            conflict = str(error)
        else:
            return failed("同键不同意图没有报冲突：第二个意图拿到了第一个意图的结果")
        if len(calls) != 1:
            return failed(f"冲突时 action 不应再执行，实际执行 {len(calls)} 次")

        return passed(
            f"同一 state 目录两个 run：账本键 {keys}",
            f"两条各执行一次：attempt 记录 {outcomes}，下游 effects_log={sink_rows} 行",
            f"不带 --resume 重跑 r1：退出码 {again.returncode}（{_payload(again).get('code')}）",
            f"默认 run id 连跑两次：退出码 {first.returncode} 然后 {second.returncode}，下游仍 1 行",
            f"同键不同意图：IdempotencyKeyConflict（{conflict}），action 执行 {len(calls)} 次",
        )


# --- S4.4 ---------------------------------------------------------------
def check_s44() -> CheckResult:
    """对照 S4.2：那里是可重放工具超时后重试并收敛；这里是不可重放工具超时后**不重试**。"""
    with tempfile.TemporaryDirectory() as tmp:
        sink_dir = Path(tmp) / "sink"
        started = time.monotonic()
        # flaky_append 每次调用都是「先追加一行，再睡 2 秒」
        process = _run(["tool-retry", "--sink-dir", str(sink_dir),
                        "--tool", "flaky_append", "--timeout", "0.1"])
        wall = time.monotonic() - started
        payload = _payload(process)

        if process.returncode != EXIT_OUTCOME_UNKNOWN:
            return failed(f"退出码应为 {EXIT_OUTCOME_UNKNOWN}（结果未知），实际 {process.returncode}",
                          process.stderr[-400:])
        rows = _effects(sink_dir)
        outcomes = _attempt_outcomes(sink_dir)
        appended = _count(sink_dir / "sink.sqlite", "append_log")
        if len(rows) != 1 or rows[0]["status"] != "unknown":
            return failed(f"账本应有 1 条 unknown：{[(r['key'], r['status']) for r in rows]}")
        if rows[0]["executions"] != 1:
            return failed(f"不可重放工具超时后被重试了：executions={rows[0]['executions']}")
        if appended != 1:
            return failed(f"下游 append_log 应为 1 行，实际 {appended} 行（重试导致重复写）")
        elapsed = payload.get("elapsed_s")
        if elapsed is None or elapsed >= 1.0:
            return failed(f"超时没有限住墙钟时间：超时设 0.1s，调用耗时 {elapsed}s（工具睡 2s）")
        if wall >= 1.5:
            return failed(f"进程退出被工具线程拖住：子进程总耗时 {wall:.2f}s（工具睡 2s）")

        return passed(
            f"工具 {payload.get('tool')} 的 replay={payload.get('replay')}（未声明，按不可重放处理）",
            f"超时后退出码 {process.returncode}（{payload.get('code')}），没有重试：executions={rows[0]['executions']}",
            f"ledger 1 条 {rows[0]['status']}；attempt 记录 {outcomes}",
            f"下游 append_log={appended} 行",
            f"超时设 0.1s，调用耗时 {elapsed}s，子进程总耗时 {wall:.2f}s（工具本身睡 2s）",
        )
