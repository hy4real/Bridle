"""M7 的回归断言（S2.5–S2.9、S4.4）：副作用判据。

和 M1 一样全部走**子进程**。区别是 M1 用 `os._exit(137)` 在账本已 `completed` 之后退出，
这里用真 `SIGKILL` 打在「意图已写、结果未写」的窗口里——那条路径 M1 没有覆盖。

出处：长任务恢复的两个追问——幂等键怎么构造；工具调用发出后进程挂了，
恢复时知不知道成功、会不会写重复。
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
from typing import Callable

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


# --- 真 SIGKILL 的脚手架 ---------------------------------------------------
def _peek(db_path: Path, sql: str) -> int | None:
    """子进程还活着时从旁边看一眼。只读连接：不会在子进程建库之前抢先建出一个空库。

    库还没建、表还没建、瞬时锁冲突，都算「还没到」。
    """
    try:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=0.2)
    except sqlite3.OperationalError:
        return None
    try:
        return conn.execute(sql).fetchone()[0]
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def _kill_when(args: list[str], ready: Callable[[], bool], *, deadline: float = 10.0) -> int | None:
    """起一个 harness 子进程，等 `ready()` 为真后发真 SIGKILL，返回它的返回码。

    不用固定 sleep：轮询的是「副作用是否已经到达下游」这类可观察的事实。
    到期仍未就绪返回 None。无论如何都不留孤儿进程。
    """
    process = subprocess.Popen(_cli(args), cwd=_REPO_ROOT, env=_ENV,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        expires = time.monotonic() + deadline
        while time.monotonic() < expires:
            if process.poll() is not None:
                return None  # 子进程自己退出了：没有停在窗口里
            if ready():
                os.kill(process.pid, signal.SIGKILL)
                return process.wait(timeout=10)
            time.sleep(0.02)
        return None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def _written(state_dir: Path, table: str) -> Callable[[], bool]:
    """就绪条件：下游已经有一行了（副作用已发生，账本的结果还没落）。"""
    return lambda: _peek(state_dir / "sink.sqlite", f"SELECT COUNT(*) FROM {table}") == 1


def _intent_recorded(state_dir: Path) -> Callable[[], bool]:
    """就绪条件：账本里已经有 pending 的意图（子进程停在写下游之前）。"""
    return lambda: _peek(state_dir / "ledger.sqlite",
                         "SELECT COUNT(*) FROM effects WHERE status = 'pending'") == 1


def _resume(state_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return _run(["run", "--state-dir", str(state_dir), "--resume", *extra])


_KILLED = -signal.SIGKILL
_UNSAFE = ("--effect-replay", "unsafe")


# --- S2.5 ---------------------------------------------------------------
def check_s25() -> CheckResult:
    """可重放副作用：真 SIGKILL 打在「下游已写、账本结果未写」之间，恢复后重放并收敛。"""
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        killed = _kill_when(["run", "--state-dir", str(state_dir), "--hold", "after-write"],
                            _written(state_dir, "effects_log"))
        if killed != _KILLED:
            return failed(f"子进程没有在窗口里被 SIGKILL：返回码 {killed}（应为 {_KILLED}）")

        before = _effects(state_dir)
        if len(before) != 1 or before[0]["status"] != "pending" or before[0]["executions"] != 1:
            return failed(f"被杀后账本应为 1 条 pending、executions=1：{before}")

        resumed = _resume(state_dir)
        after = _effects(state_dir)
        outcomes = _attempt_outcomes(state_dir)
        sink_rows = _count(state_dir / "sink.sqlite", "effects_log")
        if resumed.returncode != 0 or _payload(resumed).get("status") != "completed":
            return failed(f"恢复失败：退出码 {resumed.returncode}", resumed.stderr[-400:])
        if after[0]["status"] != "completed" or after[0]["executions"] != 2:
            return failed(f"恢复后应为 completed、executions=2：{after}")
        if sink_rows != 1:
            return failed(f"下游应仍为 1 行，实际 {sink_rows}（重放没有收敛）")

        return passed(
            f"子进程被真 SIGKILL：returncode={killed}（不是 os._exit）",
            f"被杀时：下游已写 1 行，账本 status={before[0]['status']}、executions={before[0]['executions']}",
            f"恢复后：退出码 {resumed.returncode}，账本 status={after[0]['status']}、executions={after[0]['executions']}",
            f"attempt 记录 {outcomes}；下游 effects_log={sink_rows} 行（按键 upsert 收敛）",
        )


# --- S2.6 ---------------------------------------------------------------
def check_s26() -> CheckResult:
    """不可重放副作用：同一个窗口，恢复不重放，停在 blocked。再恢复一次也一样。"""
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        killed = _kill_when(["run", "--state-dir", str(state_dir), *_UNSAFE, "--hold", "after-write"],
                            _written(state_dir, "append_log"))
        if killed != _KILLED:
            return failed(f"子进程没有在窗口里被 SIGKILL：返回码 {killed}（应为 {_KILLED}）")

        codes, statuses = [], []
        for _ in range(2):
            resumed = _resume(state_dir, *_UNSAFE)
            codes.append(resumed.returncode)
            statuses.append(_payload(resumed).get("status"))
        rows = _effects(state_dir)
        appended = _count(state_dir / "sink.sqlite", "append_log")
        payload = _payload(resumed)

        if codes != [EXIT_OUTCOME_UNKNOWN] * 2 or statuses != ["blocked"] * 2:
            return failed(f"两次恢复都应退出码 {EXIT_OUTCOME_UNKNOWN}、状态 blocked：{codes} {statuses}",
                          resumed.stderr[-400:])
        if rows[0]["status"] != "unknown" or rows[0]["executions"] != 1:
            return failed(f"账本应为 unknown、executions=1：{rows}")
        if appended != 1:
            return failed(f"下游 append_log 应为 1 行，实际 {appended} 行（不可重放的操作被重放了）")

        return passed(
            f"子进程被真 SIGKILL：returncode={killed}，此时下游 append_log 已有 1 行",
            f"连续两次 --resume：退出码 {codes}，状态 {statuses}，卡在 {payload.get('blocked_on')}",
            f"账本 status={rows[0]['status']}、executions={rows[0]['executions']}、replay={rows[0]['replay']}",
            f"下游 append_log={appended} 行（没有被重放）",
        )


# --- S2.7 ---------------------------------------------------------------
def _block_then_resolve(tmp: str, name: str, hold: str, ready_table: str | None,
                        outcome: str) -> tuple[str | None, dict]:
    """杀在指定窗口 → 恢复到 blocked → 人工解除 → 再恢复。返回 (失败原因, 观测值)。"""
    state_dir = Path(tmp) / name
    ready = _written(state_dir, ready_table) if ready_table else _intent_recorded(state_dir)
    killed = _kill_when(["run", "--state-dir", str(state_dir), *_UNSAFE, "--hold", hold], ready)
    if killed != _KILLED:
        return f"{name}：子进程没有在窗口里被 SIGKILL：返回码 {killed}", {}
    seen = {"appended_at_kill": _count(state_dir / "sink.sqlite", "append_log")}

    blocked = _resume(state_dir, *_UNSAFE)
    if blocked.returncode != EXIT_OUTCOME_UNKNOWN:
        return f"{name}：解除前恢复应退出码 {EXIT_OUTCOME_UNKNOWN}，实际 {blocked.returncode}", seen
    key = _payload(blocked).get("blocked_on")
    resolve = _run(["effect", "resolve", "--state-dir", str(state_dir), "--key", str(key),
                    "--outcome", outcome, "--by", "verify-s27"])
    if resolve.returncode != 0:
        return f"{name}：effect resolve 失败：退出码 {resolve.returncode} {resolve.stdout[-200:]}", seen

    final = _resume(state_dir, *_UNSAFE)
    row = _effects(state_dir)[0]
    seen.update(key=key, final_code=final.returncode, status=row["status"],
                executions=row["executions"], resolved_by=row["resolved_by"],
                appended=_count(state_dir / "sink.sqlite", "append_log"))
    if final.returncode != 0 or row["status"] != "completed":
        return f"{name}：解除后恢复应完成：退出码 {final.returncode}，账本 {row['status']}", seen
    return None, seen


def check_s27() -> CheckResult:
    """unknown 只有人能带走。说「已执行」就不再执行；说「没执行」就恰好再执行一次。"""
    with tempfile.TemporaryDirectory() as tmp:
        # (a) 写完之后被杀 → 人确认已执行
        error, a = _block_then_resolve(tmp, "applied", "after-write", "append_log", "applied")
        if error:
            return failed(error)
        if a["appended_at_kill"] != 1 or a["appended"] != 1 or a["executions"] != 1:
            return failed(f"(a) 确认已执行后不应再执行：{a}")
        if not a["resolved_by"]:
            return failed(f"(a) 账本没有记下是谁解除的：{a}")

        # (b) 还没写就被杀 → 人确认没执行
        error, b = _block_then_resolve(tmp, "not-applied", "before-write", None, "not-applied")
        if error:
            return failed(error)
        if b["appended_at_kill"] != 0:
            return failed(f"(b) 被杀时下游应为 0 行：{b}")
        if b["appended"] != 1 or b["executions"] != 2:
            return failed(f"(b) 确认没执行后应恰好再执行一次：{b}")

        return passed(
            f"(a) 写后被杀：下游 {a['appended_at_kill']} 行 → resolve applied（by {a['resolved_by']}）"
            f" → 恢复退出码 {a['final_code']}，executions={a['executions']}，下游仍 {a['appended']} 行",
            f"(b) 写前被杀：下游 {b['appended_at_kill']} 行 → resolve not-applied"
            f" → 恢复退出码 {b['final_code']}，executions={b['executions']}，下游 {b['appended']} 行",
            f"两种情况解除前恢复都是退出码 {EXIT_OUTCOME_UNKNOWN}：账本无法区分「写了」和「没写」，所以不猜",
            f"解除的键：{a['key']}",
        )


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
