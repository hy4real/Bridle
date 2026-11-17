"""M9 的回归断言（S12.1–S12.3）：事件日志是真相，checkpoint 是投影缓存。

出处：S12.1、S12.2 是「日志是唯一真相」（deepseek-harness 的做法）；S12.3 是
「恢复的时候知不知道自己做成了」在状态层的同一个问题。
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable

from .checks_m7 import _KILLED, _attempt_outcomes, _count, _effects, _kill_when, _payload, _peek, _run
from .registry import CheckResult, failed, passed


def _events(state_dir: Path, run_id: str = "run-1") -> list[tuple[int, str, int | None]]:
    conn = sqlite3.connect(state_dir / "ledger.sqlite")
    try:
        return [tuple(row) for row in conn.execute(
            "SELECT seq, type, step_index FROM events WHERE run_id = ? ORDER BY seq", (run_id,))]
    finally:
        conn.close()


def _run_types(events: list[tuple[int, str, int | None]]) -> list[str]:
    return [type_ for _, type_, _ in events if not type_.startswith("effect_")]


def _log_verify(state_dir: Path, run_id: str = "run-1"):
    process = _run(["log", "verify", "--state-dir", str(state_dir), "--run-id", run_id])
    return process, _payload(process)


# --- S12.1 --------------------------------------------------------------
def check_s121() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        process = _run(["run", "--state-dir", str(state_dir)])
        if process.returncode != 0:
            return failed(f"正常跑失败：退出码 {process.returncode}", process.stderr[-400:])

        events = _events(state_dir)
        run_types = _run_types(events)
        expected = ["run_started", "step_committed", "step_committed", "step_committed", "run_finished"]
        if run_types != expected:
            return failed(f"run 级事件序列不对：{run_types}")
        seqs = [seq for seq, _, _ in events]
        if seqs != list(range(1, len(events) + 1)):
            return failed(f"seq 有空洞或乱序：{seqs}")

        verify, payload = _log_verify(state_dir)
        if verify.returncode != 0 or payload.get("projection_equals_cache") is not True:
            return failed(f"log verify 应退出码 0 且投影等于缓存：{verify.returncode} {payload}",
                          verify.stderr[-400:])

        return passed(
            f"共 {len(events)} 条事件，seq 1..{len(events)} 连续；run 级：{run_types}",
            f"副作用事件：{[t for _, t, _ in events if t.startswith('effect_')]}（与账本行同事务写入）",
            f"log verify：退出码 {verify.returncode}，last_seq={payload.get('last_seq')}，"
            f"projection_equals_cache={payload.get('projection_equals_cache')}",
        )


# --- S12.2 --------------------------------------------------------------
def check_s122() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        if _run(["run", "--state-dir", str(state_dir)]).returncode != 0:
            return failed("正常跑失败")
        sink_before = _count(state_dir / "sink.sqlite", "effects_log")
        commits_before = _run_types(_events(state_dir)).count("step_committed")
        attempts_before = _effects(state_dir)[0]["attempts"]

        # 把缓存改坏：最新一条 checkpoint 的 step_index 改成 0
        conn = sqlite3.connect(state_dir / "checkpoints.sqlite")
        try:
            conn.execute(
                """UPDATE checkpoints SET state_json = json_set(state_json, '$.step_index', 0)
                   WHERE rowid = (SELECT rowid FROM checkpoints WHERE thread_id = 'run-1'
                                  ORDER BY ts DESC, checkpoint_id DESC LIMIT 1)""")
            conn.commit()
        finally:
            conn.close()

        broken, broken_payload = _log_verify(state_dir)
        if broken.returncode != 1 or broken_payload.get("projection_equals_cache") is not False:
            return failed(f"缓存改坏后 log verify 应退出码 1：{broken.returncode} {broken_payload}")

        resumed = _run(["run", "--state-dir", str(state_dir), "--resume"])
        payload = _payload(resumed)
        if resumed.returncode != 0 or payload.get("status") != "completed":
            return failed(f"以日志为准恢复应完成：{resumed.returncode} {payload.get('status')}",
                          resumed.stderr[-400:])
        sink_after = _count(state_dir / "sink.sqlite", "effects_log")
        if sink_after != sink_before:
            return failed(f"下游行数变了：{sink_before} → {sink_after}（按改坏的缓存重跑了步骤）")
        types = _run_types(_events(state_dir))
        commits_after = types.count("step_committed")
        attempts_after = _effects(state_dir)[0]["attempts"]
        if commits_after != commits_before or attempts_after != attempts_before:
            return failed(f"按改坏的缓存重跑了步骤：step_committed {commits_before} → {commits_after}，"
                          f"账本 attempts {attempts_before} → {attempts_after}")
        if "projection_repaired" not in types:
            return failed(f"没有留下 projection_repaired：{types}")

        fixed, fixed_payload = _log_verify(state_dir)
        if fixed.returncode != 0:
            return failed(f"修复后 log verify 应退出码 0：{fixed.returncode} {fixed_payload}")

        return passed(
            f"缓存 step_index 改成 0 后：log verify 退出码 {broken.returncode}，"
            f"不一致的键 {broken_payload.get('mismatched_keys')}",
            f"--resume：退出码 {resumed.returncode}，状态 {payload['status']}；没有重跑步骤："
            f"step_committed {commits_before} → {commits_after}，账本 attempts {attempts_before} → {attempts_after}，"
            f"下游 {sink_before} → {sink_after} 行",
            f"run 级事件：{types}",
            f"修复后 log verify：退出码 {fixed.returncode}",
        )


# --- S12.3 --------------------------------------------------------------
def _committed(state_dir: Path, step_index: int) -> Callable[[], bool]:
    return lambda: _peek(
        state_dir / "ledger.sqlite",
        f"SELECT COUNT(*) FROM events WHERE type = 'step_committed' AND step_index = {step_index}") == 1


def check_s123() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp) / "state"
        killed = _kill_when(["run", "--state-dir", str(state_dir), "--hold-after-commit", "1"],
                            _committed(state_dir, 1))
        if killed != _KILLED:
            return failed(f"子进程没有停在「事件已提交、缓存未写」处被 SIGKILL：返回码 {killed}")

        resumed = _run(["run", "--state-dir", str(state_dir), "--resume"])
        payload = _payload(resumed)
        if resumed.returncode != 0 or payload.get("status") != "completed":
            return failed(f"恢复失败：{resumed.returncode} {payload.get('status')}", resumed.stderr[-400:])
        if payload.get("steps_run") != ["report"]:
            return failed(f"恢复后应只执行第 2 步 report：{payload.get('steps_run')}")
        rows = _effects(state_dir)
        outcomes = _attempt_outcomes(state_dir)
        if rows[0]["executions"] != 1 or rows[0]["attempts"] != 1:
            return failed(f"已提交的副作用步被重跑了：executions={rows[0]['executions']} "
                          f"attempts={rows[0]['attempts']}")
        if "deduplicated" in outcomes:
            return failed(f"恢复走了去重，说明那一步又进了账本：{outcomes}")
        sink_rows = _count(state_dir / "sink.sqlite", "effects_log")

        return passed(
            f"第 1 步的 step_committed 写入后、缓存写入前被真 SIGKILL：returncode={killed}",
            f"恢复：退出码 {resumed.returncode}，只执行 {payload['steps_run']}",
            f"账本 executions={rows[0]['executions']}、attempts={rows[0]['attempts']}，attempt 记录 {outcomes}",
            f"下游 effects_log={sink_rows} 行；run 级事件 {_run_types(_events(state_dir))}",
        )
