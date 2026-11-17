"""M10 的回归断言（S12.4）：重启上限（毒丸保护）。

连续恢复且没有任何新 `step_committed` 达到上限（默认 5）时，
在发射 `run_resumed` 之前熔断为 `faulted`（退出码 5），不再执行任何步骤；
一旦提交新步骤，计数清零重计，后续可恢复完成。
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from ..runtime.events import Event, EventLog, restarts_since_commit
from .checks_m7 import _ENV, _REPO_ROOT, _count, _payload
from .registry import CheckResult, failed, passed

EXIT_FAULTED = 5


def _run_clean(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    """显式 Popen + try/finally 清理的子进程执行，确保任何路径不留孤儿进程。"""
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


def _read_events(state_dir: Path, run_id: str = "run-1") -> list[Event]:
    return EventLog(state_dir / "ledger.sqlite").read(run_id)


def _latest_checkpoint_kind(state_dir: Path, run_id: str = "run-1") -> tuple[str | None, str | None]:
    with sqlite3.connect(state_dir / "checkpoints.sqlite") as conn:
        row = conn.execute(
            "SELECT kind, state_json FROM checkpoints WHERE thread_id = ? "
            "ORDER BY ts DESC, checkpoint_id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
    if row is None:
        return None, None
    return row[0], json.loads(row[1]).get("status")


# --- S12.4 --------------------------------------------------------------
def check_s124() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        # ① 默认上限 5：--crash-after-step 0 连续恢复 5 次无新提交，第 6 次熔断为 faulted，第 7 次幂等返回
        dir_a = Path(tmp) / "poison"
        init_a = _run_clean(["run", "--state-dir", str(dir_a), "--crash-after-step", "0"])
        if init_a.returncode != 137:
            return failed(f"① 首次崩溃注入未生效：退出码 {init_a.returncode}", init_a.stderr[-400:])

        for attempt in range(1, 6):
            crashed = _run_clean([
                "run", "--state-dir", str(dir_a), "--resume", "--crash-after-step", "0",
            ])
            if crashed.returncode != 137:
                return failed(
                    f"① 第 {attempt} 次恢复应允许重跑并在第 0 步崩溃（137），实际 {crashed.returncode}",
                    crashed.stderr[-400:],
                )
            cnt = restarts_since_commit(_read_events(dir_a))
            if cnt != attempt:
                return failed(f"① 第 {attempt} 次恢复后 restarts_since_commit 应为 {attempt}，实际 {cnt}")

        # 第 6 次 --resume（仍带 --crash-after-step 0）：必须在执行第 0 步之前熔断，不得退出 137
        sixth = _run_clean([
            "run", "--state-dir", str(dir_a), "--resume", "--crash-after-step", "0",
        ])
        sixth_payload = _payload(sixth)
        if sixth.returncode != EXIT_FAULTED or sixth_payload.get("status") != "faulted":
            return failed(
                f"① 第 6 次 --resume 应熔断为 faulted（退出码 {EXIT_FAULTED}）："
                f"实际 {sixth.returncode} / {sixth_payload.get('status')}",
                (sixth.stderr or sixth.stdout)[-400:],
            )
        if sixth_payload.get("steps_run") != [] or sixth_payload.get("replayed_steps") != []:
            return failed(f"① 熔断时不应执行任何步骤：{sixth_payload}")
        if not sixth_payload.get("error"):
            return failed(f"① 熔断应保留错误说明：{sixth_payload}")

        events_6 = _read_events(dir_a)
        types_6 = [e.type for e in events_6]
        if types_6.count("step_committed") != 0 or types_6.count("run_resumed") != 5:
            return failed(f"① 第 6 次不应写 run_resumed 或 step_committed：{types_6}")
        if events_6[-1].type != "run_stopped" or events_6[-1].payload.get("status") != "faulted":
            return failed(f"① 末条事件应为 run_stopped(faulted)：{events_6[-1]}")
        cp_kind, cp_status = _latest_checkpoint_kind(dir_a)
        if cp_kind != "faulted" or cp_status != "faulted":
            return failed(f"① 最新缓存应为 kind=faulted / status=faulted，实际 {cp_kind}/{cp_status}")

        # 第 7 次 --resume：直接返回 faulted，不新增 run_resumed 或 run_stopped
        seventh = _run_clean(["run", "--state-dir", str(dir_a), "--resume"])
        seventh_payload = _payload(seventh)
        if seventh.returncode != EXIT_FAULTED or seventh_payload.get("status") != "faulted":
            return failed(
                f"① 第 7 次 --resume 应仍返回退出码 {EXIT_FAULTED} / faulted："
                f"实际 {seventh.returncode} / {seventh_payload.get('status')}",
                (seventh.stderr or seventh.stdout)[-400:],
            )
        if seventh_payload.get("error") != sixth_payload.get("error"):
            return failed(
                f"① 第 7 次应保留原始 faulted 错误信息：{seventh_payload.get('error')!r} vs {sixth_payload.get('error')!r}"
            )
        events_7 = _read_events(dir_a)
        if len(events_7) != len(events_6) or [e.type for e in events_7] != types_6:
            return failed(
                f"① 第 7 次不得新增事件（尤其是 run_resumed / run_stopped）："
                f"{[e.type for e in events_7]} vs {types_6}"
            )
        verify_a = _run_clean(["log", "verify", "--state-dir", str(dir_a), "--run-id", "run-1"])
        verify_a_payload = _payload(verify_a)
        if verify_a.returncode != 0 or verify_a_payload.get("projection_equals_cache") is not True:
            return failed(f"① faulted 后 log verify 应为 0：{verify_a.returncode} {verify_a_payload}")

        # ② 崩 2 次 → 提交 1 步 → 再崩 1 次后 restarts_since_commit == 1 → 继续恢复完成
        # 配 --restart-limit 3：全程共恢复 5 次（> 3），只有 step_committed 真正清零才跑得完
        dir_b = Path(tmp) / "reset"
        if _run_clean([
            "run", "--state-dir", str(dir_b), "--crash-after-step", "0", "--restart-limit", "3",
        ]).returncode != 137:
            return failed("② 首次第 0 步崩溃注入失败")
        for n in (1, 2):
            r = _run_clean([
                "run", "--state-dir", str(dir_b), "--resume", "--crash-after-step", "0", "--restart-limit", "3",
            ])
            if r.returncode != 137:
                return failed(f"② 提交前第 {n} 次恢复崩溃失败：退出码 {r.returncode}", r.stderr[-300:])

        # 提交第 0 步（plan），并在第 1 步（apply_effect）崩溃
        commit_and_crash = _run_clean([
            "run", "--state-dir", str(dir_b), "--resume", "--crash-after-step", "1", "--restart-limit", "3",
        ])
        if commit_and_crash.returncode != 137:
            return failed(f"② 第 0 步提交后应停在第 1 步崩溃（137），实际 {commit_and_crash.returncode}")
        events_after_commit = _read_events(dir_b)
        if [e.type for e in events_after_commit].count("step_committed") != 1:
            return failed(f"② 此时应恰好提交 1 步：{[e.type for e in events_after_commit]}")
        if restarts_since_commit(events_after_commit) != 0:
            return failed(f"② 提交后计数未清零：{restarts_since_commit(events_after_commit)}")

        # 提交后再崩 1 次（第 1 步）
        crash_after_commit = _run_clean([
            "run", "--state-dir", str(dir_b), "--resume", "--crash-after-step", "1", "--restart-limit", "3",
        ])
        if crash_after_commit.returncode != 137:
            return failed(f"② 提交后再崩 1 次应退出 137，实际 {crash_after_commit.returncode}")
        events_restarted = _read_events(dir_b)
        restarts_after = restarts_since_commit(events_restarted)
        if restarts_after != 1:
            return failed(f"② 提交后再崩 1 次后 restarts_since_commit 应为 1，实际 {restarts_after}")

        # 恢复完成
        done = _run_clean(["run", "--state-dir", str(dir_b), "--resume", "--restart-limit", "3"])
        done_payload = _payload(done)
        if done.returncode != 0 or done_payload.get("status") != "completed":
            return failed(
                f"② 清零后应继续恢复完成（退出码 0）：实际 {done.returncode} / {done_payload.get('status')}",
                (done.stderr or done.stdout)[-400:],
            )
        sink_rows = _count(dir_b / "sink.sqlite", "effects_log")
        if sink_rows != 1:
            return failed(f"② 下游应收敛为 1 行，实际 {sink_rows}")
        final_events = _read_events(dir_b)
        total_resumes = sum(1 for e in final_events if e.type == "run_resumed")
        if total_resumes <= 3:
            return failed(f"② 全程恢复次数应大于 restart_limit=3 以证明清零，实际 {total_resumes}")
        verify_b = _run_clean(["log", "verify", "--state-dir", str(dir_b), "--run-id", "run-1"])
        verify_b_payload = _payload(verify_b)
        if verify_b.returncode != 0 or verify_b_payload.get("projection_equals_cache") is not True:
            return failed(f"② 完成后 log verify 失败：{verify_b.returncode} {verify_b_payload}")

        return passed(
            f"① --crash-after-step 0 连续恢复 5 次无新提交后：第 6 次 --resume 退出码 {sixth.returncode}、"
            f"status={sixth_payload['status']}、steps_run={sixth_payload['steps_run']}、checkpoints={sixth_payload['checkpoints']}",
            f"① 第 7 次 --resume：仍退出码 {seventh.returncode}、status={seventh_payload['status']}，"
            f"events={len(events_7)} 条与第 6 次后完全一致（不新增 run_resumed/run_stopped），log verify=0",
            f"② --restart-limit 3：先崩 2 次再提交 1 步后计数清零，再崩 1 次 restarts_since_commit={restarts_after}",
            f"② 随后 --resume 退出码 {done.returncode}（status={done_payload['status']}，effects_log={sink_rows} 行，"
            f"全程 run_resumed={total_resumes} 次 > 上限 3，log verify=0）",
        )
