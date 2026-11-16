"""执行循环：跑步骤、落 checkpoint、恢复、取消。

刻意保持很小——它只做四件事：
1. 从最新 checkpoint 决定"从第几步开始"；
2. 每步前后检查取消；
3. 按 durability 落盘；
4. 中断/取消时把状态写清楚再退出（不留孤儿）。

**恢复 = 从 checkpoint 记录的 step_index 重跑**，不是从断点续行。
所以步骤必须假设自己会被重跑——这是 S2.2 的前提，不是缺陷。

步骤没有正常跑完（中断、取消、副作用结果未知）时，落盘的是**进入该步之前**的状态快照：
步骤可能已经改了一半 state，把半步修改存下来，恢复重跑时它们会被再应用一遍。

副作用结果未知且不可重放时，run 停在 `blocked`。恢复会重新进入这一步，
账本里的 `unknown` 被 `resolve` 之前一直停在 `blocked`——不会自动重放。
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..effects import EffectOutcomeUnknown, RunAlreadyExists, validate_run_id
from .checkpointer import Checkpointer
from .ledger import EffectLedger
from .lock import run_lock
from .types import (
    Cancelled,
    CancelToken,
    Checkpoint,
    Durability,
    NodeInterrupt,
    RunFailed,
    RunReport,
    RunStatus,
    Step,
    StepContext,
    Task,
)


class AgentRuntime:
    def __init__(
        self,
        task: Task,
        state_dir: Path,
        *,
        run_id: str = "run-1",
        durability: Durability = Durability.SYNC,
        cancel: CancelToken | None = None,
        tool_delay: float = 0.0,
        crash_after_step: int | None = None,
    ) -> None:
        self.task = task
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = validate_run_id(run_id)
        self.durability = durability
        self.cancel = cancel or CancelToken()
        self.tool_delay = tool_delay
        self.crash_after_step = crash_after_step
        self.checkpointer = Checkpointer(self.state_dir)
        self.ledger = EffectLedger(self.state_dir)

    # --- 状态 ---
    def _fresh_state(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "step_index": 0, "steps_done": [], "status": RunStatus.RUNNING.value}

    def _save(self, state: dict[str, Any], parent: str | None, *, kind: str = "step") -> str:
        checkpoint = Checkpoint(
            thread_id=self.run_id,
            checkpoint_id=Checkpointer.new_checkpoint_id(),
            state=dict(state),
            parent_checkpoint_id=parent,
            ts=time.time(),
            kind=kind,
        )
        self.checkpointer.save(checkpoint, self.durability)
        return checkpoint.checkpoint_id

    # --- 主循环 ---
    def run(self, *, resume: bool = False) -> RunReport:
        # 整个 run 期间持有 run 锁；进程被杀时内核释放
        with run_lock(self.state_dir, self.run_id):
            return self._run(resume=resume)

    def _run(self, *, resume: bool) -> RunReport:
        report = RunReport(run_id=self.run_id, status=RunStatus.RUNNING)

        if resume:
            latest = self.checkpointer.latest(self.run_id)
            if latest is None:
                raise FileNotFoundError(f"没有可恢复的 checkpoint：run_id={self.run_id}")
            state = dict(latest.state)
            parent: str | None = latest.checkpoint_id
            if state.get("status") == RunStatus.COMPLETED.value:
                report.status = RunStatus.COMPLETED
                report.steps_run = list(state.get("steps_done", []))
                return report
            state["status"] = RunStatus.RUNNING.value
        else:
            if self.checkpointer.latest(self.run_id) is not None:
                raise RunAlreadyExists(self.run_id)
            state = self._fresh_state()
            parent = None

        start = int(state["step_index"])
        base_ctx = StepContext(state=state, ledger=self.ledger, cancel=self.cancel,
                               tool_delay=self.tool_delay, run_id=self.run_id)

        while True:
            index = int(state["step_index"])
            try:
                step = self._step_at(state)
            except RunFailed as error:
                # 任务在决定下一步时就判定无法继续（例如步数用尽）：此刻没有执行到一半的步
                return self._stop(report, copy.deepcopy(state), parent, RunStatus.FAILED,
                                  error=str(error))
            if step is None:
                break  # 最后一步的 checkpoint 写完、final 还没写就被杀，恢复时也走到这里
            if resume and index == start:
                report.replayed_steps = [step.name]

            # 1) 步边界检查取消（协作式，不是 kill 进程）
            if self.cancel.cancelled:
                state["status"] = RunStatus.CANCELLED.value
                self._save(state, parent, kind="cancelled")
                report.status = RunStatus.CANCELLED
                report.error = "cancelled before step"
                self.checkpointer.flush()
                return report

            # 每步一个上下文：step_index 进键的作用域。state 是同一个 dict（replace 是浅拷贝）。
            ctx = replace(base_ctx, step_index=index)
            entry = copy.deepcopy(state)  # 步入口快照：这一步没跑完时，落盘的是它
            try:
                step.action(ctx)
            except NodeInterrupt:
                return self._stop(report, entry, parent, RunStatus.INTERRUPTED)
            except Cancelled:
                return self._stop(report, entry, parent, RunStatus.CANCELLED,
                                  error=f"cancelled during {step.name}")
            except EffectOutcomeUnknown as error:
                report.blocked_on = error.key
                return self._stop(report, entry, parent, RunStatus.BLOCKED,
                                  error=f"blocked during {step.name}: {error}")
            except RunFailed as error:
                return self._stop(report, entry, parent, RunStatus.FAILED,
                                  error=f"failed during {step.name}: {error}")

            state["step_index"] = index + 1
            state["steps_done"].append(step.name)
            report.steps_run.append(step.name)

            # 2) 崩溃注入点：副作用已发生、checkpoint 还没写。
            #    这就是真实世界里最危险的那个窗口，S2.1/S2.2 要复现它。
            if self.crash_after_step is not None and index == self.crash_after_step:
                os._exit(137)

            parent = self._save(state, parent)

        state["status"] = RunStatus.COMPLETED.value
        self._save(state, parent, kind="final")
        self.checkpointer.flush()
        report.status = RunStatus.COMPLETED
        return report

    def _step_at(self, state: dict[str, Any]) -> Step | None:
        """下一步是什么。静态任务查表，动态任务问 `next_step`。"""
        if self.task.next_step is not None:
            return self.task.next_step(state)
        index = int(state["step_index"])
        return self.task.steps[index] if index < len(self.task.steps) else None

    def _stop(
        self,
        report: RunReport,
        entry: dict[str, Any],
        parent: str | None,
        status: RunStatus,
        *,
        error: str | None = None,
    ) -> RunReport:
        """步骤没有正常跑完：把步入口快照连同状态写清楚再退出（exit 档也要落盘）。"""
        entry["status"] = status.value
        self._save(entry, parent, kind=status.value)
        self.checkpointer.flush()
        report.status = status
        report.steps_run = list(entry["steps_done"])
        report.error = error
        return report
