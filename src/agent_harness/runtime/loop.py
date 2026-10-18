"""执行循环：跑步骤、落 checkpoint、恢复、取消。

刻意保持很小——它只做四件事：
1. 从最新 checkpoint 决定"从第几步开始"；
2. 每步前后检查取消；
3. 按 durability 落盘；
4. 中断/取消时把状态写清楚再退出（不留孤儿）。

**恢复 = 从 checkpoint 记录的 step_index 重跑**，不是从断点续行。
所以步骤必须假设自己会被重跑——这是 S2.2 的前提，不是缺陷。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .checkpointer import Checkpointer
from .ledger import EffectLedger
from .types import (
    Cancelled,
    CancelToken,
    Checkpoint,
    Durability,
    NodeInterrupt,
    RunReport,
    RunStatus,
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
        self.run_id = run_id
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
            report.replayed_steps = [self.task.steps[state["step_index"]].name]
        else:
            state = self._fresh_state()
            parent = None

        start = int(state["step_index"])
        ctx = StepContext(state=state, ledger=self.ledger, cancel=self.cancel, tool_delay=self.tool_delay)

        for index in range(start, len(self.task.steps)):
            step = self.task.steps[index]

            # 1) 步边界检查取消（协作式，不是 kill 进程）
            if self.cancel.cancelled:
                state["status"] = RunStatus.CANCELLED.value
                self._save(state, parent, kind="cancelled")
                report.status = RunStatus.CANCELLED
                report.error = "cancelled before step"
                self.checkpointer.flush()
                return report

            try:
                step.action(ctx)
            except NodeInterrupt:
                state["status"] = RunStatus.INTERRUPTED.value
                self._save(state, parent, kind="interrupted")
                self.checkpointer.flush()
                report.status = RunStatus.INTERRUPTED
                report.steps_run = list(state["steps_done"])
                return report
            except Cancelled:
                state["status"] = RunStatus.CANCELLED.value
                self._save(state, parent, kind="cancelled")
                self.checkpointer.flush()
                report.status = RunStatus.CANCELLED
                report.steps_run = list(state["steps_done"])
                report.error = f"cancelled during {step.name}"
                return report

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
