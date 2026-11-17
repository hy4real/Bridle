"""执行循环：跑步骤、提交、恢复、取消。

**真相源是事件日志**（`runtime/events.py`），checkpoint 是由它算出来的缓存：
1. 新 run 先写 `run_started`（带初始 state）；
2. 每步执行完，把它对 state 的增量写成一条 `step_committed`——**这一条写进去，这一步就提交了**；
3. 然后才写 checkpoint 缓存。缓存写没写成都不影响提交；
4. 停下（中断、取消、结果未知、失败）写 `run_stopped`，跑完写 `run_finished`。

恢复：由事件重算 state（投影），再和最新的缓存比。不一致以日志为准，记一条
`projection_repaired` 并用投影重写缓存。

**恢复 = 从投影里的 step_index 重跑**，不是从断点续行。没提交的步会重跑，
所以步骤必须假设自己会被重跑——这是 S2.2 的前提，不是缺陷。

步骤没有正常跑完时，落盘的是**进入该步之前**（也就是最后一次提交）的 state：
步骤可能已经改了一半 state，把半步修改存下来，恢复重跑时它们会被再应用一遍。

副作用结果未知且不可重放，或已有审批失效/吊销时，run 停在 `blocked`。恢复会重新进入这一步，
账本里的 `unknown` 被 `resolve` 之前一直停在 `blocked`——不会自动重放。

durability：`sync` 每个事件、每份缓存立刻落盘；`exit` 都攒在内存里，run 结束或停下时
一次写掉（崩溃 = 这次运行全丢；此时缓存里的 `last_seq` 记为 0）。
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..effects import ApprovalRevoked, EffectOutcomeUnknown, RunAlreadyExists, validate_run_id
from .checkpointer import Checkpointer
from .events import EventLog, compute_delta, project
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


def _json_copy(state: dict[str, Any]) -> dict[str, Any]:
    """只留 JSON 能表达的东西：缓存与投影都经过 JSON，比较时才对得上。"""
    return json.loads(json.dumps(state))


def mismatched_keys(cache: dict[str, Any] | None, projection: dict[str, Any] | None) -> list[str]:
    cache, projection = cache or {}, projection or {}
    return sorted(k for k in set(cache) | set(projection) if cache.get(k) != projection.get(k))


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
        hold_after_commit: int | None = None,
    ) -> None:
        self.task = task
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = validate_run_id(run_id)
        self.durability = durability
        self.cancel = cancel or CancelToken()
        self.tool_delay = tool_delay
        self.crash_after_step = crash_after_step
        self.hold_after_commit = hold_after_commit
        self.checkpointer = Checkpointer(self.state_dir)
        self.ledger = EffectLedger(self.state_dir)
        self.events = EventLog(self.ledger.db_path)
        self._last_seq = 0
        self._pending_events: list[tuple[str, int | None, dict[str, Any]]] = []

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
            last_seq=self._last_seq,
        )
        self.checkpointer.save(checkpoint, self.durability)
        return checkpoint.checkpoint_id

    def _emit(self, type_: str, *, step_index: int | None = None,
              payload: dict[str, Any] | None = None) -> None:
        if self.durability is Durability.EXIT:
            self._pending_events.append((type_, step_index, payload or {}))
            return
        self._last_seq = self.events.append(self.run_id, type_, step_index=step_index, payload=payload)

    def _flush(self) -> None:
        """先写事件、再写缓存：缓存永远不会领先于日志。"""
        if self._pending_events:
            pending, self._pending_events = self._pending_events, []
            self._last_seq = self.events.append_many(self.run_id, pending)[-1]
        self.checkpointer.flush()

    # --- 主循环 ---
    def run(self, *, resume: bool = False) -> RunReport:
        # 整个 run 期间持有 run 锁；进程被杀时内核释放
        with run_lock(self.state_dir, self.run_id):
            return self._run(resume=resume)

    def _run(self, *, resume: bool) -> RunReport:
        report = RunReport(run_id=self.run_id, status=RunStatus.RUNNING)
        projection, last_seq = project(self.events.read(self.run_id))

        if not resume:
            if projection is not None:
                raise RunAlreadyExists(self.run_id)
            state = self._fresh_state()
            parent: str | None = None
            self._emit("run_started", payload={"state": state})
        else:
            if projection is None:
                raise FileNotFoundError(f"没有可恢复的 run：事件日志里没有 run_id={self.run_id}")
            self._last_seq = last_seq
            cache = self.checkpointer.latest(self.run_id)
            parent = cache.checkpoint_id if cache is not None else None
            if cache is None or cache.state != projection:
                # 缓存落后于日志（例如死在「事件已提交、缓存未写」之间）或被改坏：以日志为准
                self._emit("projection_repaired", payload={
                    "cache_checkpoint_id": parent,
                    "cache_last_seq": cache.last_seq if cache is not None else None,
                    "mismatched_keys": mismatched_keys(cache.state if cache else None, projection),
                })
                parent = self._save(projection, parent, kind="repaired")
                self._flush()
            state = copy.deepcopy(projection)
            if state.get("status") == RunStatus.COMPLETED.value:
                report.status = RunStatus.COMPLETED
                report.steps_run = list(state.get("steps_done", []))
                return report
            state["status"] = RunStatus.RUNNING.value
            self._emit("run_resumed", payload={"from_step": state["step_index"]})

        committed = _json_copy(state)  # 最后一次提交时的 state
        start = int(state["step_index"])
        base_ctx = StepContext(state=state, ledger=self.ledger, cancel=self.cancel,
                               tool_delay=self.tool_delay, run_id=self.run_id)

        while True:
            index = int(state["step_index"])
            try:
                step = self._step_at(state)
            except RunFailed as error:
                # 任务在决定下一步时就判定无法继续（例如步数用尽）：此刻没有执行到一半的步
                return self._stop(report, committed, parent, RunStatus.FAILED, error=str(error))
            if step is None:
                break
            if resume and index == start:
                report.replayed_steps = [step.name]

            # 1) 步边界检查取消（协作式，不是 kill 进程）
            if self.cancel.cancelled:
                return self._stop(report, committed, parent, RunStatus.CANCELLED,
                                  error="cancelled before step")

            # 每步一个上下文：step_index 进键的作用域。state 是同一个 dict（replace 是浅拷贝）。
            ctx = replace(base_ctx, step_index=index)
            try:
                step.action(ctx)
            except NodeInterrupt:
                return self._stop(report, committed, parent, RunStatus.INTERRUPTED)
            except Cancelled:
                return self._stop(report, committed, parent, RunStatus.CANCELLED,
                                  error=f"cancelled during {step.name}")
            except (EffectOutcomeUnknown, ApprovalRevoked) as error:
                report.blocked_on = error.key
                return self._stop(report, committed, parent, RunStatus.BLOCKED,
                                  error=f"blocked during {step.name}: {error}")
            except RunFailed as error:
                return self._stop(report, committed, parent, RunStatus.FAILED,
                                  error=f"failed during {step.name}: {error}")

            state["step_index"] = index + 1
            state["steps_done"].append(step.name)
            report.steps_run.append(step.name)

            # 2) 崩溃注入点：步骤（和它的副作用）做完了，还没提交。
            #    恢复时这一步会重跑——S2.1/S2.2 要复现的就是这个窗口。
            if self.crash_after_step is not None and index == self.crash_after_step:
                os._exit(137)

            # 3) 提交：这条事件写进去，这一步就做完了
            after = _json_copy(state)
            self._emit("step_committed", step_index=index,
                       payload={"step": step.name, "delta": compute_delta(committed, after)})
            committed = after

            # 4) 验收用的停顿点：事件已提交、缓存还没写（S12.3）
            if self.hold_after_commit is not None and index == self.hold_after_commit:
                from .demo import hold_until_killed
                hold_until_killed()

            parent = self._save(state, parent)

        state["status"] = RunStatus.COMPLETED.value
        self._emit("run_finished")
        self._save(state, parent, kind="final")
        self._flush()
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
        """步骤没有正常跑完：把最后一次提交的 state 连同状态写清楚再退出（exit 档也要落盘）。"""
        entry = copy.deepcopy(entry)
        entry["status"] = status.value
        self._emit("run_stopped", payload={"status": status.value, "error": error})
        self._save(entry, parent, kind=status.value)
        self._flush()
        report.status = status
        report.steps_run = list(entry["steps_done"])
        report.error = error
        return report
