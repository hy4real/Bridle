"""LangGraph backend：同一份 Task 跑在真框架上。

存在的理由（S10.1/S10.3）：JD 明写 LangGraph/LangChain，「用过框架」的证据是代码，
不是简历里的一行字。这里用官方 `StateGraph` + `SqliteSaver` 把同一套步骤跑一遍，
再和 native runtime 对比结果是否一致。

设计边界：**它只是可选 backend，不是主路径。** native 才是默认实现，
理由是 langgraph 会把状态语义锁在它的 channel 模型里（见 §差异）。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from ..effects import RunAlreadyExists, validate_run_id
from .ledger import EffectLedger
from .lock import run_lock
from .types import (
    Cancelled,
    CancelToken,
    RunReport,
    RunStatus,
    StepContext,
    Task,
)


class _State(TypedDict, total=False):
    step_index: int
    steps_done: list[str]


class LangGraphRuntime:
    """与 AgentRuntime 同签名的最小实现，用来做「双 runtime 等价」验收。"""

    name = "langgraph"

    def __init__(
        self,
        task: Task,
        state_dir: Path,
        *,
        run_id: str = "run-1",
        cancel: CancelToken | None = None,
        tool_delay: float = 0.0,
        crash_after_step: int | None = None,
        effect_key: str = "demo-effect-1",
    ) -> None:
        self.task = task
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = validate_run_id(run_id)
        self.cancel = cancel or CancelToken()
        self.tool_delay = tool_delay
        self.crash_after_step = crash_after_step
        self.effect_key = effect_key
        self.ledger = EffectLedger(self.state_dir)
        self.db_path = self.state_dir / "langgraph.sqlite"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.saver = SqliteSaver(self._conn)
        self.saver.setup()
        self.graph = self._build()

    # --- 图 ---
    def _build(self):
        task = self.task
        base_ctx = StepContext(state={}, ledger=self.ledger, cancel=self.cancel,
                               tool_delay=self.tool_delay, run_id=self.run_id)

        def run_step(state: _State) -> _State:
            index = state.get("step_index", 0)
            if index >= len(task.steps):
                return {"step_index": index, "steps_done": state.get("steps_done", [])}
            step = task.steps[index]
            ctx = replace(base_ctx, step_index=index)  # 与 native 相同的键作用域
            ctx.state.setdefault("plan", {"effect_key": self.effect_key})
            step.action(ctx)
            # 崩溃注入：副作用已发生、这个 superstep 的 checkpoint 还没落
            if self.crash_after_step is not None and index == self.crash_after_step:
                os._exit(137)
            return {"step_index": index + 1,
                    "steps_done": [*state.get("steps_done", []), step.name]}

        builder = StateGraph(_State)
        builder.add_node("step", run_step)
        builder.set_entry_point("step")
        builder.add_conditional_edges(
            "step",
            lambda state: "step" if state.get("step_index", 0) < len(task.steps) else END,
            {"step": "step", END: END},
        )
        return builder.compile(checkpointer=self.saver)

    # --- 执行 ---
    @property
    def _config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.run_id, "checkpoint_ns": ""}}

    def run(self, *, resume: bool = False) -> RunReport:
        with run_lock(self.state_dir, self.run_id):
            return self._run(resume=resume)

    def _run(self, *, resume: bool) -> RunReport:
        report = RunReport(run_id=self.run_id, status=RunStatus.RUNNING)
        try:
            if resume:
                snapshot = self.graph.get_state(self._config)
                previous = snapshot.values.get("step_index", 0)
                self.graph.invoke(None, self._config)
                report.replayed_steps = (
                    [self.task.steps[previous].name] if previous < len(self.task.steps) else [])
            else:
                if self.graph.get_state(self._config).values:
                    raise RunAlreadyExists(self.run_id)
                self.graph.invoke({"step_index": 0, "steps_done": []}, self._config)
        except Cancelled:
            report.status = RunStatus.CANCELLED
            return report

        final = self.graph.get_state(self._config).values
        report.steps_run = list(final.get("steps_done", []))
        report.status = RunStatus.COMPLETED
        return report

    def close(self) -> None:
        self._conn.close()
