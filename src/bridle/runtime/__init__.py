"""runtime：状态、checkpoint、恢复、取消。

几条硬约束：
1. 事件日志（`events.py`）是真相；checkpoint 是由日志算出来的投影缓存，
   主键含 namespace，形状沿用 LangGraph 的 checkpointer。
2. durability 两档：exit（结束时写）/ sync（每步写，默认）。
3. 恢复的语义是**重跑没提交的节点**，不是从断点续行：副作用要么可重放，
   要么在账本里声明不可重放，结果未知时停下等人（`ledger.py`）。
"""

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
from .checkpointer import Checkpointer
from .ledger import EffectLedger
from .loop import AgentRuntime

__all__ = [
    "AgentRuntime",
    "CancelToken",
    "Cancelled",
    "Checkpoint",
    "Checkpointer",
    "Durability",
    "EffectLedger",
    "NodeInterrupt",
    "RunFailed",
    "RunReport",
    "RunStatus",
    "Step",
    "StepContext",
    "Task",
]
