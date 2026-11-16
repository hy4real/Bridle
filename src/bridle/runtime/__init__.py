"""runtime：状态、checkpoint、恢复、取消。

三条来自调研的硬约束（`sources/github-roadmap-analysis.md` §2）：
1. checkpoint 是**三张表**：checkpoints / checkpoint_blobs / checkpoint_writes，
   主键含 namespace，子图与主图才能共存（LangGraph 的做法，我们照抄结构）。
2. durability 是**三档配置**：exit / async / sync，代价各不相同（默认 async）。
3. 恢复的语义是**重跑节点**，不是从断点续行——所以副作用必须幂等。
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
