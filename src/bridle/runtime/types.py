"""runtime 的公共类型。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from ..effects import scoped_key


class Durability(str, Enum):
    """事件与 checkpoint 缓存的写入时机。

    exit  只在运行结束时落盘。崩溃 = 这次运行全丢。性能最好，长任务最脆弱。
    sync  每步同步落盘。慢，但崩溃后一定恢复到最近提交的步。

    M1–M8 还有一档 async，实现上和 sync 一样是同步写，名不副实，M9.5 删除。
    """

    EXIT = "exit"
    SYNC = "sync"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    CRASHED = "crashed"
    BLOCKED = "blocked"  # 副作用结果未知（等人 resolve）或审批失效/吊销
    FAILED = "failed"    # 任务自己判定无法继续（模型不可用、步数用尽）；重试没有意义


class RunFailed(Exception):
    """步骤或任务判定这个 run 无法继续：模型超时、配额耗尽、步数上限。

    和取消、中断不同，它不是外部叫停；和结果未知也不同，它没有悬而未决的副作用。
    run 落成 `failed`，checkpoint 存的是步入口快照。
    """


class Cancelled(Exception):
    """协作式取消：由工具/步骤在检查点主动抛出，不是被 kill。"""


class NodeInterrupt(Exception):
    """中断：恢复时**整个节点从头重跑**（LangGraph 语义）。

    所以节点里的副作用必须幂等——这是 S2.2 要证明的东西。
    """

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        super().__init__("node interrupted")
        self.payload = payload or {}


@dataclass
class CancelToken:
    """一个事件，不是信号处理器的私货：谁都能查、谁都能等。"""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled("cancelled by token")

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


@dataclass
class Checkpoint:
    thread_id: str
    checkpoint_id: str
    state: dict[str, Any]
    parent_checkpoint_id: str | None = None
    checkpoint_ns: str = ""
    ts: float = 0.0
    kind: str = "step"
    last_seq: int = 0  # 这份缓存对应事件日志的哪一条（exit 档下为 0）


@dataclass
class Step:
    """一个可重跑的步骤。重跑是恢复的语义，不是异常路径。"""

    name: str
    action: Callable[["StepContext"], None]


@dataclass
class Task:
    """两种任务：

    静态  `steps` 是固定的步骤表，第 i 步就是 `steps[i]`（M1 起的 demo 任务）。
    动态  `next_step(state)` 根据当前 state 决定下一步，返回 None 表示结束。
          模型循环是这种：下一步是调模型还是调工具，看上一步的结果。

    动态任务的 `next_step` 必须**只读 state、不产生副作用**：恢复时它会被重新调用，
    同一份 state 必须给出同一个下一步。
    """

    name: str
    steps: list[Step] = field(default_factory=list)
    next_step: Callable[[dict[str, Any]], "Step | None"] | None = None


@dataclass
class StepContext:
    """步骤能看到的东西。

    刻意不给"当前是第几次执行"——步骤必须假设自己会被重跑，
    attempt 计数在 ledger 里，由幂等键收敛。
    """

    state: dict[str, Any]
    ledger: Any
    cancel: CancelToken
    tool_delay: float = 0.0
    run_id: str = ""
    step_index: int = 0

    def effect_key(self, business_key: str) -> str:
        """步骤里的副作用一律用它取键：带上 run 与 step 作用域，跨 run 不会误命中。"""
        return scoped_key(self.run_id, self.step_index, business_key)

    def sleep_or_cancel(self, seconds: float) -> None:
        """模拟慢工具：分段等待，保证取消能在 100ms 级别生效。"""
        remaining = seconds
        while remaining > 0:
            self.cancel.raise_if_cancelled()
            slice_s = min(0.05, remaining)
            self.cancel.wait(slice_s)
            remaining -= slice_s


@dataclass
class RunReport:
    run_id: str
    status: RunStatus
    steps_run: list[str] = field(default_factory=list)
    replayed_steps: list[str] = field(default_factory=list)
    error: str | None = None
    blocked_on: str | None = None  # blocked 时：卡在哪个副作用键或审批 ID 上
