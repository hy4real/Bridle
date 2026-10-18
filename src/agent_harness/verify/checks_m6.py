"""M6 的回归断言（S7.1、S7.2、S9.1、S10.1–S10.3）。"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_harness.concurrency import BoundedRunner
from agent_harness.eval.suite import CaseResult, FailureMode, run_suite, summarize
from agent_harness.runtime import AgentRuntime, Durability
from agent_harness.runtime.demo import EffectSink, build_demo_task
from agent_harness.runtime.langgraph_export import export_checkpoint, import_checkpoint, tables

from .registry import CheckResult, failed, passed

_REPO_ROOT = Path(__file__).resolve().parents[3]


# --- S7.1 ---------------------------------------------------------------
def check_s71() -> CheckResult:
    first = summarize(run_suite())
    second = summarize(run_suite())

    if first["total"] != 20:
        return failed(f"用例数应为 20，实际 {first['total']}")
    if first != second:
        return failed(f"两次运行结果不一致：{first} vs {second}")
    if first["failed"]:
        return failed(f"回归集有失败：{first['failures']}")

    return passed(
        f"固定 20 例，跑两遍：{first['passed']}/20 通过，pass_rate={first['pass_rate']}",
        f"两次 summary 逐字段相同（deterministic）",
        "边界声明：这是**工程回归**维度（改代码没弄坏既有行为），不是效果评测",
    )


# --- S7.2 ---------------------------------------------------------------
def check_s72() -> CheckResult:
    # 故意制造一个偏离：把配额实现换成"永远放行"，看分类器能不能点名
    import agent_harness.eval.suite as suite_module

    class LeakyQuota:
        def __init__(self, limit: float) -> None:
            self.limit = limit

        def reserve(self, amount: float, *, key: str) -> object:
            return object()  # 永远放行 = 超卖

    original = suite_module.QuotaLedger
    suite_module.QuotaLedger = LeakyQuota  # type: ignore[assignment]
    try:
        results = run_suite()
    finally:
        suite_module.QuotaLedger = original  # type: ignore[assignment]

    summary = summarize(results)
    e09 = next((r for r in results if r.id == "E09"), None)
    if e09 is None:
        return failed("找不到 E09 用例")
    if e09.mode is not FailureMode.QUOTA_OVERSOLD:
        return failed(f"偏离没有被点名：E09 mode={e09.mode}")
    if summary["failure_modes"].get("quota_oversold") != 1:
        return failed(f"失败模式统计不对：{summary['failure_modes']}")
    if summary["passed"] != 19:
        return failed(f"其余用例应仍通过，实际 {summary['passed']}/20")

    healthy = summarize(run_suite())
    return passed(
        f"注入偏离（配额永远放行）后：E09 被判为 {e09.mode.value}，失败统计 {summary['failure_modes']}",
        f"其余 19 例不受影响（passed={summary['passed']}）",
        f"恢复实现后回到 20/20（pass_rate={healthy['pass_rate']}）",
        "结论：失败按模式归因，而不是只有「没过」一个布尔值",
    )


# --- S9.1 ---------------------------------------------------------------
def check_s91() -> CheckResult:
    runner = BoundedRunner(limit=3)
    started = time.monotonic()
    runner.map([lambda: time.sleep(0.05) for _ in range(8)])
    elapsed = time.monotonic() - started
    stats = runner.stats.as_dict()

    if stats["max_in_flight"] > 3:
        return failed(f"在途数超过上限：{stats['max_in_flight']}")
    if stats["completed"] != 8:
        return failed(f"完成数不对：{stats['completed']}")
    if stats["queue_waits"] != 8 or stats["total_wait_s"] <= 0:
        return failed(f"排队时长没有记录：{stats}")
    if elapsed < 0.05 * 3:
        return failed(f"总耗时 {elapsed:.2f}s 短于 3 批 × 0.05s，说明没有真正限流")

    return passed(
        f"8 个任务、上限 3 → max_in_flight={stats['max_in_flight']}（不超限）",
        f"排队被记录：queue_waits={stats['queue_waits']}，total_wait={stats['total_wait_s']}s，max_wait={stats['max_wait_s']}s",
        f"总耗时 {elapsed:.2f}s ≥ 3 批 × 0.05s（确实串行了）",
        "结论：并发上限生效且排队可见，不是「都塞进去然后一起变慢」",
    )


# --- S10.1 --------------------------------------------------------------
def check_s101() -> CheckResult:
    from agent_harness.runtime.langgraph_backend import LangGraphRuntime

    outcomes = {}
    for runtime_name in ("native", "langgraph"):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            task = build_demo_task(state_dir)
            runtime = (AgentRuntime(task, state_dir, durability=Durability.SYNC)
                       if runtime_name == "native" else LangGraphRuntime(task, state_dir))
            report = runtime.run()
            outcomes[runtime_name] = {
                "status": report.status.value,
                "steps": list(report.steps_run),
                "sink_rows": len(EffectSink(state_dir).rows()),
            }

    if outcomes["native"] != outcomes["langgraph"]:
        return failed(f"两个 runtime 结果不一致：{outcomes}")

    return passed(
        f"native：{outcomes['native']}",
        f"langgraph：{outcomes['langgraph']}",
        "结论：同一份 Task 在两个 runtime 上跑出相同状态、相同步骤、相同下游写入次数",
    )


# --- S10.2 --------------------------------------------------------------
def check_s102() -> CheckResult:
    from agent_harness.runtime.types import Checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        task = build_demo_task(state_dir)
        runtime = AgentRuntime(task, state_dir, durability=Durability.SYNC)
        runtime.run()
        native = runtime.checkpointer.latest("run-1")
        if native is None:
            return failed("native 没有 checkpoint")

        db = export_checkpoint(native, Path(tmp) / "export")
        table_names = tables(db)
        back = import_checkpoint(db, "run-1")

        if back.state != native.state:
            return failed(f"往返后状态不一致：{back.state} vs {native.state}")
        if table_names != ["checkpoints", "writes"]:
            return failed(f"langgraph SQLite saver 的表结构不是预期：{table_names}")

        return passed(
            f"导出 → 经 langgraph SqliteSaver 读回：表 {table_names}（**两表**，Postgres 才是三表）",
            f"状态往返一致：step_index={back.state.get('step_index')}，steps={back.state.get('steps_done')}",
            "结论：互操作走的是对方代码（put/get_tuple），不是照抄 DDL",
        )


# --- S10.3 --------------------------------------------------------------
def check_s103() -> CheckResult:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        from agent_harness.runtime.langgraph_backend import LangGraphRuntime
    except ImportError as error:
        return failed(f"langgraph backend 不可导入：{error}")

    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        task = build_demo_task(state_dir)
        runtime = LangGraphRuntime(task, state_dir)
        compiled = runtime.graph
        saver = runtime.saver
        runtime.close()

    doc = _REPO_ROOT / "docs" / "framework-comparison.md"
    if not doc.exists():
        return failed(f"缺少对照文档：{doc}")
    text = doc.read_text(encoding="utf-8")
    required_sections = ["LangGraph 提供什么", "我们提供什么", "故意没做", "差异"]
    missing = [s for s in required_sections if s not in text]

    if not isinstance(saver, SqliteSaver):
        return failed("backend 不是真 SqliteSaver")
    if compiled is None:
        return failed("图没有编译出来")
    if missing:
        return failed(f"对照文档缺章节：{missing}")

    return passed(
        f"真 backend：StateGraph 编译成功，checkpointer = {type(saver).__name__}",
        f"对照文档 `docs/framework-comparison.md` 含四节：{required_sections}",
        "结论：「为什么不用现成框架」有代码与文档两个答案，不是口头解释",
    )
