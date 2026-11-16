"""20 个固定用例 + 失败模式分类。

语义约定（关键）：
- 用例返回 `None` 表示**系统表现符合设计**；
- 返回某个 `FailureMode` 表示**发生了偏离**，模式名说明偏在哪里。

所以通过率 100% 才是健康状态；任何一条偏离都会在报告里带上模式名，
而不是一句"没过"。这就是 S7.2 要的东西：失败可归因。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from ..context.budget import WeightedBudget
from ..context.manager import ContextManager
from ..context.prefix import has_dynamic_content, prefix_hash
from ..cost.ledger import CallRecord
from ..fake_model.server import FakeModelServer, ScriptedModel
from ..model.gateway import ModelGateway, ModelTimeout, QuotaExhausted, Usage
from ..obs.metrics import HighCardinalityLabel, Metrics
from ..policy import ApprovalPolicy, CommandRules, Decision, PolicyEngine, Rule
from ..quota import QuotaLedger
from ..runtime import AgentRuntime, Durability
from ..runtime.demo import EffectSink, build_demo_task
from ..runtime.ledger import EffectLedger
from ..sandbox import SandboxUnavailable, build_backend
from ..tools.base import ToolProtocolError
from ..tools.builtin import crashing_tool, flaky_write_tool, note_read
from ..tools.registry import build_fixture_registry
from ..tools.runner import ToolRunner


class FailureMode(str, Enum):
    """偏离方向。名字即诊断结论。"""

    POLICY_LEAK = "policy_leak"                 # 该拦的没拦
    SANDBOX_OPEN = "sandbox_open"               # 沙箱不可用却继续跑
    TOOL_ARGS_UNVALIDATED = "tool_args_unvalidated"
    TOOL_CRASH_UNCAUGHT = "tool_crash_uncaught"
    TOOL_MISCLASSIFIED = "tool_misclassified"   # 业务失败/协议失败搞混
    TOOL_BUDGET_IGNORED = "tool_budget_ignored"
    DUPLICATE_EFFECT = "duplicate_effect"       # 重试导致重复写
    QUOTA_OVERSOLD = "quota_oversold"
    QUOTA_NOT_RAISED = "quota_not_raised"       # 配额耗尽却没抛
    RETRY_MISSING = "retry_missing"             # 该重试没重试
    TIMEOUT_KIND_WRONG = "timeout_kind_wrong"   # 超时口径搞混
    CONTEXT_LOSS = "context_loss"
    PREFIX_UNSTABLE = "prefix_unstable"
    USAGE_INVALID_ACCEPTED = "usage_invalid_accepted"
    BUCKETS_OVERLAP = "buckets_overlap"
    BUDGET_NO_DOWNGRADE = "budget_no_downgrade"
    CARDINALITY_ACCEPTED = "cardinality_accepted"
    RUNTIME_MISMATCH = "runtime_mismatch"
    CASE_CRASHED = "case_crashed"               # 用例自身异常


@dataclass(frozen=True)
class Case:
    id: str
    name: str
    area: str
    run: Callable[[], FailureMode | None]


@dataclass(frozen=True)
class CaseResult:
    id: str
    name: str
    area: str
    mode: FailureMode | None

    @property
    def passed(self) -> bool:
        return self.mode is None


def _gateway(base_url: str, **kwargs) -> ModelGateway:
    return ModelGateway(base_url, sleep=lambda _s: None, **kwargs)


# --- A5 策略与护栏 ---
def _e01() -> FailureMode | None:
    engine = PolicyEngine(rules=CommandRules([Rule("d1", Decision.FORBID, "rm -rf *")]))
    return None if engine.decide("rm -rf /x").decision is Decision.FORBID else FailureMode.POLICY_LEAK


def _e02() -> FailureMode | None:
    engine = PolicyEngine(approval=ApprovalPolicy.NEVER)
    return None if engine.decide("ls").decision is Decision.FORBID else FailureMode.POLICY_LEAK


def _e03() -> FailureMode | None:
    os.environ["HARNESS_SANDBOX_DISABLE"] = "1"
    try:
        build_backend(writable_root=None)
    except SandboxUnavailable:
        return None
    finally:
        os.environ.pop("HARNESS_SANDBOX_DISABLE", None)
    return FailureMode.SANDBOX_OPEN


# --- A4 工具 ---
def _e04() -> FailureMode | None:
    try:
        note_read().run({})
    except ToolProtocolError:
        return None
    return FailureMode.TOOL_ARGS_UNVALIDATED


def _e05() -> FailureMode | None:
    try:
        crashing_tool().run({"path": "x"})
    except ToolProtocolError:
        return None
    return FailureMode.TOOL_CRASH_UNCAUGHT


def _e06() -> FailureMode | None:
    outcome = note_read().run({"path": "missing"})
    ok = outcome.is_error and outcome.kind.value == "business_error"
    return None if ok else FailureMode.TOOL_MISCLASSIFIED


def _e07() -> FailureMode | None:
    registry = build_fixture_registry(60)
    selection = registry.select("file", budget_tokens=300)
    ok = selection.count < 60 and selection.tokens <= 300
    return None if ok else FailureMode.TOOL_BUDGET_IGNORED


def _e08() -> FailureMode | None:
    with tempfile.TemporaryDirectory() as tmp:
        sink = EffectSink(Path(tmp))
        ledger = EffectLedger(Path(tmp))
        ToolRunner(ledger, retries=2, timeout_s=0.08).call(flaky_write_tool(sink), {"target": "a"})
        return None if len(sink.rows()) == 1 else FailureMode.DUPLICATE_EFFECT


# --- A1 / A9 配额与限流 ---
def _e09() -> FailureMode | None:
    quota = QuotaLedger(100)
    granted = sum(1 for key in ("a", "b", "c") if quota.reserve(80, key=key))
    return None if granted == 1 else FailureMode.QUOTA_OVERSOLD


def _e10() -> FailureMode | None:
    model = ScriptedModel([{"status": 429}])
    with FakeModelServer(model) as server:
        try:
            _gateway(server.base_url).complete([{"role": "user", "content": "hi"}])
        except QuotaExhausted:
            return None
    return FailureMode.QUOTA_NOT_RAISED


def _e11() -> FailureMode | None:
    model = ScriptedModel([{"status": 429, "headers": {"retry-after": "0"}}] * 2 + [{"content": "ok"}])
    with FakeModelServer(model) as server:
        completion = _gateway(server.base_url).complete([{"role": "user", "content": "hi"}])
    return None if completion.attempts == 3 else FailureMode.RETRY_MISSING


def _e12() -> FailureMode | None:
    model = ScriptedModel([{"content": "慢", "delay_s": 0.4}])
    with FakeModelServer(model) as server:
        try:
            _gateway(server.base_url, first_byte_timeout=0.1).complete(
                [{"role": "user", "content": "hi"}])
        except ModelTimeout as error:
            ok = error.kind == "first_byte" and model.calls == 1
            return None if ok else FailureMode.TIMEOUT_KIND_WRONG
    return FailureMode.TIMEOUT_KIND_WRONG


# --- A3 上下文 ---
def _e13() -> FailureMode | None:
    context = ContextManager(system_prompt="sys")
    context.add_user("记住 ORION")
    for _ in range(6):
        context.add_user("背景" * 500)
    context.compact(keep_last=1, max_tokens=context.tokens() // 3)
    return None if "ORION" in str(context.build_request()) else FailureMode.CONTEXT_LOSS


def _e14() -> FailureMode | None:
    context = ContextManager(system_prompt="sys")
    context.add_user("q1")
    first = prefix_hash(context.build_request())
    context.add_user("q2")
    return None if prefix_hash(context.build_request()) == first else FailureMode.PREFIX_UNSTABLE


def _e15() -> FailureMode | None:
    detected = bool(has_dynamic_content("今天是 2026-10-18")) and not has_dynamic_content("你是助理")
    return None if detected else FailureMode.PREFIX_UNSTABLE


# --- A6 成本 ---
def _e16() -> FailureMode | None:
    try:
        Usage(input_tokens=10, output_tokens=1, cache_read_tokens=20)
    except ValueError:
        return None
    return FailureMode.USAGE_INVALID_ACCEPTED


def _e17() -> FailureMode | None:
    try:
        CallRecord("t", "s", "u", "c", "a", "m", input_tokens=10, output_tokens=1,
                   cache_read=8, cache_write=8)
    except ValueError:
        return None
    return FailureMode.BUCKETS_OVERLAP


def _e18() -> FailureMode | None:
    model = ScriptedModel([{"content": "a"}, {"content": "b"}])
    with FakeModelServer(model) as server:
        gateway = _gateway(server.base_url, budget=WeightedBudget(limit=50))
        gateway.complete([{"role": "user", "content": "hi"}])
        second = gateway.complete([{"role": "user", "content": "hi"}])
    return None if second.model == "deepseek-flash" else FailureMode.BUDGET_NO_DOWNGRADE


# --- A8 可观测 ---
def _e19() -> FailureMode | None:
    metrics = Metrics()
    try:
        metrics.add("x", 1, session_id="s")
    except HighCardinalityLabel:
        return None
    return FailureMode.CARDINALITY_ACCEPTED


# --- A10 双 runtime ---
def _e20() -> FailureMode | None:
    from ..runtime.langgraph_backend import LangGraphRuntime

    results = []
    for runtime_name in ("native", "langgraph"):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            task = build_demo_task(state_dir)
            runtime = (AgentRuntime(task, state_dir, durability=Durability.SYNC)
                       if runtime_name == "native" else LangGraphRuntime(task, state_dir))
            report = runtime.run()
            results.append((report.status.value, tuple(report.steps_run),
                            len(EffectSink(state_dir).rows())))
    return None if results[0] == results[1] else FailureMode.RUNTIME_MISMATCH


CASES: tuple[Case, ...] = (
    Case("E01", "deny 规则拦截危险命令", "A5", _e01),
    Case("E02", "approval=never 默认拒绝", "A5", _e02),
    Case("E03", "沙箱不可用 fail-closed", "A5", _e03),
    Case("E04", "参数缺省是协议失败", "A4", _e04),
    Case("E05", "工具崩溃是协议失败", "A4", _e05),
    Case("E06", "业务失败不是异常", "A4", _e06),
    Case("E07", "工具定义受预算约束", "A4", _e07),
    Case("E08", "超时重试不重复写", "A4", _e08),
    Case("E09", "配额只放行一个", "A1", _e09),
    Case("E10", "429 无 retry-after 快速失败", "A9", _e10),
    Case("E11", "429 有 retry-after 可重试", "A9", _e11),
    Case("E12", "首字节超时单独识别且不重放", "A9", _e12),
    Case("E13", "压缩后关键事实仍在", "A3", _e13),
    Case("E14", "前缀字节稳定", "A3", _e14),
    Case("E15", "动态内容被识别", "A3", _e15),
    Case("E16", "usage 子集约束", "A6", _e16),
    Case("E17", "账本桶互斥", "A6", _e17),
    Case("E18", "超预算降级", "A6", _e18),
    Case("E19", "高基数标签被拒", "A8", _e19),
    Case("E20", "双 runtime 结果一致", "A10", _e20),
)


def run_suite() -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in CASES:
        try:
            mode = case.run()
        except Exception:
            mode = FailureMode.CASE_CRASHED
        results.append(CaseResult(case.id, case.name, case.area, mode))
    return results


def summarize(results: list[CaseResult]) -> dict:
    failures = [r for r in results if not r.passed]
    by_mode: dict[str, int] = {}
    for result in failures:
        key = result.mode.value if result.mode else "none"
        by_mode[key] = by_mode.get(key, 0) + 1
    return {
        "total": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "pass_rate": round((len(results) - len(failures)) / len(results), 4) if results else 0.0,
        "failure_modes": by_mode,
        "failures": [
            {"id": r.id, "name": r.name, "mode": r.mode.value if r.mode else "none"}
            for r in failures
        ],
    }
