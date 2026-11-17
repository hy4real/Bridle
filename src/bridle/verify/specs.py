"""46 条规格的登记表。

这张表是 `specs/a1-a10-acceptance-specs.md` 的代码镜像。两处必须同步：
改一处必须改另一处，否则 `harness verify --list` 与文档对不上。

新规格先登记为 `not_implemented`（M0 起的做法）：这是**基线**，不是缺陷。
每完成一条，把对应的 check 换成真断言，状态列从「未实现」变成「通过/失败」，
同时改 `BASELINE`。M1–M9 的 44 条已全部通过；M11（审批记录）的 2 条正在实现中。
"""

from __future__ import annotations

from typing import Callable

from . import (
    checks_m1,
    checks_m2,
    checks_m3,
    checks_m4,
    checks_m5,
    checks_m6,
    checks_m7,
    checks_m8,
    checks_m9,
    checks_m11,
)
from .registry import CheckResult, Spec, not_implemented

_SPECS: list[Spec] = []


def _spec(
    sid: str,
    area: str,
    title: str,
    command: str,
    milestone: str,
    check: Callable[[], CheckResult] | None = None,
) -> None:
    if check is None:
        check = lambda sid=sid, milestone=milestone: not_implemented(  # noqa: E731
            f"{milestone} 未开始：{sid} 的断言函数尚未实现"
        )
    _SPECS.append(
        Spec(id=sid, area=area, title=title, command=command, milestone=milestone, check=check)
    )


# A1 多租户与隔离
_spec("S1.1", "A1", "隔离单元是 session，不是同一个进程里的一个变量",
      "harness verify S1.1", "M2", checks_m2.check_s11)
_spec("S1.2", "A1", "沙箱 ≠ 权限边界：越权拦截在策略层，返回 PERMISSION_DENIED 而非 ENOENT",
      "harness verify S1.2", "M2", checks_m2.check_s12)
_spec("S1.3", "A1", "配额硬上限用「预留 + 结算」，两个并发请求不能同时穿透余额检查",
      "harness verify S1.3", "M5", checks_m5.check_s13)

# A2 长任务与状态
_spec("S2.1", "A2", "注入点退出（模拟 kill -9）后从 checkpoint 恢复，且副作用账本只有一条",
      "harness verify S2.1", "M1", checks_m1.check_s21)
_spec("S2.2", "A2", "恢复会重跑节点，节点内副作用靠幂等键收敛",
      "harness verify S2.2", "M1", checks_m1.check_s22)
_spec("S2.3", "A2", "取消是协作式传播：在途工具收到 cancel，账本记 cancelled",
      "harness verify S2.3", "M1", checks_m1.check_s23)
_spec("S2.4", "A2", "状态存「意图 + 标识 + 结果」三件套，能判定是否已执行",
      "harness verify S2.4", "M1", checks_m1.check_s24)
_spec("S2.5", "A2", "真 SIGKILL 落在「意图已写、结果未写」窗口：可重放副作用恢复后收敛",
      "harness verify S2.5", "M7", checks_m7.check_s25)
_spec("S2.6", "A2", "不可重放副作用结果未知时不自动重放：恢复停在 blocked，账本记 unknown",
      "harness verify S2.6", "M7", checks_m7.check_s26)
_spec("S2.7", "A2", "unknown 只能显式解除：resolve applied 不重跑，not-applied 恰好再执行一次",
      "harness verify S2.7", "M7", checks_m7.check_s27)
_spec("S2.8", "A2", "幂等键带 run 与 step 作用域：跨 run 不误去重，同键异意图报冲突",
      "harness verify S2.8", "M7", checks_m7.check_s28)
_spec("S2.9", "A2", "声明了反查的不可重放副作用：查到已执行则不重跑，查不到仍按未知处理",
      "harness verify S2.9", "M7", checks_m7.check_s29)

# A3 上下文工程
_spec("S3.1", "A3", "压缩是 summary + 未压缩尾部双层，压完仍记得早期事实",
      "harness verify S3.1", "M4", checks_m4.check_s31)
_spec("S3.2", "A3", "前缀字节稳定（连续两轮前缀 hash 相同），system prompt 无动态内容",
      "harness verify S3.2", "M4", checks_m4.check_s32)
_spec("S3.3", "A3", "token 报告区分 input / output / cache_read / cache_write，且子集关系成立",
      "harness verify S3.3", "M4", checks_m4.check_s33)

# A4 工具层
_spec("S4.1", "A4", "工具定义有 token 预算：60 个工具时注入的是 top-k 而不是全量",
      "harness verify S4.1", "M3", checks_m3.check_s41)
_spec("S4.2", "A4", "工具超时重试不重复写：账本仍只有一条",
      "harness verify S4.2", "M3", checks_m3.check_s42)
_spec("S4.3", "A4", "业务失败（is_error）与协议失败（异常）走不同路径且日志可区分",
      "harness verify S4.3", "M3", checks_m3.check_s43)
_spec("S4.4", "A4", "不可重放工具超时后不自动重试，且超时真正限住墙钟时间",
      "harness verify S4.4", "M7", checks_m7.check_s44)

# A5 可靠性与护栏
_spec("S5.1", "A5", "沙箱不可用时 fail-closed（拒绝启动），不是裸跑",
      "harness verify S5.1", "M2", checks_m2.check_s51)
_spec("S5.2", "A5", "deny 优先于 allow，allow 无法为 deny 开例外",
      "harness verify S5.2", "M2", checks_m2.check_s52)
_spec("S5.3", "A5", "refusal / finish_reason=length / 内容过滤三条旁路都不写入下游",
      "harness verify S5.3", "M2", checks_m2.check_s53)
_spec("S5.4", "A5", "审批记录跨进程存活：interrupted 后 approve 再恢复，不重问人直接执行",
      "harness verify S5.4", "M11", checks_m11.check_s54)
_spec("S5.5", "A5", "恢复前校验审批有效性：策略变化或过期则吊销，run 停在 blocked",
      "harness verify S5.5", "M11", checks_m11.check_s55)

# A6 成本
_spec("S6.1", "A6", "归因账本分层：task → session → turn → call → attempt",
      "harness cost --run <id> --by turn", "M5", checks_m5.check_s61)
_spec("S6.2", "A6", "超预算降级而不是硬失败：切小模型继续或明确 BUDGET_EXCEEDED",
      "harness verify S6.2", "M4", checks_m4.check_s62)
_spec("S6.3", "A6", "token 桶互斥：total = input + output，cache/reasoning 是子集不相加",
      "harness verify S6.3", "M5", checks_m5.check_s63)

# A7 评测
_spec("S7.1", "A7", "工程回归集可复现：固定用例跑两遍通过率一致",
      "harness eval --suite core", "M6", checks_m6.check_s71)
_spec("S7.2", "A7", "失败样本按失败模式分类，而不是只有「没过」",
      "harness eval --suite core --report failures", "M6", checks_m6.check_s72)

# A8 可观测
_spec("S8.1", "A8", "span 名与属性对齐 OTel GenAI 语义约定",
      "harness trace --run <id> --format otlp", "M5", checks_m5.check_s81)
_spec("S8.2", "A8", "高基数（session_id / trace_id）不进 metric 标签",
      "harness verify S8.2", "M5", checks_m5.check_s82)
_spec("S8.3", "A8", "脱敏在导出阶段生效，且 hook 失败不丢整批",
      "harness verify S8.3", "M5", checks_m5.check_s83)

# A9 并发与性能
_spec("S9.1", "A9", "并发上限生效且排队时长进指标，不是静默变慢",
      "harness verify S9.1", "M6", checks_m6.check_s91)
_spec("S9.2", "A9", "两类 429 分流：带 retry-after 重试，不带则快速失败",
      "harness verify S9.2", "M4", checks_m4.check_s92)
_spec("S9.3", "A9", "超时按口径拆分：首字节超时不触发整体重放",
      "harness verify S9.3", "M4", checks_m4.check_s93)

# A10 框架对照（岗位要求）
_spec("S10.1", "A10", "同一套验收在 native 与 langgraph 两个 runtime 上通过率一致",
      "harness verify --all --runtime langgraph", "M6", checks_m6.check_s101)
_spec("S10.2", "A10", "checkpoint 可导出为 langgraph 三表结构，并能反向导入继续跑",
      "harness checkpoint export --format langgraph", "M6", checks_m6.check_s102)
_spec("S10.3", "A10", "「为什么不用现成框架」有代码答案：真 backend + 对照文档",
      "python -c \"import bridle.runtime.langgraph_backend\"", "M6", checks_m6.check_s103)

# A11 模型循环（M8）：runtime 里跑的是 model → tool → model，恢复时不重问模型已给出的决定
_spec("S11.1", "A11", "两轮闭环：模型要求调用工具，工具结果回灌，模型给出最终回答",
      "harness verify S11.1", "M8", checks_m8.check_s111)
_spec("S11.2", "A11", "工具步执行中被真 SIGKILL：恢复后不重问模型，副作用收敛",
      "harness verify S11.2", "M8", checks_m8.check_s112)
_spec("S11.3", "A11", "循环里不可重放的工具结果未知：恢复停在 blocked，解除后才继续",
      "harness verify S11.3", "M8", checks_m8.check_s113)
_spec("S11.4", "A11", "策略拒绝的工具调用不执行，拒绝原因作为工具结果回灌给模型",
      "harness verify S11.4", "M8", checks_m8.check_s114)
_spec("S11.5", "A11", "步数上限：模型一直要工具时，到上限判失败，不无限循环",
      "harness verify S11.5", "M8", checks_m8.check_s115)

# A12 事件日志（M9）：append-only 的事件是真相，checkpoint 是可以重算的投影缓存
_spec("S12.1", "A12", "投影可自检：由事件重算的状态与 checkpoint 缓存一致",
      "harness verify S12.1", "M9", checks_m9.check_s121)
_spec("S12.2", "A12", "缓存被改坏时以日志为准：自检报不一致，恢复后修复并留下记录",
      "harness verify S12.2", "M9", checks_m9.check_s122)
_spec("S12.3", "A12", "崩在「事件已提交、缓存未写」之间：恢复不重跑已提交的步",
      "harness verify S12.3", "M9", checks_m9.check_s123)

SPECS: tuple[Spec, ...] = tuple(_SPECS)


def by_id(sid: str) -> Spec:
    for spec in SPECS:
        if spec.id == sid:
            return spec
    raise KeyError(f"没有这条规格：{sid}")


def run(spec: Spec) -> CheckResult:
    if spec.check is None:  # 防御：登记时漏了断言函数
        return not_implemented(f"{spec.id} 没有绑定断言函数")
    return spec.check()
