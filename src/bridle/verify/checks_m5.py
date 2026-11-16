"""M5 的回归断言（S1.3、S6.1、S6.3、S8.1–S8.3）。"""

from __future__ import annotations

import threading

from bridle.cost.ledger import CallRecord, CostLedger
from bridle.fake_model.server import FakeModelServer, ScriptedModel
from bridle.model.gateway import ModelGateway
from bridle.obs.metrics import HighCardinalityLabel, Metrics
from bridle.obs.redact import Redactor
from bridle.obs.tracer import Tracer
from bridle.quota import QuotaLedger

from .registry import CheckResult, failed, passed


# --- S1.3 ---------------------------------------------------------------
def check_s13() -> CheckResult:
    quota = QuotaLedger(limit=100)
    granted: list[str] = []
    denied: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(index: int) -> None:
        barrier.wait()  # 让 8 个线程真正同时冲进来
        reservation = quota.reserve(80, key=f"job-{index}")
        with lock:
            (granted if reservation else denied).append(f"job-{index}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if len(granted) != 1:
        return failed(f"准入数应为 1，实际 {len(granted)}：{granted}")
    if quota.reserved != 80 or quota.available != 20:
        return failed(f"预留后状态不对：reserved={quota.reserved} available={quota.available}")

    return passed(
        f"余额 100、8 个并发请求各要 80 → 只有 1 个被允许（{granted}），{len(denied)} 个被拒",
        f"预留后 available={quota.available}（= 100 − 0 已结算 − 80 在途预留）",
        "结论：可用额度 = 已授权 − 已结算 − 在途预留；「检查 + 扣减」做不到这件事",
    )


# --- S6.1 ---------------------------------------------------------------
def check_s61() -> CheckResult:
    ledger = CostLedger()

    def record(turn: str, call: str, attempt: str, inp: int, out: int,
               cache_read: int = 0, cache_write: int = 0) -> None:
        ledger.record(CallRecord(
            task_id="task-42", session_id="sess-1", turn_id=turn, call_id=call,
            attempt_id=attempt, model="deepseek-v4-pro",
            input_tokens=inp, output_tokens=out,
            cache_read=cache_read, cache_write=cache_write,
        ))

    record("turn-1", "call-1", "att-1", 1000, 200, 600, 100)
    record("turn-1", "call-2", "att-2", 1200, 150, 800, 0)
    record("turn-1", "call-2", "att-3", 1200, 150, 800, 0)  # 重试：同一 call，不同 attempt
    record("turn-2", "call-3", "att-4", 900, 100, 400, 0)

    report = ledger.report()
    levels = {"by_task", "by_session", "by_turn", "by_call", "by_attempt"}
    if not levels <= set(report):
        return failed(f"缺少层级：{levels - set(report)}")
    if len(report["by_turn"]) != 2:
        return failed(f"turn 数不对：{len(report['by_turn'])}")
    if len(report["by_call"]) != 3:
        return failed(f"call 数不对：{len(report['by_call'])}")
    if len(report["by_attempt"]) != 4:
        return failed(f"attempt 数不对：{len(report['by_attempt'])}")
    if report["by_turn"]["turn-1"]["calls"] != 3:
        return failed("turn-1 的调用数不对")
    if report["total"]["calls"] != 4:
        return failed("总计调用数不对")

    return passed(
        "五层齐备：task → session → turn → call → attempt",
        f"turn=2、call=3（其中 call-2 重试一次）、attempt=4",
        f"turn-1 汇总：calls={report['by_turn']['turn-1']['calls']}，"
        f"input={report['by_turn']['turn-1']['input']}，weighted={report['by_turn']['turn-1']['weighted']:.0f}",
        "结论：session 只能答「这段交互花了多少」，task 级才能答「一个长任务花了多少、恢复了几次」",
    )


# --- S6.3 ---------------------------------------------------------------
def check_s63() -> CheckResult:
    try:
        CallRecord(task_id="t", session_id="s", turn_id="u", call_id="c", attempt_id="a",
                   model="m", input_tokens=100, output_tokens=10,
                   cache_read=80, cache_write=40)
    except ValueError as error:
        rejected = str(error)
    else:
        return failed("cache 桶超过 input 竟然没被拒绝：重叠口径会虚高账单")

    record = CallRecord(task_id="t", session_id="s", turn_id="u", call_id="c", attempt_id="a",
                        model="m", input_tokens=100, output_tokens=10,
                        cache_read=60, cache_write=20)
    if record.total_tokens != 110:
        return failed(f"total 应为 input+output=110，实际 {record.total_tokens}")
    if record.total_tokens == 100 + 10 + 60 + 20:
        return failed("把 cache 加进 total 了")

    return passed(
        f"非法口径被写入层拒绝：{rejected}",
        f"合法口径 total = input + output = {record.total_tokens}（cache 60+20 不参与相加）",
        "结论：桶互斥在**写入时**强制，不靠事后看报表",
    )


# --- S8.1 ---------------------------------------------------------------
def check_s81() -> CheckResult:
    model = ScriptedModel([{"content": "ok"}])
    tracer = Tracer()
    with FakeModelServer(model) as server:
        gateway = ModelGateway(server.base_url, sleep=lambda _s: None)
        with tracer.span("invoke_agent deepseek-v4-pro",
                         **{"gen_ai.operation.name": "invoke_agent",
                            "gen_ai.request.model": "deepseek-v4-pro"}) as agent_span:
            completion = gateway.complete([{"role": "user", "content": "hi"}])
            agent_span.attributes["gen_ai.usage.input_tokens"] = completion.usage.input_tokens
            agent_span.attributes["gen_ai.usage.output_tokens"] = completion.usage.output_tokens
            agent_span.attributes["gen_ai.usage.cache_read.input_tokens"] = completion.usage.cache_read_tokens
            agent_span.attributes["gen_ai.conversation.id"] = "sess-1"
            with tracer.span("execute_tool note_read", **{"gen_ai.tool.name": "note_read"}):
                pass

    exported = tracer.export()["resourceSpans"][0]["scopeSpans"][0]["spans"]
    names = [span["name"] for span in exported]
    required = {
        "gen_ai.operation.name", "gen_ai.request.model", "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens", "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.conversation.id", "gen_ai.tool.name",
    }
    present = {attr["key"] for span in exported for attr in span["attributes"]}
    missing = required - present

    if "invoke_agent deepseek-v4-pro" not in names:
        return failed(f"span 名不符合 {operation} {model} 约定：{names}")
    if "execute_tool note_read" not in names:
        return failed(f"工具 span 名不符合 execute_tool {{tool}}：{names}")
    if missing:
        return failed(f"缺少属性：{sorted(missing)}")

    return passed(
        f"span 名：{names}",
        f"属性齐全：{len(required)} 个 OTel GenAI 键",
        "导出为 OTLP 形状 JSON（resourceSpans → scopeSpans → spans），非完整 protobuf——已写进 README",
    )


# --- S8.2 ---------------------------------------------------------------
def check_s82() -> CheckResult:
    metrics = Metrics()
    metrics.add("gen_ai.client.token.usage", 120, model="deepseek-v4-pro", operation="inference")

    rejected = []
    for label in ("session_id", "trace_id", "user_id"):
        try:
            metrics.add("gen_ai.client.token.usage", 1, **{label: "abc"})
        except HighCardinalityLabel:
            rejected.append(label)

    if len(rejected) != 3:
        return failed(f"高基数标签没有被全部拒绝：只拒了 {rejected}")
    if len(metrics.snapshot()) != 1:
        return failed(f"被拒的标签仍然写进了指标：{metrics.snapshot()}")

    return passed(
        f"session_id / trace_id / user_id 三个高基数标签全部抛 HighCardinalityLabel",
        f"只留下 1 条低基数序列：{metrics.snapshot()}",
        "结论：指标标签是白名单；高基数只能进 trace 与日志",
    )


# --- S8.3 ---------------------------------------------------------------
def check_s83() -> CheckResult:
    secret = "sk-abcdefghijklmnop1234"
    tracer = Tracer(Redactor(fail_on="execute_tool"))
    with tracer.span("invoke_agent m", **{"gen_ai.request.model": "m", "note": f"key={secret}"}):
        pass
    with tracer.span("execute_tool read", **{"gen_ai.tool.name": "read", "note": f"key={secret}"}):
        pass

    exported = tracer.export()
    spans = exported["resourceSpans"][0]["scopeSpans"][0]["spans"]
    blob = str(spans)

    if secret in blob:
        return failed("导出结果里仍然有明文密钥")
    if len(spans) != 2:
        return failed(f"脱敏 hook 失败导致整批丢失：只剩 {len(spans)} 条")
    if not exported["redaction_failures"]:
        return failed("没有记录脱敏失败")
    if "[REDACTED_API_KEY]" not in blob:
        return failed("正常 span 没有被脱敏")
    if "[REDACTION_FAILED]" not in blob:
        return failed("失败 span 没有降级为清空字段（可能把原文导出了）")

    return passed(
        f"正常 span：密钥被替换为 [REDACTED_API_KEY]",
        f"脱敏 hook 抛错的 span：字符串字段降级为 [REDACTION_FAILED]，span 本身保留",
        f"失败被记录：{exported['redaction_failures']}；导出批 2 条未丢",
        "结论：脱敏在导出阶段、失败即清空该字段——绝不因 hook 异常而导出原文或丢整批",
    )
