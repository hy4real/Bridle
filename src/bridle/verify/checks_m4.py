"""M4 的回归断言（S3.1–S3.3、S6.2、S9.2、S9.3）。

全部在进程内跑：假模型服务起在本地随机端口，不需要网络与密钥。
"""

from __future__ import annotations

from bridle.context.budget import WeightedBudget
from bridle.context.manager import ContextManager
from bridle.context.prefix import has_dynamic_content, prefix_hash
from bridle.fake_model.server import FakeModelServer, ScriptedModel
from bridle.model.gateway import (
    ModelGateway,
    ModelTimeout,
    QuotaExhausted,
)

from .registry import CheckResult, failed, passed

_NEEDLE = "项目代号 ORION-7731"


def _gateway(base_url: str, **kwargs) -> ModelGateway:
    return ModelGateway(base_url, sleep=lambda _s: None, **kwargs)


# --- S3.1 ---------------------------------------------------------------
def check_s31() -> CheckResult:
    context = ContextManager(system_prompt="你是内部助理，回答要短。")
    context.add_user(f"请记住：{_NEEDLE}")
    for index in range(12):
        context.add_user(f"第 {index} 轮的长背景资料：" + "资料" * 800)

    before = context.tokens()
    compacted = context.compact(keep_last=2, max_tokens=before // 3)
    after = context.tokens()
    request = context.build_request()
    blob = "".join(message["content"] for message in request)

    if not compacted or after >= before:
        return failed(f"压缩没有生效：{before} → {after}")
    if _NEEDLE not in blob:
        return failed("压缩后早期事实丢了：提问将无法回答")
    if _NEEDLE not in context.summary:
        return failed("早期事实不在 summary 里（说明摘要丢了主体）")
    if context.summary is None or len(context.messages) > 4:
        return failed(f"尾部没有收敛：仍有 {len(context.messages)} 条未压缩")

    return passed(
        f"压缩前 {before} token → 压缩后 {after} token（降幅 {100 * (before - after) // before}%）",
        f"摘要里保留关键事实：{_NEEDLE!r}",
        f"结构是双层：summary 1 条 + 未压缩尾部 {len(context.messages)} 条",
        "结论：压缩是「折进摘要 + 保留近文」，不是截断",
    )


# --- S3.2 ---------------------------------------------------------------
def check_s32() -> CheckResult:
    prompt = "你是内部助理，只回答被问到的内容。"
    context = ContextManager(system_prompt=prompt)

    context.add_user("第一轮问题")
    context.add_assistant("第一轮回答")
    first = prefix_hash(context.build_request(), min_tokens=1)

    context.add_user("第二轮问题")
    second = prefix_hash(context.build_request(), min_tokens=1)

    dynamic = has_dynamic_content(prompt)
    if first != second:
        return failed(f"前缀不稳定：{first} != {second}")
    if dynamic:
        return failed(f"system prompt 里有动态内容：{dynamic}")

    # 反例：把日期写进 system prompt，前缀立刻变——这就是缓存杀手
    tainted = ContextManager(system_prompt=prompt + " 今天是 2026-10-18")
    tainted.add_user("第一轮问题")
    tainted_hash = prefix_hash(tainted.build_request(), min_tokens=1)
    if not has_dynamic_content(tainted.system_prompt):
        return failed("没有识别出 system prompt 里的日期")
    if tainted_hash == first:
        return failed("反例没有体现差异：日期竟然不影响前缀 hash")

    return passed(
        f"两轮前缀 hash 相同：{first}",
        "system prompt 无日期/时间/星期表述（has_dynamic_content 返回空）",
        f"反例：写入「今天是 2026-10-18」后 hash 变为 {tainted_hash}——证明日期会毁掉缓存前缀",
        f"尾部追加不影响前缀：消息数 {len(context.messages)} 条时前缀仍为 {first}",
    )


# --- S3.3 ---------------------------------------------------------------
def check_s33() -> CheckResult:
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_tokens": 1200,
        "prompt_tokens_details": {"cached_tokens": 640},
        "cache_write_tokens": 120,
    }
    model = ScriptedModel([{"content": "ok", "usage": usage}])
    with FakeModelServer(model) as server:
        completion = _gateway(server.base_url).complete([{"role": "user", "content": "hi"}])

    u = completion.usage
    if u.cache_read_tokens != 640 or u.cache_write_tokens != 120:
        return failed(f"缓存字段没解析出来：{u.as_dict()}")
    if u.input_tokens < u.cache_read_tokens + u.cache_write_tokens:
        return failed(f"子集关系不成立：{u.as_dict()}")
    if u.total != u.input_tokens + u.output_tokens:
        return failed("total 不等于 input + output")
    if u.total == u.input_tokens + u.output_tokens + u.cache_read_tokens:
        return failed("把 cache_read 又加了一遍（这是虚高账单的错法）")

    return passed(
        f"input={u.input_tokens} output={u.output_tokens} "
        f"cache_read={u.cache_read_tokens} cache_write={u.cache_write_tokens}",
        f"input ≥ cache_read + cache_write（{u.input_tokens} ≥ {640 + 120}）",
        f"total = input + output = {u.total}；cache 是子集，不重复相加",
        "结论：token 桶互斥，报告口径与 OTel/厂商一致",
    )


# --- S6.2 ---------------------------------------------------------------
def check_s62() -> CheckResult:
    model = ScriptedModel([{"content": "第一次"}, {"content": "第二次"}])
    # 假模型的 usage 是 input=100 / output=20 ⇒ 单次花费 140；限额定 100，第一次之后即超预算
    budget = WeightedBudget(limit=100, input_weight=1.0, output_weight=2.0)
    with FakeModelServer(model) as server:
        gateway = _gateway(server.base_url, budget=budget,
                           model="deepseek-v4-pro", cheap_model="deepseek-flash")
        first = gateway.complete([{"role": "user", "content": "hi"}])
        second = gateway.complete([{"role": "user", "content": "hi again"}])

    if first.model != "deepseek-v4-pro":
        return failed(f"第一次就用降价模型了：{first.model}")
    if second.model != "deepseek-flash":
        return failed(f"超预算后没有降级：{second.model}")
    if not any("降级" in d for d in second.decisions):
        return failed(f"没有记录降级决定：{second.decisions}")
    if second.text != "第二次":
        return failed("降级后调用失败：任务被打断")

    return passed(
        f"第 1 次用 {first.model}（花费 {first.usage.total} token）",
        f"预算 {budget.limit:.0f}，累计 {budget.spent:.0f} 后第 2 次降级到 {second.model}",
        f"降级记录：{second.decisions}",
        "结论：超预算是「换档继续」，不是整单失败",
    )


# --- S9.2 ---------------------------------------------------------------
def check_s92() -> CheckResult:
    # 带 retry-after：可重试，第三次成功
    retry_model = ScriptedModel([
        {"status": 429, "headers": {"retry-after": "1"}},
        {"status": 429, "headers": {"retry-after": "1"}},
        {"content": "终于成功"},
    ])
    with FakeModelServer(retry_model) as server:
        ok = _gateway(server.base_url, retries=2).complete([{"role": "user", "content": "hi"}])

    # 不带 retry-after：配额耗尽，必须立刻失败
    quota_model = ScriptedModel([{"status": 429, "message": "quota exhausted"}])
    with FakeModelServer(quota_model) as server:
        try:
            _gateway(server.base_url, retries=2).complete([{"role": "user", "content": "hi"}])
        except QuotaExhausted:
            fast_fail = True
        else:
            fast_fail = False

    if ok.attempts != 3 or ok.text != "终于成功":
        return failed(f"带 retry-after 的重试路径不对：attempts={ok.attempts} text={ok.text!r}")
    if not fast_fail:
        return failed("不带 retry-after 的 429 竟然被重试了（应当立刻失败）")
    if quota_model.calls != 1:
        return failed(f"配额耗尽仍调用 upstream {quota_model.calls} 次")

    return passed(
        f"429 + retry-after → 退避重试，第 {ok.attempts} 次成功",
        f"429 无 retry-after → 立刻 QuotaExhausted，上游只被调用 {quota_model.calls} 次",
        "结论：两类 429 语义不同，混在一起重试会白烧配额",
    )


# --- S9.3 ---------------------------------------------------------------
def check_s93() -> CheckResult:
    # 首字节超时：响应头都还没来
    slow_headers = ScriptedModel([{"content": "太慢", "delay_s": 0.6}])
    with FakeModelServer(slow_headers) as server:
        try:
            _gateway(server.base_url, first_byte_timeout=0.15, total_timeout=5).complete(
                [{"role": "user", "content": "hi"}])
        except ModelTimeout as error:
            first_kind = error.kind
        else:
            first_kind = None
    first_attempts = slow_headers.calls

    # 总超时：头回来了，body 卡住
    slow_body = ScriptedModel([{"content": "体很慢", "body_delay_s": 0.6}])
    with FakeModelServer(slow_body) as server:
        try:
            _gateway(server.base_url, first_byte_timeout=1.0, total_timeout=0.15,
                     retries=0).complete([{"role": "user", "content": "hi"}])
        except ModelTimeout as error:
            total_kind = error.kind
        else:
            total_kind = None

    if first_kind != "first_byte":
        return failed(f"首字节超时没有被单独识别：kind={first_kind}")
    if first_attempts != 1:
        return failed(f"首字节超时竟然重放了 {first_attempts} 次")
    if total_kind != "total":
        return failed(f"总超时没有被单独识别：kind={total_kind}")

    return passed(
        f"响应头未到 → ModelTimeout(kind=first_byte)，上游只被调用 {first_attempts} 次（不重放）",
        "头已到、body 卡住 → ModelTimeout(kind=total)，可按重试策略处理",
        "结论：五种超时口径里至少首字节与总时长必须分开，失效模式不同",
    )
