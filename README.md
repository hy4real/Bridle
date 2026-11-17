# Bridle

一个**不预设用途的通用 Agent Harness**：runtime + agent loop + 有主张的默认栈。

> Agent = Model + Harness（[LangChain 文档](https://docs.langchain.com/oss/python/langchain/agents)）。
> Model 做不了，**Harness 是能由个人做完整的那一半**。

## 现在能跑什么

```bash
uv sync --dev
uv run harness verify --list      # 46 条可验收规格
uv run harness verify --all       # 逐条跑；各状态条数与 BASELINE 一致
uv run harness verify S2.1        # 只跑一条
./scripts/ci.sh                   # 单测 + 规格数 + 基线断言
```

**当前状态：M11 已交付，46 条规格全部通过。**
通过数记在 `BASELINE` 里，CI 会比对——**想加通过数必须同时改 BASELINE**，
没人能悄悄让测试变绿。

## M1 做了什么：runtime

```bash
# 1) 正常跑：三步 + 最终 checkpoint
uv run harness run --state-dir /tmp/demo

# 2) 复现最危险的窗口：副作用已写、checkpoint 未写时进程被杀
uv run harness run --state-dir /tmp/demo2 --crash-after-step 1   # 退出码 137
uv run harness run --state-dir /tmp/demo2 --resume               # 重跑那一步，下游仍 1 条

# 3) 协作式取消：SIGINT 后账本记 cancelled，下游没被写
uv run harness run --state-dir /tmp/demo3 --tool-delay 2   # 另开一个终端 kill -INT
```

第 2 条的退出码 137 来自 `os._exit` 注入点，不是真实信号，而且退出时账本已经记了 `completed`。
真 `SIGKILL` 打在「意图已写、结果未写」之间的验收在 M7。

三条设计决定：

| 决定 | 理由 |
| --- | --- |
| checkpoint 主键含 `checkpoint_ns`，形状与 LangGraph 同构 | 子图与主图才能共存。M1 还仿 PostgresSaver 多建了 blobs、writes 两张表，它们只写不读，M9 删掉（见下文 M9） |
| 恢复 = 从 `step_index` 重跑整步，不是续行 | 只存 channel 值就能恢复，不必序列化栈帧；代价是步骤必须幂等 |
| 账本先落**意图**再执行，下游按键 upsert | 端到端做不到 exactly-once；at-least-once + 幂等才是可实现的保证 |

## M2 做了什么：策略三层 + 沙箱

```bash
uv run harness sandbox-probe                      # 沙箱自检（起不来就 exit 13）
uv run harness policy-check --cmd "rm -rf /x" --deny-cmd "rm -rf *" --allow-cmd "*"
uv run harness exec --session A --state-dir /tmp/s --cmd "echo hi > f.txt" \
     --sandbox-mode workspace-write --allow-cmd "*"
uv run harness exec --session C --state-dir /tmp/s --cmd "cat secret" \
     --deny-read ~/.ssh --sandbox-mode read-only --allow-cmd "*"
uv run harness extract --fake-script refusal.json --sink-dir /tmp/sink   # 三条旁路 → exit 14
```

四件事被钉住了：

| 决定 | 证据 |
| --- | --- |
| 隔离单元是 **session** | 两个 session 的工作区不同，A 写的文件 B 看不到（S1.1） |
| 越权拦截在**策略层**，不是"文件不存在" | deny 后返回 EPERM（`Errno 1 Operation not permitted`），不是 ENOENT（S1.2） |
| 沙箱起不来就 **fail-closed** | exit 13 `SANDBOX_UNAVAILABLE`；放行必须显式 `--allow-unsandboxed` 且打印警告（S5.1） |
| **deny > ask > allow**，`approval=never` 把 ask 降级为拒绝 | 更具体的 allow（`rm *`）也突破不了 deny（`rm -rf *`）（S5.2） |
| 结构化输出三条旁路都不写下游 | `refusal` / `length` / `content_filter` → exit 14，下游 0 行（S5.3） |

**一个必须记住的坑**：Seatbelt 的路径要先 `realpath`。`/var/folders/...` 实际是
`/private/var/folders/...`，不做 realpath 时 deny 规则**不报错也不生效**——
测试 `test_seatbelt_realpath_is_required_to_match` 专门守这件事。

## M3 做了什么：工具层

```bash
uv run harness tools --count 60 --query "读取文件 read file" --budget 300
# → 注册 60、全量 3820 token、只选 4 个（260 token）、跳过 56

uv run harness tool-call --name note_read --args '{"path":"missing"}'   # 业务失败 → rc 0
uv run harness tool-call --name explode   --args '{"path":"x"}'         # 协议失败 → rc 21
uv run harness tool-retry --sink-dir /tmp/s --timeout 0.1               # 超时重试不重复写
```

| 决定 | 证据 |
| --- | --- |
| 工具定义吃 token，所以要有预算 | 60 个工具全量 3,820 token；预算 300 时只装 4 个且全是同类工具（S4.1） |
| 「未收到响应」≠「未执行」 | 第一次调用超时但下游已写；重试后 `attempts=2`、账本 1 条、下游 1 条（S4.2） |
| 失败分两类，路径不同 | 业务失败回灌 `is_error=true` 给模型（rc 0）；工具崩溃/参数缺失 → `protocol_error`（rc 21）（S4.3） |
| 空输出显式化、长输出截断 | `render()` 返回「命令成功执行，没有任何输出」或 `{truncated, head, total_chars}` |

**注意**：token 估算是 `CJK 1 字 ≈ 1 token、ASCII 4 字符 ≈ 1 token` 的粗估，
只用于预算，不用于计费——计费仍以模型返回的 `usage` 为准。这条差异写在代码注释里。

## M4 做了什么：上下文与模型网关

```bash
uv run harness verify S3.1   # 压缩：19,329 → 4,031 token，早期事实仍在摘要里
uv run harness verify S3.2   # 前缀 hash 两轮一致；system prompt 写日期会变 hash
uv run harness verify S3.3   # input=1000 / cache_read=640：子集关系，不重复相加
uv run harness verify S6.2   # 超预算 → 降级到 flash 继续，不是整单失败
uv run harness verify S9.2   # 429 带 retry-after → 第 3 次成功；不带 → 立刻失败
uv run harness verify S9.3   # 首字节超时只调用 1 次（不重放）
```

| 坑 | 具体表现 |
| --- | --- |
| 压缩 ≠ 截断 | 双层：`summary` 前置注入 + 未压缩尾部；摘要保留早期事实，压完还能回答 |
| 前缀必须字节稳定 | `build_request()` 把 system prompt 放第一位；`has_dynamic_content()` 抓日期/时间/星期 |
| token 桶是包含关系 | `input` 是总量，`cache_read`/`cache_write` 是子集；`total = input + output` |
| 超时要分口径 | 首字节超时 → 不重放（上游还没干活）；总超时 → 按重试策略 |
| 429 分两类 | 带 `retry-after` 是限速；不带是配额耗尽，重试白烧 |
| 超预算要降级 | 换便宜模型继续，并记录决策 |

**实现细节上的一个坑**：假模型默认 HTTP/1.0 会立刻断连，客户端 `conn.sock` 变 `None`，
没法给 body 设置独立超时。改成 `protocol_version = "HTTP/1.1"` 走 keep-alive 才测得出
「头已到、body 卡住」这种总超时形态。

## M5 做了什么：观测、成本账本、配额

```bash
uv run harness verify S1.3   # 8 个并发各要 80、余额 100 → 只允许 1 个
uv run harness verify S6.1   # 五层账本：task/session/turn/call/attempt
uv run harness verify S6.3   # cache 桶超出 input → 写入时直接拒绝
uv run harness verify S8.1   # span 名与 7 个 gen_ai.* 属性对齐 OTel
uv run harness verify S8.2   # session_id / trace_id 进指标 → 抛异常
uv run harness verify S8.3   # 脱敏失败 → 清空该 span 字符串字段，不丢整批、不泄露
```

| 决定 | 理由 |
| --- | --- |
| 可用额度 = 已授权 − 已结算 − **在途预留** | 「检查 + 扣减」在并发下必然穿透；预留是唯一能保证硬上限的原子操作 |
| 账本记**加权 token 单位**，不记钱 | 价格随厂商与时间变，写死假价格就是编造数字；真实成本由部署方按自己价目换算 |
| 桶互斥在**写入时**强制 | 等报表出来才发现重叠，账单已经错了 |
| 指标标签是白名单 | 高基数进指标会让监控系统自己先崩 |
| 脱敏失败 = 清空该字段 | 宁可丢信息，不可泄露；但**不能丢整批 span** |

**一条诚实标注**：导出的是 **OTLP 形状的 JSON**（`resourceSpans → scopeSpans → spans`），
不是完整 protobuf。要接真 collector 还得换 exporter——这一点写在 README 里，不夸大。

## M6 做了什么：回归集 + 并发上限 + 双 runtime

```bash
uv run harness eval --suite core --report failures   # 20 例，跑两遍结果一致
uv run harness limits --limit 3 --jobs 8             # 在途 ≤3，排队时长进指标
uv run harness run --state-dir /tmp/s --runtime langgraph
uv run harness verify S10.2                          # checkpoint 往返 langgraph saver
```

| 决定 | 证据 |
| --- | --- |
| 回归集必须能归因 | 20 例全绿；注入「配额永远放行」后 **E09 被判 `quota_oversold`**，其余 19 例不受影响 |
| 并发上限要可见 | 8 个任务、上限 3 → `max_in_flight=3`，排队等待被记录 |
| 双 runtime 要一致 | native 与 langgraph 跑同一 Task：status / steps / 下游写入次数三项相同 |
| 框架互操作走对方代码 | checkpoint 导出后由 **langgraph 自己的 `SqliteSaver.get_tuple()`** 读回，状态一致 |
| 「为什么不用框架」要有答案 | `docs/framework-comparison.md` 四节 + `--runtime langgraph` 真 backend |

**一处对原设计的修正**：规格原文写「导出为 langgraph **三表**结构」。实测：LangGraph 的
SQLite saver 是**两表**（`checkpoints` + `writes`），只有 Postgres saver 才是三表
（多 `checkpoint_blobs`）。验收按实测写，并把差异写进对照文档。

## M7 做了什么：副作用判据

M1 的账本对「意图已写、结果未写」只有一种处理：重放。下游按键 upsert 时这是对的；
下游是追加、扣减、发消息时，重放一次就多一份结果。M7 让每个带副作用的操作声明
`replay`（`safe` 或 `unsafe`，不声明按 `unsafe`），账本据此决定重放还是停下。

```bash
# 不可重放的写：写完下游后停住，另一边发真 SIGKILL
uv run harness run --state-dir /tmp/b1 --effect-replay unsafe --hold after-write &
sleep 1.5; pkill -KILL -f "state-dir /tmp/b1"

uv run harness run --state-dir /tmp/b1 --effect-replay unsafe --resume; echo $?   # 4：blocked，不重放
uv run harness effect list --state-dir /tmp/b1 --status unknown
uv run harness effect resolve --state-dir /tmp/b1 --key run-1:1:demo-effect-1 --outcome applied
uv run harness run --state-dir /tmp/b1 --effect-replay unsafe --resume; echo $?   # 0：下游仍是 1 行

# 不可重放的工具超时：不重试，0.1 秒超时就是 0.1 秒
uv run harness tool-retry --sink-dir /tmp/b2 --tool flaky_append --timeout 0.1; echo $?   # 4
```

| 决定 | 证据 |
| --- | --- |
| 账本四态：`pending` / `completed` / `cancelled` / `unknown` | `unknown` 表示结果未知且不可重放，只有 `effect resolve` 能带走（S2.6、S2.7） |
| 可重放的照旧重放 | 真 SIGKILL 后恢复：执行 2 次，下游仍 1 行（S2.5） |
| 不可重放的不重放，run 停在 `blocked` | 连续两次恢复都是退出码 4，下游 `append_log` 始终 1 行（S2.6） |
| 反查只能证明「已执行」 | 查到就不重跑；查不到仍按未知处理，因为请求可能还在路上（S2.9） |
| 幂等键带 run 与 step 作用域，外加新 run 守卫 | 两个 run 各写各的；同一个 run id 不带 `--resume` 重跑被拒（S2.8） |
| 同键不同意图报冲突 | 不返回上一个意图的结果（S2.8） |
| 超时真正限时，不可重放的工具超时后不重试 | 超时设 0.1s，调用耗时 0.104s，账本 `unknown`，下游 1 行（S4.4） |

边界：持久性只到「已提交的意图能扛进程被杀」，不承诺断电。LangGraph backend 不跟进这套语义，
`--runtime langgraph` 配 `--effect-replay unsafe` 会直接报错。恢复时 `--effect-replay` 要和第一次一致，
否则意图不同，报幂等键冲突（退出码 8）。设计与已知缺口见设计仓的 `design/decisions.md`。

## M8 做了什么：模型循环

M7 之前 runtime 跑的是三个固定的 Python 步骤，不调用模型。M8 把 model → tool → model
接进同一个 runtime：每次模型调用是一步，每次工具调用是一步，状态（对话记录、待执行的工具调用、
最终回答）跟 checkpoint 一起落盘。

```bash
uv run harness verify S11.1   # 两轮闭环：model → tool:flaky_write → model
uv run harness verify S11.2   # 工具步被真 SIGKILL：恢复后模型一共只被问 2 次
uv run harness verify S11.3   # 不可重放工具结果未知 → blocked → effect resolve → 完成
uv run harness verify S11.4   # 策略拒绝的调用不执行，POLICY_DENIED 作为工具结果交还模型
uv run harness verify S11.5   # 模型一直要工具：--max-steps 到上限判 failed
```

验收由父进程起脚本化假模型，子进程跑 `harness agent --base-url ... --prompt ...`。
脚本的消耗位置留在父进程里，所以「子进程被杀、再起一个恢复」之后，`model.calls` 就是
两次一共问了模型几次。

| 决定 | 证据 |
| --- | --- |
| 模型步提交之后才执行工具步 | 恢复落在工具步时从 checkpoint 取工具调用，不重问模型（S11.2） |
| 模型调用也经过账本（`replay=safe`） | 回答进账本后、checkpoint 写盘前被杀，恢复拿回同一个回答（`tests/test_agent_loop.py`） |
| 模型的错回灌给模型，harness 的问题才终止 run | 参数坏了、工具不存在、被策略拒绝 → 带 `is_error` 的工具结果（S11.4）；工具协议失败、模型不可用 → `failed` |
| 结果未知照样停在 `blocked` | 循环里的不可重放工具被杀后，恢复退出码 4，模型仍只被问 1 次（S11.3） |
| 步数有上限 | `--max-steps 4` 时第 5 步之前判 `failed`（S11.5） |

真模型：`--api-key-env DEEPSEEK_API_KEY` 对 `deepseek-flash` 跑过正常闭环、工具步被 SIGKILL 后恢复、
策略拒绝三个场景，全部通过，记录与三条假模型测不出的发现见 [`docs/real-model-smoke.md`](docs/real-model-smoke.md)。

边界：验收仍跑在假模型上，真模型只有这一次冒烟；不可重放工具在循环里的结果未知只在假模型上验过；
工具只有三个演示工具；对话记录不压缩（M12 的范围）；需要审批的工具调用会让 run 停在
`interrupted`，但审批本身还没有落盘（M11 的范围）；LangGraph backend 不支持动态任务。

## M9 做了什么：事件日志

M8 之前 checkpoint 就是状态本身，写坏了没有别的东西可以对照。M9 在 `ledger.sqlite` 里加了一张
append-only 的 `events` 表，run 级事件（`run_started`、`step_committed`、`run_stopped`、
`run_resumed`、`run_finished`、`projection_repaired`）和副作用事件（`effect_intent`、
`effect_outcome`、`effect_resolved`）共用一个按 run 单调、无空洞的 `seq`。checkpoint 降为
由事件算出来的投影缓存，记着自己对应到哪一条 `seq`。

```bash
uv run harness run --state-dir /tmp/ev
uv run harness log verify --state-dir /tmp/ev --run-id run-1   # 由事件重算状态，与缓存比对；不一致退出码 1
uv run harness verify S12.1   # 正常跑完：事件 seq 连续，log verify 退出码 0
uv run harness verify S12.2   # 把缓存的 step_index 改成 0：log verify 退出码 1，恢复以日志为准，不重跑
uv run harness verify S12.3   # step_committed 写入后、缓存写入前被真 SIGKILL：恢复只跑剩下的一步
```

| 决定 | 证据 |
| --- | --- |
| `step_committed` 是一步的提交点，缓存写没写成都不影响 | 提交后、缓存前被杀，恢复只执行 `report`，账本 executions=1、没有去重记录（S12.3） |
| 副作用事件与账本行同一个事务写 | 注入异常后两者都不存在（`tests/test_events.py`） |
| 状态按增量记：`set` / `extend` / `del` | `messages` 这类只增长的列表每步只记新尾巴，日志不随步数平方增长 |
| 缓存与日志对不上，以日志为准并留一条 `projection_repaired` | S12.2；S12.3 的恢复里也会出现，因为缓存落后于日志一步 |
| 收回只写不读的结构 | checkpoint 库只剩 `checkpoints` 一张表（S2.4 按新标准验）；`--durability` 只剩 `exit` / `sync`，`async` 一直是同步写，删掉 |

崩溃注入点 `--crash-after-step` 仍在提交事件之前，所以 S2.1 / S2.2 复现的「副作用已做、这一步没提交」
窗口没有变。

边界：恢复每次从第一条事件重放，没有事件快照；`log verify` 按顶层键比对；`exit` 档把事件攒到
run 结束才写，崩溃等于这次 run 全丢，和以前一样；LangGraph backend 不写事件；持久性仍只到
「进程被杀」，不承诺断电。

## M11 做了什么：审批记录

M8 里命中 `--ask-tool` 会让 run 停在 `interrupted`，但审批本身没有落盘——进程退出后再恢复，
还是要再问一次人。M11 在 `ledger.sqlite`（结构版本 3 → 4）里加了 `approvals` 表和三条审批事件
（`approval_requested`、`approval_granted`、`approval_revoked`），让审批记录跨进程存活，
并在恢复执行工具前重校验有效性与策略版本。

```bash
uv run harness approve --state-dir /tmp/ap --run-id run-1 --ttl 3600 --by why
uv run harness verify S5.4   # --ask-tool 中断（rc 3）→ approve（rc 0）→ --resume 完成（rc 0）
uv run harness verify S5.5   # 批准后改 --deny-tool 或 --ttl 0.01 过期 → 吊销并停在 blocked（rc 4）
```

| 决定 | 证据 |
| --- | --- |
| 策略版本覆盖影响判定的三项 | `PolicyEngine.version()` 对 `(rules[id,decision,pattern], approval, sandbox_mode)` 规范 JSON 做 SHA-256；`harness agent` 补齐 `--sandbox-mode` |
| 执行前**先查审批、再跑 `decide()`** | 防止批准后改成 `--deny-tool` 被 `decide()` 吞成 `POLICY_DENIED` 继续跑完，也防止删掉 `--ask-tool` 后绕过吊销 |
| 失效即吊销并停在 `blocked` | `tool_name / args_hash`、TTL、`policy_version` 任一不符，同事务写 `revoked` + `approval_revoked` 并抛 `ApprovalRevoked`（退出码 4，`blocked_on=approval_id`） |
| 审批事件与 `approvals` 行同事务 | `request_approval` 按 `(run_id, step_index, tool_call_id)` 幂等；`harness approve` 持有 `run_lock`（被占用退出码 7） |
| 工具崩溃重跑不重问人 | 有效 `approved` 跳过 `NodeInterrupt`，工具步崩溃后再次 `--resume` 直接重跑工具（`tests/test_approvals.py`） |

边界：单用户本机原型，按单条待批调用授权，不做 Agent Identity / Credential 管理或跨会话通配授权；
`revoked` 记录不可再用 `harness approve` 复活；LangGraph backend 不跟进这套审批记录。

## 收口状态

| 项 | 数字 |
| --- | --- |
| 规格 | 46 条，全部通过且带证据（`BASELINE`：46 通过 / 0 失败 / 0 未实现） |
| 单测 | 以 `uv run pytest` 的输出为准（M11 收口时 203 个） |
| 回归集 | 20 例，pass_rate 1.0，跑两遍一致 |
| runtime | native（默认）+ langgraph（可选） |

## 真模型冒烟（2026-10-18）

```bash
set -a; source ~/.env.secrets; set +a
uv run python scripts/smoke_real_model.py
```

三个场景跑通（工具调用 / 结构化输出 / 多轮记忆），并暴露两条假模型测不出的差异：

1. **`reasoning_tokens` 吃掉 `max_tokens`**：`deepseek-flash` 在结构化场景里 184 个输出 token 中
   158 是 reasoning，`max_tokens=200` 时 `content` 被截断成空串。改成 800 后通过。
2. **缓存命中字段有两套命名**：只读 `prompt_tokens_details.cached_tokens` 会在只提供
   `prompt_cache_hit_tokens` 的供应商上静默变成 0——已改成两级回退并补测。

完整记录见 [`docs/real-model-smoke.md`](docs/real-model-smoke.md)，其中也写清了**仍然没被覆盖的部分**
（真模型上无法触发 429/超时/refusal，缓存命中率没有规模数据）。

## 验收长什么样

每条规格的验收是一条命令，输出带证据行：

```
[PASS] S2.6 (M7) 不可重放副作用结果未知时不自动重放：恢复停在 blocked，账本记 unknown
        命令：harness verify S2.6
        状态：通过 —— 断言全部成立
        证据：子进程被真 SIGKILL：returncode=-9，此时下游 append_log 已有 1 行
        证据：连续两次 --resume：退出码 [4, 4]，状态 ['blocked', 'blocked']，卡在 run-1:1:demo-effect-1
        证据：账本 status=unknown、executions=1、replay=unsafe
        证据：下游 append_log=1 行（没有被重放）
```

新规格先以「未实现」登记（输出 `[TODO]`），断言写好并看到它失败之后才写实现。

设计约束：**PASSED 必须带证据**（代码里会抛异常），所以不存在「默默通过」。

## 验收默认不打真模型

`src/bridle/fake_model/` 是一个 OpenAI 兼容的脚本化模型服务，支持注入：
超时、两类 429（带/不带 `retry-after`）、`refusal`、`finish_reason=length`、慢响应。
理由：**可复现、可注入故障、可离线**。真模型只用于演示与 `eval`。

脚本耗尽时服务端返回 500 并说明原因——不静默编造回复。

## 里程碑

| M | 内容 | 规格 |
| --- | --- | --- |
| M0 | 骨架、验收框架、假模型、CI | —（基础设施） |
| M1 | runtime：checkpoint（主键含 namespace）、durability、interrupt/resume | S2.1–S2.4 |
| M2 | 策略三层（沙箱 × 审批 × 命令规则）、`SandboxBackend` | S1.1、S1.2、S5.1–S5.3 |
| M3 | 工具层：token 预算、幂等、错误分层 | S4.1–S4.3 |
| M4 | 上下文双层、前缀稳定、模型网关 | S3.1–S3.3、S6.2、S9.2、S9.3 |
| M5 | 观测、成本账本、配额预留结算 | S1.3、S6.1、S6.3、S8.1–S8.3 |
| M6 | 回归集、langgraph backend、框架对照 | S7.1、S7.2、S9.1、S10.1–S10.3 |
| M7 | 副作用判据：`replay` 声明、`unknown` 状态、键作用域、真 SIGKILL 验收 | S2.5–S2.9、S4.4 |
| M8 | 模型循环：model → tool → model 接进 runtime，恢复不重问模型 | S11.1–S11.5 |
| M9 | 事件日志：事件是真相，checkpoint 是投影缓存；删掉只写不读的两张表与 `async` 档 | S12.1–S12.3 |
| M11 | 审批记录：跨进程存活、策略版本校验与过期吊销 | S5.4–S5.5 |

## 明确不做

1. 多 agent 群聊/辩论（autogen 停更已证明不是生产默认）。
2. 内核级隔离（microVM/gVisor）——只做策略级沙箱并声明边界。
3. GUI / 编辑器集成。
4. 模型训练。
5. 跨平台（先 macOS `sandbox-exec`，Linux 留接口）。

## 与设计文档的关系

设计与证据在另一个不公开的设计仓库（`agent-harness-project/`）：
`sources/`（外部调研）、`specs/`（规格原文）、`design/`（架构与实现计划）。
**本仓库的 `src/bridle/verify/specs.py` 是 `specs/` 的代码镜像，两处必须同步。**
