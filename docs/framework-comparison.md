# 框架对照：LangGraph / LangChain / 我们自己

> 目的：回答「为什么不用现成框架」——用代码与官方实现对照，不用口头解释。
> 取数：`langgraph` 与 `langgraph-checkpoint-sqlite` 由 `uv` 装入本仓库 dev 依赖（2026-10-18）；
> 行为断言见 `harness verify S10.1–S10.3`。

## LangGraph 提供什么

| 能力 | 官方机制 | 我们的对应实现 |
| --- | --- | --- |
| 状态机执行 | `StateGraph` + conditional edges | `runtime/loop.py` 的步骤循环 |
| 持久化 | `Checkpointer` 接口；SQLite saver 两表（`checkpoints` / `writes`），Postgres saver 三表（多 `checkpoint_blobs`） | `checkpoint_blobs` 我们也拆了（三表），字段与主键对齐 |
| 恢复 | 从最新 checkpoint 继续；`interrupt` 恢复是**整个节点重跑** | 相同语义：`step_index` 之后重跑整步 |
| 流式与中断 | `stream()` / `interrupt()` / `Command(resume=...)` | 未实现流式；中断用 `NodeInterrupt` + `resume` |
| 并发与子图 | `Send` / 子图 / `checkpoint_ns` | 未实现子图；`checkpoint_ns` 字段已按对方主键预留 |

## LangChain 提供什么

| 能力 | 官方机制 | 我们的对应 |
| --- | --- | --- |
| Agent 循环 | `create_agent(model, tools, middleware)` | `AgentRuntime` + `StepContext` |
| 中间件 | 八个钩子（system prompt / tools / model call / tool call / …） | 设计定稿，实现按里程碑推进（尚未全部落地） |
| 工具定义 | `@tool` + schema 自动生成 | `Tool` dataclass + 手写 schema（含 `side_effect` / 幂等键，这是对面没有的一等字段） |
| Harness 定义 | **Agent = Model + Harness** | 同样的定义，但默认栈更薄：不内置 filesystem / subagents |

## 我们提供什么（对方没有的）

1. **幂等账本**：`intent + attempt_id + result` 三件套 + 按键 upsert 的下游约定。
   LangGraph 只存 channel 值，不记录「这次副作用做没做过」。
2. **三层策略**：沙箱档位 × 审批策略 × 命令规则（deny > ask > allow，`never` 把 ask 降级为拒绝）。
   框架层没有权限引擎。
3. **配额预留结算**：`可用 = 已授权 − 已结算 − 在途预留`。框架层没有配额概念。
4. **验收即命令**：44 条规格各自一条 `harness verify <id>`，通过的都输出证据行。

## 故意没做

1. 流式输出与 token 级中断——个人项目规模用不上，做了也无法验证。
2. 子图 / 多 agent 编排——autogen 停更已经说明这个形态不是生产默认。
3. 完整 OTLP protobuf 导出——只做 OTLP 形状 JSON，接真 collector 要换 exporter。
4. 跨平台沙箱——只做 macOS `sandbox-exec`，Linux 留接口。

## 差异（跑出来的，不是感觉）

| 项 | LangGraph | 我们 |
| --- | --- | --- |
| checkpoint 表 | SQLite 两表 / Postgres 三表 | 三表（SQLite） |
| 恢复语义 | 整节点重跑 | 整步重跑（同） |
| 双 runtime 一致性 | — | `harness verify S10.1` 断言 status / steps / 下游写入次数三项相同 |
| 框架依赖 | 必需 | **可选**：`--runtime langgraph`，native 是默认路径 |
