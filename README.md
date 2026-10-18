# agent-harness

一个**不预设用途的通用 Agent Harness**：runtime + agent loop + 有主张的默认栈。

> Agent = Model + Harness（[LangChain 文档](https://docs.langchain.com/oss/python/langchain/agents)）。
> Model 做不了，**Harness 是能由个人做完整的那一半**。

## 现在能跑什么

```bash
uv sync --dev
uv run harness verify --list      # 30 条可验收规格
uv run harness verify --all       # 逐条跑：12 通过 / 0 失败 / 18 未实现
uv run harness verify S2.1        # 只跑一条
./scripts/ci.sh                   # 单测 + 规格数 + 基线断言
```

**当前状态：M3 已交付（12 条通过），其余 18 条按里程碑逐条变绿。**
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

三条设计决定：

| 决定 | 理由 |
| --- | --- |
| checkpoint 拆三表（`checkpoints`/`blobs`/`writes`），主键含 `checkpoint_ns` | 与 LangGraph 同构；小元数据与大值分开，读元数据不必反序列化整份状态 |
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

## 验收长什么样

每条规格的验收是一条命令，输出带证据行：

```
[TODO] S2.1 (M1) kill -9 后从 checkpoint 恢复，且副作用账本只有一条
        命令：harness verify S2.1
        状态：未实现 —— M1 未开始：S2.1 的断言函数尚未实现
```

设计约束：**PASSED 必须带证据**（代码里会抛异常），所以不存在「默默通过」。

## 验收默认不打真模型

`src/agent_harness/fake_model/` 是一个 OpenAI 兼容的脚本化模型服务，支持注入：
超时、两类 429（带/不带 `retry-after`）、`refusal`、`finish_reason=length`、慢响应。
理由：**可复现、可注入故障、可离线**。真模型只用于演示与 `eval`。

脚本耗尽时服务端返回 500 并说明原因——不静默编造回复。

## 里程碑

| M | 内容 | 规格 |
| --- | --- | --- |
| M0 | 骨架、验收框架、假模型、CI | —（基础设施） |
| M1 | runtime：checkpoint 三表、durability 三档、interrupt/resume | S2.1–S2.4 |
| M2 | 策略三层（沙箱 × 审批 × 命令规则）、`SandboxBackend` | S1.1、S1.2、S5.1–S5.3 |
| M3 | 工具层：token 预算、幂等、错误分层 | S4.1–S4.3 |
| M4 | 上下文双层、前缀稳定、模型网关 | S3.1–S3.3、S6.2、S9.2、S9.3 |
| M5 | 观测、成本账本、配额预留结算 | S1.3、S6.1、S6.3、S8.1–S8.3 |
| M6 | 回归集、langgraph backend、框架对照 | S7.1、S7.2、S9.1、S10.1–S10.3 |

## 明确不做

1. 多 agent 群聊/辩论（autogen 停更已证明不是生产默认）。
2. 内核级隔离（microVM/gVisor）——只做策略级沙箱并声明边界。
3. GUI / 编辑器集成。
4. 模型训练。
5. 跨平台（先 macOS `sandbox-exec`，Linux 留接口）。

## 与设计文档的关系

设计与证据在求职仓库的 `2026-10-16/agent-harness-project/`：
`sources/`（外部调研）、`specs/`（规格原文）、`design/`（架构与实现计划）。
**本仓库的 `src/agent_harness/verify/specs.py` 是 `specs/` 的代码镜像，两处必须同步。**
