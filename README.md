# agent-harness

一个**不预设用途的通用 Agent Harness**：runtime + agent loop + 有主张的默认栈。

> Agent = Model + Harness（[LangChain 文档](https://docs.langchain.com/oss/python/langchain/agents)）。
> Model 做不了，**Harness 是能由个人做完整的那一半**。

## 现在能跑什么

```bash
uv sync --dev
uv run harness verify --list      # 30 条可验收规格
uv run harness verify --all       # 逐条跑（M0 阶段全部「未实现」，退出码 1）
uv run harness verify S2.1        # 只跑一条
./scripts/ci.sh                   # 单测 + 规格数 + 基线断言
```

**M0 的状态是「30 条全部未实现」，这是基线，不是缺陷。** CI 会断言这个基线，
防止有人把断言写成恒真来「让测试变绿」。每完成一个里程碑，对应规格从
「未实现」变成「通过」，README 的这一节同步更新。

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
