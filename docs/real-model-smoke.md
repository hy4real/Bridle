# 真模型冒烟记录（DeepSeek）

> 目的：假模型让验收可复现，但也让「协议兼容性」变成假设。这一步专门找**假模型测不出的差异**。
> 运行方式：`set -a; source ~/.env.secrets; set +a; uv run python scripts/smoke_real_model.py`
> 密钥只从环境变量读，脚本只打印 `present=True len=35`，不打印值。

## 第一次运行（2026-10-18，`deepseek-flash`）

| 场景 | 结果 | 关键读数 |
| --- | --- | --- |
| A 单轮工具调用 | 通过 | `finish_reason=tool_calls`，`get_weather({"city": "杭州"})`，耗时 0.91s |
| B 结构化输出（`response_format=json_object`，max_tokens=200） | **失败** | `finish_reason=length`，`completion_tokens=200`（打满），`json_valid=false` |
| C 多轮记忆（max_tokens=50） | 通过但内容为空 | 首轮 `content=""`、`output=50`（打满） |

## 两条真模型才暴露的差异

### 1. 推理模型的 `reasoning_tokens` 吃掉 `max_tokens`

`usage.completion_tokens_details.reasoning_tokens` 在三个场景里分别是 **13 / 158 / 33**。
场景 B 里 `output=184`，其中 **158 是 reasoning**——留给答案的只有 26 个 token，
于是 `content` 被截断成空串，`finish_reason=length`。

**工程含义**：
- 结构化输出必须给足 `max_tokens`（本仓库改成 800 后 B 通过，`finish_reason=stop`）；
- 更稳的做法是读 `reasoning_tokens` 后再判断预算，而不是拍一个 200；
- 我们的分类器把这种情况判成 `truncated` 并且**不写下游**——安全行为是对的，
  但"看起来能跑"的假模型永远测不出这条。

### 2. 缓存命中字段有两套命名

DeepSeek 同时给：

```json
"prompt_tokens_details": {"cached_tokens": 128},
"prompt_cache_hit_tokens": 128,
```

我们原来只读 `details.cached_tokens`。**如果某家供应商只给 `prompt_cache_hit_tokens`，
`cache_read` 会静默变成 0**——这正是面经里「一次任务烧了 1000 块、cacheReadTokens 全程 0 且无人感知」
的事故形态。已改成两级回退，并补了单测
`test_cache_read_falls_back_to_deepseek_field`。

## 第二次运行（修正后）

| 场景 | 结果 | 读数 |
| --- | --- | --- |
| A 工具调用 | 通过 | `input=296 output=52 cache_read=128 reasoning=13` |
| B 结构化输出（max_tokens=800） | 通过 | `finish_reason=stop`，JSON 合法，`reasoning=158` |
| C 多轮记忆 | 通过 | 第二轮答出「代号是 ORION」，`cache_read` 仍为 0（短前缀未达缓存阈值） |

## 仍然未被覆盖的部分（必须主动说）

1. **故障路径无法在真模型上触发**：429 两类、首字节超时、refusal——只能靠假模型注入。
2. **缓存命中率没有规模数据**：单次对话里 `cache_read` 时有时无，样本量 3 次，不能当结论。
3. **`gpt-*` 路径未验证**：无 OpenAI key，`strict: true` 的 refusal 旁路只能在假模型上验。
4. **成本没有换算成钱**：账本记的是加权 token 单位，真实成本要按部署方的价目表换算。
