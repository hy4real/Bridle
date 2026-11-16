"""模型循环：model → tool → model，跑在 `runtime.AgentRuntime` 的动态任务上。

放在 `runtime/` 之外：它同时依赖 runtime、tools、model、policy，而 `tools.runner` 已经依赖
`runtime.ledger`。放进 runtime 会成环。runtime 不知道有模型这回事。
"""
