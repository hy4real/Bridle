# Bridle 工程约定

## 范围

通用 Agent Harness：runtime + agent loop + 有主张的默认栈。设计文档在求职仓库
`2026-10-16/agent-harness-project/`（`design/` 与 `specs/`），**本仓库只放代码与测试**。

## 开发纪律（按优先级）

1. **验收驱动**：一条能力先有规格（`verify/specs.py` 里的一条 + 一个断言函数），再写实现。
   不许「先写功能再补测试」。规格顺序见 `README.md` 的里程碑表。
2. **不许让验收变绿的方式偷懒**：禁止把断言写成恒真、禁止 `--allow-fail` 混进 CI 主流程、
   禁止为了通过而修改判定标准。要改标准就改规格文档，并在提交信息里写清理由。
3. **PASSED 必须带证据**：`CheckResult` 在 `PASSED` 且 `evidence` 为空时抛异常，不要绕过它。
4. **失败路径优先**：新功能先写失败用例（超时/429/被 kill/权限拒绝），再写成功路径。
5. **不注释掉报错、不加 `# type: ignore` 了事**（确有必要的必须写明原因）。
6. **密钥只从环境变量读**，不进代码、不进日志、不进 commit。验证时只打印变量名与长度。

## 目录

```
src/bridle/cli.py           命令入口（verify 已实现，run/eval/cost/trace 待接）
src/bridle/verify/          规格登记表 + 执行器 + 输出
src/bridle/fake_model/      脚本化假模型服务（OpenAI 兼容）
tests/                             框架自测 + 假模型行为契约
scripts/ci.sh                      本地与 CI 同一套检查
```

`verify/specs.py` 与设计仓库的 `specs/a1-a10-acceptance-specs.md` 是**镜像关系**：
改一处必须改另一处，且 `tests/test_verify.py::test_spec_count_matches_docs` 会守住条数。

## 提交前

```bash
./scripts/ci.sh
```

任何一步失败都不提交。只在本地提交，不推送（推送需单独授权）。
