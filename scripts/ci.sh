#!/usr/bin/env bash
# 本地与 CI 同一套检查。任何一步失败即退出非零。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 单测 =="
uv run pytest

echo "== 2/3 规格清单（应为 30 条）=="
count=$(uv run harness verify --list --count)
[ "$count" = "30" ] || { echo "规格数不是 30，实际 $count"; exit 1; }
echo "规格数 $count OK"

echo "== 3/3 基线：30 条必须全部未实现（M0）=="
uv run harness verify --all --allow-fail | tee report.txt
grep -q "0 通过 / 0 失败 / 30 未实现" report.txt || { echo "基线不符：M0 应全部未实现"; exit 1; }
echo "基线 OK"
