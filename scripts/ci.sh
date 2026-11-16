#!/usr/bin/env bash
# 本地与 CI 同一套检查。任何一步失败即退出非零。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 单测 =="
uv run pytest

echo "== 2/3 规格清单（应为 41 条）=="
count=$(uv run harness verify --list --count)
[ "$count" = "41" ] || { echo "规格数不是 41，实际 $count"; exit 1; }
echo "规格数 $count OK"

echo "== 3/3 基线：不许有「失败」，通过数必须等于预期 =="
uv run harness verify --all --allow-fail | tee report.txt
grep -q "0 失败" report.txt || { echo "有规格失败"; exit 1; }
grep -q "$(cat BASELINE)" report.txt || {
  echo "通过数变化：预期 $(cat BASELINE)，实际 $(tail -1 report.txt)"
  echo "如果这是有意的，请同时更新 BASELINE 与 README 的状态节"
  exit 1
}
echo "基线 OK（$(cat BASELINE)）"
