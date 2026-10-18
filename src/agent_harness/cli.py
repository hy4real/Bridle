"""harness 命令行入口。

M0 只有 verify 一组命令；run / cost / trace / eval 在 M1–M6 逐个接上。
命令形状就是验收契约（见 specs 文档 §0），所以参数命名现在就要定死。
"""

from __future__ import annotations

import argparse
import sys

from .verify import runner, specs
from .verify.registry import Status


def _cmd_verify(args: argparse.Namespace) -> int:
    if args.list:
        print(runner.format_list())
        return 0

    if args.count:
        print(len(specs.SPECS))
        return 0

    outcomes = runner.run_one(args.spec) if args.spec else runner.run_all()
    if isinstance(outcomes, runner.Outcome):
        outcomes = [outcomes]

    print(runner.format_outcomes(outcomes))

    if args.allow_fail:
        return 0
    return 0 if all(o.result.status is Status.PASSED for o in outcomes) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="通用 Agent Harness")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="跑可验收规格")
    verify.add_argument("spec", nargs="?", help="只跑这一条，例如 S2.1")
    verify.add_argument("--all", action="store_true", help="跑全部（默认行为）")
    verify.add_argument("--list", action="store_true", help="列出规格清单")
    verify.add_argument("--count", action="store_true", help="只打印规格数")
    verify.add_argument(
        "--allow-fail",
        action="store_true",
        help="有未通过也返回 0（仅用于 M0 基线；CI 会断言基线内容）",
    )
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
