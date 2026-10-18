"""harness 命令行入口。

命令形状就是验收契约（见 specs 文档 §0），参数命名现在定死，后面不再改。

M0: verify
M1: run（含 resume / durability / 崩溃注入 / 取消）
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from .runtime import AgentRuntime, Durability, RunStatus
from .runtime.demo import EffectSink, build_demo_task
from .runtime.types import CancelToken
from .verify import runner, specs
from .verify.registry import Status

_EXIT_BY_STATUS = {
    RunStatus.COMPLETED: 0,
    RunStatus.INTERRUPTED: 3,
    RunStatus.CANCELLED: 130,
}


def _cmd_verify(args: argparse.Namespace) -> int:
    if args.count:  # 先判 count：--list --count 同时出现时以 count 为准
        print(len(specs.SPECS))
        return 0
    if args.list:
        print(runner.format_list())
        return 0

    outcomes = runner.run_one(args.spec) if args.spec else runner.run_all()
    if isinstance(outcomes, runner.Outcome):
        outcomes = [outcomes]
    print(runner.format_outcomes(outcomes))
    if args.allow_fail:
        return 0
    return 0 if all(o.result.status is Status.PASSED for o in outcomes) else 1


def _cmd_run(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir)
    cancel = CancelToken()

    def _on_sigint(signum: int, frame: object) -> None:
        # 协作式取消：只置位，不 kill。让在途步骤自己收尾并把状态写清楚。
        print("[harness] 收到 SIGINT，发起协作式取消", file=sys.stderr)
        cancel.cancel()

    previous = signal.signal(signal.SIGINT, _on_sigint)
    try:
        runtime = AgentRuntime(
            build_demo_task(state_dir, effect_key=args.effect_key),
            state_dir,
            run_id=args.run_id,
            durability=Durability(args.durability),
            cancel=cancel,
            tool_delay=args.tool_delay,
            crash_after_step=args.crash_after_step,
        )
        report = runtime.run(resume=args.resume)
    finally:
        signal.signal(signal.SIGINT, previous)

    payload = {
        "run_id": report.run_id,
        "status": report.status.value,
        "steps_run": report.steps_run,
        "replayed_steps": report.replayed_steps,
        "ledger": {
            "rows": runtime.ledger.rows(),
            "attempts": runtime.ledger.attempts_log(),
            "has_executed": runtime.ledger.has_executed(args.effect_key),
        },
        "sink_rows": len(EffectSink(state_dir).rows()),
        "checkpoints": [c.kind for c in runtime.checkpointer.history(args.run_id)],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return _EXIT_BY_STATUS.get(report.status, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="通用 Agent Harness")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="跑可验收规格")
    verify.add_argument("spec", nargs="?", help="只跑这一条，例如 S2.1")
    verify.add_argument("--all", action="store_true", help="跑全部（默认行为）")
    verify.add_argument("--list", action="store_true", help="列出规格清单")
    verify.add_argument("--count", action="store_true", help="只打印规格数")
    verify.add_argument("--allow-fail", action="store_true", help="有未通过也返回 0（仅 M0 基线用）")
    verify.set_defaults(func=_cmd_verify)

    run = sub.add_parser("run", help="跑一次演示任务（M1）")
    run.add_argument("--state-dir", required=True)
    run.add_argument("--run-id", default="run-1")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--durability", default="sync", choices=[d.value for d in Durability])
    run.add_argument("--tool-delay", type=float, default=0.0)
    run.add_argument("--effect-key", default="demo-effect-1")
    run.add_argument("--crash-after-step", type=int, default=None,
                     help="在第 N 步执行完、checkpoint 写盘前直接 os._exit(137)（复现最危险的窗口）")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
