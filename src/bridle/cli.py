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

from .effects import IdempotencyKeyConflict, RunAlreadyExists
from .model import Termination, classify_choice
from .policy import ApprovalPolicy, CommandRules, Decision, PolicyEngine, Rule, SandboxMode
from .runtime import AgentRuntime, Durability, RunStatus
from .runner_exec import run_exec
from .sandbox import SandboxUnavailable, build_backend
from .tools.base import ToolProtocolError
from .tools.builtin import crashing_tool, flaky_write_tool, note_read
from .tools.registry import build_fixture_registry
from .tools.runner import ToolRunner
from .concurrency import BoundedRunner
from .eval import run_suite
from .eval.suite import summarize
from .runtime.demo import EffectSink, build_demo_task, demo_effect_key
from .runtime.types import CancelToken
from .verify import runner, specs
from .verify.registry import Status

EXIT_RUN_EXISTS = 6
EXIT_KEY_CONFLICT = 8
EXIT_POLICY_DENIED = 12
EXIT_SANDBOX_UNAVAILABLE = 13
EXIT_STRUCTURED_ABORTED = 14
EXIT_TOOL_PROTOCOL = 21

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
        task = build_demo_task(state_dir, effect_key=args.effect_key)
        if args.runtime == "langgraph":
            from .runtime.langgraph_backend import LangGraphRuntime

            runtime = LangGraphRuntime(
                task, state_dir, run_id=args.run_id, cancel=cancel,
                tool_delay=args.tool_delay, crash_after_step=args.crash_after_step,
                effect_key=args.effect_key,
            )
        else:
            runtime = AgentRuntime(
                task, state_dir, run_id=args.run_id, durability=Durability(args.durability),
                cancel=cancel, tool_delay=args.tool_delay,
                crash_after_step=args.crash_after_step,
            )
        report = runtime.run(resume=args.resume)
    except RunAlreadyExists as error:
        print(json.dumps({"code": "RUN_ALREADY_EXISTS", "run_id": error.run_id,
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_RUN_EXISTS
    except IdempotencyKeyConflict as error:
        print(json.dumps({"code": "IDEMPOTENCY_KEY_CONFLICT", "key": error.key,
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_KEY_CONFLICT
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
            "has_executed": runtime.ledger.has_executed(
                demo_effect_key(args.run_id, args.effect_key)),
        },
        "sink_rows": len(EffectSink(state_dir).rows()),
        "runtime": getattr(runtime, "name", "native"),
        "checkpoints": ([c.kind for c in runtime.checkpointer.history(args.run_id)]
                        if hasattr(runtime, "checkpointer") else ["langgraph"]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return _EXIT_BY_STATUS.get(report.status, 1)


def _cmd_sandbox_probe(args: argparse.Namespace) -> int:
    try:
        backend = build_backend(
            writable_root=None if args.mode == "read-only" else Path(args.cwd or "."),
            allow_unsandboxed=args.allow_unsandboxed,
        )
    except SandboxUnavailable as error:
        print(json.dumps({"ok": False, "code": "SANDBOX_UNAVAILABLE", "error": str(error)},
                         ensure_ascii=False))
        return EXIT_SANDBOX_UNAVAILABLE
    caps = backend.capabilities()
    print(json.dumps({"ok": True, "backend": backend.name,
                      "caps": {"fs_read": caps.fs_read, "fs_write": caps.fs_write, "net": caps.net}},
                     ensure_ascii=False))
    return 0


def _cmd_policy_check(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    verdict = engine.decide(args.cmd)
    print(json.dumps({"command": args.cmd, "decision": verdict.decision.value,
                      "layer": verdict.layer, "rule": verdict.rule_id, "reason": verdict.reason},
                     ensure_ascii=False))
    return 0


def _build_engine(args: argparse.Namespace) -> PolicyEngine:
    rules = CommandRules()
    for index, pattern in enumerate(args.deny_cmd or [], start=1):
        rules.add(Rule(f"deny-{index}", Decision.FORBID, pattern))
    for index, pattern in enumerate(args.ask_cmd or [], start=1):
        rules.add(Rule(f"ask-{index}", Decision.PROMPT, pattern))
    for index, pattern in enumerate(args.allow_cmd or [], start=1):
        rules.add(Rule(f"allow-{index}", Decision.ALLOW, pattern))
    return PolicyEngine(
        rules=rules,
        approval=ApprovalPolicy(args.approval),
        sandbox_mode=SandboxMode(args.sandbox_mode),
    )


def _cmd_exec(args: argparse.Namespace) -> int:
    engine = _build_engine(args)
    verdict = engine.decide(args.cmd)
    base = {"session": args.session, "command": args.cmd,
            "decision": {"effect": verdict.decision.value, "layer": verdict.layer,
                         "rule": verdict.rule_id, "reason": verdict.reason}}

    if verdict.decision is not Decision.ALLOW:
        print(json.dumps({**base, "executed": False, "code": "POLICY_DENIED"}, ensure_ascii=False))
        return EXIT_POLICY_DENIED

    try:
        result, workspace = run_exec(
            session=args.session, command=args.cmd, state_dir=Path(args.state_dir),
            sandbox_mode=SandboxMode(args.sandbox_mode),
            deny_read=tuple(args.deny_read or ()), deny_write=tuple(args.deny_write or ()),
            allow_unsandboxed=args.allow_unsandboxed, timeout_s=args.timeout,
        )
    except SandboxUnavailable as error:
        print(json.dumps({**base, "executed": False, "code": "SANDBOX_UNAVAILABLE",
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_SANDBOX_UNAVAILABLE

    print(json.dumps({**base, "executed": True, "workspace": str(workspace),
                      "backend": result.backend, "exit_code": result.exit_code,
                      "stdout": result.stdout, "stderr": result.stderr,
                      "denied": result.denied}, ensure_ascii=False))
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    """结构化输出验收：三种非正常终止都不许写下游（S5.3）。"""
    import urllib.request

    from .fake_model.server import FakeModelServer, ScriptedModel

    steps = json.loads(Path(args.fake_script).read_text(encoding="utf-8"))
    model = ScriptedModel(steps)
    with FakeModelServer(model) as server:
        request = urllib.request.Request(
            f"{server.base_url}/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "extract"}]}).encode(),
            headers={"content-type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())

    choice = payload["choices"][0]
    termination, detail = classify_choice(choice)
    written = False
    if termination is Termination.OK:
        from .runtime.demo import EffectSink
        sink = EffectSink(Path(args.sink_dir))
        sink.upsert(args.key, {"content": choice["message"].get("content")})
        written = True

    print(json.dumps({"termination": termination.value, "detail": detail,
                      "written_downstream": written}, ensure_ascii=False))
    return 0 if termination is Termination.OK else EXIT_STRUCTURED_ABORTED


def _cmd_tools(args: argparse.Namespace) -> int:
    registry = build_fixture_registry(args.count)
    selection = registry.select(args.query, budget_tokens=args.budget)
    print(json.dumps({
        "registered": len(registry),
        "total_definition_tokens": registry.total_definition_tokens(),
        "budget": args.budget,
        "selected": selection.count,
        "selected_tokens": selection.tokens,
        "skipped": selection.skipped,
        "names": [tool.name for tool in selection.tools],
    }, ensure_ascii=False))
    return 0


def _cmd_tool_call(args: argparse.Namespace) -> int:
    from .runtime.demo import EffectSink

    sink_dir = Path(args.sink_dir) if args.sink_dir else Path("/tmp/bridle-tool")
    if args.name == "note_read":
        tool = note_read()
    elif args.name == "explode":
        tool = crashing_tool()
    elif args.name == "flaky_write":
        tool = flaky_write_tool(EffectSink(sink_dir))
    else:
        print(json.dumps({"kind": "protocol_error", "error": f"未知工具 {args.name}"}, ensure_ascii=False))
        return EXIT_TOOL_PROTOCOL

    try:
        outcome = tool.run(json.loads(args.args))
    except ToolProtocolError as error:
        print(json.dumps({"kind": "protocol_error", "tool": tool.name, "error": str(error)},
                         ensure_ascii=False))
        return EXIT_TOOL_PROTOCOL

    print(json.dumps({
        "kind": outcome.kind.value,
        "tool": outcome.tool,
        "is_error": outcome.is_error,
        "message": outcome.to_tool_message(),
    }, ensure_ascii=False))
    return 0


def _cmd_tool_retry(args: argparse.Namespace) -> int:
    """复现「超时了但上游已执行」，验证重试不重复写（S4.2）。"""
    from .runtime.demo import EffectSink
    from .runtime.ledger import EffectLedger

    sink_dir = Path(args.sink_dir)
    sink = EffectSink(sink_dir)
    ledger = EffectLedger(sink_dir)
    runner = ToolRunner(ledger, run_id=args.run_id, retries=2, timeout_s=args.timeout)
    tool = flaky_write_tool(sink)

    outcome = runner.call(tool, {"target": args.target})
    rows = ledger.rows()
    print(json.dumps({
        "tool": tool.name,
        "kind": outcome.kind.value,
        "attempts": rows[0]["attempts"] if rows else 0,
        "ledger_rows": len(rows),
        "sink_rows": len(sink.rows()),
    }, ensure_ascii=False))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    first = run_suite()
    second = run_suite()
    report = {
        "suite": args.suite,
        "runs": 2,
        "deterministic": summarize(first) == summarize(second),
        "summary": summarize(first),
    }
    if args.report == "failures":
        report["failures"] = summarize(first)["failures"]
    if args.report == "cases":
        report["cases"] = [{"id": r.id, "name": r.name, "area": r.area,
                            "mode": r.mode.value if r.mode else None} for r in first]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if summarize(first)["failed"] == 0 else 1


def _cmd_concurrency(args: argparse.Namespace) -> int:
    import time

    runner = BoundedRunner(args.limit)
    runner.map([lambda: time.sleep(args.job_s) for _ in range(args.jobs)])
    print(json.dumps(runner.stats.as_dict(), ensure_ascii=False))
    return 0


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
    run.add_argument("--runtime", default="native", choices=["native", "langgraph"])
    run.add_argument("--tool-delay", type=float, default=0.0)
    run.add_argument("--effect-key", default="demo-effect-1")
    run.add_argument("--crash-after-step", type=int, default=None,
                     help="在第 N 步执行完、checkpoint 写盘前直接 os._exit(137)（复现最危险的窗口）")
    run.set_defaults(func=_cmd_run)

    for name, func, help_text in (
        ("eval", _cmd_eval, "工程回归集：20 例 + 失败模式分类"),
        ("limits", _cmd_concurrency, "并发上限与排队时长"),
        ("tools", _cmd_tools, "工具定义预算：60 个工具时只注入 top-k"),
        ("tool-call", _cmd_tool_call, "调用一个内置工具（区分业务失败与协议失败）"),
        ("tool-retry", _cmd_tool_retry, "超时重试：验证不重复写"),
        ("sandbox-probe", _cmd_sandbox_probe, "检查沙箱后端是否可用（fail-closed 自检）"),
        ("policy-check", _cmd_policy_check, "只做策略判定，不执行"),
        ("exec", _cmd_exec, "在会话工作区里执行一条命令（过策略 + 过沙箱）"),
        ("extract", _cmd_extract, "结构化输出分类：三条旁路不写下游"),
    ):
        sub_parser = sub.add_parser(name, help=help_text)
        if name in ("exec", "policy-check"):
            sub_parser.add_argument("--cmd", required=True)
            sub_parser.add_argument("--deny-cmd", action="append", default=[])
            sub_parser.add_argument("--ask-cmd", action="append", default=[])
            sub_parser.add_argument("--allow-cmd", action="append", default=[])
            sub_parser.add_argument("--approval", default="on-request",
                                    choices=[a.value for a in ApprovalPolicy])
            sub_parser.add_argument("--sandbox-mode", default="read-only",
                                    choices=[m.value for m in SandboxMode])
            sub_parser.add_argument("--allow-unsandboxed", action="store_true")
        if name == "exec":
            sub_parser.add_argument("--session", required=True)
            sub_parser.add_argument("--state-dir", required=True)
            sub_parser.add_argument("--deny-read", action="append", default=[])
            sub_parser.add_argument("--deny-write", action="append", default=[])
            sub_parser.add_argument("--timeout", type=float, default=30.0)
        if name == "sandbox-probe":
            sub_parser.add_argument("--mode", default="read-only",
                                    choices=[m.value for m in SandboxMode])
            sub_parser.add_argument("--cwd", default=None)
            sub_parser.add_argument("--allow-unsandboxed", action="store_true")
        if name == "eval":
            sub_parser.add_argument("--suite", default="core")
            sub_parser.add_argument("--report", choices=["summary", "failures", "cases"],
                                    default="summary")
        if name == "limits":
            sub_parser.add_argument("--limit", type=int, default=3)
            sub_parser.add_argument("--jobs", type=int, default=8)
            sub_parser.add_argument("--job-s", type=float, default=0.05)
        if name == "tools":
            sub_parser.add_argument("--count", type=int, default=60)
            sub_parser.add_argument("--query", default="")
            sub_parser.add_argument("--budget", type=int, default=400)
        if name == "tool-call":
            sub_parser.add_argument("--name", required=True)
            sub_parser.add_argument("--args", default="{}")
            sub_parser.add_argument("--sink-dir", default=None)
        if name == "tool-retry":
            sub_parser.add_argument("--sink-dir", required=True)
            sub_parser.add_argument("--target", default="notes/a.md")
            sub_parser.add_argument("--run-id", default="run-1")
            sub_parser.add_argument("--timeout", type=float, default=0.1)
        if name == "extract":
            sub_parser.add_argument("--fake-script", required=True)
            sub_parser.add_argument("--sink-dir", required=True)
            sub_parser.add_argument("--key", default="extract-1")
        sub_parser.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
