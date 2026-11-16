"""harness 命令行入口。

命令形状就是验收契约（见 specs 文档 §0），参数命名现在定死，后面不再改。

M0: verify
M1: run（含 resume / durability / 崩溃注入 / 取消）
M7: effect list / effect resolve（查看与解除结果未知的副作用）
M8: agent（模型循环：model → tool → model，跑在 native runtime 上）

退出码：0 完成；3 中断；4 副作用结果未知（run 停在 blocked）；6 run id 已存在；
7 run 被另一个进程持有；8 幂等键冲突；12 策略拒绝；13 沙箱不可用；14 结构化输出中止；
21 工具协议失败；130 取消。
"""

from __future__ import annotations

import argparse
import getpass
import json
import signal
import sys
from pathlib import Path

from .effects import (
    EffectOutcomeUnknown,
    IdempotencyKeyConflict,
    Replay,
    RunAlreadyExists,
    RunLocked,
    run_id_of,
)
from .model import Termination, classify_choice
from .policy import ApprovalPolicy, CommandRules, Decision, PolicyEngine, Rule, SandboxMode
from .runtime import AgentRuntime, Durability, RunStatus
from .runner_exec import run_exec
from .sandbox import SandboxUnavailable, build_backend
from .tools.base import ToolProtocolError
from .tools.builtin import crashing_tool, flaky_append_tool, flaky_write_tool, note_read
from .tools.registry import build_fixture_registry
from .tools.runner import ToolRunner
from .concurrency import BoundedRunner
from .eval import run_suite
from .eval.suite import summarize
from .runtime.demo import HOLD_POINTS, EffectSink, build_demo_task, demo_effect_key
from .runtime.types import CancelToken
from .verify import runner, specs
from .verify.registry import Status

EXIT_OUTCOME_UNKNOWN = 4
EXIT_RUN_EXISTS = 6
EXIT_RUN_LOCKED = 7
EXIT_KEY_CONFLICT = 8
EXIT_POLICY_DENIED = 12
EXIT_SANDBOX_UNAVAILABLE = 13
EXIT_STRUCTURED_ABORTED = 14
EXIT_TOOL_PROTOCOL = 21

_EXIT_BY_STATUS = {
    RunStatus.COMPLETED: 0,
    RunStatus.INTERRUPTED: 3,
    RunStatus.BLOCKED: EXIT_OUTCOME_UNKNOWN,
    RunStatus.FAILED: 1,
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
    replay = Replay(args.effect_replay)
    if args.runtime == "langgraph" and (replay is Replay.UNSAFE or args.hold):
        # langgraph backend 不跟进 M7 的语义（blocked / unknown）。悄悄降级比报错危险。
        print(json.dumps({"code": "UNSUPPORTED_ON_LANGGRAPH",
                          "error": "--effect-replay unsafe 与 --hold 只支持 native runtime"},
                         ensure_ascii=False))
        return 2
    cancel = CancelToken()

    def _on_sigint(signum: int, frame: object) -> None:
        # 协作式取消：只置位，不 kill。让在途步骤自己收尾并把状态写清楚。
        print("[harness] 收到 SIGINT，发起协作式取消", file=sys.stderr)
        cancel.cancel()

    previous = signal.signal(signal.SIGINT, _on_sigint)
    try:
        task = build_demo_task(state_dir, effect_key=args.effect_key, replay=replay,
                               hold=args.hold, verify=args.effect_verify)
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
    except RunLocked as error:
        print(json.dumps({"code": "RUN_LOCKED", "run_id": error.run_id,
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_RUN_LOCKED
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
        "error": report.error,
        "blocked_on": report.blocked_on,
        "ledger": {
            "rows": runtime.ledger.rows(),
            "attempts": runtime.ledger.attempts_log(),
            "has_executed": runtime.ledger.has_executed(
                demo_effect_key(args.run_id, args.effect_key)),
        },
        "sink_rows": len(EffectSink(state_dir).rows()),
        "append_rows": len(EffectSink(state_dir).append_rows()),
        "runtime": getattr(runtime, "name", "native"),
        "checkpoints": ([c.kind for c in runtime.checkpointer.history(args.run_id)]
                        if hasattr(runtime, "checkpointer") else ["langgraph"]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return _EXIT_BY_STATUS.get(report.status, 1)


def _agent_tools(state_dir: Path, names: list[str] | None):
    """agent 能用的工具。写下游的两个都落到 state_dir 下的 sink：一个按键 upsert，一个追加。"""
    from .tools.registry import ToolRegistry

    sink = EffectSink(state_dir)
    available = {
        "note_read": note_read(),
        "flaky_write": flaky_write_tool(sink, slow_first=0.0),   # 可重放
        "flaky_append": flaky_append_tool(sink, slow=0.0),        # 不可重放（未声明）
    }
    unknown = sorted(set(names or []) - set(available))
    if unknown:
        raise ValueError(f"没有这些工具：{unknown}；可用 {sorted(available)}")
    registry = ToolRegistry()
    for name, tool in available.items():
        if names is None or name in names:
            registry.register(tool)
    return registry


def _cmd_agent(args: argparse.Namespace) -> int:
    from .agent.task import AgentConfig, build_agent_task
    from .model.gateway import ModelGateway

    state_dir = Path(args.state_dir)
    rules = CommandRules()
    for index, pattern in enumerate(args.deny_tool or [], start=1):
        rules.add(Rule(f"deny-tool-{index}", Decision.FORBID, pattern))
    for index, pattern in enumerate(args.ask_tool or [], start=1):
        rules.add(Rule(f"ask-tool-{index}", Decision.PROMPT, pattern))
    rules.add(Rule("allow-tools", Decision.ALLOW, "tool:*"))  # deny 永远先判，allow 开不了例外
    try:
        registry = _agent_tools(state_dir, args.tools.split(",") if args.tools else None)
    except ValueError as error:
        print(json.dumps({"code": "BAD_TOOLS", "error": str(error)}, ensure_ascii=False))
        return 2
    config = AgentConfig(
        gateway=ModelGateway(args.base_url, model=args.model, retries=1,
                             first_byte_timeout=args.model_timeout, total_timeout=args.model_timeout),
        registry=registry,
        policy=PolicyEngine(rules=rules, approval=ApprovalPolicy(args.approval)),
        max_steps=args.max_steps,
        tool_timeout_s=args.tool_timeout,
    )

    cancel = CancelToken()
    previous = signal.signal(signal.SIGINT, lambda signum, frame: cancel.cancel())
    try:
        runtime = AgentRuntime(build_agent_task(args.prompt, config), state_dir,
                               run_id=args.run_id, cancel=cancel)
        report = runtime.run(resume=args.resume)
    except RunAlreadyExists as error:
        print(json.dumps({"code": "RUN_ALREADY_EXISTS", "run_id": error.run_id,
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_RUN_EXISTS
    except RunLocked as error:
        print(json.dumps({"code": "RUN_LOCKED", "run_id": error.run_id,
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_RUN_LOCKED
    except IdempotencyKeyConflict as error:
        print(json.dumps({"code": "IDEMPOTENCY_KEY_CONFLICT", "key": error.key,
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_KEY_CONFLICT
    finally:
        signal.signal(signal.SIGINT, previous)

    latest = runtime.checkpointer.latest(args.run_id)
    state = latest.state if latest is not None else {}
    sink = EffectSink(state_dir)
    print(json.dumps({
        "run_id": report.run_id,
        "status": report.status.value,
        "steps_run": report.steps_run,
        "replayed_steps": report.replayed_steps,
        "error": report.error,
        "blocked_on": report.blocked_on,
        "final": state.get("final"),
        "steps_done": state.get("steps_done", []),
        "model_steps": state.get("model_steps", 0),
        "messages": len(state.get("messages", [])),
        "ledger": [{"key": r["key"], "status": r["status"], "executions": r["executions"]}
                   for r in runtime.ledger.rows()],
        "sink_rows": len(sink.rows()),
        "append_rows": len(sink.append_rows()),
    }, ensure_ascii=False, indent=2))
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
    """复现「超时了但上游已执行」。

    flaky_write（可重放，下游 upsert）：重试，不重复写（S4.2）。
    flaky_append（不可重放）：不重试，账本记 unknown，退出码 4（S4.4）。
    """
    import time

    from .runtime.demo import EffectSink
    from .runtime.ledger import EffectLedger

    sink_dir = Path(args.sink_dir)
    sink = EffectSink(sink_dir)
    ledger = EffectLedger(sink_dir)
    runner = ToolRunner(ledger, run_id=args.run_id, retries=2, timeout_s=args.timeout)
    tool = flaky_append_tool(sink) if args.tool == "flaky_append" else flaky_write_tool(sink)

    started = time.monotonic()
    kind, code, exit_code = None, None, 0
    try:
        kind = runner.call(tool, {"target": args.target}).kind.value
    except EffectOutcomeUnknown as error:
        code, exit_code = "EFFECT_OUTCOME_UNKNOWN", EXIT_OUTCOME_UNKNOWN
        print(f"[harness] {error}", file=sys.stderr)
    elapsed = time.monotonic() - started

    rows = ledger.rows()
    print(json.dumps({
        "tool": tool.name,
        "replay": tool.replay_mode().value,
        "kind": kind,
        "code": code,
        "status": rows[0]["status"] if rows else None,
        "attempts": rows[0]["attempts"] if rows else 0,
        "executions": rows[0]["executions"] if rows else 0,
        "ledger_rows": len(rows),
        "sink_rows": len(sink.rows()),
        "append_rows": len(sink.append_rows()),
        "elapsed_s": round(elapsed, 3),
    }, ensure_ascii=False))
    return exit_code


def _open_ledger(state_dir: Path):
    """只打开已存在的账本。目录敲错时报错，不凭空建一个空库。"""
    from .runtime.ledger import EffectLedger

    try:
        return EffectLedger(state_dir, create=False)
    except FileNotFoundError as error:
        print(json.dumps({"code": "NO_LEDGER", "error": str(error)}, ensure_ascii=False))
        return None


def _cmd_effect_list(args: argparse.Namespace) -> int:
    ledger = _open_ledger(Path(args.state_dir))
    if ledger is None:
        return 1
    rows = ledger.rows(status=args.status)
    print(json.dumps({
        "count": len(rows),
        "effects": [{
            "key": row["key"], "status": row["status"], "replay": row["replay"],
            "attempts": row["attempts"], "executions": row["executions"],
            "intent": json.loads(row["intent_json"]),
            "resolved_by": row["resolved_by"],
        } for row in rows],
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_effect_resolve(args: argparse.Namespace) -> int:
    """把一条 unknown 解除成「已执行」或「没执行」。这是人的判断，账本只负责记下是谁判的。"""
    from .runtime.lock import run_lock

    state_dir = Path(args.state_dir)
    ledger = _open_ledger(state_dir)
    if ledger is None:
        return 1
    outcome = args.outcome.replace("-", "_")
    try:
        # 必须持有 run 锁：那个 run 的进程如果还活着，此刻下结论就是在猜
        with run_lock(state_dir, run_id_of(args.key)):
            ledger.resolve(args.key, outcome, resolved_by=args.by)
    except RunLocked as error:
        print(json.dumps({"code": "RUN_LOCKED", "run_id": error.run_id,
                          "error": str(error)}, ensure_ascii=False))
        return EXIT_RUN_LOCKED
    except KeyError:
        print(json.dumps({"code": "NO_SUCH_EFFECT", "key": args.key}, ensure_ascii=False))
        return 1
    except ValueError as error:
        print(json.dumps({"code": "NOT_RESOLVABLE", "key": args.key, "error": str(error)},
                         ensure_ascii=False))
        return 1
    row = next(row for row in ledger.rows() if row["key"] == args.key)
    print(json.dumps({"key": args.key, "outcome": outcome, "status": row["status"],
                      "resolved_by": row["resolved_by"]}, ensure_ascii=False))
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
    run.add_argument("--effect-replay", default="safe", choices=[r.value for r in Replay],
                     help="demo 副作用的可重放性：safe 按键 upsert；unsafe 追加（重跑就多一行）。"
                          "恢复时要传和第一次相同的值，否则意图不同，报幂等键冲突")
    run.add_argument("--effect-verify", action="store_true",
                     help="给不可重放的 demo 副作用配反查：恢复时按键查下游，查到已执行就不重跑；"
                          "查不到仍按结果未知处理")
    run.add_argument("--hold", default=None, choices=list(HOLD_POINTS),
                     help="验收用：停在写下游之前/之后等着被 SIGKILL（30 秒后自行退出）")
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
            sub_parser.add_argument("--tool", default="flaky_write",
                                    choices=["flaky_write", "flaky_append"])
            sub_parser.add_argument("--timeout", type=float, default=0.1)
        if name == "extract":
            sub_parser.add_argument("--fake-script", required=True)
            sub_parser.add_argument("--sink-dir", required=True)
            sub_parser.add_argument("--key", default="extract-1")
        sub_parser.set_defaults(func=func)

    agent = sub.add_parser("agent", help="跑一次模型循环（M8）：model → tool → model")
    agent.add_argument("--state-dir", required=True)
    agent.add_argument("--base-url", required=True, help="OpenAI 兼容的模型服务地址（验收用脚本化假模型）")
    agent.add_argument("--prompt", required=True,
                       help="用户的任务。恢复时要传同一个：第一次模型调用的请求由它构成")
    agent.add_argument("--run-id", default="run-1")
    agent.add_argument("--resume", action="store_true")
    agent.add_argument("--model", default="deepseek-v4-pro")
    agent.add_argument("--max-steps", type=int, default=8, help="模型步与工具步合计的上限")
    agent.add_argument("--tools", default=None, help="逗号分隔的工具名；默认全部")
    agent.add_argument("--deny-tool", action="append", default=[], help="如 tool:flaky_write*")
    agent.add_argument("--ask-tool", action="append", default=[], help="命中时需要审批")
    agent.add_argument("--approval", default="on-request", choices=[a.value for a in ApprovalPolicy])
    agent.add_argument("--tool-timeout", type=float, default=5.0)
    agent.add_argument("--model-timeout", type=float, default=30.0)
    agent.set_defaults(func=_cmd_agent)

    effect = sub.add_parser("effect", help="查看与解除副作用账本里的记录")
    effect_sub = effect.add_subparsers(dest="effect_command", required=True)
    effect_list = effect_sub.add_parser("list", help="列出账本记录")
    effect_list.add_argument("--state-dir", required=True)
    effect_list.add_argument("--status", default=None,
                             choices=["pending", "completed", "cancelled", "unknown"])
    effect_list.set_defaults(func=_cmd_effect_list)
    effect_resolve = effect_sub.add_parser(
        "resolve", help="把一条 unknown 解除成已执行（不再执行）或没执行（下次执行）")
    effect_resolve.add_argument("--state-dir", required=True)
    effect_resolve.add_argument("--key", required=True)
    effect_resolve.add_argument("--outcome", required=True, choices=["applied", "not-applied"])
    effect_resolve.add_argument("--by", default=getpass.getuser(), help="解除人，写进账本")
    effect_resolve.set_defaults(func=_cmd_effect_resolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
