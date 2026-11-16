"""M8 的回归断言（S11.1–S11.5）：模型循环。

假模型服务跑在**验收这个父进程**里，`harness agent` 子进程通过 `--base-url` 访问它。
脚本的消耗位置留在父进程内存里，所以子进程被杀、再起一个恢复，`model.calls` 就是
两个进程加起来一共问了模型几次——这正是「恢复不重问模型」要数的东西。

出处：S11.1–S11.3 是「以一个普通的 ReAct Agent 来说，你在之上解决了哪些长任务相关的问题」
与「调用发出后进程挂了，恢复时知不知道成功」；S11.4 是「拦住并回灌模型」；
S11.5 是「一直在执行，但看起来任务没有推进」。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ..fake_model.server import FakeModelServer, ScriptedModel
from .checks_m7 import (
    EXIT_OUTCOME_UNKNOWN,
    _KILLED,
    _count,
    _effects,
    _kill_when,
    _payload,
    _run,
    _written,
)
from .registry import CheckResult, failed, passed

PROMPT = "把 notes/a.md 写进下游，然后告诉我写好了"
EXIT_FAILED = 1


def _write_call(call_id: str = "call_w1", tool: str = "flaky_write", target: str = "notes/a.md") -> dict:
    return {"tool_calls": [{"id": call_id, "name": tool, "arguments": {"target": target}}]}


def _agent(state_dir: Path, server: FakeModelServer, *extra: str) -> list[str]:
    return ["agent", "--state-dir", str(state_dir), "--base-url", server.base_url,
            "--prompt", PROMPT, *extra]


def _last_message(model: ScriptedModel, request_index: int) -> dict:
    return model.requests[request_index]["messages"][-1]


def _body(message: dict) -> dict:
    try:
        return json.loads(message.get("content") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


# --- S11.1 --------------------------------------------------------------
def check_s111() -> CheckResult:
    model = ScriptedModel([_write_call(), {"content": "已写入 notes/a.md"}])
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"
        process = _run(_agent(state_dir, server))
        payload = _payload(process)
        if process.returncode != 0 or payload.get("status") != "completed":
            return failed(f"agent 没有完成：退出码 {process.returncode}，状态 {payload.get('status')}",
                          process.stderr[-400:])

        steps = payload.get("steps_run")
        if steps != ["model", "tool:flaky_write", "model"]:
            return failed(f"步骤序列不对：{steps}")
        if model.calls != 2:
            return failed(f"模型应被调用 2 次，实际 {model.calls} 次")
        fed_back = _last_message(model, 1)
        if fed_back.get("role") != "tool" or fed_back.get("tool_call_id") != "call_w1":
            return failed(f"第 2 次请求的最后一条消息不是对应的工具结果：{fed_back}")
        if _body(fed_back).get("is_error") is not False:
            return failed(f"工具结果应为成功：{fed_back}")
        sink_rows = _count(state_dir / "sink.sqlite", "effects_log")
        if sink_rows != 1:
            return failed(f"下游应有 1 行，实际 {sink_rows}")
        if payload.get("final") != "已写入 notes/a.md":
            return failed(f"最终回答不对：{payload.get('final')!r}")

        return passed(
            f"步骤序列 {steps}，退出码 {process.returncode}",
            f"模型被调用 {model.calls} 次；第 2 次请求带回工具结果 tool_call_id={fed_back['tool_call_id']}",
            f"工具结果 content={fed_back['content'][:80]}",
            f"下游 effects_log={sink_rows} 行；最终回答「{payload['final']}」",
        )


# --- S11.2 --------------------------------------------------------------
def check_s112() -> CheckResult:
    model = ScriptedModel([_write_call(), {"content": "已写入 notes/a.md"}])
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"
        killed = _kill_when(_agent(state_dir, server, "--hold-after-write", "flaky_write"),
                            _written(state_dir, "effects_log"))
        if killed != _KILLED:
            return failed(f"工具步执行中没有被 SIGKILL：返回码 {killed}（应为 {_KILLED}）")
        calls_at_kill = model.calls

        resumed = _run(_agent(state_dir, server, "--resume"))
        payload = _payload(resumed)
        if resumed.returncode != 0 or payload.get("status") != "completed":
            return failed(f"恢复失败：退出码 {resumed.returncode}，状态 {payload.get('status')}",
                          resumed.stderr[-400:])
        if calls_at_kill != 1 or model.calls != 2:
            return failed(f"模型调用次数：被杀时 {calls_at_kill}，最终 {model.calls}（应为 1 与 2）")
        sink_rows = _count(state_dir / "sink.sqlite", "effects_log")
        if sink_rows != 1:
            return failed(f"下游应有 1 行，实际 {sink_rows}")
        tool_rows = [row for row in _effects(state_dir) if ":flaky_write:" in row["key"]]

        return passed(
            f"工具写完下游后被真 SIGKILL：returncode={killed}，此时模型已被调用 {calls_at_kill} 次",
            f"恢复：退出码 {resumed.returncode}，重跑的步骤 {payload.get('replayed_steps')}，"
            f"本次执行 {payload.get('steps_run')}",
            f"模型总调用 {model.calls} 次：恢复后只问了最终回答，没有重问「要不要调工具」",
            f"工具副作用 executions={tool_rows[0]['executions'] if tool_rows else '?'}（可重放，重放 1 次），"
            f"下游 effects_log={sink_rows} 行",
        )


# --- S11.3 --------------------------------------------------------------
def check_s113() -> CheckResult:
    model = ScriptedModel([_write_call("call_a1", "flaky_append"), {"content": "已追加"}])
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"
        killed = _kill_when(_agent(state_dir, server, "--hold-after-write", "flaky_append"),
                            _written(state_dir, "append_log"))
        if killed != _KILLED:
            return failed(f"工具步执行中没有被 SIGKILL：返回码 {killed}（应为 {_KILLED}）")

        blocked = _run(_agent(state_dir, server, "--resume"))
        blocked_payload = _payload(blocked)
        calls_when_blocked = model.calls
        if blocked.returncode != EXIT_OUTCOME_UNKNOWN or blocked_payload.get("status") != "blocked":
            return failed(f"恢复应停在 blocked（退出码 {EXIT_OUTCOME_UNKNOWN}）："
                          f"实际 {blocked.returncode} / {blocked_payload.get('status')}", blocked.stderr[-400:])
        if calls_when_blocked != 1:
            return failed(f"blocked 时模型应只被调用过 1 次，实际 {calls_when_blocked}")

        key = blocked_payload.get("blocked_on")
        resolve = _run(["effect", "resolve", "--state-dir", str(state_dir), "--key", str(key),
                        "--outcome", "applied", "--by", "verify-s113"])
        if resolve.returncode != 0:
            return failed(f"effect resolve 失败：{resolve.returncode} {resolve.stdout[-200:]}")

        final = _run(_agent(state_dir, server, "--resume"))
        final_payload = _payload(final)
        appended = _count(state_dir / "sink.sqlite", "append_log")
        if final.returncode != 0 or final_payload.get("status") != "completed":
            return failed(f"解除后恢复应完成：{final.returncode} / {final_payload.get('status')}",
                          final.stderr[-400:])
        if model.calls != 2 or appended != 1:
            return failed(f"解除后：模型调用 {model.calls} 次（应 2），append_log {appended} 行（应 1）")
        fed_back = _body(_last_message(model, 1))

        return passed(
            f"不可重放工具写完下游后被真 SIGKILL：returncode={killed}",
            f"恢复：退出码 {blocked.returncode}，状态 blocked，卡在 {key}；模型仍只被调用 {calls_when_blocked} 次",
            f"effect resolve --outcome applied 之后恢复：退出码 {final.returncode}，模型共调用 {model.calls} 次",
            f"下游 append_log={appended} 行（没有重放）；回灌给模型的工具结果：{json.dumps(fed_back, ensure_ascii=False)[:90]}",
        )


# --- S11.4 --------------------------------------------------------------
def check_s114() -> CheckResult:
    model = ScriptedModel([_write_call(), {"content": "被拒绝了，不写了"}])
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"
        process = _run(_agent(state_dir, server, "--deny-tool", "tool:flaky_write*"))
        payload = _payload(process)
        if process.returncode != 0 or payload.get("status") != "completed":
            return failed(f"被拒后 run 应正常完成：退出码 {process.returncode}，状态 {payload.get('status')}",
                          process.stderr[-400:])
        sink_rows = _count(state_dir / "sink.sqlite", "effects_log")
        if sink_rows != 0:
            return failed(f"被拒的工具不应执行：下游有 {sink_rows} 行")
        if model.calls != 2:
            return failed(f"模型应被调用 2 次（拿到拒绝后再回答），实际 {model.calls}")
        fed_back = _last_message(model, 1)
        body = _body(fed_back)
        if fed_back.get("tool_call_id") != "call_w1" or body.get("is_error") is not True \
                or body.get("code") != "POLICY_DENIED":
            return failed(f"第 2 次请求没有带回拒绝结果：{fed_back}")
        ledger_rows = len(_effects(state_dir))

        return passed(
            f"--deny-tool 'tool:flaky_write*'：步骤 {payload.get('steps_run')}，退出码 {process.returncode}",
            f"回灌给模型：{fed_back['content'][:110]}",
            f"下游 effects_log={sink_rows} 行；账本里只有模型调用的记录（{ledger_rows} 条），没有工具副作用",
            "结论：拒绝发生在执行之前，并作为工具结果交还模型，而不是让 run 崩掉",
        )


# --- S11.5 --------------------------------------------------------------
def check_s115() -> CheckResult:
    endless = [{"tool_calls": [{"id": f"call_r{i}", "name": "note_read", "arguments": {"path": "notes/a.md"}}]}
               for i in range(6)]
    model = ScriptedModel(endless)
    with tempfile.TemporaryDirectory() as tmp, FakeModelServer(model) as server:
        state_dir = Path(tmp) / "state"
        process = _run(_agent(state_dir, server, "--max-steps", "4"))
        payload = _payload(process)
        if process.returncode != EXIT_FAILED or payload.get("status") != "failed":
            return failed(f"到上限应判失败（退出码 {EXIT_FAILED}）：实际 {process.returncode} / "
                          f"{payload.get('status')}", process.stderr[-400:])
        if "max_steps" not in (payload.get("error") or ""):
            return failed(f"错误信息里没有 max_steps：{payload.get('error')!r}")
        if model.calls != 2:
            return failed(f"4 步里模型应被调用 2 次，实际 {model.calls}")

        return passed(
            f"脚本每轮都要求调用工具；--max-steps 4：退出码 {process.returncode}，状态 {payload['status']}",
            f"执行的步骤 {payload.get('steps_run')}；模型被调用 {model.calls} 次（脚本里还剩 {len(endless) - model.calls} 轮没用）",
            f"错误信息：{payload['error']}",
        )
