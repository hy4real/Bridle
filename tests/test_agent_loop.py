"""模型循环的进程内测试（M8.4）。跨进程的真 SIGKILL 在 verify/checks_m8.py。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridle.agent.task import AgentConfig, build_agent_task
from bridle.fake_model.server import FakeModelServer, ScriptedModel
from bridle.model.gateway import ModelGateway
from bridle.policy import ApprovalPolicy, CommandRules, Decision, PolicyEngine, Rule
from bridle.runtime import AgentRuntime, Checkpointer, EffectLedger, RunStatus
from bridle.runtime.demo import EffectSink
from bridle.tools.builtin import flaky_write_tool, note_read
from bridle.tools.registry import ToolRegistry


def _config(server: FakeModelServer, sink: EffectSink, *, rules: list[Rule] | None = None,
            approval: ApprovalPolicy = ApprovalPolicy.ON_REQUEST, max_steps: int = 8) -> AgentConfig:
    registry = ToolRegistry()
    registry.register(note_read())
    registry.register(flaky_write_tool(sink, slow_first=0.0))
    policy_rules = CommandRules(list(rules or []) + [Rule("allow", Decision.ALLOW, "tool:*")])
    return AgentConfig(
        gateway=ModelGateway(server.base_url, retries=0, sleep=lambda _s: None,
                             first_byte_timeout=5.0, total_timeout=5.0),
        registry=registry, policy=PolicyEngine(rules=policy_rules, approval=approval),
        max_steps=max_steps)


def _call(name: str, arguments, call_id: str = "c1") -> dict:
    return {"tool_calls": [{"id": call_id, "name": name, "arguments": arguments}]}


def _fed_back(model: ScriptedModel, index: int = 1) -> dict:
    return json.loads(model.requests[index]["messages"][-1]["content"])


def _run(tmp_path: Path, steps: list[dict], **kwargs):
    model = ScriptedModel(steps)
    with FakeModelServer(model) as server:
        sink = EffectSink(tmp_path)
        report = AgentRuntime(build_agent_task("做事", _config(server, sink, **kwargs)), tmp_path).run()
    return report, model, sink


def test_multiple_tool_calls_in_one_reply_run_as_separate_steps_in_order(tmp_path: Path) -> None:
    reply = {"tool_calls": [
        {"id": "a", "name": "note_read", "arguments": {"path": "x"}},
        {"id": "b", "name": "flaky_write", "arguments": {"target": "y"}},
    ]}
    report, model, sink = _run(tmp_path, [reply, {"content": "好了"}])

    assert report.status is RunStatus.COMPLETED
    assert report.steps_run == ["model", "tool:note_read", "tool:flaky_write", "model"]
    roles = [(m["role"], m.get("tool_call_id")) for m in model.requests[1]["messages"]]
    assert roles[-3:] == [("assistant", None), ("tool", "a"), ("tool", "b")]
    assert len(sink.rows()) == 1


def test_unknown_tool_and_bad_arguments_are_fed_back_not_fatal(tmp_path: Path) -> None:
    report, model, _ = _run(tmp_path, [
        _call("no_such_tool", {}, "u1"),
        _call("note_read", "{oops", "u2"),
        _call("note_read", {}, "u3"),          # 缺必填参数 path
        {"content": "放弃"},
    ])

    assert report.status is RunStatus.COMPLETED
    assert _fed_back(model, 1)["code"] == "UNKNOWN_TOOL"
    assert _fed_back(model, 2)["code"] == "BAD_ARGUMENTS"
    assert _fed_back(model, 3)["code"] == "INVALID_ARGUMENTS"
    assert all(_fed_back(model, i)["is_error"] for i in (1, 2, 3))


def test_business_error_from_a_tool_is_fed_back_with_is_error(tmp_path: Path) -> None:
    report, model, _ = _run(tmp_path, [_call("note_read", {"path": "missing"}), {"content": "没有这条笔记"}])
    assert report.status is RunStatus.COMPLETED
    body = _fed_back(model)
    assert body["is_error"] is True and body["error"] == "note not found"


def test_prompt_rule_interrupts_the_run_without_executing(tmp_path: Path) -> None:
    report, model, sink = _run(tmp_path, [_call("flaky_write", {"target": "y"}), {"content": "x"}],
                               rules=[Rule("ask", Decision.PROMPT, "tool:flaky_write")])
    assert report.status is RunStatus.INTERRUPTED
    assert sink.rows() == [] and model.calls == 1
    latest = Checkpointer(tmp_path).latest("run-1")
    assert latest.kind == "interrupted" and latest.state["pending_tool_calls"]  # 恢复后还会再问一次审批


def test_prompt_rule_unattended_becomes_policy_denied(tmp_path: Path) -> None:
    report, model, sink = _run(tmp_path, [_call("flaky_write", {"target": "y"}), {"content": "x"}],
                               rules=[Rule("ask", Decision.PROMPT, "tool:flaky_write")],
                               approval=ApprovalPolicy.NEVER)
    assert report.status is RunStatus.COMPLETED and sink.rows() == []
    assert _fed_back(model)["code"] == "POLICY_DENIED"


@pytest.mark.parametrize("step, needle", [
    ({"status": 500}, "模型不可用"),
    ({"status": 429}, "模型不可用"),           # 不带 retry-after：配额耗尽
    ({"content": "写到一半", "finish_reason": "length"}, "finish_reason=length"),
])
def test_model_failures_end_the_run_as_failed(tmp_path: Path, step: dict, needle: str) -> None:
    report, model, _ = _run(tmp_path, [step])
    assert report.status is RunStatus.FAILED and needle in report.error
    assert Checkpointer(tmp_path).latest("run-1").kind == "failed"


def test_model_answer_is_not_asked_again_if_killed_before_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回答已经落进账本、checkpoint 还没写时进程死了：恢复拿回同一个回答，不重问模型。"""

    class _Died(BaseException):
        pass

    model = ScriptedModel([_call("note_read", {"path": "x"}), {"content": "第二轮"}, {"content": "不该被用到"}])
    with FakeModelServer(model) as server:
        config = _config(server, EffectSink(tmp_path))
        original_save = AgentRuntime._save

        def die_on_first_step_save(self, state, parent, *, kind="step"):
            if kind == "step" and state["step_index"] == 1:
                raise _Died()
            return original_save(self, state, parent, kind=kind)

        monkeypatch.setattr(AgentRuntime, "_save", die_on_first_step_save)
        with pytest.raises(_Died):
            AgentRuntime(build_agent_task("做事", config), tmp_path).run()
        monkeypatch.setattr(AgentRuntime, "_save", original_save)
        assert model.calls == 1
        assert Checkpointer(tmp_path).latest("run-1") is None  # 第一步的 checkpoint 没写成

        report = AgentRuntime(build_agent_task("做事", config), tmp_path).run(resume=False)

    assert report.status is RunStatus.COMPLETED
    assert model.calls == 2  # 第一轮的回答来自账本；只为第二轮问了一次
    model_rows = [r for r in EffectLedger(tmp_path).rows() if r["key"].endswith(":model")]
    assert [r["executions"] for r in model_rows] == [1, 1]


def test_max_steps_must_be_positive(tmp_path: Path) -> None:
    with FakeModelServer(ScriptedModel([])) as server:
        with pytest.raises(ValueError):
            build_agent_task("x", _config(server, EffectSink(tmp_path), max_steps=0))


# --- harness agent 的密钥参数 ---------------------------------------------
def test_cli_agent_reads_the_api_key_from_the_named_env_var_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from bridle import cli
    from bridle.model import gateway as gateway_module

    seen: list[str] = []
    original = gateway_module.ModelGateway.with_api_key

    def spy(self, api_key: str):
        seen.append(api_key)
        return original(self, api_key)

    monkeypatch.setattr(gateway_module.ModelGateway, "with_api_key", spy)
    monkeypatch.setenv("BRIDLE_TEST_KEY", "k-123456")
    with FakeModelServer(ScriptedModel([{"content": "好"}])) as server:
        code = cli.main(["agent", "--state-dir", str(tmp_path / "a"), "--base-url", server.base_url,
                         "--prompt", "x", "--api-key-env", "BRIDLE_TEST_KEY"])
    captured = capsys.readouterr()

    assert code == 0 and seen == ["k-123456"]
    assert "BRIDLE_TEST_KEY present=True len=8" in captured.err
    assert "k-123456" not in captured.out + captured.err  # 值不出现在任何输出里


def test_cli_agent_refuses_to_start_when_the_key_env_var_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from bridle import cli

    monkeypatch.delenv("BRIDLE_TEST_KEY", raising=False)
    code = cli.main(["agent", "--state-dir", str(tmp_path / "a"), "--base-url", "http://127.0.0.1:9/v1",
                     "--prompt", "x", "--api-key-env", "BRIDLE_TEST_KEY"])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["code"] == "MISSING_API_KEY"
    assert not (tmp_path / "a" / "checkpoints.sqlite").exists()  # 没有开始跑
