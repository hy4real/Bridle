"""切点摘要与原文保留单元测试（M12.2）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bridle import cli
from bridle.agent.compaction import (
    build_model_messages,
    compute_cut_point,
    estimate_message_tokens,
    format_compaction_messages,
    message_seqs_from_events,
    recompute_cut_point_from_events,
)
from bridle.runtime import AgentRuntime, Checkpointer, Durability, EffectLedger, RunStatus
from bridle.runtime.events import Event, EventLog, project
from bridle.runtime.types import Step, StepContext, Task


def test_estimate_message_tokens_handles_none_content_and_tool_calls() -> None:
    bare = {"role": "assistant", "content": None}
    with_calls = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "flaky_write", "arguments": json.dumps({"target": "a" * 200})},
            }
        ],
    }
    assert estimate_message_tokens(bare) >= 1
    assert estimate_message_tokens(with_calls) > estimate_message_tokens(bare) + 30


def test_compute_cut_point_aligns_orphan_tool_to_assistant_boundary() -> None:
    messages = [
        {"role": "system", "content": "系统提示词"},
        {"role": "user", "content": "最早的问题：" + "甲" * 40},
        {"role": "assistant", "content": "好的"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "note_read", "arguments": '{"path":"a"}'}},
                {"id": "c2", "type": "function", "function": {"name": "note_read", "arguments": '{"path":"b"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true, "text": "' + "乙" * 20 + '"}'},
        {"role": "tool", "tool_call_id": "c2", "content": '{"ok": true, "text": "' + "丙" * 20 + '"}'},
        {"role": "assistant", "content": "读取完成"},
    ]
    # keep_recent_tokens 只够覆盖 assistant("读取完成") + tool(c2)，切点原本会落在 tool(c2) 上
    budget = estimate_message_tokens(messages[6]) + estimate_message_tokens(messages[5])
    cut = compute_cut_point(messages, keep_recent_tokens=budget)

    # 必须向前对齐到对应的 assistant(tool_calls)（下标 3），绝不能以 tool 消息开头
    assert cut == 3
    assert messages[cut]["role"] == "assistant"
    assert messages[cut].get("tool_calls")


def test_compute_cut_point_never_cuts_system_and_handles_edge_cases() -> None:
    assert compute_cut_point([], keep_recent_tokens=100) == 0
    assert compute_cut_point([{"role": "system", "content": "sys"}], keep_recent_tokens=10) == 1
    assert compute_cut_point(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        keep_recent_tokens=0,
    ) == 1

    # 单条超大消息在尾部时，仍保留它（至少留尾）而不切走 system
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "早期消息"},
        {"role": "user", "content": "超大" * 500},
    ]
    assert compute_cut_point(msgs, keep_recent_tokens=10) == 2

    with pytest.raises(ValueError):
        compute_cut_point(msgs, keep_recent_tokens=-1)


def test_build_model_messages_and_format_compaction_messages() -> None:
    msgs = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "第一句"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "note_read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        {"role": "user", "content": "最近问题"},
    ]
    built = build_model_messages(msgs, summary="早期摘要", kept_from_index=4)
    assert built[0] == msgs[0]
    assert built[1]["role"] == "system" and "早期摘要" in built[1]["content"]
    assert built[2:] == [msgs[4]]

    prompt = format_compaction_messages(msgs[1:4], prev_summary="前次摘要文本")
    assert len(prompt) == 2
    assert "前次摘要文本" in prompt[1]["content"]
    assert "tool_calls" in prompt[1]["content"]


def test_runtime_writes_step_committed_and_summary_together_and_recomputes_cut(
    tmp_path: Path, capsys
) -> None:
    """loop.py 同事务写 step_committed + summary，append-only 原文不减，可重算切点，log verify 为 0。"""
    keep_tokens = 35
    scripted = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "早期事实 ORION-1：" + "甲" * 40},
        {"role": "assistant", "content": "收到甲：" + "乙" * 40},
        {"role": "user", "content": "中期事实：" + "丙" * 40},
        {"role": "user", "content": "最近问题"},
    ]

    def append_initial(ctx: StepContext) -> None:
        ctx.state["messages"] = scripted[:3]

    def append_mid(ctx: StepContext) -> None:
        ctx.state["messages"] = [*ctx.state["messages"], scripted[3]]

    def append_recent(ctx: StepContext) -> None:
        ctx.state["messages"] = [*ctx.state["messages"], scripted[4]]

    def compact(ctx: StepContext) -> None:
        events = EventLog(ctx.ledger.db_path).read(ctx.run_id)
        msgs, seqs = message_seqs_from_events(events)
        assert msgs == ctx.state["messages"]
        cut = compute_cut_point(msgs, keep_recent_tokens=keep_tokens)
        ctx.emit_summary("折叠了 ORION-1", first_kept_seq=seqs[cut], kept_from_index=cut)

    task = Task(
        name="compact-test",
        steps=[
            Step("init", append_initial),
            Step("mid", append_mid),
            Step("recent", append_recent),
            Step("compact", compact),
        ],
    )
    report = AgentRuntime(task, tmp_path).run()
    assert report.status is RunStatus.COMPLETED

    events = EventLog(tmp_path / "ledger.sqlite").read("run-1")
    summary_events = [e for e in events if e.type == "summary"]
    assert len(summary_events) == 1
    sev = summary_events[0]
    assert events[sev.seq - 2].type == "step_committed"
    assert "summary" not in events[sev.seq - 2].payload["delta"].get("set", {})

    # 确认 messages 增量在初始化后全部走 extend（append-only）
    msg_commits = [
        e for e in events if e.type == "step_committed" and "messages" in e.payload["delta"].get("extend", {})
    ]
    assert len(msg_commits) == 2

    # 用 seq < summary_event.seq（跳过 "summary"）重算切点，必须与现场 first_kept_seq 一致
    re_cut, re_seq = recompute_cut_point_from_events(
        events, before_seq=sev.seq, keep_recent_tokens=keep_tokens
    )
    assert re_cut == sev.payload["kept_from_index"]
    assert re_seq == sev.payload["first_kept_seq"]

    # 投影与缓存一致，且原文一条没少
    latest = Checkpointer(tmp_path).latest("run-1")
    assert latest is not None and latest.state["messages"] == scripted
    assert latest.state["summary"] == "折叠了 ORION-1"
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 0
    verify_out = json.loads(capsys.readouterr().out)
    assert verify_out["projection_equals_cache"] is True


def test_recompute_cut_point_skips_earlier_summary_events() -> None:
    events = [
        Event("r", 1, "run_started", None, {"state": {"step_index": 0, "status": "running"}}, 0.0),
        Event("r", 2, "step_committed", 0, {"delta": {"set": {"messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "m1 " + "x" * 120},
            {"role": "assistant", "content": "m2 " + "y" * 120},
        ]}}}, 0.0),
        Event("r", 3, "step_committed", 1, {"delta": {"extend": {"messages": [
            {"role": "user", "content": "m3"},
        ]}}}, 0.0),
        Event("r", 4, "summary", 2, {"summary": "s1", "first_kept_seq": 3, "kept_from_index": 3}, 0.0),
        Event("r", 5, "step_committed", 3, {"delta": {"extend": {"messages": [
            {"role": "assistant", "content": "m4 " + "z" * 120},
            {"role": "user", "content": "m5"},
        ]}}}, 0.0),
        Event("r", 6, "summary", 4, {"summary": "s2", "first_kept_seq": 5, "kept_from_index": 5}, 0.0),
    ]
    cut, seq = recompute_cut_point_from_events(events, before_seq=6, keep_recent_tokens=20)
    assert (cut, seq) == (5, 5)
    assert project(events)[0]["summary"] == "s2"


def test_step_context_emit_summary_guards(tmp_path: Path) -> None:
    from bridle.runtime.types import CancelToken

    ctx = StepContext(state={}, ledger=None, cancel=CancelToken(), run_id="r", step_index=0)
    with pytest.raises(ValueError):
        ctx.emit_summary("s", first_kept_seq=0, kept_from_index=1)
    with pytest.raises(ValueError):
        ctx.emit_summary("s", first_kept_seq=1, kept_from_index=-1)
    ctx.emit_summary("s", first_kept_seq=1, kept_from_index=1)
    with pytest.raises(ValueError):
        ctx.emit_summary("s2", first_kept_seq=1, kept_from_index=1)


def test_exit_durability_flushes_summary_and_passes_log_verify(tmp_path: Path) -> None:
    def init_step(ctx: StepContext) -> None:
        ctx.state["messages"] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "user", "content": "new"},
        ]

    def compact_step(ctx: StepContext) -> None:
        ctx.emit_summary("exit-summary", first_kept_seq=2, kept_from_index=2)

    task = Task("exit-sum", steps=[Step("init", init_step), Step("compact", compact_step)])
    report = AgentRuntime(task, tmp_path, durability=Durability.EXIT).run()
    assert report.status is RunStatus.COMPLETED
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 0


def test_resume_repairs_stale_cache_preserving_summary_fields(tmp_path: Path) -> None:
    def step0(ctx: StepContext) -> None:
        ctx.state["messages"] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "user", "content": "recent"},
        ]

    def step1(ctx: StepContext) -> None:
        ctx.emit_summary("摘要内容", first_kept_seq=2, kept_from_index=2)

    task = Task("repair-sum", steps=[Step("s0", step0), Step("s1", step1)])
    assert AgentRuntime(task, tmp_path).run().status is RunStatus.COMPLETED

    # 模拟写完事件后、缓存未更新/被改坏
    import sqlite3

    with sqlite3.connect(tmp_path / "checkpoints.sqlite") as conn:
        conn.execute(
            """UPDATE checkpoints
               SET state_json = json_remove(state_json, '$.summary', '$.first_kept_seq', '$.kept_from_index')"""
        )
        conn.commit()
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 1

    resumed = AgentRuntime(task, tmp_path).run(resume=True)
    assert resumed.status is RunStatus.COMPLETED
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 0
    latest = Checkpointer(tmp_path).latest("run-1")
    assert latest is not None
    assert latest.state["summary"] == "摘要内容"
    assert latest.state["first_kept_seq"] == 2
    assert latest.state["kept_from_index"] == 2


# --- M12.3：agent 循环与模型摘要边界 --------------------------------------
def _agent_config(server, sink, **kwargs):
    from bridle.agent.task import AgentConfig
    from bridle.model.gateway import ModelGateway
    from bridle.policy import ApprovalPolicy, CommandRules, Decision, PolicyEngine, Rule
    from bridle.tools.builtin import note_read
    from bridle.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(note_read())
    rules = CommandRules([Rule("allow", Decision.ALLOW, "tool:*")])
    return AgentConfig(
        gateway=ModelGateway(
            server.base_url,
            retries=0,
            sleep=lambda _s: None,
            first_byte_timeout=5.0,
            total_timeout=5.0,
        ),
        registry=registry,
        policy=PolicyEngine(rules=rules, approval=ApprovalPolicy.ON_REQUEST),
        **kwargs,
    )


def test_compact_step_does_not_reask_summary_model_after_crash_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """摘要模型回答已落账本、step_committed + summary 还没落盘时崩溃：恢复不重问摘要模型。"""
    from bridle.agent.task import build_agent_task
    from bridle.fake_model.server import FakeModelServer, ScriptedModel
    from bridle.runtime.demo import EffectSink

    class _Died(BaseException):
        pass

    model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "note_read", "arguments": {"path": "p1-" + "甲" * 60}}]},
        {"tool_calls": [{"id": "c2", "name": "note_read", "arguments": {"path": "p2-" + "乙" * 60}}]},
        {"content": "崩溃前摘要"},       # compact_step 调用（只能被调用 1 次）
        {"content": "恢复后完成"},
    ])
    with FakeModelServer(model) as server:
        config = _agent_config(
            server, EffectSink(tmp_path), max_steps=12, context_budget_tokens=300, keep_recent_tokens=150
        )
        original_append_many = EventLog.append_many

        def die_before_summary_commit(self, run_id, items):
            if any(type_ == "summary" for type_, _, _ in items):
                raise _Died()
            return original_append_many(self, run_id, items)

        monkeypatch.setattr(EventLog, "append_many", die_before_summary_commit)
        with pytest.raises(_Died):
            AgentRuntime(build_agent_task("测试崩溃恢复", config), tmp_path).run()
        monkeypatch.setattr(EventLog, "append_many", original_append_many)
        assert model.calls == 3  # 2 轮模型 + 1 次摘要；已落账本，事件还没提交

        report = AgentRuntime(build_agent_task("测试崩溃恢复", config), tmp_path).run(resume=True)

    assert report.status is RunStatus.COMPLETED
    assert report.replayed_steps == ["compact"]
    assert model.calls == 4  # 恢复时重跑 compact 但不重问摘要模型，只问最终回答 1 次
    compact_rows = [r for r in EffectLedger(tmp_path).rows() if r["key"].endswith(":compact")]
    assert len(compact_rows) == 1
    assert compact_rows[0]["executions"] == 1
    assert compact_rows[0]["attempts"] == 2  # 第 2 次走账本去重
    assert cli.main(["log", "verify", "--state-dir", str(tmp_path), "--run-id", "run-1"]) == 0


def test_single_huge_message_does_not_trigger_infinite_compaction_loop(tmp_path: Path) -> None:
    """尾部单条消息就超过 budget 时，new_cut == kept_from_index 不得循环压缩。"""
    from bridle.agent.task import build_agent_task
    from bridle.fake_model.server import FakeModelServer, ScriptedModel
    from bridle.runtime.demo import EffectSink

    # 第 1 轮只有 [sys, user(超大)], budget=50 仍无法继续切（new_cut=1 == kept_from_index=1），直接走 model
    huge_prompt = "超大提示词：" + "甲" * 200
    model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "note_read", "arguments": {"path": "大路径-" + "乙" * 200}}]},
        {"content": "折叠了 user"},  # compact 只折叠一次（把 huge_prompt 折掉，留 [asst, tool]）
        {"content": "最终完成"},      # 即使 [asst, tool] 仍 > budget，new_cut == prev_cut，不再循环压缩
    ])
    with FakeModelServer(model) as server:
        config = _agent_config(
            server, EffectSink(tmp_path), max_steps=8, context_budget_tokens=100, keep_recent_tokens=50
        )
        report = AgentRuntime(build_agent_task(huge_prompt, config), tmp_path).run()

    assert report.status is RunStatus.COMPLETED
    assert report.steps_run == ["model", "tool:note_read", "compact", "model"]
    assert model.calls == 3


@pytest.mark.parametrize(
    "step, expected_needle",
    [
        ({"status": 500}, "摘要模型不可用"),
        ({"content": "半截", "finish_reason": "length"}, "finish_reason=length"),
        ({"content": "   "}, "摘要模型返回空摘要"),
    ],
)
def test_compact_step_failures_fail_the_run(tmp_path: Path, step: dict, expected_needle: str) -> None:
    from bridle.agent.task import build_agent_task
    from bridle.fake_model.server import FakeModelServer, ScriptedModel
    from bridle.runtime.demo import EffectSink

    model = ScriptedModel([
        {"tool_calls": [{"id": "c1", "name": "note_read", "arguments": {"path": "p1-" + "甲" * 60}}]},
        {"tool_calls": [{"id": "c2", "name": "note_read", "arguments": {"path": "p2-" + "乙" * 60}}]},
        step,
    ])
    with FakeModelServer(model) as server:
        config = _agent_config(
            server, EffectSink(tmp_path), max_steps=10, context_budget_tokens=300, keep_recent_tokens=150
        )
        report = AgentRuntime(build_agent_task("测试摘要失败", config), tmp_path).run()

    assert report.status is RunStatus.FAILED
    assert report.error is not None and expected_needle in report.error


def test_agent_config_and_cli_reject_invalid_budget_settings(tmp_path: Path, capsys) -> None:
    from bridle.agent.task import build_agent_task
    from bridle.fake_model.server import FakeModelServer, ScriptedModel
    from bridle.runtime.demo import EffectSink

    with FakeModelServer(ScriptedModel([])) as server:
        with pytest.raises(ValueError):
            build_agent_task(
                "x",
                _agent_config(server, EffectSink(tmp_path), context_budget_tokens=0),
            )
        with pytest.raises(ValueError):
            build_agent_task(
                "x",
                _agent_config(server, EffectSink(tmp_path), keep_recent_tokens=-1),
            )

    code = cli.main([
        "agent", "--state-dir", str(tmp_path / "bad"), "--base-url", "http://127.0.0.1:9/v1",
        "--prompt", "x", "--context-budget", "0",
    ])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["code"] == "BAD_CONFIG"

