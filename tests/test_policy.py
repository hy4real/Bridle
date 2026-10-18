"""策略层与沙箱后端的单元测试。

沙箱相关测试会真的调 `/usr/bin/sandbox-exec`（macOS）：
如果这块不测，"沙箱生效"就只是配置看起来对而已。
非 macOS 环境自动跳过。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agent_harness.policy import (
    ApprovalPolicy,
    CommandRules,
    Decision,
    PolicyEngine,
    Rule,
    SandboxMode,
)
from agent_harness.sandbox import SandboxUnavailable, build_backend
from agent_harness.workspace import SessionWorkspace

MACOS = sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()


def _engine(**kwargs) -> PolicyEngine:
    rules = CommandRules([
        Rule("deny-1", Decision.FORBID, "rm -rf *"),
        Rule("ask-1", Decision.PROMPT, "git push *"),
        Rule("allow-1", Decision.ALLOW, "ls"),
    ])
    return PolicyEngine(rules=rules, **kwargs)


def test_deny_beats_allow_regardless_of_specificity() -> None:
    verdict = _engine().decide("rm -rf /tmp/x")
    assert verdict.decision is Decision.FORBID
    assert verdict.rule_id == "deny-1"


def test_approval_never_downgrades_prompt_to_forbid() -> None:
    verdict = _engine(approval=ApprovalPolicy.NEVER).decide("git push origin main")
    assert verdict.decision is Decision.FORBID
    assert verdict.layer == "approval"
    assert "never" in verdict.reason


def test_unknown_command_defaults_to_prompt_not_allow() -> None:
    verdict = _engine().decide("curl http://example.com")
    assert verdict.decision is Decision.PROMPT
    assert verdict.layer == "default"


def test_whitespace_cannot_bypass_deny() -> None:
    assert _engine().decide("rm    -rf   /tmp/x").decision is Decision.FORBID


def test_workspace_is_per_session(tmp_path: Path) -> None:
    a = SessionWorkspace(tmp_path, "A").ensure()
    b = SessionWorkspace(tmp_path, "B").ensure()
    assert a.workspace != b.workspace
    assert a.workspace.is_dir() and b.workspace.is_dir()


@pytest.mark.skipif(not MACOS, reason="需要 macOS sandbox-exec")
def test_seatbelt_denies_read_with_eperm(tmp_path: Path) -> None:
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    (secret_dir / "s.txt").write_text("x", encoding="utf-8")

    backend = build_backend(writable_root=None)
    result = backend.execute(
        f'python3 -c "open(\'{secret_dir}/s.txt\').read()"',
        cwd=tmp_path,
        deny_read=(str(secret_dir),),
    )
    assert result.denied is True
    assert "Operation not permitted" in result.stderr or "Errno 1" in result.stderr


@pytest.mark.skipif(not MACOS, reason="需要 macOS sandbox-exec")
def test_seatbelt_realpath_is_required_to_match(tmp_path: Path) -> None:
    """/var 与 /private/var 是同一路径的两种写法：不做 realpath 就拦不住。"""
    secret = tmp_path / "s.txt"
    secret.write_text("x", encoding="utf-8")
    backend = build_backend(writable_root=None)
    result = backend.execute(f'cat "{secret}"', cwd=tmp_path, deny_read=(str(tmp_path),))
    assert result.denied is True


def test_read_only_backend_has_no_write_capability(tmp_path: Path) -> None:
    backend = build_backend(writable_root=None)
    assert backend.capabilities().fs_write is False


def test_sandbox_disable_env_raises_unless_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_SANDBOX_DISABLE", "1")
    with pytest.raises(SandboxUnavailable):
        build_backend(writable_root=None)
    backend = build_backend(writable_root=None, allow_unsandboxed=True)
    assert backend.name == "unsafe-local"


def test_sandbox_mode_default_is_read_only() -> None:
    assert SandboxMode("read-only") is SandboxMode.READ_ONLY
