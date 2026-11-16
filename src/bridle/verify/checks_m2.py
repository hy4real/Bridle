"""M2 的回归断言（S1.1、S1.2、S5.1–S5.3）。

三条走真实沙箱（macOS `sandbox-exec`），两条走策略与解析器。
注意每个 check 都自带临时目录，互不污染。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .registry import CheckResult, failed, passed

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "bridle.cli", *args],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src"), **(env or {})},
    )


def _json(process: subprocess.CompletedProcess[str]) -> dict:
    try:
        return json.loads(process.stdout or "{}")
    except json.JSONDecodeError:
        return {}


# --- S1.1 ---------------------------------------------------------------
def check_s11() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "state"
        write = _run(["exec", "--session", "A", "--state-dir", str(state),
                      "--cmd", "echo hi > only-a.txt; ls",
                      "--sandbox-mode", "workspace-write", "--allow-cmd", "*"])
        read = _run(["exec", "--session", "B", "--state-dir", str(state),
                     "--cmd", "ls", "--sandbox-mode", "workspace-write", "--allow-cmd", "*"])
        a, b = _json(write), _json(read)

        if write.returncode != 0 or read.returncode != 0:
            return failed("exec 失败", write.stderr[-300:], read.stderr[-300:])
        if "only-a.txt" not in a.get("stdout", ""):
            return failed(f"session A 没写出文件：{a.get('stdout')!r}")
        if "only-a.txt" in b.get("stdout", ""):
            return failed(f"session B 看到了 A 的文件：{b.get('stdout')!r}")
        if a.get("workspace") == b.get("workspace"):
            return failed(f"两个 session 共用同一工作区：{a.get('workspace')}")

        return passed(
            f"session A 工作区：{a['workspace']}",
            f"session B 工作区：{b['workspace']}",
            f"A 的 ls 含 only-a.txt，B 的 ls 为 {b.get('stdout')!r}",
            "结论：隔离单元是 session，工作区按 session 落盘",
        )


# --- S1.2 ---------------------------------------------------------------
def check_s12() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        secret_dir = Path(tmp) / "secret"
        secret_dir.mkdir()
        secret = secret_dir / "token.txt"
        secret.write_text("top-secret", encoding="utf-8")

        process = _run([
            "exec", "--session", "C", "--state-dir", str(Path(tmp) / "state"),
            "--cmd", f'python3 -c "open(\'{secret}\').read(); print(1)"',
            "--deny-read", str(secret_dir), "--allow-cmd", "*", "--sandbox-mode", "read-only",
        ])
        payload = _json(process)
        stderr = payload.get("stderr", "")

        if process.returncode != 0:
            return failed(f"exec 本身失败：{process.returncode}", process.stderr[-300:])
        if not payload.get("denied"):
            return failed("沙箱没有拦住：denied=false", stderr[-200:])
        if "No such file" in stderr or "ENOENT" in stderr:
            return failed("返回的是 ENOENT——说明拦截发生在文件不存在这一层，不是策略层", stderr[-200:])
        if "Operation not permitted" not in stderr and "Errno 1" not in stderr:
            return failed(f"没有观察到 EPERM：{stderr[-200:]}")

        return passed(
            f"命令退出码 {payload.get('exit_code')}，denied={payload.get('denied')}",
            "stderr 含 「Operation not permitted / Errno 1」（EPERM），不含 ENOENT",
            "结论：越权拦截发生在策略层（OS 沙箱），不是「文件恰好不存在」",
        )


# --- S5.1 ---------------------------------------------------------------
def check_s51() -> CheckResult:
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "state"
        blocked = _run(
            ["exec", "--session", "D", "--state-dir", str(state),
             "--cmd", "echo x", "--allow-cmd", "*"],
            env={"HARNESS_SANDBOX_DISABLE": "1"},
        )
        allowed = _run(
            ["exec", "--session", "D", "--state-dir", str(state),
             "--cmd", "echo x", "--allow-cmd", "*", "--allow-unsandboxed"],
            env={"HARNESS_SANDBOX_DISABLE": "1"},
        )

        payload = _json(blocked)
        if blocked.returncode != 13 or payload.get("code") != "SANDBOX_UNAVAILABLE":
            return failed(
                f"没有 fail-closed：rc={blocked.returncode} code={payload.get('code')}",
                blocked.stdout[-300:])
        if payload.get("executed") is not False:
            return failed("沙箱不可用时仍然执行了命令（裸跑）")

        if allowed.returncode != 0 or not _json(allowed).get("executed"):
            return failed("显式 --allow-unsandboxed 后仍被拒绝", allowed.stdout[-300:])
        if "未启用沙箱" not in allowed.stderr:
            return failed("放行时没有打印警告", allowed.stderr[-200:])

        return passed(
            "沙箱不可用 → 退出码 13 / code=SANDBOX_UNAVAILABLE / executed=false",
            "显式 --allow-unsandboxed → 执行成功，stderr 有「未启用沙箱」警告",
            "结论：fail-closed 是默认，放行必须显式且可见",
        )


# --- S5.2 ---------------------------------------------------------------
def check_s52() -> CheckResult:
    cases = [
        (["--cmd", "rm -rf /tmp/x", "--deny-cmd", "rm -rf *", "--allow-cmd", "*"], "forbid", "deny-1"),
        (["--cmd", "rm -rf /tmp/x", "--allow-cmd", "rm *", "--deny-cmd", "rm -rf *"], "forbid", "deny-1"),
        (["--cmd", "git push --force", "--ask-cmd", "git push *", "--approval", "never"], "forbid", "ask-1"),
        (["--cmd", "ls", "--allow-cmd", "ls"], "allow", "allow-1"),
    ]
    evidence: list[str] = []
    for args, expected, rule in cases:
        process = _run(["policy-check", *args])
        payload = _json(process)
        if payload.get("decision") != expected or payload.get("rule") != rule:
            return failed(
                f"判定不符：cmd={args[1]} 期望 {expected}/{rule}，实际 "
                f"{payload.get('decision')}/{payload.get('rule')} —— {payload.get('reason')}")
        evidence.append(f"{args[1]} → {payload['decision']}（层 {payload['layer']}，规则 {payload['rule']}）")

    evidence.append("allow 规则更具体（rm *）也不能突破 deny（rm -rf *）；approval=never 把 ask 降级为拒绝")
    evidence.append("结论：deny > ask > allow，且 never 不是「放行」")
    return passed(*evidence)


# --- S5.3 ---------------------------------------------------------------
def check_s53() -> CheckResult:
    scripts = {
        "refusal": [{"content": "", "refusal": "我拒绝"}],
        "truncated": [{"content": "{\"a\":1}", "finish_reason": "length"}],
        "filtered": [{"content": "{\"a\":1}", "finish_reason": "content_filter"}],
        "ok": [{"content": "{\"a\":1}"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        evidence: list[str] = []
        for name, steps in scripts.items():
            script_path = Path(tmp) / f"{name}.json"
            script_path.write_text(json.dumps(steps), encoding="utf-8")
            sink_dir = Path(tmp) / f"sink-{name}"
            process = _run(["extract", "--fake-script", str(script_path),
                            "--sink-dir", str(sink_dir), "--key", "k1"])
            payload = _json(process)

            if name == "ok":
                if process.returncode != 0 or payload.get("written_downstream") is not True:
                    return failed("正常终止没有写下游", process.stdout[-200:])
                evidence.append("ok → 退出码 0，写入下游")
                continue

            if process.returncode != 14:
                return failed(f"{name} 的退出码应为 14，实际 {process.returncode}", process.stdout[-200:])
            if payload.get("written_downstream") is not False:
                return failed(f"{name} 仍写了下游（把空对象当合法数据）", process.stdout[-200:])
            if (sink_dir / "sink.sqlite").exists():
                import sqlite3
                count = sqlite3.connect(sink_dir / "sink.sqlite").execute(
                    "SELECT COUNT(*) FROM effects_log").fetchone()[0]
                if count:
                    return failed(f"{name} 向下游写了 {count} 行")
            evidence.append(f"{name} → 退出码 14，termination={payload.get('termination')}，未写下游")

        evidence.append("结论：refusal / length / content_filter 三条旁路都进「非正常终止」分支")
        return passed(*evidence)
