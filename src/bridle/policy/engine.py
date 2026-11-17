"""三层策略引擎。"""

from __future__ import annotations

import hashlib
import json

from .rules import CommandRules
from .types import ApprovalPolicy, Decision, PolicyVerdict, SandboxMode


class PolicyEngine:
    def __init__(
        self,
        *,
        rules: CommandRules | None = None,
        approval: ApprovalPolicy = ApprovalPolicy.ON_REQUEST,
        sandbox_mode: SandboxMode = SandboxMode.READ_ONLY,
    ) -> None:
        self.rules = rules or CommandRules()
        self.approval = approval
        self.sandbox_mode = sandbox_mode

    def version(self) -> str:
        """策略快照的规范 SHA-256：(rules[id,decision,pattern], approval, sandbox_mode)。"""
        payload = {
            "rules": [
                {"id": rule.id, "decision": rule.decision.value, "pattern": rule.pattern}
                for rule in self.rules.rules
            ],
            "approval": self.approval.value,
            "sandbox_mode": self.sandbox_mode.value,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def decide(self, command: str) -> PolicyVerdict:
        deny = self.rules.first_match(command, Decision.FORBID)
        if deny is not None:
            return PolicyVerdict(Decision.FORBID, "rules", deny.id,
                                 f"命中 deny 规则 {deny.id}：{deny.pattern}")

        ask = self.rules.first_match(command, Decision.PROMPT)
        if ask is not None:
            if self.approval is ApprovalPolicy.NEVER:
                # 无人值守：需要人点头的请求直接拒绝，绝不静默放行
                return PolicyVerdict(Decision.FORBID, "approval", ask.id,
                                     f"规则 {ask.id} 需要审批，但 approval=never，降级为拒绝")
            return PolicyVerdict(Decision.PROMPT, "rules", ask.id,
                                 f"命中 ask 规则 {ask.id}：{ask.pattern}")

        allow = self.rules.first_match(command, Decision.ALLOW)
        if allow is not None:
            return PolicyVerdict(Decision.ALLOW, "rules", allow.id,
                                 f"命中 allow 规则 {allow.id}：{allow.pattern}")

        # 没有规则命中：按审批策略决定默认
        if self.approval is ApprovalPolicy.NEVER:
            return PolicyVerdict(Decision.FORBID, "default", None,
                                 "没有匹配规则，approval=never，默认拒绝")
        return PolicyVerdict(Decision.PROMPT, "default", None,
                             "没有匹配规则，按默认要求审批")
