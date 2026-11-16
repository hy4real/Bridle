"""验收执行与输出。

输出必须能被**另一个人在旁边看懂**：每行有 ID、里程碑、状态、原因/证据。
不做花哨表格，因为这段输出将来要贴进 README 和面试稿。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .registry import CheckResult, Spec, Status
from . import specs as spec_module


@dataclass(frozen=True)
class Outcome:
    spec: Spec
    result: CheckResult


def run_all() -> list[Outcome]:
    return [Outcome(spec, spec_module.run(spec)) for spec in spec_module.SPECS]


def run_one(sid: str) -> Outcome:
    spec = spec_module.by_id(sid)
    return Outcome(spec, spec_module.run(spec))


def summary_line(outcomes: list[Outcome]) -> str:
    counts = Counter(o.result.status for o in outcomes)
    total = len(outcomes)
    return (
        f"{total} 条规格："
        f"{counts[Status.PASSED]} 通过 / "
        f"{counts[Status.FAILED]} 失败 / "
        f"{counts[Status.NOT_IMPLEMENTED]} 未实现"
    )


def format_list() -> str:
    lines = ["ID     里程碑  维度  要证明的能力", "-" * 78]
    for spec in spec_module.SPECS:
        lines.append(f"{spec.id:<6} {spec.milestone:<7} {spec.area:<5} {spec.title}")
    lines.append("")
    lines.append(summary_line([Outcome(s, spec_module.run(s)) for s in spec_module.SPECS]))
    return "\n".join(lines)


def format_outcomes(outcomes: list[Outcome], *, show_evidence: bool = True) -> str:
    lines: list[str] = []
    for outcome in outcomes:
        spec, result = outcome.spec, outcome.result
        mark = {
            Status.PASSED: "PASS",
            Status.FAILED: "FAIL",
            Status.NOT_IMPLEMENTED: "TODO",
        }[result.status]
        lines.append(f"[{mark}] {spec.id} ({spec.milestone}) {spec.title}")
        lines.append(f"        命令：{spec.command}")
        lines.append(f"        状态：{result.status.value} —— {result.reason}")
        if show_evidence:
            for item in result.evidence:
                lines.append(f"        证据：{item}")
    lines.append("")
    lines.append(summary_line(outcomes))
    return "\n".join(lines)
