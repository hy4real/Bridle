"""验收框架与规格表的自测。

这些测试保护的是**框架本身**，不是业务逻辑：
- 规格数必须与文档一致（48），否则文档与代码会漂移；
- 已交付里程碑（M1–M9、M11）的 46 条必须全部通过，新登记的规格先是「未实现」，
  各状态条数与 BASELINE 逐字一致，否则有人会偷偷把断言写成恒真；
- PASSED 必须带证据，否则「通过」就等于没跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bridle.verify import runner, specs
from bridle.verify.registry import CheckResult, Status, failed, not_implemented, passed


_DELIVERED_MILESTONES = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M11"}
_BASELINE = Path(__file__).resolve().parents[1] / "BASELINE"


def test_spec_count_matches_docs() -> None:
    assert len(specs.SPECS) == 48


def test_spec_ids_are_unique_and_sorted() -> None:
    ids = [spec.id for spec in specs.SPECS]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids, key=lambda s: (int(s[1:].split(".")[0]), int(s.split(".")[1])))


def test_every_spec_has_milestone_and_command() -> None:
    for spec in specs.SPECS:
        assert spec.milestone.startswith("M"), spec.id
        assert spec.command.startswith(("harness", "python")), spec.id
        assert spec.title, spec.id


def test_m1_specs_are_wired_to_real_checks() -> None:
    """M1 的四条必须绑定真断言，不能退回 not_implemented。"""
    for sid in ("S2.1", "S2.2", "S2.3", "S2.4"):
        assert specs.by_id(sid).check is not None
        assert specs.by_id(sid).check().__class__.__name__ == "CheckResult"


def test_delivered_specs_pass_and_baseline_matches() -> None:
    """已交付里程碑的规格必须全部通过且带证据；任何规格不许「失败」；条数与 BASELINE 一致。

    新规格先登记为「未实现」再逐条变绿（M0 的做法），所以这里要求的是
    「已交付的不许退、未交付的不许假装」。里程碑交付时把它加进 `_DELIVERED_MILESTONES`。
    """
    outcomes = runner.run_all()
    assert len(outcomes) == 48
    delivered = [o for o in outcomes if o.spec.milestone in _DELIVERED_MILESTONES]
    assert len(delivered) == 46
    for outcome in delivered:
        assert outcome.result.status is Status.PASSED, (
            outcome.spec.id, outcome.result.status, outcome.result.reason)
        assert outcome.result.evidence, outcome.spec.id
    for outcome in outcomes:
        assert outcome.result.status is not Status.FAILED, (
            outcome.spec.id, outcome.result.reason)
    baseline = _BASELINE.read_text(encoding="utf-8").strip()
    assert runner.summary_line(outcomes).endswith(baseline), (
        runner.summary_line(outcomes), baseline)


def test_helpers_and_run_one() -> None:
    assert passed("x") .status is Status.PASSED
    assert failed("坏").status is Status.FAILED
    assert not_implemented("没写").status is Status.NOT_IMPLEMENTED

    done = runner.run_one("S2.1")
    assert done.spec.id == "S2.1"
    assert done.result.status is Status.PASSED
    assert done.result.evidence  # 通过必须有证据

    # 这条断言守护「规格表没有被悄悄改空」
    assert len(specs.SPECS) == 48
    assert all(spec.check is not None for spec in specs.SPECS)


def test_unknown_spec_raises() -> None:
    with pytest.raises(KeyError):
        specs.by_id("S99.9")


def test_a_raising_check_is_reported_as_failed_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    from bridle.verify.registry import Spec

    def explode() -> CheckResult:
        raise TimeoutError("子进程没回来")

    broken = Spec(id="S0.0", area="A0", title="会抛异常的断言", command="harness verify S0.0",
                  milestone="M0", check=explode)
    monkeypatch.setattr(specs, "SPECS", (broken, specs.by_id("S5.2")))

    outcomes = runner.run_all()

    assert [o.result.status for o in outcomes] == [Status.FAILED, Status.PASSED]
    assert "TimeoutError" in outcomes[0].result.reason  # 另一条规格照常出结果
