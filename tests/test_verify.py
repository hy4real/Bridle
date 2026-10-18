"""验收框架与规格表的自测。

这些测试保护的是**框架本身**，不是业务逻辑：
- 规格数必须与文档一致（30），否则文档与代码会漂移；
- M0 基线必须全部「未实现」，否则有人会偷偷把断言写成恒真；
- PASSED 必须带证据，否则「通过」就等于没跑。
"""

from __future__ import annotations

import pytest

from agent_harness.verify import runner, specs
from agent_harness.verify.registry import CheckResult, Status, failed, not_implemented, passed


def test_spec_count_matches_docs() -> None:
    assert len(specs.SPECS) == 30


def test_spec_ids_are_unique_and_sorted() -> None:
    ids = [spec.id for spec in specs.SPECS]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids, key=lambda s: (int(s[1:].split(".")[0]), int(s.split(".")[1])))


def test_every_spec_has_milestone_and_command() -> None:
    for spec in specs.SPECS:
        assert spec.milestone.startswith("M"), spec.id
        assert spec.command.startswith(("harness", "python")), spec.id
        assert spec.title, spec.id


def test_baseline_is_all_not_implemented() -> None:
    outcomes = runner.run_all()
    assert {o.result.status for o in outcomes} == {Status.NOT_IMPLEMENTED}
    assert "0 通过 / 0 失败 / 30 未实现" in runner.summary_line(outcomes)


def test_passed_requires_evidence() -> None:
    with pytest.raises(ValueError):
        CheckResult(Status.PASSED, "看起来没问题")


def test_helpers_and_run_one() -> None:
    assert passed("x") .status is Status.PASSED
    assert failed("坏").status is Status.FAILED
    assert not_implemented("没写").status is Status.NOT_IMPLEMENTED

    outcome = runner.run_one("S2.1")
    assert outcome.spec.id == "S2.1"
    assert outcome.result.status is Status.NOT_IMPLEMENTED


def test_unknown_spec_raises() -> None:
    with pytest.raises(KeyError):
        specs.by_id("S99.9")
