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


def test_m1_specs_are_wired_to_real_checks() -> None:
    """M1 的四条必须绑定真断言，不能退回 not_implemented。"""
    for sid in ("S2.1", "S2.2", "S2.3", "S2.4"):
        assert specs.by_id(sid).check is not None
        assert specs.by_id(sid).check().__class__.__name__ == "CheckResult"


def test_unfinished_specs_stay_honest() -> None:
    """M2 之后的规格必须还是「未实现」——不允许提前标绿。"""
    finished = {
        "S2.1", "S2.2", "S2.3", "S2.4", "S1.1", "S1.2",
        "S5.1", "S5.2", "S5.3", "S4.1", "S4.2", "S4.3",
        "S3.1", "S3.2", "S3.3", "S6.2", "S9.2", "S9.3",
    }
    unfinished = {f"S{area}.{i}" for area in range(1, 11) for i in range(1, 5)} - finished
    for outcome in runner.run_all():
        if outcome.spec.id in unfinished:
            assert outcome.result.status is Status.NOT_IMPLEMENTED, outcome.spec.id


def test_passed_requires_evidence() -> None:
    with pytest.raises(ValueError):
        CheckResult(Status.PASSED, "看起来没问题")


def test_helpers_and_run_one() -> None:
    assert passed("x") .status is Status.PASSED
    assert failed("坏").status is Status.FAILED
    assert not_implemented("没写").status is Status.NOT_IMPLEMENTED

    done = runner.run_one("S2.1")
    assert done.spec.id == "S2.1"
    assert done.result.status is Status.PASSED
    assert done.result.evidence  # 通过必须有证据

    todo = runner.run_one("S6.1")  # M5 才做，当前必须仍是未实现
    assert todo.result.status is Status.NOT_IMPLEMENTED


def test_unknown_spec_raises() -> None:
    with pytest.raises(KeyError):
        specs.by_id("S99.9")
