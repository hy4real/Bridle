"""工程回归集：证明「改代码没弄坏已有行为」。

注意边界（HANDOFF §8 红线）：这是**工程回归**维度，不是效果评测维度。
它回答「这次提交有没有破坏既有能力」，不回答「这个 agent 好不好用」。
两者混着说会被当场抓。
"""

from .suite import Case, CaseResult, FailureMode, run_suite

__all__ = ["Case", "CaseResult", "FailureMode", "run_suite"]
