"""Reachable-solution frontier estimation (brief section 3.2)."""

from __future__ import annotations

from cbs.frontier.estimators import (
    Chao1Estimate,
    ConfidenceInterval,
    chao1,
    clopper_pearson,
    good_turing_unseen_mass,
    pass_at_k,
    rule_of_three,
    sample_coverage,
    solution_rarefaction,
    success_probability_curve,
    zero_success_upper_bound,
)
from cbs.frontier.records import DEFAULT_BUDGET_GRID, FrontierRecord, build_record
from cbs.frontier.sampler import (
    DEFAULT_SCHEDULE,
    FrontierSampler,
    TemperatureSchedule,
)
from cbs.frontier.validate import (
    CoverageValidation,
    PipelineValidation,
    validate_ci_coverage,
    validate_pipeline,
)

__all__ = [
    "Chao1Estimate",
    "ConfidenceInterval",
    "CoverageValidation",
    "DEFAULT_BUDGET_GRID",
    "DEFAULT_SCHEDULE",
    "FrontierRecord",
    "FrontierSampler",
    "PipelineValidation",
    "TemperatureSchedule",
    "build_record",
    "chao1",
    "clopper_pearson",
    "good_turing_unseen_mass",
    "pass_at_k",
    "rule_of_three",
    "sample_coverage",
    "solution_rarefaction",
    "success_probability_curve",
    "validate_ci_coverage",
    "validate_pipeline",
    "zero_success_upper_bound",
]
