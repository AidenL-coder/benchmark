"""Per-task frontier records.

One `FrontierRecord` is the complete measurement of one (model, task) pair at
budget `N_max`: the estimate, its confidence bounds, the observed solution
diversity, the unseen-mass estimates, and -- as importantly -- the conditions
under which it was produced.

The proposal (section 8) names the frontier estimator's lower-bound character as
"the single most important limitation and must be stated plainly". That is
enforced structurally here rather than left to the write-up: a record cannot
express absolute unreachability. `beyond_frontier` is always qualified by
`n_samples`, and `p_upper_bound` is always finite and positive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from cbs.budget import Usage
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

__all__ = ["FrontierRecord", "build_record", "DEFAULT_BUDGET_GRID"]

#: Budgets at which pass@k and rarefaction are reported. Log-spaced because the
#: interesting behaviour (does the solution set saturate?) is multiplicative.
DEFAULT_BUDGET_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)


@dataclass
class FrontierRecord:
    """The frontier measurement for one (model, task) at a given budget."""

    task_id: str
    model_id: str
    split: str | None
    family: str

    n_samples: int
    n_correct: int
    p_hat: float
    p_ci_low: float
    p_ci_high: float
    #: Upper bound on `p(x)` -- the Clopper-Pearson upper limit. Always > 0.
    p_upper_bound: float
    #: The familiar `3/n` form, reported alongside the exact bound.
    rule_of_three_bound: float
    exact_zero_success_bound: float

    #: Zero successes within `n_samples`. Read as "not reached at compute
    #: n_samples", never as "unreachable".
    beyond_frontier: bool

    # -- solution diversity ----------------------------------------------
    n_distinct_solutions: int
    species_counts: dict[str, int]
    good_turing_unseen_mass: float | None
    sample_coverage: float | None
    chao1: dict | None

    # -- budget curves ----------------------------------------------------
    pass_at_k: dict[int, float]
    rarefaction: dict[int, float]

    # -- provenance -------------------------------------------------------
    temperature_schedule: list[dict]
    scaffold_fingerprint: dict
    usage: Usage
    verifier_backend: str
    verifier_is_security_boundary: bool
    n_verifier_errors: int = 0
    n_budget_exhausted: int = 0
    canonicalization_methods: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def solution_set_saturated(self) -> bool:
        """Whether the distinct-solution count appears to have stopped growing.

        Uses Good-Turing coverage: a coverage near 1 means new samples rarely
        produce novel solutions. An unsaturated task is one whose frontier
        estimate is visibly budget-limited, which is the caveat that must travel
        with any crossing claim about it.
        """
        coverage = self.sample_coverage
        return coverage is not None and coverage >= 0.99

    def summary_line(self) -> str:
        if self.beyond_frontier:
            return (
                f"{self.task_id:28s} 0/{self.n_samples} correct  "
                f"p < {self.p_upper_bound:.5f} (95%)  "
                f"BEYOND-FRONTIER@N={self.n_samples}"
            )
        coverage = (
            "n/a" if self.sample_coverage is None else f"{self.sample_coverage:.3f}"
        )
        return (
            f"{self.task_id:28s} {self.n_correct}/{self.n_samples} correct  "
            f"p_hat={self.p_hat:.4f} [{self.p_ci_low:.4f},{self.p_ci_high:.4f}]  "
            f"distinct={self.n_distinct_solutions}  coverage={coverage}"
        )

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "split": self.split,
            "family": self.family,
            "n_samples": self.n_samples,
            "n_correct": self.n_correct,
            "p_hat": self.p_hat,
            "p_ci_low": self.p_ci_low,
            "p_ci_high": self.p_ci_high,
            "p_upper_bound": self.p_upper_bound,
            "rule_of_three_bound": self.rule_of_three_bound,
            "exact_zero_success_bound": self.exact_zero_success_bound,
            "beyond_frontier": self.beyond_frontier,
            "beyond_frontier_qualifier": (
                f"not reached at compute N={self.n_samples}; "
                f"p(x) < {self.p_upper_bound:.6g} at 95% confidence. "
                "This is a budget-relative statement, not unreachability."
            ),
            "n_distinct_solutions": self.n_distinct_solutions,
            "species_counts": self.species_counts,
            "good_turing_unseen_mass": self.good_turing_unseen_mass,
            "sample_coverage": self.sample_coverage,
            "solution_set_saturated": self.solution_set_saturated,
            "chao1": self.chao1,
            "pass_at_k": {str(k): v for k, v in self.pass_at_k.items()},
            "rarefaction": {str(k): v for k, v in self.rarefaction.items()},
            "temperature_schedule": self.temperature_schedule,
            "scaffold_fingerprint": self.scaffold_fingerprint,
            "usage": self.usage.as_dict(),
            "verifier_backend": self.verifier_backend,
            "verifier_is_security_boundary": self.verifier_is_security_boundary,
            "n_verifier_errors": self.n_verifier_errors,
            "n_budget_exhausted": self.n_budget_exhausted,
            "canonicalization_methods": self.canonicalization_methods,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


def build_record(
    *,
    task_id: str,
    model_id: str,
    split: str | None,
    family: str,
    n_samples: int,
    species_counts: dict[str, int],
    temperature_schedule: list[dict],
    scaffold_fingerprint: dict,
    usage: Usage,
    verifier_backend: str,
    verifier_is_security_boundary: bool,
    n_verifier_errors: int = 0,
    n_budget_exhausted: int = 0,
    canonicalization_methods: dict[str, int] | None = None,
    confidence: float = 0.95,
    budget_grid: tuple[int, ...] = DEFAULT_BUDGET_GRID,
    metadata: dict | None = None,
) -> FrontierRecord:
    """Assemble a record from raw counts, computing every derived statistic."""
    n_correct = sum(species_counts.values())
    if n_correct > n_samples:
        raise ValueError(
            f"{task_id}: {n_correct} correct exceeds {n_samples} samples -- "
            "species counts and sample count disagree"
        )

    ci: ConfidenceInterval = clopper_pearson(n_correct, n_samples, confidence)
    chao: Chao1Estimate | None = chao1(species_counts, confidence=confidence)

    grid = [k for k in budget_grid if 0 < k <= n_samples]
    if n_samples and n_samples not in grid:
        grid.append(n_samples)

    return FrontierRecord(
        task_id=task_id,
        model_id=model_id,
        split=split,
        family=family,
        n_samples=n_samples,
        n_correct=n_correct,
        p_hat=ci.point,
        p_ci_low=ci.low,
        p_ci_high=ci.high,
        p_upper_bound=ci.high,
        rule_of_three_bound=rule_of_three(n_samples),
        exact_zero_success_bound=zero_success_upper_bound(n_samples, confidence),
        beyond_frontier=(n_correct == 0 and n_samples > 0),
        n_distinct_solutions=len(species_counts),
        species_counts=dict(species_counts),
        good_turing_unseen_mass=good_turing_unseen_mass(species_counts),
        sample_coverage=sample_coverage(species_counts),
        chao1=chao.as_dict() if chao else None,
        pass_at_k=success_probability_curve(n_samples, n_correct, grid)
        if n_samples
        else {},
        rarefaction=solution_rarefaction(
            species_counts, [k for k in grid if k <= max(1, n_correct)]
        ),
        temperature_schedule=temperature_schedule,
        scaffold_fingerprint=scaffold_fingerprint,
        usage=usage,
        verifier_backend=verifier_backend,
        verifier_is_security_boundary=verifier_is_security_boundary,
        n_verifier_errors=n_verifier_errors,
        n_budget_exhausted=n_budget_exhausted,
        canonicalization_methods=canonicalization_methods or {},
        metadata=metadata or {},
    )
