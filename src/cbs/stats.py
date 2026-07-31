"""Aggregate statistics for analysis (brief section 6, "Statistical plan";
section 9.4). Two pieces the pre-registered plan requires and nothing in
Phases 0-4 needed yet:

*   **bootstrap CIs for aggregate metrics** -- a per-task point estimate (a
    crossing rate, an elicitation gain) aggregated across many tasks needs its
    own confidence interval, distinct from the per-task Clopper-Pearson
    interval `cbs.frontier.estimators` already provides.
*   **multiple-comparison correction** -- brief section 6: "Correct for
    multiple comparisons across tasks when aggregating crossing claims."
    Testing each of, say, 100 held-out tasks individually at nominal 95%
    confidence means roughly 5 false "crossings" are expected by chance alone
    even under the null; Benjamini-Hochberg controls the expected proportion
    of those among the claimed crossings, not the per-test rate.

Implemented from scratch for the same reason as the estimators in
`cbs.frontier.estimators` (D-07): no scientific-stack dependency, and every
function is checked against a textbook reference calculation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from cbs.frontier.estimators import ConfidenceInterval

__all__ = ["bootstrap_ci", "benjamini_hochberg", "BHResult"]


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / len(xs),
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap CI for an arbitrary statistic of `values`.

    The point estimate is `statistic` applied to the *original* data, not the
    mean of the bootstrap replicates (which is a common and avoidable source
    of small extra bias). The interval is the empirical
    `[alpha/2, 1 - alpha/2]` percentile range of the resampled statistic.

    Percentile bootstrap rather than a parametric interval because the
    aggregate metrics this backs (a crossing rate across a heterogeneous set
    of tasks, an elicitation gain) have no assumed distributional form the way
    a single binomial proportion does -- that is exactly the case
    Clopper-Pearson already covers, in `cbs.frontier.estimators`.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    n = len(values)
    point = statistic(values)
    rng = random.Random(seed)
    replicates = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        replicates.append(statistic(resample))
    replicates.sort()

    alpha = 1.0 - confidence
    low_idx = max(0, int((alpha / 2) * n_resamples))
    high_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples) - 1)
    return ConfidenceInterval(
        point=point,
        low=replicates[low_idx],
        high=replicates[high_idx],
        method="bootstrap-percentile",
        confidence=confidence,
    )


@dataclass
class BHResult:
    """One Benjamini-Hochberg multiple-comparison correction pass."""

    alpha: float
    #: Same order as the input p-values.
    rejected: list[bool]
    #: Benjamini-Hochberg adjusted p-values ("q-values"), same order as input.
    adjusted_p: list[float]
    n_rejected: int

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "rejected": self.rejected,
            "adjusted_p": self.adjusted_p,
            "n_rejected": self.n_rejected,
        }


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> BHResult:
    """Benjamini-Hochberg step-up FDR control.

    Brief section 6: aggregating a crossing claim across many held-out tasks
    without this means the *number* of claimed crossings overstates how many
    are real, in direct proportion to how many tasks were tested. This
    controls the expected false-discovery proportion at `alpha`, not the
    per-task false-positive rate (which is what each Clopper-Pearson interval
    already controls individually).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    m = len(p_values)
    if m == 0:
        return BHResult(alpha=alpha, rejected=[], adjusted_p=[], n_rejected=0)

    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]

    # Largest k such that p_(k) <= (k/m) * alpha; reject all ranks <= k.
    threshold_rank = 0
    for rank, p in enumerate(sorted_p, start=1):
        if p <= (rank / m) * alpha:
            threshold_rank = rank

    rejected_sorted = [rank <= threshold_rank for rank in range(1, m + 1)]

    # Adjusted p-values (q-values): running minimum from the largest rank down
    # of min(1, p_(k) * m / k), the standard BH monotone adjustment.
    adjusted_sorted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        candidate = sorted_p[rank - 1] * m / rank
        running_min = min(running_min, candidate)
        adjusted_sorted[rank - 1] = min(1.0, running_min)

    rejected = [False] * m
    adjusted_p = [0.0] * m
    for sorted_idx, original_idx in enumerate(order):
        rejected[original_idx] = rejected_sorted[sorted_idx]
        adjusted_p[original_idx] = adjusted_sorted[sorted_idx]

    return BHResult(
        alpha=alpha,
        rejected=rejected,
        adjusted_p=adjusted_p,
        n_rejected=sum(rejected),
    )
