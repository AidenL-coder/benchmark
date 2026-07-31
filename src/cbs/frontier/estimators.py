"""Frontier statistics.

Implements the estimators the brief (section 3.2) and proposal (sections 5.4,
6) require, from scratch, so the measurement core has no scientific-stack
dependency and can be unit-tested against published reference values.

Two quantities are easy to conflate and are therefore named apart here. The
distinction matters because they answer different questions and only one of them
bears on whether a task is beyond the frontier:

*   :func:`good_turing_unseen_mass` -- among the *correct* solutions the model
    produces, what share of the correct-solution distribution have we not yet
    seen? This is a statement about solution **diversity**. It is computed from
    the counts-of-counts of observed correct solutions and is undefined (and
    reported as ``None``) for a task with zero successes, since there is no
    sample to compute it from.

*   :func:`clopper_pearson` / :func:`rule_of_three` -- given `c` successes in `n`
    draws, what range of true `p(x)` is consistent with that? This is a statement
    about **solvability**, and it is the only one of the two that can bound
    `p(x)` for a zero-success task.

Reading a zero unseen-mass estimate on a zero-success task as evidence that the
task is unsolvable would be exactly backwards: it reflects an empty sample, not a
covered distribution. Every function below documents which question it answers.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "ConfidenceInterval",
    "clopper_pearson",
    "rule_of_three",
    "zero_success_upper_bound",
    "pass_at_k",
    "good_turing_unseen_mass",
    "sample_coverage",
    "Chao1Estimate",
    "chao1",
    "counts_of_counts",
    "solution_rarefaction",
    "success_probability_curve",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Regularised incomplete beta, and its inverse. Needed for exact binomial CIs.
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, max_iter: int = 300, tol: float = 1e-14) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """`I_x(a, b)`, the CDF of a Beta(a, b) distribution."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - ln_beta)
    # The continued fraction converges quickly only on one side of this pivot;
    # use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) on the other side.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    back = math.exp(b * math.log1p(-x) + a * math.log(x) - ln_beta)
    return 1.0 - back * _betacf(b, a, 1.0 - x) / b


def inverse_regularized_incomplete_beta(a: float, b: float, p: float) -> float:
    """Solve `I_x(a, b) = p` for x, by bisection.

    Bisection rather than Newton: it is slower but cannot diverge, and these are
    called once per task, not in the sampling inner loop.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Binomial intervals -- "is this task solvable, and how often?"
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    method: str
    confidence: float = 0.95

    def contains(self, value: float) -> bool:
        return self.low - _EPS <= value <= self.high + _EPS

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "method": self.method,
            "confidence": self.confidence,
        }


def clopper_pearson(
    successes: int, trials: int, confidence: float = 0.95
) -> ConfidenceInterval:
    """Exact (conservative) binomial CI for `p(x)`.

    Required by the proposal's statistical plan (section 6). Exact rather than
    normal-approximation because the interesting regime here is `c` at or near
    zero out of a large `n`, where a Wald interval is badly wrong (it collapses
    to the degenerate [0, 0]).
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid counts: {successes} successes in {trials} trials")
    if trials == 0:
        return ConfidenceInterval(float("nan"), 0.0, 1.0, "clopper-pearson", confidence)

    alpha = 1.0 - confidence
    point = successes / trials
    low = (
        0.0
        if successes == 0
        else inverse_regularized_incomplete_beta(
            successes, trials - successes + 1, alpha / 2.0
        )
    )
    high = (
        1.0
        if successes == trials
        else inverse_regularized_incomplete_beta(
            successes + 1, trials - successes, 1.0 - alpha / 2.0
        )
    )
    return ConfidenceInterval(point, low, high, "clopper-pearson", confidence)


def rule_of_three(trials: int) -> float:
    """The `3/n` upper bound on `p` after zero successes in `n` trials.

    The brief quotes this directly (section 3.2). It is a first-order
    approximation to :func:`zero_success_upper_bound` and is reported because it
    is the form readers recognise; the exact bound is reported alongside it.
    """
    if trials <= 0:
        return 1.0
    return min(1.0, 3.0 / trials)


def zero_success_upper_bound(trials: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper bound on `p` given zero successes in `n` trials.

    `1 - alpha**(1/n)`, of which `3/n` is the small-`alpha` approximation.
    """
    if trials <= 0:
        return 1.0
    alpha = 1.0 - confidence
    return 1.0 - alpha ** (1.0 / trials)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased `pass@k` from `c` successes in `n` samples (Chen et al., 2021).

    `1 - C(n-c, k) / C(n, k)`, evaluated as a product to avoid overflow and
    catastrophic cancellation at the `n ~ 10^4` budgets this study uses.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if n <= 0 or k > n:
        raise ValueError(f"require 0 < k <= n, got n={n}, k={k}")
    if c < 0 or c > n:
        raise ValueError(f"require 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(n - c + 1, n + 1):
        result *= 1.0 - k / i
    return 1.0 - result


# ---------------------------------------------------------------------------
# Species estimators -- "how much solution diversity have we not seen?"
# ---------------------------------------------------------------------------
def counts_of_counts(species_counts: Mapping[str, int] | Sequence[int]) -> dict[int, int]:
    """Frequency-of-frequencies `f_r`: how many species were seen exactly `r` times."""
    values = (
        list(species_counts.values())
        if isinstance(species_counts, Mapping)
        else list(species_counts)
    )
    return dict(Counter(v for v in values if v > 0))


def good_turing_unseen_mass(
    species_counts: Mapping[str, int] | Sequence[int],
) -> float | None:
    """Good-Turing estimate `f1 / N` of unseen **correct-solution** mass.

    Answers: if the model produces one more *correct* solution, what is the
    probability it is one we have never seen? High values mean the observed set
    of distinct solutions under-represents the model's true solution diversity.

    Returns ``None`` when no correct solutions were observed. That case is not
    "zero unseen mass" -- it is no information, and a caller that renders it as
    0.0 would report a fully-covered distribution where none was sampled. For
    zero-success tasks use :func:`clopper_pearson` or
    :func:`zero_success_upper_bound` instead; those are the estimators that
    speak to solvability.
    """
    counts = counts_of_counts(species_counts)
    total = sum(r * fr for r, fr in counts.items())
    if total == 0:
        return None
    return counts.get(1, 0) / total


def sample_coverage(
    species_counts: Mapping[str, int] | Sequence[int],
) -> float | None:
    """Good-Turing sample coverage `1 - f1/N`. ``None`` on an empty sample."""
    unseen = good_turing_unseen_mass(species_counts)
    return None if unseen is None else 1.0 - unseen


@dataclass(frozen=True)
class Chao1Estimate:
    """Estimated species richness with a log-normal CI (Chao 1987)."""

    observed: int
    estimate: float
    f1: int
    f2: int
    variance: float
    low: float
    high: float
    method: str
    confidence: float = 0.95

    @property
    def estimated_unseen(self) -> float:
        return max(0.0, self.estimate - self.observed)

    def as_dict(self) -> dict:
        return {
            "observed_species": self.observed,
            "estimated_species": self.estimate,
            "estimated_unseen_species": self.estimated_unseen,
            "f1": self.f1,
            "f2": self.f2,
            "variance": self.variance,
            "low": self.low,
            "high": self.high,
            "method": self.method,
            "confidence": self.confidence,
        }


def chao1(
    species_counts: Mapping[str, int] | Sequence[int],
    bias_corrected: bool = True,
    confidence: float = 0.95,
) -> Chao1Estimate | None:
    """Chao1 lower-bound estimate of the number of distinct correct solutions.

    Answers: how many distinct correct solutions does the model have available
    for this task, including ones this sample missed? Chao1 is a *lower bound* on
    richness, which suits the study's posture -- it can say the observed solution
    set is incomplete, never that it is complete.

    The bias-corrected form is the default and is used unconditionally when
    `f2 == 0`, where the classic form divides by zero.

    Returns ``None`` for an empty sample, for the same reason as
    :func:`good_turing_unseen_mass`.
    """
    counts = counts_of_counts(species_counts)
    observed = sum(counts.values())
    if observed == 0:
        return None

    f1 = counts.get(1, 0)
    f2 = counts.get(2, 0)

    if bias_corrected or f2 == 0:
        method = "chao1-bias-corrected"
        f0 = f1 * (f1 - 1) / (2.0 * (f2 + 1))
        # Chao & Shen (2003) variance for the bias-corrected estimator.
        variance = (
            f1 * (f1 - 1) / (2.0 * (f2 + 1))
            + f1 * (2 * f1 - 1) ** 2 / (4.0 * (f2 + 1) ** 2)
            + f1**2 * f2 * (f1 - 1) ** 2 / (4.0 * (f2 + 1) ** 4)
        )
    else:
        method = "chao1-classic"
        f0 = f1**2 / (2.0 * f2)
        ratio = f1 / f2
        variance = f2 * (0.5 * ratio**2 + ratio**3 + 0.25 * ratio**4)

    estimate = observed + f0

    # Log-normal CI on the *unseen* count, which keeps the lower limit at or
    # above `observed` -- richness cannot be below what was actually seen.
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else _normal_quantile(
        1.0 - (1.0 - confidence) / 2.0
    )
    if f0 <= _EPS or variance <= _EPS:
        low = high = float(observed)
    else:
        k = math.exp(z * math.sqrt(math.log1p(variance / (f0**2))))
        low = observed + f0 / k
        high = observed + f0 * k

    return Chao1Estimate(
        observed=observed,
        estimate=estimate,
        f1=f1,
        f2=f2,
        variance=variance,
        low=max(float(observed), low),
        high=high,
        method=method,
        confidence=confidence,
    )


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


# ---------------------------------------------------------------------------
# Budget curves -- "the frontier as a function of compute"
# ---------------------------------------------------------------------------
def solution_rarefaction(
    species_counts: Mapping[str, int] | Sequence[int], sizes: Iterable[int]
) -> dict[int, float]:
    """Expected distinct correct solutions if only `m` correct samples were drawn.

    Individual-based rarefaction (Hurlbert 1971). Used to show that the observed
    solution set is still growing at `N_max`, which is direct evidence that the
    frontier estimate is budget-limited rather than saturated -- the proposal's
    single most important caveat (section 8).
    """
    values = (
        list(species_counts.values())
        if isinstance(species_counts, Mapping)
        else list(species_counts)
    )
    values = [v for v in values if v > 0]
    n = sum(values)
    out: dict[int, float] = {}
    for m in sizes:
        if m <= 0 or n == 0:
            out[m] = 0.0
            continue
        if m >= n:
            out[m] = float(len(values))
            continue
        # E[S_m] = sum_i (1 - C(n - n_i, m) / C(n, m)), computed in log space.
        total = 0.0
        log_denom = _log_binom(n, m)
        for ni in values:
            if n - ni >= m:
                total += 1.0 - math.exp(_log_binom(n - ni, m) - log_denom)
            else:
                total += 1.0
        out[m] = total
    return out


def _log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def success_probability_curve(
    n: int, c: int, budgets: Iterable[int]
) -> dict[int, float]:
    """`pass@k` across a range of budgets `k`.

    This is "the frontier as a function of budget" that the brief asks for
    (section 3.2): the probability that a budget of `k` samples solves the task
    at least once, given `c` successes observed in `n` draws.
    """
    return {k: pass_at_k(n, c, k) for k in budgets if 0 < k <= n}
