"""Estimator validation -- the Phase 2 Definition of Done.

The brief requires: "on a toy task with known ground-truth solve rate, estimator
recovers it within CI". That is split into two checks, because they validate
different things and a single check would conflate them:

1.  :func:`validate_pipeline` runs the **whole instrument** -- mock model, `S0`,
    sandbox, verifier, canonicaliser, estimators -- against the toy family whose
    true `p(x)` and true species richness are known by construction. This catches
    plumbing faults: a broken extractor, a verifier that mis-scores, a
    canonicaliser that merges distinct solutions.

2.  :func:`validate_ci_coverage` checks the **statistical procedure** by direct
    simulation over many replicates. A confidence interval is a claim about
    long-run coverage, so one run containing the true value is weak evidence and
    one run missing it is not a bug -- at 95% nominal, roughly 1 task in 20 is
    *expected* to miss. Only repeated sampling can distinguish a correct
    conservative interval from a broken one.

Running only (1) would let a systematically miscalibrated interval pass whenever
it happened to be wide; running only (2) would validate arithmetic against
arithmetic and never touch the pipeline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from cbs.budget import BudgetAccountant
from cbs.frontier.estimators import chao1, clopper_pearson
from cbs.frontier.records import FrontierRecord
from cbs.frontier.sampler import DEFAULT_SCHEDULE, FrontierSampler, TemperatureSchedule
from cbs.models.base import CompletionRequest
from cbs.models.mock import MockModelClient
from cbs.sandbox import select_backend
from cbs.tasks import Verifier
from cbs.tasks.families.toy import TOY_TASKS, toy_behaviours, toy_suite

__all__ = [
    "TaskValidation",
    "PipelineValidation",
    "CoverageValidation",
    "validate_pipeline",
    "validate_ci_coverage",
]


@dataclass
class TaskValidation:
    task_id: str
    true_p: float
    n_samples: int
    n_correct: int
    p_hat: float
    ci_low: float
    ci_high: float
    p_recovered: bool
    true_species: int
    observed_species: int
    chao1_estimate: float | None
    chao1_low: float | None
    chao1_high: float | None
    species_recovered: bool | None
    beyond_frontier: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def line(self) -> str:
        mark = "OK " if self.p_recovered else "MISS"
        sp = (
            "n/a"
            if self.chao1_estimate is None
            else f"{self.chao1_estimate:5.2f} [{self.chao1_low:.2f},{self.chao1_high:.2f}]"
        )
        return (
            f"{mark} {self.task_id:24s} true_p={self.true_p:.3f} "
            f"p_hat={self.p_hat:.3f} CI=[{self.ci_low:.3f},{self.ci_high:.3f}] "
            f"| species true={self.true_species} obs={self.observed_species} "
            f"chao1={sp}"
        )


def binomial_tail_ge(k: int, n: int, p: float) -> float:
    """`P(X >= k)` for `X ~ Binomial(n, p)`."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * p**i * (1 - p) ** (n - i)
    return total


@dataclass
class PipelineValidation:
    results: list[TaskValidation] = field(default_factory=list)
    records: list[FrontierRecord] = field(default_factory=list)
    n_samples: int = 0
    confidence: float = 0.95
    #: Fail only if the miss count is this unlikely under nominal coverage.
    alpha: float = 0.01

    @property
    def n_recovered(self) -> int:
        return sum(1 for r in self.results if r.p_recovered)

    @property
    def n_missed(self) -> int:
        return len(self.results) - self.n_recovered

    @property
    def miss_p_value(self) -> float:
        """`P(at least this many misses)` if the interval is correctly calibrated."""
        return binomial_tail_ge(self.n_missed, len(self.results), 1 - self.confidence)

    @property
    def ok(self) -> bool:
        """Whether the observed misses are consistent with nominal coverage.

        Demanding zero misses would be wrong, not strict: a 95% interval is
        *supposed* to miss about 1 task in 20, so over 6 tasks the probability of
        at least one miss is about 26%. A pass criterion that fails a quarter of
        the time on a healthy pipeline trains you to ignore it.

        Instead this asks whether the miss count is implausible under correct
        calibration, and fails only then. Gross plumbing faults -- a broken
        verifier, a mis-wired extractor -- produce many misses at once and are
        caught easily; a single boundary miss is not evidence of anything.
        `validate_ci_coverage` is the high-power check on calibration itself.
        """
        return self.miss_p_value >= self.alpha

    def report(self) -> str:
        lines = [
            f"Pipeline validation ({self.n_samples} samples/task, "
            f"{len(self.results)} tasks)",
            "-" * 78,
        ]
        lines += [r.line() for r in self.results]
        lines.append("-" * 78)
        lines.append(
            f"true p recovered within {self.confidence:.0%} CI: "
            f"{self.n_recovered}/{len(self.results)}"
        )
        lines.append(
            f"P(>= {self.n_missed} misses | correctly calibrated) = "
            f"{self.miss_p_value:.4f}  "
            f"(fail threshold {self.alpha}) -> {'PASS' if self.ok else 'FAIL'}"
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_samples_per_task": self.n_samples,
            "n_tasks": len(self.results),
            "n_recovered": self.n_recovered,
            "n_missed": self.n_missed,
            "miss_p_value": self.miss_p_value,
            "alpha": self.alpha,
            "ok": self.ok,
            "tasks": [r.as_dict() for r in self.results],
        }


def validate_pipeline(
    n_samples: int = 200,
    output_dir: Path | str = "runs/validation",
    seed: int = 0,
    schedule: TemperatureSchedule = DEFAULT_SCHEDULE,
    p_overrides: dict[str, float] | None = None,
) -> PipelineValidation:
    """Run the full instrument against known ground truth."""
    suite = toy_suite()
    behaviours = toy_behaviours(p_overrides)
    model = MockModelClient(behaviours=behaviours, seed=seed, model_id="mock-validate")
    verifier = Verifier(select_backend("auto"))
    sampler = FrontierSampler(
        model=model,
        verifier=verifier,
        schedule=schedule,
        output_dir=output_dir,
    )
    accountant = BudgetAccountant("validate")

    defs = {d.task_id: d for d in TOY_TASKS}
    out = PipelineValidation(n_samples=n_samples)

    for task in suite.tasks:
        record = sampler.estimate_task(task, n_samples, accountant, resume=False)
        definition = defs[task.task_id]
        true_p = behaviours[task.task_id].p_correct
        true_species = definition.to_behaviour().true_species_count()

        ci = clopper_pearson(record.n_correct, record.n_samples)
        chao = record.chao1

        # Richness is only checkable when something was actually sampled, and
        # Chao1 is a *lower-bound* estimator: the honest check is that the true
        # richness is not below the interval, not that it sits inside it.
        species_recovered: bool | None
        if chao is None:
            species_recovered = None
        else:
            species_recovered = true_species >= chao["low"] - 1e-9

        out.results.append(
            TaskValidation(
                task_id=task.task_id,
                true_p=true_p,
                n_samples=record.n_samples,
                n_correct=record.n_correct,
                p_hat=record.p_hat,
                ci_low=ci.low,
                ci_high=ci.high,
                p_recovered=ci.contains(true_p),
                true_species=true_species,
                observed_species=record.n_distinct_solutions,
                chao1_estimate=chao["estimated_species"] if chao else None,
                chao1_low=chao["low"] if chao else None,
                chao1_high=chao["high"] if chao else None,
                species_recovered=species_recovered,
                beyond_frontier=record.beyond_frontier,
            )
        )
        out.records.append(record)

    return out


@dataclass
class CoverageValidation:
    n_replicates: int
    n_samples: int
    cases: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Clopper-Pearson is exact-conservative: coverage must be >= nominal.

        Allows a small Monte-Carlo tolerance, and checks only the lower side --
        over-coverage is the estimator behaving as designed, not a fault.
        """
        return all(c["coverage"] >= c["nominal"] - c["mc_tolerance"] for c in self.cases)

    def report(self) -> str:
        lines = [
            f"CI coverage validation ({self.n_replicates} replicates x "
            f"{self.n_samples} samples)",
            "-" * 78,
        ]
        for c in self.cases:
            mark = "OK " if c["coverage"] >= c["nominal"] - c["mc_tolerance"] else "FAIL"
            lines.append(
                f"{mark} true_p={c['true_p']:.4f}  empirical coverage="
                f"{c['coverage']:.4f}  (nominal {c['nominal']:.2f}, "
                f"tol {c['mc_tolerance']:.4f}, mean CI width {c['mean_width']:.4f})"
            )
        lines.append("-" * 78)
        lines.append(
            "Clopper-Pearson is conservative by construction, so coverage at or "
            "above nominal is the expected result."
        )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_replicates": self.n_replicates,
            "n_samples": self.n_samples,
            "ok": self.ok,
            "cases": self.cases,
        }


def validate_ci_coverage(
    true_ps: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 0.5, 0.9),
    n_samples: int = 300,
    n_replicates: int = 400,
    confidence: float = 0.95,
    seed: int = 12345,
) -> CoverageValidation:
    """Empirically verify the interval attains its nominal coverage.

    Simulates binomial draws directly rather than running the model and sandbox:
    this validates the *statistical procedure*, and interposing the pipeline
    would only add cost and noise. `validate_pipeline` is what exercises the
    pipeline.

    `true_p = 0` is included deliberately -- it is the regime every
    beyond-frontier claim lives in, and the one where a normal-approximation
    interval fails outright.
    """
    rng = random.Random(seed)
    out = CoverageValidation(n_replicates=n_replicates, n_samples=n_samples)

    for true_p in true_ps:
        hits = 0
        width_total = 0.0
        for _ in range(n_replicates):
            successes = sum(1 for _ in range(n_samples) if rng.random() < true_p)
            ci = clopper_pearson(successes, n_samples, confidence)
            if ci.contains(true_p):
                hits += 1
            width_total += ci.high - ci.low
        coverage = hits / n_replicates
        # 3 Monte-Carlo standard errors of the coverage estimate.
        se = (confidence * (1 - confidence) / n_replicates) ** 0.5
        out.cases.append(
            {
                "true_p": true_p,
                "coverage": coverage,
                "nominal": confidence,
                "mc_tolerance": 3 * se,
                "mean_width": width_total / n_replicates,
            }
        )
    return out


def validate_verifier_agreement(
    n_per_task: int = 120,
    seed: int = 1,
    task_ids: tuple[str, ...] | None = None,
) -> dict:
    """Check the verifier's verdict against the mock's known ground truth.

    The mock knows whether each emitted sample came from the correct or the
    incorrect variant pool, so every sample is a labelled example. Any
    disagreement is a verifier defect, and the two directions mean different
    things:

    *   a **false positive** (verifier passes a known-wrong solution) inflates
        `p_hat(x)`, shrinks the beyond-frontier set, and -- if it happened under
        `S_evo` -- would manufacture an apparent frontier-crossing out of
        nothing. This is the failure mode the brief demands be bounded by manual
        audit (section 7, Phase 1);
    *   a **false negative** deflates `p_hat(x)` and would push solvable tasks
        into the beyond-frontier set, making crossings look easier to achieve.

    Neither is tolerable, so the pass condition is exact agreement.
    """
    behaviours = toy_behaviours()
    suite = toy_suite().by_id()
    verifier = Verifier(select_backend("auto"))
    schedule = DEFAULT_SCHEDULE.allocate(max(n_per_task, 1))

    ids = task_ids or tuple(
        t for t in suite if behaviours[t].p_correct not in (0.0, 1.0)
    )

    false_positives = 0
    false_negatives = 0
    checked = 0
    disagreements: list[dict] = []

    for task_id in ids:
        model = MockModelClient(behaviours, seed=seed, model_id=f"agree-{seed}")
        task = suite[task_id]
        accountant = BudgetAccountant("verifier-agreement")
        for i in range(n_per_task):
            completion = model.complete(
                CompletionRequest(
                    prompt=task.prompt,
                    temperature=schedule[i],
                    seed=i,
                    meta={"task_id": task_id},
                ),
                accountant,
            )
            truth = bool(completion.meta["_ground_truth_correct"])
            passed = verifier.verify_completion(task, completion.text).passed
            checked += 1
            if passed and not truth:
                false_positives += 1
                disagreements.append({"task_id": task_id, "i": i, "kind": "false_positive"})
            elif truth and not passed:
                false_negatives += 1
                disagreements.append({"task_id": task_id, "i": i, "kind": "false_negative"})

    return {
        "n_checked": checked,
        "n_tasks": len(ids),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "agreement_rate": (checked - len(disagreements)) / checked if checked else 0.0,
        "disagreements": disagreements[:20],
        "ok": checked > 0 and not disagreements,
    }


def validate_chao1_lower_bound(
    n_species: int = 12,
    n_samples: int = 150,
    n_replicates: int = 200,
    seed: int = 999,
) -> dict:
    """Check Chao1 behaves as a lower bound on richness under undersampling.

    Draws from a skewed (Zipf-like) species distribution, where a finite sample
    always misses rare species. The expected behaviour is
    `observed <= chao1_estimate <= true_richness` most of the time; Chao1
    over-estimating would be the notable failure, since the study relies on it to
    say "you have not seen everything", never "you have seen everything".
    """
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(n_species)]
    total = sum(weights)
    probs = [w / total for w in weights]

    n_at_or_below = 0
    n_above_observed = 0
    estimates = []
    observed_counts = []

    for _ in range(n_replicates):
        counts: dict[str, int] = {}
        for _ in range(n_samples):
            u = rng.random()
            acc = 0.0
            for i, p in enumerate(probs):
                acc += p
                if u <= acc:
                    counts[f"s{i}"] = counts.get(f"s{i}", 0) + 1
                    break
        est = chao1(counts)
        if est is None:
            continue
        estimates.append(est.estimate)
        observed_counts.append(est.observed)
        if est.estimate <= n_species + 1e-9:
            n_at_or_below += 1
        if est.estimate >= est.observed - 1e-9:
            n_above_observed += 1

    n = len(estimates)
    return {
        "true_richness": n_species,
        "n_replicates": n,
        "mean_observed": sum(observed_counts) / n if n else 0.0,
        "mean_estimate": sum(estimates) / n if n else 0.0,
        "frac_not_exceeding_truth": n_at_or_below / n if n else 0.0,
        "frac_at_least_observed": n_above_observed / n if n else 0.0,
        "ok": n > 0 and n_above_observed == n,
    }
