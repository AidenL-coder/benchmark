"""`S0` vs `S_star` at matched compute -- the elicitation control.

Brief section 9.1: "Elicitation-adjusted gain = `S_evo` - `S_star` at matched
compute." Before that comparison means anything, `S_star` itself has to be
benchmarked against `S0` at matched compute -- otherwise there is no baseline
to know whether *any* gain over `S0` is unusual. That is what this module does,
and it is also the template Phase 5 reuses for `S_evo` vs `S_star`.

Reframing "matched compute" for a multi-call scaffold
------------------------------------------------------
`S0` is one call by construction (brief section 3.2), so its "solve rate at a
budget of `B` calls" is exactly `pass@B` on its own frontier record -- already
built and validated in Phase 2 (`cbs.frontier.estimators.pass_at_k`). No new
sampling is needed for `S0`'s side of the comparison; it is read off an existing
record.

`S_star` spends its budget across candidates and repairs however its algorithm
decides, in a single run. So its "solve rate at a budget of `B` calls" is
measured empirically: run `K` independent repetitions, each with a **fresh**
per-repetition allowance capped at `B` calls, and take the empirical rate with a
Clopper-Pearson CI -- the same estimator already validated in Phase 2, applied
to a different unit (one scaffold run, not one model sample).

Each repetition is registered with `MatchedComputeHarness` under its own
pseudo-task id (`f"{task_id}::rep{i}"`), so the harness's existing matched-
compute verification applies at full per-repetition resolution, not just on
average.

Why `S_star` under-spending its allowance is not itself a red flag
--------------------------------------------------------------------
`S_star` stops once it has a candidate that passes what it can check
(`stop_on_first_public_pass`, on by default) -- a real deployed agent does not
keep burning calls once it is satisfied. So `S_star`'s *realised* spend is
routinely below its `B`-call allowance, and `MatchReport.matched` will often
read `False` for that reason alone. That is expected and is not the confound
the harness exists to catch: the dangerous direction is a system *exceeding*
its allowance to manufacture a result, which the hard cap already prevents
structurally. `ComparisonRecord` therefore reports realised spend for both
systems explicitly rather than gating on `matched`, and callers should read
under-spending by the *stronger* system as making a positive result more
credible, not less.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cbs.budget import BudgetCaps, MatchedComputeHarness, Usage
from cbs.frontier.estimators import clopper_pearson, pass_at_k
from cbs.frontier.records import FrontierRecord
from cbs.models.base import ModelClient
from cbs.scaffolds.base import Scaffold
from cbs.scaffolds.s0 import S0
from cbs.tasks.schema import Task, TaskSuite
from cbs.tasks.verifier import Verifier

__all__ = [
    "ScaffoldRunSummary",
    "ComparisonRecord",
    "ScaffoldComparator",
    "measure_scaffold_solve_rate",
]


@dataclass
class ScaffoldRunSummary:
    """Empirical solve rate of one scaffold on one task at a per-run budget `B`."""

    scaffold_name: str
    task_id: str
    budget_calls: int
    n_reps: int
    n_solved: int
    p_hat: float
    ci_low: float
    ci_high: float
    mean_calls: float
    mean_total_tokens: float
    n_budget_exhausted: int
    op_counts: dict[str, int]
    expanding_dependency: float  # fraction of SOLVED reps that used >=1 expanding op

    def as_dict(self) -> dict:
        return {
            "scaffold_name": self.scaffold_name,
            "task_id": self.task_id,
            "budget_calls": self.budget_calls,
            "n_reps": self.n_reps,
            "n_solved": self.n_solved,
            "p_hat": self.p_hat,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "mean_calls": self.mean_calls,
            "mean_total_tokens": self.mean_total_tokens,
            "n_budget_exhausted": self.n_budget_exhausted,
            "op_counts": self.op_counts,
            "expanding_dependency": self.expanding_dependency,
        }


def measure_scaffold_solve_rate(
    scaffold: Scaffold,
    task: Task,
    model: ModelClient,
    verifier: Verifier,
    budget_calls: int,
    n_reps: int,
    harness: MatchedComputeHarness,
    confidence: float = 0.95,
    system_label: str | None = None,
) -> ScaffoldRunSummary:
    """Empirical solve rate of any `Scaffold` over `n_reps` independent
    repetitions, each under a fresh per-rep budget of `budget_calls`.

    Shared by `ScaffoldComparator` (`S_star` vs `S0`) and `cbs.ablation`
    (a scaffold vs an ablated variant of itself): both are exactly this
    measurement applied to a different pair of scaffold instances.
    """
    label = system_label or scaffold.name
    solved = 0
    total_calls = 0
    total_tokens = 0
    n_exhausted = 0
    op_counts: dict[str, int] = {}
    expanding_in_solved = 0

    for rep in range(n_reps):
        pseudo_task_id = f"{task.task_id}::rep{rep}"
        accountant = harness.allowance_for(label, pseudo_task_id)
        result = scaffold.solve(task, model, accountant, verifier=verifier, seed=rep)
        total_calls += accountant.spent.calls
        total_tokens += accountant.spent.total_tokens
        if result.budget_exhausted:
            n_exhausted += 1
        if result.passed:
            solved += 1
            if result.trace.used_expanding:
                expanding_in_solved += 1
        for name, count in result.trace.op_counts().items():
            op_counts[name] = op_counts.get(name, 0) + count

    ci = clopper_pearson(solved, n_reps, confidence)
    return ScaffoldRunSummary(
        scaffold_name=label,
        task_id=task.task_id,
        budget_calls=budget_calls,
        n_reps=n_reps,
        n_solved=solved,
        p_hat=ci.point,
        ci_low=ci.low,
        ci_high=ci.high,
        mean_calls=total_calls / n_reps if n_reps else 0.0,
        mean_total_tokens=total_tokens / n_reps if n_reps else 0.0,
        n_budget_exhausted=n_exhausted,
        op_counts=op_counts,
        expanding_dependency=(expanding_in_solved / solved if solved else 0.0),
    )


@dataclass
class ComparisonRecord:
    """`S0` vs `S_star` on one task at matched per-run compute `B`."""

    task_id: str
    budget_calls: int
    s0: ScaffoldRunSummary
    s_star: ScaffoldRunSummary
    #: S_star.p_hat - S0.p_hat. Positive means S_star elicits more at this budget.
    elicitation_gain: float
    #: Advisory only -- see module docstring for why under-spending is not a red flag.
    realised_spend_matched: bool
    realised_spend_ratio: dict[str, float]

    def summary_line(self) -> str:
        return (
            f"{self.task_id:26s} B={self.budget_calls:3d}  "
            f"S0={self.s0.p_hat:.3f}[{self.s0.ci_low:.3f},{self.s0.ci_high:.3f}]  "
            f"S*={self.s_star.p_hat:.3f}[{self.s_star.ci_low:.3f},{self.s_star.ci_high:.3f}]  "
            f"gain={self.elicitation_gain:+.3f}  "
            f"mean_calls S0={self.s0.mean_calls:.1f} S*={self.s_star.mean_calls:.1f}"
        )

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "budget_calls": self.budget_calls,
            "s0": self.s0.as_dict(),
            "s_star": self.s_star.as_dict(),
            "elicitation_gain": self.elicitation_gain,
            "realised_spend_matched": self.realised_spend_matched,
            "realised_spend_ratio": self.realised_spend_ratio,
        }


class ScaffoldComparator:
    """Runs `S0` (read off its frontier record) against `S_star` (freshly
    sampled) at a matched per-run call budget `B`, task by task."""

    def __init__(
        self,
        model: ModelClient,
        verifier: Verifier,
        s_star: Scaffold,
        budget_calls: int,
        n_reps: int = 30,
        confidence: float = 0.95,
        harness: MatchedComputeHarness | None = None,
    ):
        self.model = model
        self.verifier = verifier
        self.s_star = s_star
        self.budget_calls = budget_calls
        self.n_reps = n_reps
        self.confidence = confidence
        self.harness = harness or MatchedComputeHarness(BudgetCaps(calls=budget_calls))

    def s0_summary_from_record(
        self, record: FrontierRecord, task_id: str
    ) -> ScaffoldRunSummary:
        """`S0`'s solve rate at budget `B` is `pass@B` on its own frontier record.

        No new sampling: this reuses the already-validated pass@k estimator
        (Phase 2) rather than re-deriving the same quantity a second way.
        """
        if record.n_samples < self.budget_calls:
            raise ValueError(
                f"{task_id}: S0 frontier record has only {record.n_samples} samples, "
                f"fewer than the comparison budget B={self.budget_calls}. Re-run "
                "frontier estimation with N_max >= B first."
            )
        p_hat = pass_at_k(record.n_samples, record.n_correct, self.budget_calls)
        # pass@k has no closed-form CI of its own; the underlying per-sample
        # rate's Clopper-Pearson interval, propagated through pass@k's
        # monotone map, is the honest bound available without a bespoke
        # derivation, and it is conservative in the same direction as CP itself.
        base_ci = clopper_pearson(record.n_correct, record.n_samples, self.confidence)
        ci_low = pass_at_k(record.n_samples, max(0, int(base_ci.low * record.n_samples)), self.budget_calls)
        ci_high = pass_at_k(record.n_samples, min(record.n_samples, int(base_ci.high * record.n_samples) + 1), self.budget_calls)
        return ScaffoldRunSummary(
            scaffold_name="S0",
            task_id=task_id,
            budget_calls=self.budget_calls,
            n_reps=record.n_samples,  # informational: draws the record was built from
            n_solved=record.n_correct,
            p_hat=p_hat,
            ci_low=min(ci_low, ci_high),
            ci_high=max(ci_low, ci_high),
            mean_calls=1.0,  # S0 is one call per attempt by construction
            mean_total_tokens=(
                record.usage.total_tokens / record.n_samples if record.n_samples else 0.0
            ),
            n_budget_exhausted=record.n_budget_exhausted,
            op_counts={"single_call": record.n_samples, "format_extract": record.n_samples},
            expanding_dependency=0.0,  # S0 uses no support-expanding operations, by design
        )

    def s_star_summary(self, task: Task) -> ScaffoldRunSummary:
        """Run `S_star` for `n_reps` independent repetitions at budget `B` each."""
        return measure_scaffold_solve_rate(
            scaffold=self.s_star,
            task=task,
            model=self.model,
            verifier=self.verifier,
            budget_calls=self.budget_calls,
            n_reps=self.n_reps,
            harness=self.harness,
            confidence=self.confidence,
        )

    def compare_task(self, task: Task, s0_record: FrontierRecord) -> ComparisonRecord:
        s0_summary = self.s0_summary_from_record(s0_record, task.task_id)
        s_star_summary = self.s_star_summary(task)

        # Fold S0's per-attempt (1-call) spend into the harness under the same
        # pseudo-task ids, purely so `overall_report()` sees both systems.
        for rep in range(self.n_reps):
            acct = self.harness.allowance_for("S0", f"{task.task_id}::rep{rep}")
            if acct.spent.calls == 0:
                acct.charge(
                    Usage(calls=1, completion_tokens=int(s0_summary.mean_total_tokens))
                )

        matched_flags = []
        for rep in range(self.n_reps):
            report = self.harness.report_for(f"{task.task_id}::rep{rep}")
            matched_flags.append(report.matched)

        return ComparisonRecord(
            task_id=task.task_id,
            budget_calls=self.budget_calls,
            s0=s0_summary,
            s_star=s_star_summary,
            elicitation_gain=s_star_summary.p_hat - s0_summary.p_hat,
            realised_spend_matched=all(matched_flags),
            realised_spend_ratio={
                "S0": s0_summary.mean_calls / self.budget_calls,
                self.s_star.name: s_star_summary.mean_calls / self.budget_calls,
            },
        )

    def compare_suite(
        self, suite: TaskSuite, s0_records: dict[str, FrontierRecord], progress: bool = True
    ) -> list[ComparisonRecord]:
        results = []
        for i, task in enumerate(suite.tasks, start=1):
            if task.task_id not in s0_records:
                raise KeyError(
                    f"no S0 frontier record for {task.task_id}; run frontier "
                    "estimation for S0 first (needs N_max >= the comparison budget)"
                )
            if progress:
                print(f"[{i}/{len(suite.tasks)}] {task.task_id}", flush=True)
            record = self.compare_task(task, s0_records[task.task_id])
            results.append(record)
            if progress:
                print(f"    {record.summary_line()}", flush=True)
        return results


def load_frontier_records(path: Path | str) -> dict[str, FrontierRecord]:
    """Load `records.jsonl` written by `cbs frontier estimate` and re-key by task id.

    Only the fields `ScaffoldComparator` actually needs are reconstructed; this
    is a read path for comparison, not a full round-trip of `FrontierRecord`.
    """
    from cbs.budget import Usage as _Usage

    out: dict[str, FrontierRecord] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            usage = data.get("usage", {})
            out[data["task_id"]] = FrontierRecord(
                task_id=data["task_id"],
                model_id=data["model_id"],
                split=data.get("split"),
                family=data.get("family", ""),
                n_samples=data["n_samples"],
                n_correct=data["n_correct"],
                p_hat=data["p_hat"],
                p_ci_low=data["p_ci_low"],
                p_ci_high=data["p_ci_high"],
                p_upper_bound=data["p_upper_bound"],
                rule_of_three_bound=data["rule_of_three_bound"],
                exact_zero_success_bound=data["exact_zero_success_bound"],
                beyond_frontier=data["beyond_frontier"],
                n_distinct_solutions=data["n_distinct_solutions"],
                species_counts=data.get("species_counts", {}),
                good_turing_unseen_mass=data.get("good_turing_unseen_mass"),
                sample_coverage=data.get("sample_coverage"),
                chao1=data.get("chao1"),
                pass_at_k={int(k): v for k, v in data.get("pass_at_k", {}).items()},
                rarefaction={int(k): v for k, v in data.get("rarefaction", {}).items()},
                temperature_schedule=data.get("temperature_schedule", []),
                scaffold_fingerprint=data.get("scaffold_fingerprint", {}),
                usage=_Usage(
                    calls=usage.get("calls", 0),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    usd=usage.get("usd", 0.0),
                ),
                verifier_backend=data.get("verifier_backend", ""),
                verifier_is_security_boundary=data.get(
                    "verifier_is_security_boundary", False
                ),
                n_verifier_errors=data.get("n_verifier_errors", 0),
                n_budget_exhausted=data.get("n_budget_exhausted", 0),
                canonicalization_methods=data.get("canonicalization_methods", {}),
                metadata=data.get("metadata", {}),
            )
    return out
