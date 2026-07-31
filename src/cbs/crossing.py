"""Frontier-crossing determination (brief section 3.3, 5.5).

The operational definition (brief section 3.3 / project brief section 3.3) is
a conjunction of four conditions. This module evaluates all four from objects
already produced elsewhere in the pipeline -- a `FrontierRecord` (`S0`'s side,
Phase 2), a `ScaffoldRunSummary` (the evolved scaffold's side, Phase 3's
`cbs.compare`), and optionally an `AblationResult` (Phase 4's `cbs.ablation`)
-- so that "did this count as a crossing" is computed the same mechanical way
every time, not re-argued per task during analysis.

An evolved agent "crosses the frontier" on task `x` iff ALL of:

1.  it solves `x` reliably (>= k/K runs);
2.  `M`'s base success satisfies `p_hat(x) < 1/N_max` under `S0`;
3.  the comparison is at inference compute matched to `N_max`;
4.  ablating support-expanding operations from the evolved scaffold removes
    the crossing.

Condition 2 collapses to `FrontierRecord.beyond_frontier` exactly when the
scaffold's per-run budget equals `S0`'s sampling budget `N_max`
(`p_hat(x) < 1/N_max` <=> zero successes in exactly `N_max` draws), which is
the setup `evaluate_crossing` requires -- see its budget-mismatch check.
"""

from __future__ import annotations

from dataclasses import dataclass

from cbs.ablation import AblationResult
from cbs.compare import ScaffoldRunSummary
from cbs.frontier.records import FrontierRecord
from cbs.scaffolds.tagging import contested_operations

__all__ = ["CrossingVerdict", "evaluate_crossing"]


@dataclass
class CrossingVerdict:
    """The four-part determination for one task, plus which parts held."""

    task_id: str
    k: int
    K: int
    reliable_solve: bool
    beyond_frontier: bool
    compute_matched: bool
    #: `None` when no ablation was supplied -- crossing cannot be confirmed
    #: without one, but the other three conditions are still worth reporting.
    ablation_removes_crossing: bool | None
    crossed: bool
    #: Flagged, not blocking: a crossing attributed to a contested operation
    #: (docs D-04/D-05, e.g. `temperature_schedule`) is a weaker claim and
    #: must be reported with that caveat, per `contested_operations()`.
    rests_on_contested_op: bool

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "k": self.k,
            "K": self.K,
            "reliable_solve": self.reliable_solve,
            "beyond_frontier": self.beyond_frontier,
            "compute_matched": self.compute_matched,
            "ablation_removes_crossing": self.ablation_removes_crossing,
            "crossed": self.crossed,
            "rests_on_contested_op": self.rests_on_contested_op,
        }

    def summary_line(self) -> str:
        if not self.beyond_frontier:
            return f"{self.task_id:26s} not beyond-frontier at this budget -- not a crossing candidate"
        parts = [
            f"reliable={self.reliable_solve}",
            f"compute_matched={self.compute_matched}",
            f"ablation_removes={self.ablation_removes_crossing}",
        ]
        verdict = "CROSSED" if self.crossed else "not crossed"
        caveat = " [rests on a contested op]" if self.rests_on_contested_op else ""
        return f"{self.task_id:26s} {verdict}{caveat}  ({', '.join(parts)})"


def evaluate_crossing(
    s0_record: FrontierRecord,
    evolved_summary: ScaffoldRunSummary,
    k: int,
    K: int,
    ablation: AblationResult | None = None,
    expanding_ops_used: frozenset[str] = frozenset(),
) -> CrossingVerdict:
    """Evaluate the four-part crossing definition for one task.

    `k`, `K`: the "solves reliably" threshold as `k` successes out of `K`
    evaluated as a *rate* `k/K` applied to `evolved_summary.n_reps` (which need
    not equal `K` exactly -- see brief section 6, "pre-register... thresholds
    (k/K...)").

    `expanding_ops_used`: the support-expanding operations observed in the
    evolved scaffold's solved runs (e.g. from aggregating
    `ScaffoldRunSummary.op_counts` against the registry). Used only to flag
    `rests_on_contested_op`; does not itself gate `crossed`.
    """
    if evolved_summary.task_id != s0_record.task_id:
        raise ValueError(
            f"mismatched task ids: S0 record is for {s0_record.task_id!r}, "
            f"evolved summary is for {evolved_summary.task_id!r}"
        )

    reliable = (evolved_summary.n_solved / evolved_summary.n_reps) >= (k / K) if evolved_summary.n_reps else False

    # p_hat(x) < 1/N_max <=> zero successes in exactly N_max draws, which is
    # what `beyond_frontier` already means -- but only when N_max equals the
    # budget this comparison is actually matched against; otherwise the
    # thresholds silently refer to different budgets.
    if s0_record.n_samples != evolved_summary.budget_calls:
        raise ValueError(
            f"{s0_record.task_id}: S0 was sampled at N_max={s0_record.n_samples} "
            f"but the evolved scaffold's budget is {evolved_summary.budget_calls}; "
            "'p_hat(x) < 1/N_max' is only well-defined when these match. Re-sample "
            "S0 at the same N_max as the comparison budget."
        )
    beyond_frontier = s0_record.beyond_frontier

    compute_matched = evolved_summary.budget_calls <= s0_record.n_samples

    ablation_removes = None
    if ablation is not None:
        if ablation.task_id != s0_record.task_id:
            raise ValueError(
                f"ablation is for {ablation.task_id!r}, not {s0_record.task_id!r}"
            )
        ablated_rate = (
            ablation.ablated.n_solved / ablation.ablated.n_reps
            if ablation.ablated.n_reps
            else 0.0
        )
        ablation_removes = ablated_rate < (k / K)

    crossed = bool(
        reliable and beyond_frontier and compute_matched and (ablation_removes is True)
    )

    return CrossingVerdict(
        task_id=s0_record.task_id,
        k=k,
        K=K,
        reliable_solve=reliable,
        beyond_frontier=beyond_frontier,
        compute_matched=compute_matched,
        ablation_removes_crossing=ablation_removes,
        crossed=crossed,
        rests_on_contested_op=bool(expanding_ops_used & contested_operations()),
    )
