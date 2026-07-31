"""Ablation runner (brief section 9.2, "Core tests and ablations").

Measures whether a scaffold's solve rate is *load-bearing* on a given
operation by comparing it against an ablated variant -- the mechanism behind
brief section 9.2's ablation matrix ("Disable execution feedback / tool calls
/ retrieval / decomposition -- is it load-bearing for crossings?") and the
crossing test's requirement that "ablating support-expanding operations from
the evolved scaffold removes the crossing" (brief section 3.3).

This is deliberately generic over any two `Scaffold` instances, not specific to
`S_evo`: "ablated" just means "a second scaffold instance that cannot perform
the operation under test". For `S0`/`S_star`, which we author, that is simply a
different constructor argument (e.g. `SStar(max_repairs_per_candidate=0)` has
no execution-feedback loop at all). For an untrusted evolved scaffold we do not
control the source of, the same idea is implemented at the interception layer
(`cbs.scaffolds.evolved`) instead -- blank the error text an
`InterceptionSession` hands back, and the agent genuinely cannot condition on
it, whatever its own code tries to do.

Validated here against `S_star`, which is safe to run today; the exact same
function is what Phase 5 points at `S_evo` once a Docker-capable host exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from cbs.budget import BudgetCaps, MatchedComputeHarness
from cbs.compare import ScaffoldRunSummary, measure_scaffold_solve_rate
from cbs.models.base import ModelClient
from cbs.scaffolds.base import Scaffold
from cbs.tasks.schema import Task
from cbs.tasks.verifier import Verifier

__all__ = ["AblationResult", "run_ablation"]


@dataclass
class AblationResult:
    """Whether disabling one operation removed solves that depended on it."""

    task_id: str
    ablated_op: str
    baseline: ScaffoldRunSummary
    ablated: ScaffoldRunSummary

    @property
    def solves_removed(self) -> int:
        return max(0, self.baseline.n_solved - self.ablated.n_solved)

    @property
    def fraction_removed(self) -> float:
        """Brief section 9.1's "support-expansion dependency" metric: the
        fraction of baseline solves that vanished under ablation."""
        if self.baseline.n_solved == 0:
            return 0.0
        return self.solves_removed / self.baseline.n_solved

    @property
    def load_bearing(self) -> bool:
        """Whether ablation removed solves at all, ignoring CI overlap.

        A cheap, structural signal -- `fraction_removed > 0` -- not a
        significance test. Use the two summaries' CIs directly for anything
        that needs a formal claim about the size of the effect.
        """
        return self.solves_removed > 0

    def summary_line(self) -> str:
        return (
            f"{self.task_id:26s} ablate={self.ablated_op:20s} "
            f"baseline={self.baseline.p_hat:.3f}({self.baseline.n_solved}/{self.baseline.n_reps})  "
            f"ablated={self.ablated.p_hat:.3f}({self.ablated.n_solved}/{self.ablated.n_reps})  "
            f"removed={self.solves_removed} ({self.fraction_removed:.0%})"
        )

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "ablated_op": self.ablated_op,
            "baseline": self.baseline.as_dict(),
            "ablated": self.ablated.as_dict(),
            "solves_removed": self.solves_removed,
            "fraction_removed": self.fraction_removed,
            "load_bearing": self.load_bearing,
        }


def run_ablation(
    ablated_op: str,
    baseline_scaffold: Scaffold,
    ablated_scaffold: Scaffold,
    task: Task,
    model: ModelClient,
    verifier: Verifier,
    budget_calls: int,
    n_reps: int = 30,
    confidence: float = 0.95,
) -> AblationResult:
    """Compare `baseline_scaffold` against `ablated_scaffold` on one task.

    The two scaffold instances must differ *only* in their ability to perform
    `ablated_op` -- constructing that pair correctly is the caller's
    responsibility (this function only measures and compares; see module
    docstring for how `S_star` and `S_evo` each construct such a pair).
    """
    harness = MatchedComputeHarness(BudgetCaps(calls=budget_calls))
    baseline = measure_scaffold_solve_rate(
        scaffold=baseline_scaffold,
        task=task,
        model=model,
        verifier=verifier,
        budget_calls=budget_calls,
        n_reps=n_reps,
        harness=harness,
        confidence=confidence,
        system_label="baseline",
    )
    ablated = measure_scaffold_solve_rate(
        scaffold=ablated_scaffold,
        task=task,
        model=model,
        verifier=verifier,
        budget_calls=budget_calls,
        n_reps=n_reps,
        harness=harness,
        confidence=confidence,
        system_label=f"ablated:{ablated_op}",
    )
    return AblationResult(
        task_id=task.task_id, ablated_op=ablated_op, baseline=baseline, ablated=ablated
    )
