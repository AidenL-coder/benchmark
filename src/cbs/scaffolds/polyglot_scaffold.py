"""`S0` on the Polyglot benchmark (D-31/D-42).

Concrete counterpart to `cbs.scaffolds.swebench_scaffold` for HGM's *other*
confirmed native `S_evo` substrate (D-31). Follows the same injected-function
pattern as `cbs.scaffolds.evolved.EvolvedScaffold` and `swebench_scaffold`:
everything Docker/HGM-specific stays behind one injected callable, supplied
by a caller living outside `cbs` (e.g. a `scripts/polyglot_glue.py`, the
Polyglot analogue of `scripts/swebench_glue.py`).

**Only one injected function, not two** -- the one real structural
difference from `swebench_scaffold`. HGM's own `polyglot.harness.
process_entry` runs the agent trajectory *and* grades the result in a
single container, in one call (confirmed by reading it, D-42): it copies
the agent in, runs it to produce a patch, then -- still inside that same
container -- reveals the hidden test content via `git reset --hard
test_commit` and runs the real eval command, returning `eval_result`
directly. There is no natural seam here the way SWE-bench Verified's
harness had (D-40 had to call `docker_build`/`test_spec` directly to get
separate agent-only and verify-only containers) -- `process_entry` is
already atomic. So `S0Polyglot` needs only `PolyglotAgentFunction`, which
both runs the trajectory and reports the real grading verdict.

**Why there is no `SStarPolyglot` here yet**: `S_star`'s execution-feedback
repair loop needs a way to check a candidate against *some* signal before
the hidden oracle is queried, without querying the oracle itself early.
Polyglot's data has no separate public/hidden test split at the data level
(one `files["test"]` entry per instance, not `Task.tests`/`public_tests`,
D-42) -- oracle safety instead comes from the hidden test content only
being revealed by `test_commit`, inside `process_entry`'s own single
container, after the agent has already stopped. Building a genuine
mid-trajectory execution-feedback hook for Polyglot would need the same
kind of container-splitting surgery D-40 did for SWE-bench Verified's
harness (a separate agent-only container whose patch can be checked against
something before the real grading container runs) -- real, unbuilt
engineering, not a design mistake to route around quietly.

`SStarPolyglotBestOfN` (D-47) is the part that *can* be built without that
surgery: N independent trajectories with **oracle-blind** selection by
self-consistency. It is a genuine elicitation control -- how much of an
apparent gain is reachable by sampling the same frozen model harder, with no
evolution -- and it is what supplies the missing rung between `S0` and an
evolved scaffold.

**The oracle-safety subtlety that shapes its design.** Polyglot's
`process_entry` runs *and grades* a trajectory in one atomic call, so N
trajectories yield N grades whether or not we want them. Selecting the
candidate whose grade is best would be oracle-assisted and is *not* a
scaffold a real deployment could run. `select_by_consensus` therefore never
inspects `passed`: it votes on literal diff text alone. The per-candidate
grades are still recorded, because they license a second, clearly-separated
quantity:

*   ``resolved`` -- the oracle-blind scaffold's own result: did the
    self-consistency-selected candidate pass? This is the elicitation
    control proper, comparable to `S0`.
*   ``pass_at_n`` -- did *any* of the N trajectories pass? This is an
    upper bound on what *any* selection rule over the same N samples could
    achieve, i.e. a budget-relative estimate of the frozen model's reachable
    set. It must never be reported as scaffold performance; it is the
    ceiling that a perfect selector would hit, and the gap between it and
    ``resolved`` is exactly the headroom a better selection rule could win.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from cbs.budget import BudgetAccountant, BudgetExceeded, Usage
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.polyglot import PolyglotInstance

__all__ = [
    "PolyglotRunResult",
    "PolyglotResult",
    "PolyglotAgentFunction",
    "S0Polyglot",
    "SStarPolyglotBestOfN",
]


@dataclass(frozen=True)
class PolyglotRunResult:
    """Result of running one real agent-trajectory-plus-grading pass.

    `eval_result` is kept in HGM's own vocabulary verbatim
    (`"resolved"`/`"unresolved"`/`"empty_patch"`) rather than translated,
    so a caller inspecting a real result sees exactly what the real harness
    reported, not a `cbs`-invented relabeling.
    """

    solution: str  # the produced model_patch diff text, "" if none
    eval_result: str
    trace: OperationTrace
    usage: Usage
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.eval_result == "resolved"


@dataclass
class PolyglotResult:
    instance_id: str
    solution: str
    trace: OperationTrace
    usage: Usage
    eval_result: str = ""
    passed: bool = False
    budget_exhausted: bool = False
    error: str | None = None
    metadata: dict = field(default_factory=dict)


#: `(instance, problem_statement) -> PolyglotRunResult`. `problem_statement`
#: is threaded explicitly rather than read off `instance`, mirroring
#: `SweBenchAgentFunction` -- not used by `S0Polyglot` itself (which always
#: passes `instance.problem_statement` unchanged), but keeps the same seam
#: available if a future repair-capable scaffold needs to substitute an
#: augmented prompt.
PolyglotAgentFunction = Callable[[PolyglotInstance, str], PolyglotRunResult]


class S0Polyglot:
    """`S0`'s Polyglot analogue (D-31/D-42).

    One agent trajectory, no retries, no repair, no selection -- mirrors
    `cbs.scaffolds.s0.S0`'s "one call, no repair" spirit exactly. Unlike
    `S0SweBench`, there is no separate `verify_fn` to invoke afterward:
    `agent_fn` itself already returns the real grading verdict, since
    `process_entry` produces both in one atomic call (see module
    docstring).
    """

    name = "S0_polyglot"

    def config_fingerprint(self) -> dict:
        return {"name": self.name}

    def solve(
        self,
        instance: PolyglotInstance,
        agent_fn: PolyglotAgentFunction,
        accountant: BudgetAccountant,
        *,
        seed: int | None = None,
    ) -> PolyglotResult:
        run = agent_fn(instance, instance.problem_statement)

        budget_exhausted = False
        try:
            accountant.charge(run.usage)
        except BudgetExceeded:
            # Post-hoc, same limitation as S0SweBench/SStarSweBench (D-40):
            # a Docker-run trajectory's real cost isn't known until it has
            # already run, so this can only surface an overrun after the
            # fact, not prevent it.
            budget_exhausted = True

        return PolyglotResult(
            instance_id=instance.instance_id,
            solution=run.solution,
            trace=run.trace,
            usage=run.usage,
            eval_result=run.eval_result,
            passed=run.passed,
            budget_exhausted=budget_exhausted,
            error=run.error or None,
            metadata={"seed": seed},
        )


class SStarPolyglotBestOfN:
    """Elicitation control for Polyglot: N trajectories, oracle-blind
    selection by self-consistency (D-47).

    This is the rung between `S0Polyglot` and any evolved scaffold. It asks
    the question an evolved-scaffold result must be compared against: how
    much is reachable by sampling the *same frozen model* harder, with no
    evolution and no privileged information?

    **Selection never inspects the grade.** `_select_by_consensus` votes on
    literal diff text only. This is load-bearing rather than stylistic:
    Polyglot's harness grades every trajectory as a side effect of running
    it, so picking the best-scoring candidate would be trivial *and* would
    not correspond to any scaffold a real deployment could run. See the
    module docstring.

    **Two distinct numbers come out, and conflating them would be a serious
    error.** `passed` is what this oracle-blind scaffold actually achieved.
    `metadata["pass_at_n"]` is whether *any* of the N trajectories passed --
    an upper bound on what any selection rule over the same samples could
    reach, and a budget-relative estimate of the frozen model's reachable
    set. Only the former is scaffold performance.

    Like `s_star.py`'s own `_select_by_consensus`, matching is on literal
    text, so two different-but-equivalent diffs never cluster. That is a
    weaker approximation than canonicalised comparison, not a silent one.
    """

    name = "S_star_polyglot_bestofn"

    def __init__(self, n_candidates: int = 3):
        self.n_candidates = n_candidates

    def config_fingerprint(self) -> dict:
        return {"name": self.name, "n_candidates": self.n_candidates}

    @staticmethod
    def _select_by_consensus(diffs: list[str]) -> str:
        """Majority vote by literal diff text; earliest member of the
        winning cluster breaks ties. Deliberately blind to pass/fail."""
        real = [d for d in diffs if d]
        if not real:
            return ""
        counts = Counter(real)
        winning = max(counts, key=lambda d: counts[d])
        for d in real:
            if d == winning:
                return d
        return real[0]  # unreachable

    def solve(
        self,
        instance: PolyglotInstance,
        agent_fn: PolyglotAgentFunction,
        accountant: BudgetAccountant,
        *,
        seed: int | None = None,
    ) -> PolyglotResult:
        trace = OperationTrace()
        runs: list[PolyglotRunResult] = []
        total_usage = Usage()
        budget_exhausted = False

        for i in range(self.n_candidates):
            run = agent_fn(instance, instance.problem_statement)
            runs.append(run)
            total_usage = total_usage + run.usage
            for r in run.trace.records:
                trace.record_instant(r.name, r.duration_s, **r.metadata)
            try:
                accountant.charge(run.usage)
            except BudgetExceeded:
                budget_exhausted = True
                break

        if not runs:
            return PolyglotResult(
                instance_id=instance.instance_id,
                solution="",
                trace=trace,
                usage=total_usage,
                budget_exhausted=budget_exhausted,
                error="no trajectory ran",
                metadata={"seed": seed, "n_candidates": 0, "pass_at_n": False},
            )

        with trace.record("self_consistency", n_candidates=len(runs)):
            chosen = self._select_by_consensus([r.solution for r in runs])

        # The grade belonging to the candidate consensus actually picked.
        # Falls back to the first run only when no candidate produced a diff.
        selected = next((r for r in runs if r.solution and r.solution == chosen), runs[0])

        return PolyglotResult(
            instance_id=instance.instance_id,
            solution=chosen,
            trace=trace,
            usage=total_usage,
            eval_result=selected.eval_result,
            passed=selected.passed,
            budget_exhausted=budget_exhausted,
            error=None,
            metadata={
                "seed": seed,
                "n_candidates": len(runs),
                # Upper bound over the same samples -- NOT scaffold performance.
                "pass_at_n": any(r.passed for r in runs),
                "n_passing_candidates": sum(1 for r in runs if r.passed),
                "per_candidate_eval": [r.eval_result for r in runs],
            },
        )
