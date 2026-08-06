"""`S0`/`S_star` on SWE-bench Verified (D-40).

Concrete implementation of D-40's design: `cbs.tasks.swebench.SweBenchInstance`
is not a `Task`, so `S0`/`S_star` (`cbs.scaffolds.s0`/`s_star`) do not apply
directly -- a real repository-editing task cannot be solved by one blind
completion, and grading is a real Docker-based test run, not an in-process
assert. This module is the SWE-bench-flavored analogue of those two
scaffolds, following the same *injected function* pattern `cbs.scaffolds.
evolved.EvolvedScaffold` already established for agent code this project
does not implement itself: everything Docker/HGM-specific (building a
container, copying agent code in, running it, applying/testing a patch)
stays behind `SweBenchAgentFunction`/`SweBenchVerifyFunction`, supplied by
the caller (a script living outside `cbs`, exactly like `fork_bridge`'s own
`scripts/hgm_run_task_with_interception.py`). Everything in this module is
plain Python, unit-testable with synthetic functions, the same way
`tests/test_evolved.py` tests `EvolvedScaffold` without a real fork.

**The atomic unit**: one call to `SweBenchAgentFunction` is one full agent
trajectory (e.g. one call to HGM's/HyperAgents' own `chat_with_agent`,
already exercised extensively this session) -- the natural generalization
of `S0`'s "one call to `M`" for a task whose minimum viable execution unit
is a whole tool-using conversation, not a single completion (D-40).

**A real, honest limitation this task family has that `S0`/`S_star` do not**:
`cbs.budget.BudgetAccountant` gates a raw completion *before* it happens
(`model.complete(request, accountant)` checks affordability first). A
Docker-run agent trajectory has already spent whatever it spent by the time
`SweBenchAgentFunction` returns -- there is no way to pre-check "can I
afford one more trajectory" the way `s_star.py` pre-checks "can I afford one
more `single_call`", because a trajectory's real cost is not known until it
has run. `accountant.charge(attempt.usage)` here is therefore always
*post-hoc*: it can still raise `BudgetExceeded` and stop further candidates/
repairs from starting, but it cannot prevent one already-running trajectory
from overshooting the allowance. This is a genuine measurement-fidelity gap
for this family, not an oversight to quietly work around -- report it
alongside any matched-compute claim that uses this scaffold.

**Self-consistency uses literal diff-text matching**, not canonicalized
structure the way `s_star.py`'s `_select_by_consensus` canonicalizes function
bodies (`cbs.tasks.canonicalize`). Two differently-worded patches that fix
the same bug in different ways will never be treated as "the same answer"
here, unlike two syntactically-different-but-equivalent function bodies for
`S_star`. A weaker approximation, not a silent one -- worth revisiting if
self-consistency's hit rate on real data turns out to matter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from cbs.budget import BudgetAccountant, BudgetExceeded, Usage
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.swebench import SweBenchInstance

__all__ = [
    "SweBenchAttempt",
    "SweBenchVerifyResult",
    "SweBenchResult",
    "SweBenchAgentFunction",
    "SweBenchVerifyFunction",
    "S0SweBench",
    "SStarSweBench",
]


@dataclass(frozen=True)
class SweBenchAttempt:
    """Result of running one agent trajectory against a SWE-bench instance.

    `trace` is that trajectory's *own* operations (`tool_call`/`single_call`,
    from `fork_bridge.reconstruct_trace_from_events` in the real deployment)
    -- this module merges it into the scaffold-level trace rather than
    replacing it, so a `SStarSweBench` run's trace shows both what each
    trajectory did internally and what the scaffold did across trajectories
    (`execution_feedback`, `self_consistency`).
    """

    diff: str
    trace: OperationTrace
    usage: Usage
    error: str = ""


@dataclass(frozen=True)
class SweBenchVerifyResult:
    """Result of running a specific set of tests against a produced diff."""

    passed: bool
    failed_tests: tuple[str, ...] = ()
    raw_output: str = ""


#: `(instance, problem_statement) -> SweBenchAttempt`. `problem_statement` is
#: passed explicitly, not read off `instance`, so a repair attempt can
#: substitute an augmented prompt (original + previous diff + failure text)
#: without this module knowing anything about how the trajectory actually
#: runs.
SweBenchAgentFunction = Callable[[SweBenchInstance, str], SweBenchAttempt]

#: `(instance, diff, test_node_ids) -> SweBenchVerifyResult`. Also
#: Docker-backed (applies the diff to a fresh checkout and runs exactly the
#: given tests) but injected for the same reason.
SweBenchVerifyFunction = Callable[
    [SweBenchInstance, str, "tuple[str, ...]"], SweBenchVerifyResult
]


@dataclass
class SweBenchResult:
    instance_id: str
    solution: str  # the chosen diff, "" if none was produced
    trace: OperationTrace
    usage: Usage
    verification: SweBenchVerifyResult | None = None
    budget_exhausted: bool = False
    error: str | None = None
    metadata: dict = field(default_factory=dict)


def _merge_trace(into: OperationTrace, sub: OperationTrace) -> None:
    """Append every record of `sub` onto `into`, re-tagged through
    `record_instant` rather than copied verbatim -- `sub`'s own `seq` values
    are only meaningful within `sub`'s independently-numbered record list,
    so reusing them here would produce duplicate/out-of-order sequence
    numbers in the combined trace."""
    for r in sub.records:
        into.record_instant(r.name, r.duration_s, **r.metadata)


class S0SweBench:
    """`S0`'s SWE-bench Verified analogue (D-40).

    One agent trajectory, no retries, no repair, no selection -- whatever
    diff it produces is submitted as-is. Mirrors `cbs.scaffolds.s0.S0`'s "one
    call, no repair" spirit exactly, generalized to an atomic unit that is a
    whole trajectory rather than a single completion.
    """

    name = "S0_swebench"

    def config_fingerprint(self) -> dict:
        return {"name": self.name}

    def solve(
        self,
        instance: SweBenchInstance,
        agent_fn: SweBenchAgentFunction,
        accountant: BudgetAccountant,
        verify_fn: SweBenchVerifyFunction | None = None,
        *,
        seed: int | None = None,
    ) -> SweBenchResult:
        trace = OperationTrace()
        attempt = agent_fn(instance, instance.problem_statement)
        _merge_trace(trace, attempt.trace)

        budget_exhausted = False
        try:
            accountant.charge(attempt.usage)
        except BudgetExceeded:
            # Post-hoc: the trajectory already ran and already spent this --
            # see module docstring. The charge attempt still surfaces the
            # overrun rather than silently absorbing it.
            budget_exhausted = True

        verification = None
        if verify_fn is not None and attempt.diff and not attempt.error:
            # Exactly one hidden-oracle (FAIL_TO_PASS) query, on the only
            # candidate there is -- identical in spirit to S0's single
            # hidden-test check.
            verification = verify_fn(instance, attempt.diff, instance.fail_to_pass)

        return SweBenchResult(
            instance_id=instance.instance_id,
            solution=attempt.diff,
            trace=trace,
            usage=attempt.usage,
            verification=verification,
            budget_exhausted=budget_exhausted,
            error=attempt.error or None,
            metadata={"seed": seed},
        )


class SStarSweBench:
    """`S_star`'s SWE-bench Verified analogue (D-40).

    *   **Best-of-N** -- up to `max_candidates` independent agent
        trajectories, each from a fresh checkout.
    *   **Execution feedback** -- after each trajectory, run its diff
        against `instance.pass_to_pass` only, never `fail_to_pass` (D-40's
        confirmed `Task.public_tests`/`Task.tests` analogue). On failure,
        feed the failure back as an augmented problem statement for a
        repair attempt, up to `max_repairs_per_candidate` times.
    *   **Standard tool use** -- already present, not a separate mechanism:
        the trajectory's own bash/edit tool use (tagged inside `attempt.
        trace`) fills the role `S_star`'s bolt-on compile check plays for
        atomic-function tasks.
    *   **Self-consistency** -- majority vote by literal diff text among
        whichever candidates passed `pass_to_pass` (falling back to all
        candidates if none did), mirroring `s_star.py`'s own fallback
        exactly.

    Same rule as `S_star`: the hidden oracle (`fail_to_pass`) is queried
    exactly once, on the final chosen candidate.
    """

    name = "S_star_swebench"

    def __init__(
        self,
        max_candidates: int = 4,
        max_repairs_per_candidate: int = 2,
        stop_on_first_public_pass: bool = True,
    ):
        self.max_candidates = max_candidates
        self.max_repairs_per_candidate = max_repairs_per_candidate
        #: See `s_star.py`'s identical field: a real deployed agent would
        #: not keep spending once it has a candidate that passes what it can
        #: check. Set False to force full budget consumption for a stricter
        #: matched-compute comparison.
        self.stop_on_first_public_pass = stop_on_first_public_pass

    def config_fingerprint(self) -> dict:
        return {
            "name": self.name,
            "max_candidates": self.max_candidates,
            "max_repairs_per_candidate": self.max_repairs_per_candidate,
            "stop_on_first_public_pass": self.stop_on_first_public_pass,
        }

    @staticmethod
    def _repair_problem_statement(instance: SweBenchInstance, diff: str, failure: str) -> str:
        return (
            f"{instance.problem_statement}\n\n"
            "Your previous attempt produced this patch:\n"
            f"```diff\n{diff}\n```\n\n"
            "Running the repository's existing test suite against it produced "
            f"this failure:\n{failure}\n\n"
            "Fix the patch so the existing tests keep passing while still "
            "addressing the original issue."
        )

    @staticmethod
    def _select_by_consensus(diffs: list[str]) -> str:
        """Majority vote by literal diff text -- see module docstring for
        why this is a weaker approximation than `s_star.py`'s canonicalized
        version, not an oversight."""
        if not diffs:
            return ""
        counts = Counter(diffs)
        winning = max(counts, key=lambda d: counts[d])
        for d in diffs:  # earliest candidate in the winning cluster
            if d == winning:
                return d
        return diffs[0]  # unreachable

    def solve(
        self,
        instance: SweBenchInstance,
        agent_fn: SweBenchAgentFunction,
        accountant: BudgetAccountant,
        verify_fn: SweBenchVerifyFunction,
        *,
        seed: int | None = None,
    ) -> SweBenchResult:
        trace = OperationTrace()
        candidates: list[str] = []
        public_passing: list[str] = []
        total_usage = Usage()
        exhausted_before_any_candidate = False
        budget_exhausted = False
        last_agent_error = ""

        for candidate_idx in range(self.max_candidates):
            problem_statement = instance.problem_statement
            final_diff = ""
            final_public_ok: bool | None = None

            for step in range(self.max_repairs_per_candidate + 1):
                attempt = agent_fn(instance, problem_statement)
                _merge_trace(trace, attempt.trace)
                total_usage = total_usage + attempt.usage
                final_diff = attempt.diff
                if attempt.error:
                    last_agent_error = attempt.error

                try:
                    accountant.charge(attempt.usage)
                except BudgetExceeded:
                    # Post-hoc, same limitation as S0SweBench -- see module
                    # docstring. Stop spawning further attempts once caught.
                    budget_exhausted = True
                    if candidate_idx == 0 and step == 0:
                        exhausted_before_any_candidate = True
                    break

                if attempt.error or not attempt.diff:
                    break  # the trajectory itself failed; nothing to verify

                if not instance.pass_to_pass:
                    # No public signal for this instance -- same honest
                    # degradation s_star.py applies when task.public_tests
                    # is empty: no feedback available, take this attempt.
                    break

                with trace.record(
                    "execution_feedback", candidate=candidate_idx, step=step
                ):
                    result = verify_fn(instance, attempt.diff, instance.pass_to_pass)
                final_public_ok = result.passed

                if result.passed:
                    break
                if step == self.max_repairs_per_candidate:
                    break  # out of repair budget for this candidate
                if budget_exhausted:
                    break

                with trace.record(
                    "adaptive_prompt_rewrite", candidate=candidate_idx, step=step
                ):
                    problem_statement = self._repair_problem_statement(
                        instance,
                        attempt.diff,
                        result.raw_output or "; ".join(result.failed_tests),
                    )

            candidates.append(final_diff)
            if final_public_ok:
                public_passing.append(final_diff)

            if budget_exhausted:
                break
            if self.stop_on_first_public_pass and final_public_ok:
                break

        real_candidates = [c for c in candidates if c]
        if not real_candidates:
            if exhausted_before_any_candidate:
                error = "budget exhausted before any candidate was generated"
            elif last_agent_error:
                error = f"no candidate produced a usable diff; last trajectory error: {last_agent_error}"
            else:
                error = "no candidate produced a usable diff"
            return SweBenchResult(
                instance_id=instance.instance_id,
                solution="",
                trace=trace,
                usage=total_usage,
                budget_exhausted=exhausted_before_any_candidate,
                error=error,
            )

        pool = public_passing or real_candidates
        with trace.record(
            "self_consistency",
            n_candidates=len(real_candidates),
            n_public_passing=len(public_passing),
        ):
            chosen = self._select_by_consensus(pool)

        # Exactly one hidden-oracle (FAIL_TO_PASS) query, on the final
        # choice only -- mirrors S0SweBench and s_star.py.
        verification = None
        if chosen:
            verification = verify_fn(instance, chosen, instance.fail_to_pass)

        return SweBenchResult(
            instance_id=instance.instance_id,
            solution=chosen,
            trace=trace,
            usage=total_usage,
            verification=verification,
            budget_exhausted=budget_exhausted,
            metadata={
                "seed": seed,
                "n_candidates": len(real_candidates),
                "n_public_passing": len(public_passing),
            },
        )
