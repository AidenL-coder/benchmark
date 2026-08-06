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
engineering, not a design mistake to route around quietly. Best-of-N
without execution feedback (`max_candidates` independent trajectories,
self-consistency over whichever ones the hidden oracle -- queried once per
candidate at most, or once overall if candidates are pre-filtered some
other way -- confirms) is a smaller, well-defined next step if a Polyglot
`S_star` analogue is wanted before that surgery is done; not built here
either, to avoid a half-finished implementation.
"""

from __future__ import annotations

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
