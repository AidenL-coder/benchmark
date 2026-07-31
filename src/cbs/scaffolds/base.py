"""Scaffold interface.

A scaffold `S` is a program that may query the frozen model `M`, execute code,
call tools and select among candidates, ultimately emitting a solution to a task
(proposal section 4). The three systems compared are all scaffolds:

*   `S0`     -- minimal fixed scaffold (this package, `s0.py`)
*   `S_star` -- strong fixed expert baseline (Phase 3)
*   `S_evo`  -- the evolved scaffold from the forked DGM-style loop (Phase 4)

Every scaffold must (a) draw all model compute from the accountant it is given,
so matched-compute comparisons hold, and (b) tag every operation it performs, so
solved tasks can be attributed to support classes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from cbs.budget import BudgetAccountant, Usage
from cbs.models.base import ModelClient
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.schema import Task
from cbs.tasks.verifier import VerificationResult, Verifier

__all__ = ["ScaffoldResult", "Scaffold"]


@dataclass
class ScaffoldResult:
    """One solve attempt: what was emitted, how it was reached, what it cost."""

    task_id: str
    solution: str
    trace: OperationTrace
    usage: Usage
    raw_completion: str = ""
    verification: VerificationResult | None = None
    #: Set when the attempt stopped early because the budget ran out. A failed
    #: attempt that ran out of budget is not evidence about capability, and the
    #: analysis must exclude it rather than score it as a miss.
    budget_exhausted: bool = False
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.verification and self.verification.passed)

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "budget_exhausted": self.budget_exhausted,
            "error": self.error,
            "usage": self.usage.as_dict(),
            "trace": self.trace.as_dict(),
            "verification": (
                self.verification.as_dict() if self.verification else None
            ),
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class Scaffold(abc.ABC):
    name: str = "unset"

    @abc.abstractmethod
    def solve(
        self,
        task: Task,
        model: ModelClient,
        accountant: BudgetAccountant,
        verifier: Verifier | None = None,
        *,
        seed: int | None = None,
        temperature: float | None = None,
    ) -> ScaffoldResult:
        """Attempt `task`, charging all model compute to `accountant`.

        Implementations must not raise on ordinary failure (a wrong answer, a
        budget exhaustion, a malformed completion); those are recorded on the
        result. Raising is reserved for programming errors and for
        `UnregisteredOperation`, which signals the tagging invariant was broken.
        """

    def describe(self) -> dict:
        return {"name": self.name, "class": type(self).__name__}
