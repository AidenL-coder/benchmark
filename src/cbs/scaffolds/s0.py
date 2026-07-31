"""`S0` -- the minimal fixed scaffold.

Brief section 3.2 defines the reachable-solution frontier in terms of `M`'s base
success probability "under a minimal fixed scaffold (single call plus trivial
formatting)". `S0` is that scaffold, and it is therefore load-bearing in a way
the other systems are not: *the frontier is defined relative to it*. If `S0`
quietly does anything clever, the estimated frontier moves and every crossing
claim moves with it.

So `S0` does exactly two things, both support-preserving:

1.  one call to `M` with the task's prompt, verbatim;
2.  deterministic extraction of the code block from the completion.

No retries. No repair. No selection. No feedback. Best-of-N over `S0` is
performed by the *sampler* (`cbs.frontier.sampler`), not by `S0` itself, so that
the per-sample unit stays a single call and `p_hat(x)` keeps its meaning.
"""

from __future__ import annotations

from cbs.budget import BudgetAccountant, BudgetExceeded, Usage
from cbs.models.base import CompletionRequest, ModelClient
from cbs.scaffolds.base import Scaffold, ScaffoldResult
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.schema import Task
from cbs.tasks.verifier import Verifier, extract_code

__all__ = ["S0"]

#: Kept minimal and fixed. Any change alters the measured frontier, so this
#: string is part of the experimental configuration and is hashed into results.
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful Python programmer. Respond with a single Python code "
    "block containing the requested function and nothing else."
)


class S0(Scaffold):
    name = "S0"

    def __init__(
        self,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

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
        trace = OperationTrace()
        temp = self.temperature if temperature is None else temperature

        request = CompletionRequest(
            prompt=task.prompt,
            system=self.system_prompt,
            temperature=temp,
            max_tokens=self.max_tokens,
            seed=seed,
            meta={"task_id": task.task_id},
        )

        raw = ""
        usage = Usage()
        try:
            with trace.record("single_call", temperature=temp, seed=seed):
                completion = model.complete(request, accountant)
                raw = completion.text
                usage = completion.usage
        except BudgetExceeded as exc:
            return ScaffoldResult(
                task_id=task.task_id,
                solution="",
                trace=trace,
                usage=usage,
                budget_exhausted=True,
                error=str(exc),
            )

        with trace.record("format_extract"):
            solution = extract_code(raw)

        verification = None
        if verifier is not None:
            # Verification is not a scaffold operation: it is the measurement
            # apparatus deciding correctness, not the agent choosing an output.
            # `S0` never sees the result, so it cannot act on it.
            verification = verifier.verify_code(task, solution)

        return ScaffoldResult(
            task_id=task.task_id,
            solution=solution,
            trace=trace,
            usage=usage,
            raw_completion=raw,
            verification=verification,
            metadata={"temperature": temp, "seed": seed},
        )

    def config_fingerprint(self) -> dict:
        """Everything about `S0` that could move the frontier."""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
