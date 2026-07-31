"""Reference `AgentFunction` implementations.

These are not scaffolds `cbs` claims are competitive -- they exist to validate
`EvolvedScaffold`/`InterceptionSession` against known, simple behaviour before
either is trusted with a real forked loop's output. Each one exercises a
different classification path interception has to get right:

*   `single_call_agent` -- one model call, no verifier use at all. Should
    produce exactly one `single_call` operation and nothing else.
*   `blind_best_of_n_agent` -- draws several candidates, checks each against
    the verifier, returns the first that passes, but never lets a later prompt
    reference an earlier failure. Every verifier call should classify as
    `test_guided_selection` (preserving) -- proof the classifier does not
    default to calling every verifier use "expanding".
*   `feedback_repair_agent` -- draws a candidate, and on failure builds its
    *next* prompt out of the verifier's own error text. Those verifier calls
    should classify as `execution_feedback` (expanding) -- proof the classifier
    catches real conditioning rather than defaulting to "everything is safe".
*   `crashing_agent` -- raises partway through, to exercise `EvolvedScaffold`'s
    handling of untrusted code that fails outright.
"""

from __future__ import annotations

from cbs.budget import BudgetAccountant, BudgetExceeded
from cbs.models.base import CompletionRequest, ModelClient
from cbs.scaffolds.evolved import AgentVerifier
from cbs.tasks.schema import Task
from cbs.tasks.verifier import extract_code

__all__ = [
    "single_call_agent",
    "blind_best_of_n_agent",
    "feedback_repair_agent",
    "crashing_agent",
]


def single_call_agent(
    task: Task,
    model: ModelClient,
    verifier: AgentVerifier,
    accountant: BudgetAccountant,
    seed: int | None,
) -> str:
    request = CompletionRequest(
        prompt=task.prompt, temperature=0.8, seed=seed, meta={"task_id": task.task_id}
    )
    completion = model.complete(request, accountant)
    return extract_code(completion.text)


def blind_best_of_n_agent(
    task: Task,
    model: ModelClient,
    verifier: AgentVerifier,
    accountant: BudgetAccountant,
    seed: int | None,
    max_candidates: int = 4,
) -> str:
    last_code = ""
    base_seed = (seed or 0) * 1000
    for i in range(max_candidates):
        request = CompletionRequest(
            prompt=task.prompt,  # always the ORIGINAL prompt -- never touched by feedback
            temperature=0.8,
            seed=base_seed + i,
            meta={"task_id": task.task_id},
        )
        try:
            completion = model.complete(request, accountant)
        except BudgetExceeded:
            break
        code = extract_code(completion.text)
        last_code = code
        result = verifier.verify_code(task, code)
        if result.passed:
            return code
    return last_code


def feedback_repair_agent(
    task: Task,
    model: ModelClient,
    verifier: AgentVerifier,
    accountant: BudgetAccountant,
    seed: int | None,
    max_repairs: int = 3,
) -> str:
    base_seed = (seed or 0) * 1000
    prompt = task.prompt
    code = ""
    for i in range(max_repairs + 1):
        request = CompletionRequest(
            prompt=prompt, temperature=0.8, seed=base_seed + i, meta={"task_id": task.task_id}
        )
        try:
            completion = model.complete(request, accountant)
        except BudgetExceeded:
            break
        code = extract_code(completion.text)
        result = verifier.verify_code(task, code)
        if result.passed:
            return code
        error_text = (
            result.exec_result.stderr[-800:]
            if result.exec_result and result.exec_result.stderr
            else result.reason
        )
        # The defining move: the NEXT prompt is built from THIS failure's
        # error text -- exactly the behavioural evidence InterceptionSession
        # looks for to classify this verifier call as execution_feedback.
        prompt = (
            f"{task.prompt}\n\nPrevious attempt:\n{code}\n\n"
            f"It failed with:\n{error_text}\n\nFix it."
        )
    return code


def crashing_agent(
    task: Task,
    model: ModelClient,
    verifier: AgentVerifier,
    accountant: BudgetAccountant,
    seed: int | None,
) -> str:
    raise RuntimeError("this evolved variant is broken")
