"""`S_star` -- the strong fixed baseline.

Brief section 4: "expert-designed generic agentic scaffold: best-of-N,
self-consistency, execution feedback, standard tool use -- held fixed, no
evolution. This is the critical control (represents 'known good scaffolding
without self-improvement')."

`S_star` is what an `S_evo` crossing must beat to count as genuine expansion
rather than automated rediscovery of tricks a human already knows (brief
section 9.3, "elicitation control"). Its four mechanisms and their support
classes:

*   **best-of-N** -- draw up to `max_candidates` independent completions
    (`single_call`, preserving).
*   **execution feedback** -- run each candidate against `task.public_tests`
    (never the hidden `tests`; see `Task.public_tests`) and, on failure, feed
    the error back for a repair attempt (`execution_feedback` /
    `adaptive_prompt_rewrite`, both expanding).
*   **standard tool use** -- a static compile/syntax check before spending a
    repair call on a candidate that does not even parse (`tool_call`,
    expanding; genuinely cheaper and more targeted than an execution result).
*   **self-consistency** -- majority vote by canonical form among the
    candidates that passed the public check, falling back to a vote over all
    candidates if none did (`self_consistency`, preserving).

Design choice that matters: **the hidden oracle is queried exactly once**, to
score the final chosen candidate -- identical to `S0`. Internal selection uses
only the public signal plus consensus among `M`'s own samples. A version that
let selection query the hidden tests would still be support-preserving (see
`tagging.py`, `test_guided_selection`), but it would make `S_star` an
unrealistic baseline: a deployed agent does not get to try candidates against
the grading suite and keep the first that happens to pass. Keeping that
channel closed means any elicitation gain `S_star` shows is earned the way a
real strong scaffold would earn it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from cbs.budget import BudgetAccountant, BudgetExceeded, Usage
from cbs.models.base import CompletionRequest, ModelClient
from cbs.scaffolds.base import Scaffold, ScaffoldResult
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.canonicalize import canonicalize_solution
from cbs.tasks.schema import Task
from cbs.tasks.verifier import Verifier, extract_code

__all__ = ["DEFAULT_SYSTEM_PROMPT", "SStar"]

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python programmer working carefully and iteratively. "
    "Respond with a single Python code block containing the requested "
    "function and nothing else."
)


class SStar(Scaffold):
    name = "S_star"

    def __init__(
        self,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        max_candidates: int = 4,
        max_repairs_per_candidate: int = 2,
        stop_on_first_public_pass: bool = True,
    ):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_candidates = max_candidates
        self.max_repairs_per_candidate = max_repairs_per_candidate
        #: A real deployed agent would not keep burning calls once it has a
        #: candidate that passes what it can check. Set False to force full
        #: budget consumption for a stricter matched-compute comparison.
        self.stop_on_first_public_pass = stop_on_first_public_pass

    def config_fingerprint(self) -> dict:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_candidates": self.max_candidates,
            "max_repairs_per_candidate": self.max_repairs_per_candidate,
            "stop_on_first_public_pass": self.stop_on_first_public_pass,
        }

    # -- tool use: static syntax check, no execution, no sandbox needed ----
    @staticmethod
    def _compiles(code: str) -> tuple[bool, str]:
        """`compile()` only parses and assembles bytecode; it never executes
        top-level code, so this is safe to run on untrusted text directly."""
        try:
            compile(code, "<candidate>", "exec")
            return True, ""
        except SyntaxError as exc:
            return False, f"{type(exc).__name__}: {exc}"
        except (ValueError, RecursionError) as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # -- execution feedback: run against the PUBLIC subset only ------------
    @staticmethod
    def _run_public_tests(
        task: Task, code: str, verifier: Verifier
    ) -> tuple[bool | None, str]:
        """`None` means no public signal exists for this task (see
        `Task.public_tests`); the compile check is all the feedback available."""
        if not task.public_tests.strip():
            return None, ""
        shadow = replace(task, tests=task.public_tests)
        result = verifier.verify_code(shadow, code)
        if result.passed:
            return True, ""
        err = result.exec_result.stderr[-800:] if result.exec_result else ""
        return False, err or result.reason

    @staticmethod
    def _repair_prompt(task: Task, code: str, error: str) -> str:
        return (
            f"{task.prompt}\n\n"
            "Your previous attempt:\n```python\n"
            f"{code}\n```\n\n"
            "Running it against example checks produced this error:\n"
            f"{error}\n\n"
            "Fix the function. Respond with only the corrected Python code block."
        )

    # -- self-consistency: majority vote by canonical form ------------------
    @staticmethod
    def _select_by_consensus(pool: list[str]) -> str:
        if not pool:
            return ""
        keyed = [(canonicalize_solution(c).key, c) for c in pool]
        counts = Counter(key for key, _ in keyed)
        winning_key = max(counts, key=lambda k: counts[k])
        for key, code in keyed:  # earliest candidate in the winning cluster
            if key == winning_key:
                return code
        return pool[0]  # unreachable

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
        seed_base = (seed if seed is not None else 0) * 1000
        call_ordinal = 0

        def next_seed() -> int:
            nonlocal call_ordinal
            value = seed_base + call_ordinal
            call_ordinal += 1
            return value

        candidates: list[str] = []
        public_passing: list[str] = []
        total_usage = Usage()
        exhausted_before_any_candidate = False

        for candidate_idx in range(self.max_candidates):
            if not accountant.can_afford(Usage(calls=1)):
                exhausted_before_any_candidate = candidate_idx == 0
                break

            request = CompletionRequest(
                prompt=task.prompt,
                system=self.system_prompt,
                temperature=temp,
                max_tokens=self.max_tokens,
                seed=next_seed(),
                meta={"task_id": task.task_id},
            )
            try:
                with trace.record("single_call", candidate=candidate_idx, kind="initial"):
                    completion = model.complete(request, accountant)
            except BudgetExceeded:
                break
            total_usage = total_usage + completion.usage
            with trace.record("format_extract", candidate=candidate_idx):
                code = extract_code(completion.text)

            final_public_ok: bool | None = None
            for step in range(self.max_repairs_per_candidate + 1):
                with trace.record(
                    "tool_call", tool="compile_check", candidate=candidate_idx, step=step
                ):
                    compiles, compile_err = self._compiles(code)

                public_ok: bool | None = None
                feedback_err = compile_err
                if compiles and verifier is not None:
                    with trace.record(
                        "execution_feedback", candidate=candidate_idx, step=step
                    ):
                        public_ok, pub_err = self._run_public_tests(task, code, verifier)
                    if public_ok is False:
                        feedback_err = pub_err
                final_public_ok = public_ok

                if compiles and public_ok is not False:
                    break  # passed the public check, or no public signal exists
                if step == self.max_repairs_per_candidate:
                    break  # out of repair budget for this candidate
                if not accountant.can_afford(Usage(calls=1)):
                    break

                with trace.record(
                    "adaptive_prompt_rewrite", candidate=candidate_idx, step=step
                ):
                    repair_prompt = self._repair_prompt(task, code, feedback_err)

                repair_request = CompletionRequest(
                    prompt=repair_prompt,
                    system=self.system_prompt,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                    seed=next_seed(),
                    meta={"task_id": task.task_id},
                )
                try:
                    with trace.record(
                        "single_call", candidate=candidate_idx, kind="repair", step=step
                    ):
                        completion = model.complete(repair_request, accountant)
                except BudgetExceeded:
                    break
                total_usage = total_usage + completion.usage
                with trace.record("format_extract", candidate=candidate_idx):
                    code = extract_code(completion.text)

            candidates.append(code)
            if final_public_ok:
                public_passing.append(code)

            if self.stop_on_first_public_pass and final_public_ok:
                break

        if not candidates:
            return ScaffoldResult(
                task_id=task.task_id,
                solution="",
                trace=trace,
                usage=total_usage,
                budget_exhausted=exhausted_before_any_candidate,
                error=(
                    "budget exhausted before any candidate was generated"
                    if exhausted_before_any_candidate
                    else None
                ),
            )

        pool = public_passing or candidates
        with trace.record(
            "self_consistency",
            n_candidates=len(candidates),
            n_public_passing=len(public_passing),
        ):
            chosen = self._select_by_consensus(pool)

        # Exactly one hidden-oracle query, on the final choice only -- mirrors
        # S0. This is measurement deciding correctness, not a scaffold
        # operation the algorithm observes and acts on.
        verification = None
        if verifier is not None and chosen:
            verification = verifier.verify_code(task, chosen)

        return ScaffoldResult(
            task_id=task.task_id,
            solution=chosen,
            trace=trace,
            usage=total_usage,
            verification=verification,
            metadata={
                "temperature": temp,
                "seed": seed,
                "n_candidates": len(candidates),
                "n_public_passing": len(public_passing),
            },
        )
