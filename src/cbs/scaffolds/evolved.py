"""`EvolvedScaffold` -- the `S_evo` integration point (Phase 4).

Brief section 5.1: fork an existing DGM-style loop; do not rebuild it. Section
5.2 lists the measurement layer -- "support-tagging instrumentation on the
scaffold" among it -- as the actual contribution. This module is that boundary:
where a forked loop's agent code plugs into `cbs`'s measurement machinery,
not a reimplementation of the loop itself.

Why nothing here runs a real evolution loop yet
------------------------------------------------
A DGM-style loop executes model-generated, **self-modifying** scaffold code
that must itself call back out to a live model endpoint. That is a different
trust boundary than verifying a candidate *solution* (`cbs.tasks.verifier`): a
solution's blast radius is one process, run once, against fixed tests. An
evolved *scaffold*'s blast radius is arbitrary orchestration code with live
network access, executed repeatedly across generations, indefinitely. Brief
section 10 makes Docker mandatory for exactly this reason.

This machine has no Docker (`docs/DECISIONS.md` D-02), and Colab cannot nest
Docker either -- so there is currently nowhere in this project's
infrastructure that can safely run real self-modifying scaffold code. See D-23.
What is safe to build now -- and is built here -- is the adapter contract any
forked loop's agents must satisfy, validated against synthetic agent functions
that make ordinary, non-self-modifying calls. None of it requires trusting
self-modifying code, and all of it is exactly what plugs in unchanged once a
Docker-capable, GPU-capable host is available.

Tagging by interception, not self-report
------------------------------------------
`S0` and `S_star` are scaffolds *we* author, so trusting their own
`trace.record(...)` calls is reasonable -- we wrote the control flow. `S_evo`'s
agent code is the opposite: it is produced by an optimisation process whose
objective is "solve more benchmark tasks", not "report its own operations
honestly". An agent that discovered it could skip tagging an expanding
operation to look more like bounded search would have no reason not to. So
`S_evo` cannot rely on `OperationTrace.record()` called from within the
untrusted code at all -- classification has to happen from *outside*, by
intercepting the only channels through which the agent can act (querying the
frozen model, running/verifying code) and classifying each intercepted call
from what was actually observed, never from what the agent claims.

The one genuine ambiguity interception cannot resolve by construction is
selection vs. conditioning: a verifier call's pass/fail result might be used
only to *choose* among already-generated candidates (`test_guided_selection`,
preserving) or to *condition the next generation* on the failure
(`execution_feedback`, expanding) -- both look identical as "a sandboxed run
happened". `InterceptionSession` resolves this with behavioural evidence
instead of a blind default: if a later prompt to the model contains a
substantial fragment of an earlier verification's error text, that verifier
call is classified as feedback; otherwise it is classified as selection. A
default in either direction would be worse than this -- always-expanding makes
the study's central claim unfalsifiable (everything "depends on expansion"
trivially), and always-preserving would systematically hide real expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from cbs.budget import BudgetAccountant, BudgetExceeded, Usage
from cbs.models.base import Completion, CompletionRequest, ModelClient
from cbs.scaffolds.base import Scaffold, ScaffoldResult
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.schema import Task
from cbs.tasks.verifier import VerificationResult, Verifier, extract_code

__all__ = [
    "AgentFunction",
    "AgentVerifier",
    "ArchiveEntry",
    "EvolvedScaffold",
    "InterceptionSession",
]

#: Minimum length of an error-text fragment that counts as evidence of
#: conditioning. Shorter overlaps ("Error", "at line") are common enough by
#: coincidence that treating them as evidence would manufacture false
#: "execution_feedback" classifications.
MIN_FEEDBACK_MATCH_LEN = 24


@dataclass(frozen=True)
class ArchiveEntry:
    """Identity and provenance of one evolved scaffold variant.

    Mirrors DGM's archive-of-agents concept (proposal section 2.1) without
    committing to any one fork's on-disk format -- `source_excerpt` is kept
    short and is what `cbs.archive.hard_coding_heuristic` scans.
    """

    variant_id: str
    generation: int
    parent_id: str | None = None
    description: str = ""
    source_excerpt: str = ""


class AgentVerifier(Protocol):
    """Duck-typed subset of `Verifier` that intercepted agent code sees."""

    def verify_code(self, task: Task, code: str) -> VerificationResult: ...
    def verify_completion(self, task: Task, completion_text: str) -> VerificationResult: ...


#: `(task, model, verifier, accountant, seed) -> candidate source code`.
#: The forked loop's evolved agent, reduced to the minimal call shape cbs needs.
AgentFunction = Callable[
    [Task, ModelClient, AgentVerifier, BudgetAccountant, "int | None"], str
]


@dataclass
class _ModelCallEvent:
    seq: int
    prompt: str


@dataclass
class _VerifierCallEvent:
    seq: int
    passed: bool
    error_text: str


class InterceptionSession:
    """Captures every model/verifier call an agent function makes and
    classifies each one after the fact from what was actually observed.

    Construct one per `solve()` attempt; hand `model_client`/`verifier` to the
    untrusted agent function instead of the real objects.
    """

    def __init__(
        self,
        model: ModelClient,
        verifier: Verifier,
        disabled_ops: frozenset[str] = frozenset(),
    ):
        self._model_events: list[_ModelCallEvent] = []
        self._verifier_events: list[_VerifierCallEvent] = []
        self._seq = 0
        self.total_usage = Usage()
        #: Ablation: operations the agent is denied, not merely operations we
        #: hope its code happens not to use. See `_InterceptingVerifier`.
        self.disabled_ops = disabled_ops
        self.model_client: ModelClient = _InterceptingModelClient(model, self)
        self.verifier: AgentVerifier = _InterceptingVerifier(verifier, self)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _on_model_call(self, request: CompletionRequest, completion: Completion) -> None:
        self._model_events.append(_ModelCallEvent(self._next_seq(), request.prompt))
        self.total_usage = self.total_usage + completion.usage

    def _on_verifier_call(self, result: VerificationResult) -> None:
        error_text = ""
        if not result.passed:
            error_text = (
                result.exec_result.stderr[-800:]
                if result.exec_result and result.exec_result.stderr
                else result.reason
            )
        self._verifier_events.append(
            _VerifierCallEvent(self._next_seq(), result.passed, error_text)
        )

    def finalize(self, trace: OperationTrace) -> None:
        """Classify every buffered event and append it to `trace`, in the
        order the calls actually happened."""
        entries: list[tuple[int, str, object]] = [
            (e.seq, "model", e) for e in self._model_events
        ] + [(e.seq, "verifier", e) for e in self._verifier_events]
        entries.sort(key=lambda item: item[0])

        for _, kind, event in entries:
            if kind == "model":
                trace.record_instant("single_call", intercepted=True)
                continue
            conditioned = self._was_conditioned_on(event)
            op_name = "execution_feedback" if conditioned else "test_guided_selection"
            trace.record_instant(
                op_name, intercepted=True, passed=event.passed, conditioned=conditioned
            )

    def _was_conditioned_on(self, event: _VerifierCallEvent) -> bool:
        if not event.error_text or len(event.error_text) < MIN_FEEDBACK_MATCH_LEN:
            return False
        fragment = event.error_text[:MIN_FEEDBACK_MATCH_LEN]
        return any(
            fragment in later.prompt
            for later in self._model_events
            if later.seq > event.seq
        )


class _InterceptingModelClient(ModelClient):
    """Routes every `complete()` through the session; `_generate()` is
    intentionally unreachable so nothing can bypass interception."""

    def __init__(self, inner: ModelClient, session: InterceptionSession):
        self._inner = inner
        self._session = session
        self.model_id = inner.model_id

    def _generate(self, request: CompletionRequest) -> Completion:
        raise RuntimeError(
            "_InterceptingModelClient routes through complete(), not _generate(); "
            "if this fired, something bypassed interception"
        )

    def complete(self, request: CompletionRequest, accountant: BudgetAccountant) -> Completion:
        completion = self._inner.complete(request, accountant)
        self._session._on_model_call(request, completion)
        return completion

    def describe(self) -> dict:
        return {**self._inner.describe(), "intercepted": True}


class _InterceptingVerifier:
    """Duck-types `Verifier`'s agent-facing surface; every call is logged to
    the session for post-hoc classification before being returned.

    If `"execution_feedback"` is in the session's `disabled_ops`, the error
    detail behind a failing verification is withheld from whatever is *returned
    to the agent* -- the real result (with its real error text) is still what
    gets recorded for the session's own bookkeeping, so an ablation run's
    records honestly show what the agent *could have* conditioned on, while the
    agent itself genuinely never receives that text and so cannot embed it in a
    later prompt. That is what makes this a causal ablation (the agent is
    denied the capability) rather than a hope that its code happens not to
    exercise it.
    """

    def __init__(self, inner: Verifier, session: InterceptionSession):
        self._inner = inner
        self._session = session
        self.sandbox = inner.sandbox
        self.SUCCESS_MARKER = inner.SUCCESS_MARKER

    def verify_code(self, task: Task, code: str) -> VerificationResult:
        result = self._inner.verify_code(task, code)
        self._session._on_verifier_call(result)
        if "execution_feedback" in self._session.disabled_ops and not result.passed:
            return VerificationResult(
                passed=result.passed,
                reason="ablated: error detail withheld",
                exec_result=None,
                extracted_code=result.extracted_code,
            )
        return result

    def verify_completion(self, task: Task, completion_text: str) -> VerificationResult:
        return self.verify_code(task, extract_code(completion_text))


class EvolvedScaffold(Scaffold):
    """Bridges one evolved agent function into `cbs`'s measurement layer.

    All model and verifier access the agent function performs is routed
    through an `InterceptionSession`, so support-class tagging does not depend
    on the agent's own cooperation (see module docstring). The hidden oracle is
    queried exactly once at the end to score the final answer, identical to
    `S0`/`S_star` -- the agent function only ever sees the intercepted verifier.
    """

    def __init__(
        self,
        agent_fn: AgentFunction,
        archive_entry: ArchiveEntry,
        name: str | None = None,
        disabled_ops: frozenset[str] = frozenset(),
    ):
        self.agent_fn = agent_fn
        self.archive_entry = archive_entry
        self.name = name or f"S_evo:{archive_entry.variant_id}"
        #: Ablation set (cbs.ablation): operations withheld from this instance
        #: regardless of what the agent function's own code tries to do.
        self.disabled_ops = disabled_ops

    def config_fingerprint(self) -> dict:
        return {
            "name": self.name,
            "variant_id": self.archive_entry.variant_id,
            "generation": self.archive_entry.generation,
            "parent_id": self.archive_entry.parent_id,
            "disabled_ops": sorted(self.disabled_ops),
        }

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
        if verifier is None:
            raise ValueError(
                "EvolvedScaffold requires a verifier: interception wraps it so "
                "the agent function never touches the hidden oracle directly"
            )

        trace = OperationTrace()
        session = InterceptionSession(model, verifier, disabled_ops=self.disabled_ops)
        solution = ""
        budget_exhausted = False
        error: str | None = None

        try:
            solution = self.agent_fn(
                task, session.model_client, session.verifier, accountant, seed
            ) or ""
        except BudgetExceeded as exc:
            budget_exhausted = len(session._model_events) == 0
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 -- agent_fn is untrusted, third-party code
            error = f"{type(exc).__name__}: {exc}"

        session.finalize(trace)

        verification = None
        if solution:
            # Exactly one hidden-oracle query, on the agent's final answer --
            # measurement deciding correctness, not something the agent can see.
            verification = verifier.verify_code(task, solution)

        return ScaffoldResult(
            task_id=task.task_id,
            solution=solution,
            trace=trace,
            usage=session.total_usage,
            verification=verification,
            budget_exhausted=budget_exhausted,
            error=error,
            metadata={
                "variant_id": self.archive_entry.variant_id,
                "generation": self.archive_entry.generation,
                "seed": seed,
            },
        )
