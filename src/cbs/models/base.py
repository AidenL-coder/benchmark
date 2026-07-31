"""Provider-agnostic interface to the frozen foundation model `M`.

Everything in this project treats `M` as frozen and reaches it only through
`ModelClient`. Swapping vLLM for Ollama for a hosted API is a config change, not
a code change (docs/DECISIONS.md D-01).

Every call is charged to a `BudgetAccountant`, so matched-compute comparisons are
enforced at the only place model compute can be spent.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from cbs.budget import BudgetAccountant, Usage

__all__ = [
    "CompletionRequest",
    "Completion",
    "ModelClient",
    "ModelUnavailable",
]


class ModelUnavailable(RuntimeError):
    """The configured backend cannot be reached (missing dep, server down, ...)."""


@dataclass(frozen=True)
class CompletionRequest:
    """One sampling request against the frozen model.

    `seed` is threaded through explicitly rather than left to global RNG state:
    the frontier estimator draws thousands of samples per task and reproducibility
    of *which* samples were drawn is a stated requirement (brief section 10).
    """

    prompt: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int | None = None
    stop: tuple[str, ...] = ()
    system: str | None = None
    # Free-form routing hints (e.g. task_id) used by the mock backend and logs.
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    """One sampled completion plus the compute it cost."""

    text: str
    usage: Usage
    finish_reason: str = "stop"
    model_id: str = ""
    seed: int | None = None
    meta: dict = field(default_factory=dict)


class ModelClient(abc.ABC):
    """A frozen model. Implementations must not mutate model weights (obviously)
    and must not carry state across calls -- `M` is memoryless here, so that all
    adaptation lives in the scaffold, which is the object of study."""

    #: Stable identifier recorded in every frontier record.
    model_id: str = "unset"

    @abc.abstractmethod
    def _generate(self, request: CompletionRequest) -> Completion:
        """Backend-specific generation. Must populate `Completion.usage`."""

    def complete(
        self, request: CompletionRequest, accountant: BudgetAccountant
    ) -> Completion:
        """Sample once, charging the cost to `accountant`.

        The charge happens *after* generation because true token counts are only
        known then. To keep a cap from being overshot by the final call, callers
        that are near a limit should pre-check with `accountant.can_afford(...)`;
        `cbs.frontier.sampler` does exactly this.
        """
        completion = self._generate(request)
        accountant.charge(completion.usage)
        return completion

    def describe(self) -> dict:
        return {"model_id": self.model_id, "backend": type(self).__name__}


def estimate_tokens(text: str) -> int:
    """Crude token estimate used only where a backend reports no usage.

    Roughly 4 characters per token. Any record produced with estimated rather
    than reported usage is flagged, because matched-compute claims built on
    estimates are weaker than ones built on reported counts.
    """
    return max(1, len(text) // 4)
