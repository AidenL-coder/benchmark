"""Deterministic mock frozen model with *known* ground truth.

Why this exists
---------------
Phase 2's Definition of Done is: "on a toy task with known ground-truth solve
rate, estimator recovers it within CI". That requires a model whose true `p(x)`
and whose true set of distinct correct solutions are known exactly. No real model
can provide that, so the estimator validation runs against this backend.

Crucially the mock emits *real source code* drawn from per-task pools, so the
whole pipeline is exercised end to end -- sample, execute in the sandbox, verify
against tests, canonicalise, count species -- rather than short-circuiting the
parts under test. A mock that returned "correct"/"incorrect" labels would
validate the arithmetic while leaving the pipeline unvalidated.

Determinism
-----------
Sample `i` for task `x` under global seed `s` is a pure function of
`(s, x, i, temperature)`. Python's builtin `hash()` is salted per process and is
therefore unusable here; we derive seeds with BLAKE2b instead.
"""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass, field

from cbs.budget import Usage
from cbs.models.base import Completion, CompletionRequest, ModelClient, estimate_tokens

__all__ = ["MockTaskBehaviour", "MockModelClient"]


def _derive_seed(*parts: object) -> int:
    """Stable 64-bit seed from arbitrary parts (process-independent)."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return struct.unpack("<Q", h.digest())[0]


@dataclass
class MockTaskBehaviour:
    """Ground truth for one task under the mock model.

    Attributes
    ----------
    p_correct:
        The true per-sample probability of emitting a correct solution. This is
        the quantity the frontier estimator must recover.
    correct_variants:
        Distinct correct solutions ("species"). Their number is the true species
        richness that Chao1 must recover.
    incorrect_variants:
        Distinct wrong solutions, emitted when the draw comes up incorrect.
    species_weights:
        Relative probabilities over `correct_variants`, conditional on being
        correct. Defaults to uniform. Skewed weights are the interesting case for
        Good-Turing: they produce singletons and therefore non-zero unseen mass.
    temperature_sharpening:
        If True, species weights are raised to the power `1/T` (renormalised), so
        low temperature concentrates on the modal solution and high temperature
        flattens the distribution. This makes the temperature schedule
        (brief section 5.4, "tuned for solution diversity") a live variable rather
        than a no-op. `p_correct` is deliberately *unaffected* by temperature so
        that the ground truth under validation stays unambiguous; set
        `p_correct_temperature_exponent` to model the accuracy/diversity tradeoff
        when that realism is wanted instead.
    """

    p_correct: float
    correct_variants: list[str]
    incorrect_variants: list[str] = field(default_factory=list)
    species_weights: list[float] | None = None
    temperature_sharpening: bool = True
    p_correct_temperature_exponent: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_correct <= 1.0:
            raise ValueError(f"p_correct must be in [0, 1], got {self.p_correct}")
        if self.p_correct > 0 and not self.correct_variants:
            raise ValueError("p_correct > 0 requires at least one correct variant")
        if self.p_correct < 1 and not self.incorrect_variants:
            raise ValueError("p_correct < 1 requires at least one incorrect variant")
        if self.species_weights is not None:
            if len(self.species_weights) != len(self.correct_variants):
                raise ValueError("species_weights must align with correct_variants")
            if any(w <= 0 for w in self.species_weights):
                raise ValueError("species_weights must be strictly positive")

    def effective_p_correct(self, temperature: float) -> float:
        """True solve probability at this temperature."""
        if self.p_correct_temperature_exponent == 0.0:
            return self.p_correct
        # Higher temperature -> lower accuracy, bounded to [0, 1].
        factor = (1.0 + temperature) ** self.p_correct_temperature_exponent
        return min(1.0, max(0.0, self.p_correct * factor))

    def effective_species_weights(self, temperature: float) -> list[float]:
        """Normalised species distribution at this temperature."""
        weights = self.species_weights or [1.0] * len(self.correct_variants)
        if self.temperature_sharpening and temperature > 0:
            power = 1.0 / max(temperature, 1e-6)
            # Clamp the exponent: at very low T this otherwise underflows to a
            # degenerate all-zero weight vector.
            power = min(power, 50.0)
            weights = [w**power for w in weights]
        total = sum(weights)
        if total <= 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]

    def true_species_count(self) -> int:
        return len(self.correct_variants)


class MockModelClient(ModelClient):
    """A frozen model with exactly known behaviour, for validating the instrument.

    The behaviour of a task is looked up by `request.meta["task_id"]`. Tasks with
    no registered behaviour fall back to `default_behaviour`, or raise if none is
    set -- silently inventing behaviour for an unregistered task would let a
    validation run pass while measuring nothing.
    """

    def __init__(
        self,
        behaviours: dict[str, MockTaskBehaviour],
        seed: int = 0,
        model_id: str = "mock",
        default_behaviour: MockTaskBehaviour | None = None,
        usd_per_1k_tokens: float = 0.0,
    ):
        self.behaviours = dict(behaviours)
        self.seed = seed
        self.model_id = model_id
        self.default_behaviour = default_behaviour
        self.usd_per_1k_tokens = usd_per_1k_tokens
        #: Incremented per (task, temperature) so repeated calls draw fresh samples
        #: while remaining reproducible from `seed`.
        self._draw_counter: dict[tuple[str, float], int] = {}

    def behaviour_for(self, task_id: str) -> MockTaskBehaviour:
        if task_id in self.behaviours:
            return self.behaviours[task_id]
        if self.default_behaviour is not None:
            return self.default_behaviour
        raise KeyError(
            f"no mock behaviour registered for task {task_id!r} and no default set"
        )

    def _generate(self, request: CompletionRequest) -> Completion:
        task_id = request.meta.get("task_id")
        if task_id is None:
            raise ValueError("MockModelClient requires request.meta['task_id']")
        behaviour = self.behaviour_for(task_id)

        key = (task_id, request.temperature)
        if request.seed is not None:
            index = request.seed
        else:
            index = self._draw_counter.get(key, 0)
            self._draw_counter[key] = index + 1

        rng = random.Random(
            _derive_seed(self.seed, self.model_id, task_id, request.temperature, index)
        )

        p = behaviour.effective_p_correct(request.temperature)
        is_correct = rng.random() < p
        if is_correct:
            weights = behaviour.effective_species_weights(request.temperature)
            text = rng.choices(behaviour.correct_variants, weights=weights, k=1)[0]
            species = behaviour.correct_variants.index(text)
        else:
            text = rng.choice(behaviour.incorrect_variants)
            species = -1

        prompt_tokens = estimate_tokens(request.prompt)
        completion_tokens = estimate_tokens(text)
        usage = Usage(
            calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd=(prompt_tokens + completion_tokens) / 1000.0 * self.usd_per_1k_tokens,
        )
        return Completion(
            text=text,
            usage=usage,
            model_id=self.model_id,
            seed=index,
            meta={
                # Ground truth, for validating the pipeline. Analysis code must
                # never read these fields -- they do not exist for a real model.
                "_ground_truth_correct": is_correct,
                "_ground_truth_species": species,
                "task_id": task_id,
                "temperature": request.temperature,
            },
        )

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": "MockModelClient",
            "seed": self.seed,
            "n_tasks": len(self.behaviours),
            "has_default": self.default_behaviour is not None,
        }
