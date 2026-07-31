"""Frozen-model clients and their registry."""

from __future__ import annotations

from cbs.models.base import (
    Completion,
    CompletionRequest,
    ModelClient,
    ModelUnavailable,
    estimate_tokens,
)
from cbs.models.mock import MockModelClient, MockTaskBehaviour
from cbs.models.openai_compat import OpenAICompatClient, OpenAICompatConfig

__all__ = [
    "Completion",
    "CompletionRequest",
    "ModelClient",
    "ModelUnavailable",
    "MockModelClient",
    "MockTaskBehaviour",
    "OpenAICompatClient",
    "OpenAICompatConfig",
    "build_model",
    "estimate_tokens",
]


def build_model(spec: dict) -> ModelClient:
    """Construct a model client from a config block.

    Expected shape::

        backend: openai_compat | mock
        # ...backend-specific keys
    """
    spec = dict(spec)
    backend = spec.pop("backend", None)
    if backend is None:
        raise ValueError("model config requires a 'backend' key")

    if backend == "openai_compat":
        return OpenAICompatClient(OpenAICompatConfig(**spec))
    if backend == "mock":
        behaviours: dict[str, MockTaskBehaviour] = {}

        # Pull ground truth from a task family rather than restating it in YAML.
        # The true p(x) and solution variants live with the tasks they describe,
        # so a config can never drift out of sync with the family it validates.
        family = spec.pop("behaviours_from_family", None)
        if family:
            behaviours.update(_family_behaviours(family))

        behaviours.update(
            {
                task_id: MockTaskBehaviour(**b)
                for task_id, b in (spec.pop("behaviours", {}) or {}).items()
            }
        )
        default = spec.pop("default_behaviour", None)
        return MockModelClient(
            behaviours=behaviours,
            default_behaviour=MockTaskBehaviour(**default) if default else None,
            **spec,
        )
    raise ValueError(f"unknown model backend {backend!r}")


def _family_behaviours(family: str) -> dict:
    if family == "toy":
        from cbs.tasks.families.toy import toy_behaviours

        return toy_behaviours()
    raise ValueError(
        f"no mock ground truth defined for family {family!r}. Only families with "
        "declared correct/incorrect variants can drive the mock backend."
    )
