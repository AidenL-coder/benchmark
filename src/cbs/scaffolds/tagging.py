"""Support-class instrumentation -- the theoretical spine (brief section 3.1).

Every scaffold operation is one of:

*   **support-preserving** -- the emitted solution is a sample from `M`'s
    conditional distribution followed by selection or aggregation (best-of-N,
    majority vote, self-consistency, reranking). Such a scaffold can raise the
    success *rate* at fixed budget, but cannot emit a correct completion that has
    ~zero probability under `M`.
*   **support-expanding** -- information is routed outside `M`'s single-shot
    distribution: execution/error feedback, external tool or solver calls,
    retrieval, or decomposition into subproblems each individually inside `M`'s
    frontier.

The falsifiable structural claim the study tests is that *any* genuine
frontier-crossing depends on at least one support-expanding operation. That claim
is only testable if tagging is complete and honest, which drives two design
choices here:

1.  **Unknown operations raise.** An evolved `S_evo` scaffold that invents a new
    operation cannot silently emit it as untagged; the run fails loudly until a
    human classifies it. Defaulting unknown ops to "preserving" would let
    genuine expansion masquerade as bounded search, and defaulting to
    "expanding" would do the reverse.
2.  **The registry is central and auditable**, not scattered decorators, so the
    partition can be reviewed as a single artefact.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "SupportClass",
    "UnregisteredOperation",
    "OpRecord",
    "OperationTrace",
    "register_operation",
    "support_class_of",
    "registry_snapshot",
]


class SupportClass(str, Enum):
    PRESERVING = "support-preserving"
    EXPANDING = "support-expanding"


class UnregisteredOperation(KeyError):
    """A scaffold used an operation with no declared support class."""


@dataclass(frozen=True)
class OpDefinition:
    """One entry in the partition, with the argument for its classification.

    The rationale is stored, not merely commented, because the partition *is* the
    theoretical contribution (brief section 11.2) and has to be reviewable as a
    standalone artefact. `registry_snapshot()` ships it with every result.
    """

    support_class: SupportClass
    rationale: str
    #: True where the classification is genuinely arguable. These are the
    #: operations whose treatment a reviewer is most likely to challenge, and any
    #: crossing that depends on one must be reported with that caveat attached.
    contested: bool = False


#: The partition. Adding an entry is a scientific claim and should be reviewed.
_REGISTRY: dict[str, OpDefinition] = {
    # -- support-preserving: emit a sample from M, then select/aggregate ----
    "single_call": OpDefinition(
        SupportClass.PRESERVING,
        "One sample from M under the fixed prompt. The definitional base case.",
    ),
    "format_extract": OpDefinition(
        SupportClass.PRESERVING,
        "Deterministic extraction of code from the completion. Adds no "
        "information; it only reads what M already emitted.",
    ),
    "best_of_n": OpDefinition(
        SupportClass.PRESERVING,
        "Draws N samples from M and returns one of them. The emitted solution is "
        "always a sample from M, so it cannot have ~zero probability under M.",
    ),
    "majority_vote": OpDefinition(
        SupportClass.PRESERVING,
        "Aggregates over samples of M; the returned answer is one M produced.",
    ),
    "self_consistency": OpDefinition(
        SupportClass.PRESERVING,
        "Marginalises over sampled reasoning paths. Selection among M's samples.",
    ),
    "rerank": OpDefinition(
        SupportClass.PRESERVING,
        "Reorders candidates sampled from M; emits one of them unchanged.",
    ),
    "test_guided_selection": OpDefinition(
        SupportClass.PRESERVING,
        "Best-of-N where the selector is the verifier. The oracle signal comes "
        "from outside M, but it only *chooses among* M's samples and can never "
        "emit one M did not produce. Classifying this as expanding would make "
        "the frontier definition circular: the frontier is itself defined as "
        "N-sample best-of-N with a verifier (brief section 3.2), so the baseline "
        "estimator would count as an expanding scaffold.",
    ),
    "temperature_schedule": OpDefinition(
        SupportClass.PRESERVING,
        "Changes sampling temperature. Reweights M's distribution and can only "
        "reach completions with non-zero probability under it.",
        contested=True,
    ),
    "prompt_template": OpDefinition(
        SupportClass.PRESERVING,
        "A fixed, human-authored prompt template applied identically to every "
        "task. Held constant across systems, so it shifts the conditional but "
        "not differentially between the systems being compared.",
        contested=True,
    ),
    # -- support-expanding: route information outside M's single-shot dist --
    "execution_feedback": OpDefinition(
        SupportClass.EXPANDING,
        "Executes a candidate and conditions the next sample on the resulting "
        "error text. That text is produced by the interpreter, not by M, so the "
        "follow-up sample is drawn from a conditional M could not have formed "
        "single-shot.",
    ),
    "tool_call": OpDefinition(
        SupportClass.EXPANDING,
        "Injects the output of an external program into the context.",
    ),
    "retrieval": OpDefinition(
        SupportClass.EXPANDING,
        "Injects documents from an external corpus into the context.",
    ),
    "decomposition": OpDefinition(
        SupportClass.EXPANDING,
        "Splits a task into subproblems each individually within M's frontier "
        "and composes the results. The composed artefact can be one M would "
        "never sample end-to-end.",
    ),
    "external_solver": OpDefinition(
        SupportClass.EXPANDING,
        "Delegates a subproblem to a non-M solver (SMT, CAS, search).",
    ),
    "adaptive_prompt_rewrite": OpDefinition(
        SupportClass.EXPANDING,
        "Rewrites the prompt conditioned on the outcome of prior attempts. The "
        "conditioning signal originates outside M (execution results, verifier "
        "output), which makes this self-refinement rather than reprompting. "
        "Contrast `prompt_template`, which is fixed and task-independent.",
    ),
}


def register_operation(
    name: str,
    support_class: SupportClass,
    rationale: str = "",
    contested: bool = False,
) -> None:
    """Declare an operation's support class.

    Re-registering under a *different* class raises: silently reclassifying an
    operation mid-study would invalidate every record produced before the change.
    """
    existing = _REGISTRY.get(name)
    if existing is not None and existing.support_class != support_class:
        raise ValueError(
            f"operation {name!r} is already registered as "
            f"{existing.support_class.value}; refusing to reclassify it as "
            f"{support_class.value}"
        )
    if existing is None and not rationale:
        raise ValueError(
            f"registering new operation {name!r} requires a rationale: the "
            "support partition is a scientific claim, not a lookup table"
        )
    _REGISTRY[name] = existing or OpDefinition(support_class, rationale, contested)


def support_class_of(name: str) -> SupportClass:
    return definition_of(name).support_class


def definition_of(name: str) -> OpDefinition:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise UnregisteredOperation(
            f"operation {name!r} has no declared support class. Every scaffold "
            "operation must be classified as support-preserving or "
            "support-expanding before it can be used, because the study's central "
            "claim is stated in terms of that partition. Call "
            "cbs.scaffolds.tagging.register_operation() to classify it."
        ) from exc


def registry_snapshot() -> dict[str, dict]:
    """The full partition with rationales, recorded alongside every result."""
    return {
        name: {
            "support_class": d.support_class.value,
            "rationale": d.rationale,
            "contested": d.contested,
        }
        for name, d in sorted(_REGISTRY.items())
    }


def contested_operations() -> set[str]:
    """Operations whose classification is arguable; crossings touching these
    must be reported with the caveat attached."""
    return {name for name, d in _REGISTRY.items() if d.contested}


@dataclass(frozen=True)
class OpRecord:
    name: str
    support_class: SupportClass
    seq: int
    duration_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "support_class": self.support_class.value,
            "seq": self.seq,
            "duration_s": round(self.duration_s, 6),
            **({"metadata": self.metadata} if self.metadata else {}),
        }


@dataclass
class OperationTrace:
    """Ordered record of the operations one solve attempt used."""

    records: list[OpRecord] = field(default_factory=list)

    def record_instant(self, name: str, duration_s: float = 0.0, **metadata) -> None:
        """Append an already-completed operation, tagged now rather than at a
        call site.

        `record()` assumes the calling code can be trusted to name its own
        operation as it happens -- true for `S0`/`S_star`, which we author. It
        is not true for an evolved, untrusted agent (`cbs.scaffolds.evolved`):
        whether a given verifier call was *selection* (support-preserving) or
        *conditioning the next sample on its error* (support-expanding) can
        only be decided by watching what the agent did with the result
        afterward, which is necessarily after the fact. Interception buffers
        raw events and calls this once classification is complete, in the
        order the events actually occurred.
        """
        support_class = support_class_of(name)
        self.records.append(
            OpRecord(
                name=name,
                support_class=support_class,
                seq=len(self.records),
                duration_s=duration_s,
                metadata=dict(metadata),
            )
        )

    @contextmanager
    def record(self, name: str, **metadata):
        """Time and tag one operation.

        The support class is resolved on *entry*, so an unregistered operation
        fails before it can do any work.
        """
        support_class = support_class_of(name)
        seq = len(self.records)
        started = time.monotonic()
        try:
            yield
        finally:
            self.records.append(
                OpRecord(
                    name=name,
                    support_class=support_class,
                    seq=seq,
                    duration_s=time.monotonic() - started,
                    metadata=dict(metadata),
                )
            )

    # -- attribution ------------------------------------------------------
    @property
    def classes_used(self) -> set[SupportClass]:
        return {r.support_class for r in self.records}

    @property
    def used_expanding(self) -> bool:
        """Whether this attempt used any support-expanding operation.

        This is the field the crossing analysis keys on.
        """
        return SupportClass.EXPANDING in self.classes_used

    @property
    def expanding_ops(self) -> set[str]:
        return {
            r.name for r in self.records if r.support_class is SupportClass.EXPANDING
        }

    @property
    def contested_ops_used(self) -> set[str]:
        """Operations used here whose support classification is arguable.

        A crossing attributed to a contested operation is a weaker claim than one
        attributed to, say, `execution_feedback`, and must be reported as such.
        """
        return {r.name for r in self.records} & contested_operations()

    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.name] = counts.get(r.name, 0) + 1
        return counts

    def as_dict(self) -> dict:
        return {
            "n_ops": len(self.records),
            "classes_used": sorted(c.value for c in self.classes_used),
            "used_expanding": self.used_expanding,
            "expanding_ops": sorted(self.expanding_ops),
            "contested_ops_used": sorted(self.contested_ops_used),
            "op_counts": self.op_counts(),
            "records": [r.as_dict() for r in self.records],
        }
