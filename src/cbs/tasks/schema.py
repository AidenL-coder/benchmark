"""Task and split schema.

A `Task` is the unit the frontier is estimated over. It must carry everything
needed to verify a candidate solution deterministically, because `p_hat(x)` is
only meaningful if "correct" means the same thing on every one of the `N_max`
draws.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum

__all__ = ["Split", "Task", "TaskSuite"]


class Split(str, Enum):
    """Brief section 8. `TRAIN` is the only split `S_evo` may evaluate on."""

    TRAIN = "train"
    HELD_OUT = "held_out"
    TRANSFER = "transfer"


@dataclass(frozen=True)
class Task:
    """One verifiable problem.

    Attributes
    ----------
    prompt:
        Exactly what is shown to the frozen model. Held fixed across all systems
        for the minimal scaffold `S0`; scaffolds may build their own prompts on
        top, which is itself a scaffold operation and is tagged as such.
    tests:
        Python source asserting correctness of `entry_point`. Executed in the
        sandbox alongside the candidate. Must exit non-zero on failure. This is
        the hidden oracle: it decides `p_hat(x)` and every pass/fail verdict, and
        no scaffold operation may read its outcome and feed it back into
        generation (see `public_tests`).
    public_tests:
        A weaker, deliberately partial subset of `tests` that a scaffold is
        permitted to execute and condition on. Real coding agents do not see the
        held-out grading suite while working; execution-feedback loops (brief
        section 3.1) run the candidate against something the agent itself has
        access to. If `execution_feedback` read the hidden `tests` and fed the
        failure back into the prompt, that would leak the grading oracle into
        the generative process -- which would make an "expanding" operation
        trivially able to walk straight to a passing answer, and would make any
        resulting frontier-crossing an artefact of oracle access rather than of
        the operation. Empty string means no public signal is available for
        this task; a scaffold falls back to a plain compile/parse check.
    setup:
        Optional preamble prepended before the candidate (imports, fixtures).
    reference_solution:
        A known-good solution. Used to smoke-test the verifier itself: if the
        reference fails, the task is broken, not the model.
    """

    task_id: str
    family: str
    prompt: str
    tests: str
    entry_point: str = ""
    public_tests: str = ""
    setup: str = ""
    reference_solution: str | None = None
    timeout_s: float = 10.0
    memory_mb: int = 1024
    split: Split | None = None
    metadata: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        """Hash of the semantically load-bearing fields.

        Excludes `split` and `metadata`, so re-splitting does not invalidate a
        task's identity, but any change to the prompt or tests does.
        """
        payload = json.dumps(
            {
                "task_id": self.task_id,
                "family": self.family,
                "prompt": self.prompt,
                "tests": self.tests,
                "public_tests": self.public_tests,
                "entry_point": self.entry_point,
                "setup": self.setup,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["split"] = self.split.value if self.split else None
        d["content_hash"] = self.content_hash()
        return d


@dataclass
class TaskSuite:
    """A named, hashable collection of tasks."""

    name: str
    tasks: list[Task]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def by_id(self) -> dict[str, Task]:
        return {t.task_id: t for t in self.tasks}

    def filter_split(self, split: Split) -> "TaskSuite":
        return TaskSuite(
            name=f"{self.name}:{split.value}",
            tasks=[t for t in self.tasks if t.split == split],
        )

    def suite_hash(self) -> str:
        """Stable hash over the whole suite, including split assignment.

        This is the number quoted in results to prove the splits were frozen
        before the run (brief section 7, Phase 1 DoD).
        """
        h = hashlib.sha256()
        for task in sorted(self.tasks, key=lambda t: t.task_id):
            h.update(task.task_id.encode("utf-8"))
            h.update(b"\x00")
            h.update(task.content_hash().encode("utf-8"))
            h.update(b"\x00")
            h.update((task.split.value if task.split else "none").encode("utf-8"))
            h.update(b"\x01")
        return h.hexdigest()

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for task in self.tasks:
            key = task.split.value if task.split else "unassigned"
            out[key] = out.get(key, 0) + 1
        return out
