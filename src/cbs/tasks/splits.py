"""Deterministic, frozen, hashed task splits (brief section 7, Phase 1 DoD).

Assignment is a pure function of `task_id` and a salt, not of iteration order or
RNG state, so the same suite always splits the same way and a split can be
reproduced from the manifest alone.

The split boundary is the study's main defence against the overfitting confound
(brief section 3.4): `S_evo` may evaluate on `train` only. `held_out` is drawn
from the same distribution and never shown to the loop; `transfer` is a different
family entirely. A manifest is written before any sampling so the assignment
provably predates the results.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from cbs.tasks.schema import Split, Task, TaskSuite

__all__ = ["SplitRatios", "SplitManifest", "assign_splits", "write_manifest", "verify_manifest"]


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.5
    held_out: float = 0.5
    transfer: float = 0.0

    def __post_init__(self) -> None:
        total = self.train + self.held_out + self.transfer
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        if min(self.train, self.held_out, self.transfer) < 0:
            raise ValueError("split ratios must be non-negative")

    def as_dict(self) -> dict:
        return {
            "train": self.train,
            "held_out": self.held_out,
            "transfer": self.transfer,
        }


def _unit_hash(task_id: str, salt: str) -> float:
    """Map a task id into [0, 1) deterministically and portably."""
    h = hashlib.blake2b(f"{salt}\x00{task_id}".encode("utf-8"), digest_size=8)
    return struct.unpack("<Q", h.digest())[0] / 2**64


def assign_splits(
    suite: TaskSuite,
    ratios: SplitRatios,
    salt: str = "cbs-v1",
    transfer_families: set[str] | None = None,
) -> TaskSuite:
    """Return a copy of `suite` with `split` assigned on every task.

    `transfer_families` overrides ratio-based assignment: any task whose family
    is listed goes to `transfer` wholesale. That is the correct treatment for the
    reasoning/transfer family (brief section 8), which is a *different
    distribution* rather than a random slice of the same one -- splitting it by
    ratio would leak transfer-distribution tasks into `train` and destroy the
    only test of RQ4.
    """
    transfer_families = transfer_families or set()
    out: list[Task] = []
    for task in suite.tasks:
        if task.family in transfer_families:
            split = Split.TRANSFER
        else:
            u = _unit_hash(task.task_id, salt)
            if u < ratios.train:
                split = Split.TRAIN
            elif u < ratios.train + ratios.held_out:
                split = Split.HELD_OUT
            else:
                split = Split.TRANSFER
        out.append(replace(task, split=split))
    return TaskSuite(name=suite.name, tasks=out)


@dataclass
class SplitManifest:
    suite_name: str
    suite_hash: str
    salt: str
    ratios: dict
    counts: dict
    created_utc: str
    task_splits: dict[str, str] = field(default_factory=dict)
    task_content_hashes: dict[str, str] = field(default_factory=dict)
    cbs_version: str = ""

    def as_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "suite_hash": self.suite_hash,
            "salt": self.salt,
            "ratios": self.ratios,
            "counts": self.counts,
            "created_utc": self.created_utc,
            "cbs_version": self.cbs_version,
            "task_splits": self.task_splits,
            "task_content_hashes": self.task_content_hashes,
        }


def build_manifest(suite: TaskSuite, ratios: SplitRatios, salt: str) -> SplitManifest:
    from cbs import __version__

    return SplitManifest(
        suite_name=suite.name,
        suite_hash=suite.suite_hash(),
        salt=salt,
        ratios=ratios.as_dict(),
        counts=suite.counts(),
        created_utc=datetime.now(timezone.utc).isoformat(),
        task_splits={
            t.task_id: (t.split.value if t.split else "unassigned") for t in suite.tasks
        },
        task_content_hashes={t.task_id: t.content_hash() for t in suite.tasks},
        cbs_version=__version__,
    )


def write_manifest(
    suite: TaskSuite, ratios: SplitRatios, path: Path, salt: str = "cbs-v1"
) -> SplitManifest:
    """Freeze the split to disk. Refuses to silently overwrite a differing one."""
    manifest = build_manifest(suite, ratios, salt)
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("suite_hash") != manifest.suite_hash:
            raise FileExistsError(
                f"{path} already holds a frozen split with a different suite hash "
                f"({existing.get('suite_hash')[:12]}... vs "
                f"{manifest.suite_hash[:12]}...). Refusing to overwrite: a split "
                "must be frozen before results are produced against it. Delete it "
                "deliberately if the suite genuinely changed."
            )
        return manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    return manifest


def verify_manifest(suite: TaskSuite, path: Path) -> dict:
    """Check a suite against a frozen manifest.

    Reports drift rather than raising, so `cbs splits verify` can enumerate every
    problem in one pass.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []

    if data.get("suite_hash") != suite.suite_hash():
        problems.append("suite_hash mismatch: tasks or split assignment changed")

    recorded_splits = data.get("task_splits", {})
    recorded_hashes = data.get("task_content_hashes", {})
    current = {t.task_id: t for t in suite.tasks}

    for task_id in sorted(set(recorded_splits) - set(current)):
        problems.append(f"task missing from suite: {task_id}")
    for task_id in sorted(set(current) - set(recorded_splits)):
        problems.append(f"task not in manifest (added after freeze): {task_id}")

    for task_id, task in sorted(current.items()):
        if task_id in recorded_hashes and recorded_hashes[task_id] != task.content_hash():
            problems.append(f"task content changed since freeze: {task_id}")
        expected = recorded_splits.get(task_id)
        actual = task.split.value if task.split else "unassigned"
        if expected is not None and expected != actual:
            problems.append(
                f"split changed for {task_id}: manifest={expected} current={actual}"
            )

    return {
        "ok": not problems,
        "problems": problems,
        "manifest_suite_hash": data.get("suite_hash"),
        "current_suite_hash": suite.suite_hash(),
    }
