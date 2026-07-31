"""Archive persistence and overfitting/transfer analysis (brief section 9.1-9.2).

An `S_evo` loop persists an archive of scaffold variants across generations
(proposal section 2.1). This module is the analysis layer that reads such an
archive, independent of which fork produced it: it computes the overfitting
gap and transfer retention (brief section 9.1), and offers a crude triage
heuristic for eval-specific hard-coding (brief section 7, Phase 4: "inspect the
archive for eval-specific hard-coding").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cbs.scaffolds.evolved import ArchiveEntry

__all__ = [
    "ArchiveRecord",
    "overfitting_gap",
    "transfer_retention",
    "hard_coding_heuristic",
    "persist_archive",
    "load_archive",
]

#: Below this length, a literal is common enough (a small integer, "True",
#: empty-string) that a match is more likely coincidence than hard-coding.
MIN_HARDCODE_LITERAL_LEN = 6


@dataclass
class ArchiveRecord:
    """One archived variant's identity plus its measured solve rates.

    `held_out_solve_rate` / `transfer_solve_rate` are `None` until measured --
    an archive persisted mid-evolution (train only, per brief section 8: "the
    loop may evaluate on train only") should not silently read as a zero rate
    on splits it was never run against.
    """

    entry: ArchiveEntry
    train_solve_rate: float
    held_out_solve_rate: float | None = None
    transfer_solve_rate: float | None = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "entry": asdict(self.entry),
            "train_solve_rate": self.train_solve_rate,
            "held_out_solve_rate": self.held_out_solve_rate,
            "transfer_solve_rate": self.transfer_solve_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArchiveRecord":
        return cls(
            entry=ArchiveEntry(**data["entry"]),
            train_solve_rate=data["train_solve_rate"],
            held_out_solve_rate=data.get("held_out_solve_rate"),
            transfer_solve_rate=data.get("transfer_solve_rate"),
            metadata=data.get("metadata", {}),
        )


def overfitting_gap(record: ArchiveRecord) -> float | None:
    """`train - held_out` for one variant (brief section 9.1, "Overfitting gap").

    `None` if the variant has never been evaluated on `held_out` -- distinct
    from a gap of `0.0`, which would claim "no overfitting" about a variant
    that was simply never checked.
    """
    if record.held_out_solve_rate is None:
        return None
    return record.train_solve_rate - record.held_out_solve_rate


def transfer_retention(
    record: ArchiveRecord,
    s0_held_out_rate: float,
    s0_transfer_rate: float,
) -> float | None:
    """`transfer gain / held-out gain`, both relative to `S0` (brief section 9.1).

    Gains, not raw rates: `S_evo`'s raw transfer solve rate is not informative
    on its own (a task family can simply be easier or harder overall). What
    the retention metric asks is whether whatever `S_evo` gained over `S0` on
    `held_out` survives, proportionally, on a disjoint family it never
    optimised on. Returns `None` if the held-out gain is ~zero (division would
    be meaningless: there is no gain for a transfer gain to "retain a fraction
    of") or if `transfer_solve_rate` was never measured.
    """
    if record.transfer_solve_rate is None or record.held_out_solve_rate is None:
        return None
    held_out_gain = record.held_out_solve_rate - s0_held_out_rate
    if abs(held_out_gain) < 1e-9:
        return None
    transfer_gain = record.transfer_solve_rate - s0_transfer_rate
    return transfer_gain / held_out_gain


def hard_coding_heuristic(source_excerpt: str, task_literals: list[str]) -> list[str]:
    """Flag literal task inputs/outputs appearing verbatim in scaffold source.

    A crude, deliberately high-recall / low-precision triage tool for "inspect
    the archive for eval-specific hard-coding" (brief section 7, Phase 5) --
    not a verdict. A hit means a human should read the surrounding code before
    concluding the variant memorised an answer rather than, say, using a
    genuinely generic constant that happens to coincide with one. Silence does
    not certify a variant clean either: hard-coding via anything less literal
    than an exact string match (e.g. an off-by-one-transformed constant, or
    logic keyed on a task's hash) will not be caught by this.
    """
    hits = []
    for literal in task_literals:
        text = literal.strip()
        if len(text) >= MIN_HARDCODE_LITERAL_LEN and text in source_excerpt:
            hits.append(literal)
    return hits


def persist_archive(records: list[ArchiveRecord], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.as_dict()) + "\n")


def load_archive(path: Path | str) -> list[ArchiveRecord]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(ArchiveRecord.from_dict(json.loads(line)))
    return records
