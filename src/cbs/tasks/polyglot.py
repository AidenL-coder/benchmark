"""Polyglot benchmark instance representation (D-31/D-42).

Deliberately **not** a `cbs.tasks.schema.Task`, for the same reason
`cbs.tasks.swebench.SweBenchInstance` isn't: a Polyglot exercise is a small
git repository (an Exercism-style language exercise) at a `base_commit`,
solved by editing its solution file(s) and graded by revealing hidden test
content at a separate `test_commit` and running the language's real test
command inside a Docker image -- not a prompt-plus-code-string `Task`.

Confirmed against HGM's own `polyglot/polyglot_benchmark_metadata.json`
(225 real entries) and `polyglot/harness.py:process_entry`, not assumed from
reading `hgm.py` alone:

*   There is no `Task.tests`/`Task.public_tests`-style split at the data
    level -- each entry has exactly one `files["test"]` file. Oracle safety
    instead comes from *when* that file's real content becomes visible:
    the agent's own working container is checked out at `base_commit`
    (`entry["test_commit"]` is not applied there), and `process_entry` only
    reveals the true test content by `git reset --hard {test_commit}`
    *after* the agent has already produced its patch and stopped -- so the
    agent genuinely cannot see the grading tests while working, without
    `cbs` needing to construct a separate public/hidden split itself.
*   Several languages' real eval commands (confirmed in
    `polyglot/constants.py`, e.g. C++'s `cmake -DEXERCISM_RUN_ALL_TESTS=1`)
    compile-gate most test cases behind a flag that IS set for real grading
    -- so `process_entry`'s `eval_result` already reflects the full hidden
    test suite, not just an always-visible smoke subset. Nothing extra is
    needed here to get a true hidden-oracle check.
*   Unlike `SweBenchInstance`, no dataset field needs decoding (no
    JSON-encoded-string-that-prints-like-a-list gotcha the way
    `FAIL_TO_PASS`/`PASS_TO_PASS` had) -- `process_entry` consumes each raw
    entry dict directly and unmodified. `PolyglotInstance` therefore keeps
    the complete original dict (`raw`) rather than re-deriving an adapter
    that reconstructs it, precisely to avoid the class of bug D-40 hit
    repeatedly translating `SweBenchInstance` back into swebench's expected
    shape (a missing `hints_text` key, JSON-list fields, ...) -- there is
    nothing to translate here if the original dict is simply kept.

Unlike `load_swebench_verified`, there is no stable, `cbs`-independent
external host for this data: it lives inside HGM's own vendored
`polyglot-benchmark/` git submodule, not a public HuggingFace dataset. So
`load_polyglot_benchmark` takes an explicit metadata-file path rather than
reaching out to a fixed URI -- this makes `cbs.tasks.polyglot` correctly
dependent on wherever a real HGM checkout happens to sit (there is no
"the" copy the way there is for SWE-bench Verified), not a design gap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["PolyglotInstance", "PolyglotSuite", "load_polyglot_benchmark"]


@dataclass(frozen=True)
class PolyglotInstance:
    """One Polyglot benchmark exercise.

    Attributes
    ----------
    instance_id:
        e.g. ``"cpp__all-your-base"``.
    language:
        e.g. ``"cpp"`` -- selects the real test command (`polyglot.
        constants.TEST_COMMANDS`) and Docker image family.
    problem_statement:
        The natural-language exercise instructions shown to the agent.
    raw:
        The complete original entry dict from HGM's own
        `polyglot_benchmark_metadata.json` -- passed to `process_entry`
        (or the analogous real glue) unmodified, not reconstructed from
        named fields. Includes `base_commit`, `test_commit`, `files`,
        `test_patch`, `reference_tests`, `reference_answers` (the gold
        solution, this family's `reference_solution` analogue), `repo`
        (path inside the vendored `polyglot-benchmark/` checkout), etc.
    """

    instance_id: str
    language: str
    problem_statement: str
    raw: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        """Hash of the semantically load-bearing fields -- the whole `raw`
        dict, since (unlike `SweBenchInstance`) there is no separate set of
        named grading fields to single out; any change to `raw` changes
        what is graded or how."""
        payload = json.dumps(
            {
                "instance_id": self.instance_id,
                "language": self.language,
                "problem_statement": self.problem_statement,
                "raw": self.raw,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PolyglotSuite:
    """A named, hashable collection of `PolyglotInstance`s.

    Mirrors `cbs.tasks.swebench.SweBenchSuite`'s frozen-hashed-split
    discipline (brief section 7, Phase 1 DoD).
    """

    name: str
    instances: tuple[PolyglotInstance, ...]

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def by_id(self) -> dict[str, PolyglotInstance]:
        return {inst.instance_id: inst for inst in self.instances}

    def suite_hash(self) -> str:
        """Stable hash over the whole suite -- the number quoted in results
        to prove the split was frozen before sampling, same role as
        `TaskSuite.suite_hash`/`SweBenchSuite.suite_hash`."""
        h = hashlib.sha256()
        for inst in sorted(self.instances, key=lambda i: i.instance_id):
            h.update(inst.instance_id.encode("utf-8"))
            h.update(b"\x00")
            h.update(inst.content_hash().encode("utf-8"))
            h.update(b"\x01")
        return h.hexdigest()


def load_polyglot_benchmark(
    metadata_path: str | Path, instance_ids: list[str] | None = None
) -> PolyglotSuite:
    """Load Polyglot benchmark entries from a real HGM checkout's metadata
    file (`polyglot/polyglot_benchmark_metadata.json`, 225 real entries as
    of D-42).

    Parameters
    ----------
    metadata_path:
        Path to `polyglot_benchmark_metadata.json` inside a real HGM
        checkout. Not a fixed default -- see module docstring on why this
        data has no stable `cbs`-independent host the way SWE-bench
        Verified does.
    instance_ids:
        If given, load only these instances rather than all 225.
    """
    path = Path(metadata_path)
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)

    if instance_ids is not None:
        wanted = set(instance_ids)
        entries = [e for e in entries if e["instance_id"] in wanted]

    instances = tuple(
        PolyglotInstance(
            instance_id=entry["instance_id"],
            language=entry["language"],
            problem_statement=entry["problem_statement"],
            raw=entry,
        )
        for entry in entries
    )
    return PolyglotSuite(name="polyglot_benchmark", instances=instances)
