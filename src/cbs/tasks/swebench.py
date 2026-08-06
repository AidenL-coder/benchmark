"""SWE-bench Verified instance representation (D-40).

Deliberately **not** a `cbs.tasks.schema.Task`. A SWE-bench Verified instance
is a git repository at a commit plus a natural-language problem statement,
solved by producing a diff and graded by running a real test suite inside a
per-instance Docker image -- none of `Task`'s assumptions (a prompt shown
verbatim to `M`, a single candidate code string, assert-based `tests` a
sandbox concatenates and executes) hold here, for the same reason D-33 found
LiveCodeBench did not fit as "just another loader". Forcing this into `Task`
would either silently drop information (`fail_to_pass`/`pass_to_pass` don't
have a slot) or quietly change what `content_hash`/verification mean.

The `tests` / `public_tests` analogue, confirmed against the real dataset
(`princeton-nlp/SWE-bench_Verified`), not assumed from documentation:

*   ``fail_to_pass`` -- the specific regression test(s) that must go from
    failing to passing after the fix. This is the hidden grading oracle,
    queried exactly once on the final chosen candidate (D-40) -- the direct
    analogue of `Task.tests`.
*   ``pass_to_pass`` -- the repository's own pre-existing tests, which must
    keep passing. Legitimately runnable by a scaffold working the problem
    (a real engineer would run the existing suite) without revealing
    whether the target bug is fixed, since it contains no test *of* that
    bug -- the direct analogue of `Task.public_tests`.

Both fields arrive from the HuggingFace dataset as JSON-encoded strings, not
native lists -- confirmed by inspecting the actual loaded row's field types
before writing this loader, not assumed from how they print.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

__all__ = ["SweBenchInstance", "SweBenchSuite", "load_swebench_verified"]

#: The dataset this project measures S0/S_star/S_evo against for the
#: SWE-bench family (D-31/D-39) -- loaded directly, not via HGM's own
#: SWE-bench-then-filter-by-Verified-IDs indirection (D-39's flagged loose
#: end: that indirection is unverified to be equivalent, and there is no
#: reason to replicate it when the dataset is available directly).
HF_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"


@dataclass(frozen=True)
class SweBenchInstance:
    """One SWE-bench Verified problem.

    Attributes
    ----------
    instance_id:
        e.g. ``"astropy__astropy-12907"``.
    repo:
        e.g. ``"astropy/astropy"`` -- a GitHub ``owner/name`` slug.
    base_commit:
        Commit the repo is checked out at before any fix is applied.
    problem_statement:
        The natural-language issue text shown to the agent.
    fail_to_pass:
        Test node IDs that must go from failing to passing. The hidden
        oracle -- queried exactly once, on the final chosen candidate,
        mirroring `Task.tests`.
    pass_to_pass:
        Test node IDs that must remain passing. Legitimately runnable
        mid-attempt for execution feedback, mirroring `Task.public_tests`.
    patch:
        The reference (gold) fix, as a unified diff. Used to smoke-test the
        verification pipeline itself, the same role `Task.reference_solution`
        plays: if the reference patch does not resolve the instance under
        this project's own harness wiring, the wiring is broken, not the
        model.
    test_patch:
        Diff that introduces the `fail_to_pass`/`pass_to_pass` tests
        themselves, applied before grading regardless of what the model
        produced.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    patch: str
    test_patch: str
    environment_setup_commit: str = ""
    version: str = ""
    difficulty: str = ""
    metadata: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        """Hash of the semantically load-bearing fields.

        Excludes `metadata`, mirroring `Task.content_hash`'s exclusion of
        `split`/`metadata` -- re-splitting must not invalidate identity, but
        any change to the problem or its grading tests must.
        """
        payload = json.dumps(
            {
                "instance_id": self.instance_id,
                "repo": self.repo,
                "base_commit": self.base_commit,
                "problem_statement": self.problem_statement,
                "fail_to_pass": list(self.fail_to_pass),
                "pass_to_pass": list(self.pass_to_pass),
                "patch": self.patch,
                "test_patch": self.test_patch,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SweBenchSuite:
    """A named, hashable collection of `SweBenchInstance`s.

    Deliberately not `cbs.tasks.schema.TaskSuite` -- see module docstring --
    but mirrors its frozen-hashed-split discipline (brief section 7, Phase 1
    DoD) rather than skipping it for this family.
    """

    name: str
    instances: tuple[SweBenchInstance, ...]

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    def by_id(self) -> dict[str, SweBenchInstance]:
        return {inst.instance_id: inst for inst in self.instances}

    def suite_hash(self) -> str:
        """Stable hash over the whole suite -- the number quoted in results
        to prove the split was frozen before sampling, same role as
        `TaskSuite.suite_hash`."""
        h = hashlib.sha256()
        for inst in sorted(self.instances, key=lambda i: i.instance_id):
            h.update(inst.instance_id.encode("utf-8"))
            h.update(b"\x00")
            h.update(inst.content_hash().encode("utf-8"))
            h.update(b"\x01")
        return h.hexdigest()


def load_swebench_verified(instance_ids: list[str] | None = None) -> SweBenchSuite:
    """Load the real SWE-bench Verified dataset directly from HuggingFace.

    Requires the `swebench` optional extra (`datasets`). Not vendored --
    unlike `humanevalplus`/`mbppplus` (a few hundred KB each), SWE-bench
    Verified's repos-at-a-commit shape makes "vendor a copy" meaningless;
    the actual per-instance content lives in the real git repos HGM's own
    harness clones and builds Docker images from, not in this dataset row.

    Parameters
    ----------
    instance_ids:
        If given, load only these instances (already known to exist in the
        dataset) rather than all 500 -- useful for a smoke test or a small
        frozen split without paying to materialise the whole dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise ImportError(
            "load_swebench_verified needs the 'swebench' optional extra "
            "(pip install 'cbs[swebench]') for the `datasets` package."
        ) from exc

    rows = load_dataset(HF_DATASET_NAME, split="test")
    if instance_ids is not None:
        wanted = set(instance_ids)
        rows = [row for row in rows if row["instance_id"] in wanted]

    instances = tuple(
        SweBenchInstance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            fail_to_pass=tuple(json.loads(row["FAIL_TO_PASS"])),
            pass_to_pass=tuple(json.loads(row["PASS_TO_PASS"])),
            patch=row["patch"],
            test_patch=row["test_patch"],
            environment_setup_commit=row.get("environment_setup_commit", ""),
            version=row.get("version", ""),
            difficulty=row.get("difficulty", ""),
        )
        for row in rows
    )
    return SweBenchSuite(name="swebench_verified", instances=instances)
