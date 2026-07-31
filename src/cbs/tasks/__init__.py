"""Task schema, verification, canonicalisation and frozen splits."""

from __future__ import annotations

from cbs.tasks.canonicalize import CanonicalForm, canonicalize_solution, species_key
from cbs.tasks.schema import Split, Task, TaskSuite
from cbs.tasks.splits import (
    SplitManifest,
    SplitRatios,
    assign_splits,
    build_manifest,
    verify_manifest,
    write_manifest,
)
from cbs.tasks.verifier import VerificationResult, Verifier, extract_code

__all__ = [
    "CanonicalForm",
    "Split",
    "SplitManifest",
    "SplitRatios",
    "Task",
    "TaskSuite",
    "VerificationResult",
    "Verifier",
    "assign_splits",
    "build_manifest",
    "canonicalize_solution",
    "extract_code",
    "species_key",
    "verify_manifest",
    "write_manifest",
]
