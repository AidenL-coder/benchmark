"""Task families.

`toy` is self-contained and vendored (no download), so the instrument can be
validated end to end offline. Real benchmark families (HumanEval+, MBPP+,
LiveCodeBench, SWE-bench Verified) are loaded from data fetched separately --
see `cbs data status`.
"""

from __future__ import annotations

from cbs.tasks.families.toy import (
    TOY_TASKS,
    ToyTaskDef,
    toy_behaviours,
    toy_defs_by_id,
    toy_suite,
)

__all__ = [
    "TOY_TASKS",
    "ToyTaskDef",
    "toy_behaviours",
    "toy_defs_by_id",
    "toy_suite",
]
