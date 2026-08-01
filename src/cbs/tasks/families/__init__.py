"""Task families.

`toy` is synthetic and vendored as code, with fully known ground truth (used
to validate the instrument itself). `humaneval` is a real, vendored benchmark
(brief section 8) -- see `data/vendored/humaneval/ATTRIBUTION.md` for
provenance and the caveats that apply before it is used for anything beyond
instrument validation (original HumanEval, not HumanEval+; likely pretraining
contamination). Further real families (MBPP+, LiveCodeBench, SWE-bench
Verified, a transfer/reasoning family) remain open -- D-13/D-17.
"""

from __future__ import annotations

from cbs.tasks.families.humaneval import DEFAULT_HUMANEVAL_PATH, humaneval_suite
from cbs.tasks.families.humanevalplus import (
    DEFAULT_HUMANEVALPLUS_PATH,
    humanevalplus_suite,
)
from cbs.tasks.families.mbpp import DEFAULT_MBPP_PATH, mbpp_suite
from cbs.tasks.families.mbppplus import DEFAULT_MBPPPLUS_PATH, mbppplus_suite
from cbs.tasks.families.toy import (
    TOY_TASKS,
    ToyTaskDef,
    toy_behaviours,
    toy_defs_by_id,
    toy_suite,
)
from cbs.tasks.families.transfer_reasoning import (
    TRANSFER_TASKS,
    TransferTaskDef,
    transfer_suite,
)

__all__ = [
    "DEFAULT_HUMANEVAL_PATH",
    "DEFAULT_HUMANEVALPLUS_PATH",
    "DEFAULT_MBPP_PATH",
    "DEFAULT_MBPPPLUS_PATH",
    "TOY_TASKS",
    "TRANSFER_TASKS",
    "ToyTaskDef",
    "TransferTaskDef",
    "humaneval_suite",
    "humanevalplus_suite",
    "mbpp_suite",
    "mbppplus_suite",
    "toy_behaviours",
    "toy_defs_by_id",
    "toy_suite",
    "transfer_suite",
]
