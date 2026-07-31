"""Support-tagged scaffolds."""

from __future__ import annotations

from cbs.scaffolds.base import Scaffold, ScaffoldResult
from cbs.scaffolds.s0 import DEFAULT_SYSTEM_PROMPT, S0
from cbs.scaffolds.s_star import SStar
from cbs.scaffolds.tagging import (
    OperationTrace,
    OpDefinition,
    OpRecord,
    SupportClass,
    UnregisteredOperation,
    contested_operations,
    definition_of,
    register_operation,
    registry_snapshot,
    support_class_of,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "OpDefinition",
    "OpRecord",
    "OperationTrace",
    "S0",
    "SStar",
    "Scaffold",
    "ScaffoldResult",
    "SupportClass",
    "UnregisteredOperation",
    "contested_operations",
    "definition_of",
    "register_operation",
    "registry_snapshot",
    "support_class_of",
]
