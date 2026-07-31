"""Matched-compute accounting.

The brief (section 4) requires that `S0`, `S_star` and `S_evo` be compared at
**matched total inference compute**, and section 3.3 requires that any claimed
frontier-crossing be measured at compute matched to the frontier estimator's
`N_max`. That guarantee is only as good as the accounting, so every model call
in this codebase is charged to an accountant and no system may spend outside one.

Two distinct notions live here and must not be conflated:

*   A **cap** is a hard ceiling. Exceeding it raises `BudgetExceeded`. Caps exist
    for safety/cost control (brief section 10) and are typically per-phase.
*   An **allowance** is the matched-compute quantity: the *identical* per-task
    budget handed to each system being compared. Two systems are compared fairly
    iff they drew from equal allowances.

`MatchedComputeHarness` issues allowances and afterwards *verifies* that the
systems actually consumed comparable compute, because an allowance that one
system under-spends is not by itself evidence of a matched comparison -- it is
evidence that the comparison needs reporting with the realised spend attached.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Iterator, Mapping

__all__ = [
    "Usage",
    "BudgetCaps",
    "BudgetExceeded",
    "BudgetAccountant",
    "MatchedComputeHarness",
    "MatchReport",
]


@dataclass(frozen=True)
class Usage:
    """Resource consumption of one model call (or an aggregate of several)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            usd=self.usd + other.usd,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usd": round(self.usd, 6),
        }


ZERO_USAGE = Usage()


@dataclass(frozen=True)
class BudgetCaps:
    """Hard ceilings. `None` means "no cap on this dimension"."""

    calls: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    usd: float | None = None

    @classmethod
    def unlimited(cls) -> "BudgetCaps":
        return cls()

    def scaled(self, factor: float) -> "BudgetCaps":
        def s(v):
            return None if v is None else type(v)(v * factor)

        return BudgetCaps(
            calls=None if self.calls is None else int(self.calls * factor),
            prompt_tokens=s(self.prompt_tokens),
            completion_tokens=s(self.completion_tokens),
            total_tokens=s(self.total_tokens),
            usd=s(self.usd),
        )

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usd": self.usd,
        }


class BudgetExceeded(RuntimeError):
    """Raised when a charge would push an accountant past one of its caps.

    Carries the offending dimension so callers (and the run log) can report
    *which* budget bound the run, which matters for interpreting a result as
    "compute-limited" rather than "capability-limited".
    """

    def __init__(self, label: str, dimension: str, attempted: float, cap: float):
        super().__init__(
            f"budget '{label}' exceeded on {dimension}: "
            f"attempted {attempted} > cap {cap}"
        )
        self.label = label
        self.dimension = dimension
        self.attempted = attempted
        self.cap = cap


class BudgetAccountant:
    """Thread-safe ledger of inference compute, optionally capped.

    Accountants nest: a child's charges also propagate to its parent, so a
    per-task allowance can sit inside a per-phase cap and both are enforced.
    """

    def __init__(
        self,
        label: str,
        caps: BudgetCaps | None = None,
        parent: "BudgetAccountant | None" = None,
    ):
        self.label = label
        self.caps = caps or BudgetCaps.unlimited()
        self.parent = parent
        self._spent = ZERO_USAGE
        self._lock = threading.RLock()
        self._children: list[BudgetAccountant] = []
        if parent is not None:
            parent._children.append(self)

    # -- inspection -------------------------------------------------------
    @property
    def spent(self) -> Usage:
        with self._lock:
            return self._spent

    def remaining(self) -> dict[str, float | int | None]:
        """Headroom per dimension; `None` where uncapped."""
        s, c = self.spent, self.caps
        return {
            "calls": None if c.calls is None else max(0, c.calls - s.calls),
            "prompt_tokens": (
                None if c.prompt_tokens is None else max(0, c.prompt_tokens - s.prompt_tokens)
            ),
            "completion_tokens": (
                None
                if c.completion_tokens is None
                else max(0, c.completion_tokens - s.completion_tokens)
            ),
            "total_tokens": (
                None if c.total_tokens is None else max(0, c.total_tokens - s.total_tokens)
            ),
            "usd": None if c.usd is None else max(0.0, c.usd - s.usd),
        }

    def would_exceed(self, usage: Usage) -> tuple[str, float, float] | None:
        """Return `(dimension, attempted, cap)` if `usage` would breach a cap."""
        s, c = self.spent, self.caps
        prospective = s + usage
        checks = (
            ("calls", prospective.calls, c.calls),
            ("prompt_tokens", prospective.prompt_tokens, c.prompt_tokens),
            ("completion_tokens", prospective.completion_tokens, c.completion_tokens),
            ("total_tokens", prospective.total_tokens, c.total_tokens),
            ("usd", prospective.usd, c.usd),
        )
        for dim, attempted, cap in checks:
            if cap is not None and attempted > cap:
                return dim, attempted, cap
        return None

    def can_afford(self, usage: Usage) -> bool:
        """True iff `usage` fits in this accountant *and every ancestor*."""
        node: BudgetAccountant | None = self
        while node is not None:
            if node.would_exceed(usage) is not None:
                return False
            node = node.parent
        return True

    # -- mutation ---------------------------------------------------------
    def charge(self, usage: Usage) -> None:
        """Record `usage`, or raise `BudgetExceeded` and record nothing.

        The check runs over the whole ancestor chain *before* any mutation, so a
        rejected charge leaves every ledger untouched. Without that, a call
        rejected by a parent cap would still be counted by the child and silently
        corrupt the matched-compute comparison.
        """
        chain: list[BudgetAccountant] = []
        node: BudgetAccountant | None = self
        while node is not None:
            chain.append(node)
            node = node.parent

        for acct in chain:
            breach = acct.would_exceed(usage)
            if breach is not None:
                dim, attempted, cap = breach
                raise BudgetExceeded(acct.label, dim, attempted, cap)

        for acct in chain:
            with acct._lock:
                acct._spent = acct._spent + usage

    def child(self, label: str, caps: BudgetCaps | None = None) -> "BudgetAccountant":
        return BudgetAccountant(label=f"{self.label}/{label}", caps=caps, parent=self)

    def snapshot(self) -> dict:
        return {
            "label": self.label,
            "caps": self.caps.as_dict(),
            "spent": self.spent.as_dict(),
            "remaining": self.remaining(),
            "children": [c.snapshot() for c in self._children],
        }


@dataclass
class MatchReport:
    """Post-hoc evidence that a comparison really was compute-matched."""

    allowance: BudgetCaps
    spent: dict[str, Usage] = field(default_factory=dict)
    tolerance: float = 0.05

    @property
    def matched(self) -> bool:
        """True iff every system's realised spend is within `tolerance` of the max.

        Compares realised spend, not allowances. Equal allowances are necessary
        but not sufficient: a system that under-spends its allowance is not
        running at matched compute, and reporting it as such would understate the
        compute advantage of whichever system spent more.
        """
        if len(self.spent) < 2:
            return True
        for dim in ("calls", "total_tokens"):
            values = [getattr(u, dim) for u in self.spent.values()]
            hi = max(values)
            if hi == 0:
                continue
            if min(values) < hi * (1.0 - self.tolerance):
                return False
        return True

    def discrepancies(self) -> dict[str, dict[str, float]]:
        """Per-dimension realised spend as a fraction of the largest spender."""
        out: dict[str, dict[str, float]] = {}
        for dim in ("calls", "total_tokens"):
            values = {k: getattr(u, dim) for k, u in self.spent.items()}
            hi = max(values.values()) if values else 0
            out[dim] = {
                k: (1.0 if hi == 0 else v / hi) for k, v in values.items()
            }
        return out

    def as_dict(self) -> dict:
        return {
            "allowance": self.allowance.as_dict(),
            "spent": {k: v.as_dict() for k, v in self.spent.items()},
            "tolerance": self.tolerance,
            "matched": self.matched,
            "relative_spend": self.discrepancies(),
        }


class MatchedComputeHarness:
    """Hands identical per-task allowances to each system under comparison.

    Usage::

        harness = MatchedComputeHarness(allowance, root_accountant, tolerance=0.05)
        for system in ("S0", "S_star", "S_evo"):
            acct = harness.allowance_for(system, task_id)
            ...run the system against `acct`...
        report = harness.report_for(task_id)
        assert report.matched
    """

    def __init__(
        self,
        allowance: BudgetCaps,
        root: BudgetAccountant | None = None,
        tolerance: float = 0.05,
    ):
        self.allowance = allowance
        self.root = root or BudgetAccountant("root")
        self.tolerance = tolerance
        self._accounts: dict[tuple[str, str], BudgetAccountant] = {}

    def allowance_for(self, system: str, task_id: str) -> BudgetAccountant:
        key = (system, task_id)
        if key not in self._accounts:
            self._accounts[key] = self.root.child(
                f"{system}:{task_id}", caps=self.allowance
            )
        return self._accounts[key]

    def report_for(self, task_id: str) -> MatchReport:
        spent = {
            system: acct.spent
            for (system, tid), acct in self._accounts.items()
            if tid == task_id
        }
        return MatchReport(
            allowance=self.allowance, spent=spent, tolerance=self.tolerance
        )

    def systems(self) -> set[str]:
        return {system for system, _ in self._accounts}

    def task_ids(self) -> set[str]:
        return {task_id for _, task_id in self._accounts}

    def overall_report(self) -> dict:
        """Aggregate match evidence across every task."""
        per_task = {tid: self.report_for(tid) for tid in sorted(self.task_ids())}
        unmatched = [tid for tid, r in per_task.items() if not r.matched]
        return {
            "allowance": self.allowance.as_dict(),
            "tolerance": self.tolerance,
            "n_tasks": len(per_task),
            "n_unmatched": len(unmatched),
            "unmatched_task_ids": unmatched,
            "per_task": {tid: r.as_dict() for tid, r in per_task.items()},
        }
