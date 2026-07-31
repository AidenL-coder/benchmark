"""Budget accountant tests.

Matched-compute is the control that makes every comparison in the study
meaningful (brief section 4), so the accountant's guarantees are tested
directly -- in particular that a rejected charge leaves no partial state, since
a charge counted by a child but rejected by a parent would corrupt the
comparison silently.
"""

from __future__ import annotations

import pytest

from cbs.budget import (
    BudgetAccountant,
    BudgetCaps,
    BudgetExceeded,
    MatchedComputeHarness,
    Usage,
)


class TestUsage:
    def test_addition_and_totals(self):
        a = Usage(calls=1, prompt_tokens=10, completion_tokens=5, usd=0.1)
        b = Usage(calls=2, prompt_tokens=20, completion_tokens=7, usd=0.2)
        total = a + b
        assert total.calls == 3
        assert total.total_tokens == 42
        assert total.usd == pytest.approx(0.3)


class TestAccountant:
    def test_charges_accumulate(self):
        acct = BudgetAccountant("t")
        acct.charge(Usage(calls=1, prompt_tokens=5))
        acct.charge(Usage(calls=1, prompt_tokens=7))
        assert acct.spent.calls == 2
        assert acct.spent.prompt_tokens == 12

    def test_cap_enforced(self):
        acct = BudgetAccountant("t", BudgetCaps(calls=2))
        acct.charge(Usage(calls=1))
        acct.charge(Usage(calls=1))
        with pytest.raises(BudgetExceeded) as exc:
            acct.charge(Usage(calls=1))
        assert exc.value.dimension == "calls"

    def test_rejected_charge_is_not_recorded(self):
        acct = BudgetAccountant("t", BudgetCaps(calls=1))
        acct.charge(Usage(calls=1))
        with pytest.raises(BudgetExceeded):
            acct.charge(Usage(calls=1))
        assert acct.spent.calls == 1

    def test_charge_propagates_to_parent(self):
        parent = BudgetAccountant("parent")
        child = parent.child("child")
        child.charge(Usage(calls=3, prompt_tokens=30))
        assert parent.spent.calls == 3
        assert parent.spent.prompt_tokens == 30

    def test_parent_cap_rejects_child_charge_atomically(self):
        """The important one: a parent-rejected charge must not land on the child.

        Otherwise the child's ledger would over-report spend for a call that
        never happened, and a matched-compute comparison built on those ledgers
        would be wrong in favour of whichever system hit the cap.
        """
        parent = BudgetAccountant("parent", BudgetCaps(calls=2))
        child = parent.child("child")
        child.charge(Usage(calls=2))
        with pytest.raises(BudgetExceeded) as exc:
            child.charge(Usage(calls=1))
        assert exc.value.label == "parent"
        assert child.spent.calls == 2
        assert parent.spent.calls == 2

    def test_can_afford_checks_whole_ancestor_chain(self):
        parent = BudgetAccountant("parent", BudgetCaps(calls=2))
        child = parent.child("child", BudgetCaps(calls=100))
        assert child.can_afford(Usage(calls=2))
        assert not child.can_afford(Usage(calls=3))

    def test_usd_cap(self):
        acct = BudgetAccountant("t", BudgetCaps(usd=1.0))
        acct.charge(Usage(usd=0.75))
        with pytest.raises(BudgetExceeded) as exc:
            acct.charge(Usage(usd=0.5))
        assert exc.value.dimension == "usd"

    def test_total_tokens_cap_spans_both_directions(self):
        acct = BudgetAccountant("t", BudgetCaps(total_tokens=100))
        acct.charge(Usage(prompt_tokens=60, completion_tokens=30))
        with pytest.raises(BudgetExceeded):
            acct.charge(Usage(prompt_tokens=20))

    def test_remaining(self):
        acct = BudgetAccountant("t", BudgetCaps(calls=10))
        acct.charge(Usage(calls=4))
        assert acct.remaining()["calls"] == 6
        assert acct.remaining()["usd"] is None


class TestMatchedCompute:
    def test_equal_spend_is_matched(self):
        harness = MatchedComputeHarness(BudgetCaps(calls=10))
        for system in ("S0", "S_star", "S_evo"):
            harness.allowance_for(system, "task-1").charge(Usage(calls=5, prompt_tokens=50))
        assert harness.report_for("task-1").matched

    def test_unequal_realised_spend_is_not_matched(self):
        """Equal allowances are not enough; realised spend is what is compared."""
        harness = MatchedComputeHarness(BudgetCaps(calls=10))
        harness.allowance_for("S0", "task-1").charge(Usage(calls=10, prompt_tokens=100))
        harness.allowance_for("S_evo", "task-1").charge(Usage(calls=2, prompt_tokens=20))
        report = harness.report_for("task-1")
        assert not report.matched
        assert report.discrepancies()["calls"]["S_evo"] == pytest.approx(0.2)

    def test_within_tolerance_is_matched(self):
        harness = MatchedComputeHarness(BudgetCaps(calls=100), tolerance=0.05)
        harness.allowance_for("A", "t").charge(Usage(calls=100, prompt_tokens=1000))
        harness.allowance_for("B", "t").charge(Usage(calls=97, prompt_tokens=970))
        assert harness.report_for("t").matched

    def test_single_system_is_trivially_matched(self):
        harness = MatchedComputeHarness(BudgetCaps(calls=10))
        harness.allowance_for("S0", "t").charge(Usage(calls=1))
        assert harness.report_for("t").matched

    def test_allowance_is_identical_across_systems(self):
        harness = MatchedComputeHarness(BudgetCaps(calls=7, total_tokens=70))
        a = harness.allowance_for("S0", "t")
        b = harness.allowance_for("S_evo", "t")
        assert a.caps == b.caps
        assert a is not b

    def test_overall_report_flags_unmatched_tasks(self):
        harness = MatchedComputeHarness(BudgetCaps(calls=10))
        harness.allowance_for("A", "ok").charge(Usage(calls=5, prompt_tokens=5))
        harness.allowance_for("B", "ok").charge(Usage(calls=5, prompt_tokens=5))
        harness.allowance_for("A", "bad").charge(Usage(calls=10, prompt_tokens=10))
        harness.allowance_for("B", "bad").charge(Usage(calls=1, prompt_tokens=1))
        report = harness.overall_report()
        assert report["n_unmatched"] == 1
        assert report["unmatched_task_ids"] == ["bad"]
