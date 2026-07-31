"""Tests for the frontier-crossing determination (brief section 3.3).

Builds each of `FrontierRecord` / `ScaffoldRunSummary` / `AblationResult`
directly with synthetic fields rather than running the full pipeline: the
object under test is the four-condition combination logic, not the objects
that feed it (those are covered in test_sampler.py, test_compare.py,
test_ablation.py respectively).
"""

from __future__ import annotations

import pytest

from cbs.ablation import AblationResult
from cbs.budget import Usage
from cbs.compare import ScaffoldRunSummary
from cbs.crossing import evaluate_crossing
from cbs.frontier.records import FrontierRecord


def make_s0_record(task_id="t", n_samples=1000, n_correct=0, beyond_frontier=True):
    return FrontierRecord(
        task_id=task_id,
        model_id="m",
        split="held_out",
        family="f",
        n_samples=n_samples,
        n_correct=n_correct,
        p_hat=n_correct / n_samples,
        p_ci_low=0.0,
        p_ci_high=0.01,
        p_upper_bound=0.01,
        rule_of_three_bound=3 / n_samples,
        exact_zero_success_bound=0.003,
        beyond_frontier=beyond_frontier,
        n_distinct_solutions=0,
        species_counts={},
        good_turing_unseen_mass=None,
        sample_coverage=None,
        chao1=None,
        pass_at_k={},
        rarefaction={},
        temperature_schedule=[],
        scaffold_fingerprint={},
        usage=Usage(),
        verifier_backend="subprocess",
        verifier_is_security_boundary=False,
    )


def make_summary(task_id="t", budget_calls=1000, n_reps=10, n_solved=8):
    return ScaffoldRunSummary(
        scaffold_name="S_evo",
        task_id=task_id,
        budget_calls=budget_calls,
        n_reps=n_reps,
        n_solved=n_solved,
        p_hat=n_solved / n_reps,
        ci_low=0.3,
        ci_high=0.9,
        mean_calls=budget_calls * 0.8,
        mean_total_tokens=1000.0,
        n_budget_exhausted=0,
        op_counts={"single_call": n_reps},
        expanding_dependency=1.0,
    )


def make_ablation(task_id="t", baseline_solved=8, ablated_solved=1, n_reps=10):
    baseline = make_summary(task_id=task_id, n_reps=n_reps, n_solved=baseline_solved)
    ablated = make_summary(task_id=task_id, n_reps=n_reps, n_solved=ablated_solved)
    return AblationResult(
        task_id=task_id, ablated_op="execution_feedback", baseline=baseline, ablated=ablated
    )


class TestFourConditions:
    def test_full_crossing_when_all_four_hold(self):
        s0 = make_s0_record(n_samples=1000, n_correct=0)
        summary = make_summary(budget_calls=1000, n_reps=10, n_solved=8)
        ablation = make_ablation(baseline_solved=8, ablated_solved=1, n_reps=10)
        verdict = evaluate_crossing(s0, summary, k=6, K=10, ablation=ablation)
        assert verdict.reliable_solve
        assert verdict.beyond_frontier
        assert verdict.compute_matched
        assert verdict.ablation_removes_crossing is True
        assert verdict.crossed

    def test_not_reliable_prevents_crossing(self):
        s0 = make_s0_record(n_samples=1000, n_correct=0)
        summary = make_summary(budget_calls=1000, n_reps=10, n_solved=2)  # 20% < 60%
        ablation = make_ablation(baseline_solved=2, ablated_solved=0, n_reps=10)
        verdict = evaluate_crossing(s0, summary, k=6, K=10, ablation=ablation)
        assert not verdict.reliable_solve
        assert not verdict.crossed

    def test_not_beyond_frontier_prevents_crossing_even_if_solved_reliably(self):
        """S0 already solves this sometimes -- not a beyond-frontier task, so
        even a reliable evolved-scaffold solve is not a 'crossing'."""
        s0 = make_s0_record(n_samples=1000, n_correct=50, beyond_frontier=False)
        summary = make_summary(budget_calls=1000, n_reps=10, n_solved=9)
        ablation = make_ablation(baseline_solved=9, ablated_solved=0, n_reps=10)
        verdict = evaluate_crossing(s0, summary, k=6, K=10, ablation=ablation)
        assert not verdict.beyond_frontier
        assert not verdict.crossed

    def test_mismatched_budget_raises_rather_than_silently_comparing(self):
        """p_hat(x) < 1/N_max is only well-defined when N_max is the budget
        actually being compared against -- silently allowing a mismatch would
        let 'beyond frontier at N_max=1000' back a claim at a totally
        different compute budget."""
        s0 = make_s0_record(n_samples=1000, n_correct=0)
        summary = make_summary(budget_calls=50, n_reps=10, n_solved=8)
        with pytest.raises(ValueError, match="only well-defined"):
            evaluate_crossing(s0, summary, k=6, K=10)

    def test_no_ablation_leaves_ablation_condition_unresolved(self):
        """Without an ablation, the fourth condition cannot be confirmed, so
        the verdict must not claim a crossing -- but the other three
        conditions are still worth reporting."""
        s0 = make_s0_record(n_samples=1000, n_correct=0)
        summary = make_summary(budget_calls=1000, n_reps=10, n_solved=8)
        verdict = evaluate_crossing(s0, summary, k=6, K=10, ablation=None)
        assert verdict.ablation_removes_crossing is None
        assert not verdict.crossed  # ablation_removes is None, not True
        assert verdict.reliable_solve and verdict.beyond_frontier and verdict.compute_matched

    def test_ablation_not_removing_the_solve_prevents_crossing(self):
        """If the ablated variant still solves reliably, the solve did not
        depend on the ablated support-expanding operation -- not a crossing
        attributable to that mechanism."""
        s0 = make_s0_record(n_samples=1000, n_correct=0)
        summary = make_summary(budget_calls=1000, n_reps=10, n_solved=8)
        ablation = make_ablation(baseline_solved=8, ablated_solved=7, n_reps=10)  # still reliable
        verdict = evaluate_crossing(s0, summary, k=6, K=10, ablation=ablation)
        assert verdict.ablation_removes_crossing is False
        assert not verdict.crossed

    def test_mismatched_task_ids_between_s0_and_summary_raises(self):
        s0 = make_s0_record(task_id="a")
        summary = make_summary(task_id="b", budget_calls=1000)
        with pytest.raises(ValueError, match="mismatched task ids"):
            evaluate_crossing(s0, summary, k=6, K=10)

    def test_mismatched_task_id_in_ablation_raises(self):
        s0 = make_s0_record(task_id="a", n_samples=1000)
        summary = make_summary(task_id="a", budget_calls=1000, n_reps=10, n_solved=8)
        ablation = make_ablation(task_id="different", n_reps=10)
        with pytest.raises(ValueError, match="ablation is for"):
            evaluate_crossing(s0, summary, k=6, K=10, ablation=ablation)


class TestContestedOpFlag:
    def test_flags_when_a_contested_op_was_used(self):
        s0 = make_s0_record(n_samples=100, n_correct=0)
        summary = make_summary(budget_calls=100, n_reps=10, n_solved=8)
        verdict = evaluate_crossing(
            s0, summary, k=6, K=10, expanding_ops_used=frozenset({"temperature_schedule"})
        )
        assert verdict.rests_on_contested_op

    def test_not_flagged_for_uncontested_ops(self):
        s0 = make_s0_record(n_samples=100, n_correct=0)
        summary = make_summary(budget_calls=100, n_reps=10, n_solved=8)
        verdict = evaluate_crossing(
            s0, summary, k=6, K=10, expanding_ops_used=frozenset({"execution_feedback"})
        )
        assert not verdict.rests_on_contested_op


class TestReporting:
    def test_summary_line_for_non_beyond_frontier_task(self):
        s0 = make_s0_record(n_samples=100, n_correct=50, beyond_frontier=False)
        summary = make_summary(budget_calls=100, n_reps=10, n_solved=9)
        verdict = evaluate_crossing(s0, summary, k=6, K=10)
        assert "not beyond-frontier" in verdict.summary_line()

    def test_as_dict_round_trips_fields(self):
        s0 = make_s0_record(n_samples=100, n_correct=0)
        summary = make_summary(budget_calls=100, n_reps=10, n_solved=8)
        ablation = make_ablation(baseline_solved=8, ablated_solved=1, n_reps=10)
        verdict = evaluate_crossing(s0, summary, k=6, K=10, ablation=ablation)
        payload = verdict.as_dict()
        assert payload["crossed"] == verdict.crossed
        assert payload["task_id"] == "t"
