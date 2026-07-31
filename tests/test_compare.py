"""Tests for the S0-vs-S_star matched-compute comparator.

The comparator's core correctness property is that S0's side is read off an
*existing* frontier record via pass@k rather than resampled, and that this
degenerates to an uninformative 1.0/0.0 exactly when n_samples == k -- which is
why the CLI oversamples S0 by default (see cbs/cli.py cmd_compare). These tests
pin both the correct (oversampled) and degenerate (n==k) cases so the failure
mode cannot silently regress.
"""

from __future__ import annotations

import json

import pytest

from cbs.budget import BudgetAccountant
from cbs.compare import ScaffoldComparator, load_frontier_records
from cbs.frontier.sampler import FrontierSampler
from cbs.models.mock import MockModelClient
from cbs.sandbox import select_backend
from cbs.scaffolds.s_star import SStar
from cbs.tasks import Verifier
from cbs.tasks.families.toy import toy_behaviours, toy_suite


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


def make_s0_record(tmp_path, task_id, n_max, seed=0, verifier=None):
    suite = toy_suite().by_id()
    model = MockModelClient(toy_behaviours(), seed=seed, model_id="cmp-test")
    sampler = FrontierSampler(model=model, verifier=verifier, output_dir=tmp_path)
    return sampler.estimate_task(suite[task_id], n_max, BudgetAccountant("s0"), resume=False)


class TestS0SummaryFromRecord:
    def test_oversampled_record_gives_informative_ci(self, tmp_path, verifier):
        """N_max well above the comparison budget: a real, non-degenerate CI."""
        record = make_s0_record(tmp_path, "toy/gcd", n_max=200, verifier=verifier)
        comparator = ScaffoldComparator(
            model=None, verifier=verifier, s_star=SStar(), budget_calls=10, n_reps=1
        )
        summary = comparator.s0_summary_from_record(record, "toy/gcd")
        assert 0.0 < summary.p_hat < 1.0
        assert summary.ci_low < summary.ci_high
        # A real (non-trivial) interval, not collapsed to a point.
        assert summary.ci_high - summary.ci_low > 1e-6

    def test_n_equals_k_degenerates_when_any_success_seen(self, tmp_path, verifier):
        """Documents the degeneracy the CLI's oversampling exists to avoid."""
        record = make_s0_record(tmp_path, "toy/gcd", n_max=10, verifier=verifier)
        assume_any_success = record.n_correct > 0
        comparator = ScaffoldComparator(
            model=None, verifier=verifier, s_star=SStar(), budget_calls=10, n_reps=1
        )
        summary = comparator.s0_summary_from_record(record, "toy/gcd")
        if assume_any_success:
            assert summary.p_hat == pytest.approx(1.0)

    def test_rejects_a_record_smaller_than_the_budget(self, tmp_path, verifier):
        record = make_s0_record(tmp_path, "toy/gcd", n_max=5, verifier=verifier)
        comparator = ScaffoldComparator(
            model=None, verifier=verifier, s_star=SStar(), budget_calls=10, n_reps=1
        )
        with pytest.raises(ValueError, match="fewer"):
            comparator.s0_summary_from_record(record, "toy/gcd")

    def test_beyond_frontier_task_gives_zero_solve_rate(self, tmp_path, verifier):
        record = make_s0_record(tmp_path, "toy/impossible_parity", n_max=50, verifier=verifier)
        comparator = ScaffoldComparator(
            model=None, verifier=verifier, s_star=SStar(), budget_calls=10, n_reps=1
        )
        summary = comparator.s0_summary_from_record(record, "toy/impossible_parity")
        assert summary.p_hat == 0.0


class TestSStarSummary:
    def test_solve_rate_has_a_valid_ci(self, verifier):
        model = MockModelClient(toy_behaviours(), seed=0, model_id="cmp-test")
        comparator = ScaffoldComparator(
            model=model, verifier=verifier, s_star=SStar(max_candidates=3),
            budget_calls=15, n_reps=20,
        )
        suite = toy_suite().by_id()
        summary = comparator.s_star_summary(suite["toy/sum_list"])
        assert 0.0 <= summary.p_hat <= 1.0
        assert summary.ci_low <= summary.p_hat <= summary.ci_high
        assert summary.n_reps == 20
        assert summary.mean_calls > 0

    def test_zero_true_p_task_never_solved(self, verifier):
        model = MockModelClient(toy_behaviours(), seed=0, model_id="cmp-test")
        comparator = ScaffoldComparator(
            model=model, verifier=verifier, s_star=SStar(max_candidates=3),
            budget_calls=15, n_reps=15,
        )
        suite = toy_suite().by_id()
        summary = comparator.s_star_summary(suite["toy/impossible_parity"])
        assert summary.n_solved == 0
        assert summary.p_hat == 0.0

    def test_respects_the_call_budget_every_repetition(self, verifier):
        model = MockModelClient(toy_behaviours(), seed=0, model_id="cmp-test")
        budget = 4
        comparator = ScaffoldComparator(
            model=model, verifier=verifier,
            s_star=SStar(max_candidates=8, max_repairs_per_candidate=4, stop_on_first_public_pass=False),
            budget_calls=budget, n_reps=10,
        )
        suite = toy_suite().by_id()
        comparator.s_star_summary(suite["toy/gcd"])
        for (system, task_pseudo), acct in comparator.harness._accounts.items():
            if system == "S_star":
                assert acct.spent.calls <= budget


class TestComparisonRecord:
    def test_elicitation_gain_is_the_signed_difference(self, tmp_path, verifier):
        record = make_s0_record(tmp_path, "toy/gcd", n_max=200, verifier=verifier)
        model = MockModelClient(toy_behaviours(), seed=1, model_id="cmp-test")
        comparator = ScaffoldComparator(
            model=model, verifier=verifier, s_star=SStar(max_candidates=3),
            budget_calls=10, n_reps=20,
        )
        suite = toy_suite().by_id()
        result = comparator.compare_task(suite["toy/gcd"], record)
        assert result.elicitation_gain == pytest.approx(
            result.s_star.p_hat - result.s0.p_hat, abs=1e-9
        )

    def test_realised_spend_ratio_reports_both_systems(self, tmp_path, verifier):
        record = make_s0_record(tmp_path, "toy/gcd", n_max=200, verifier=verifier)
        model = MockModelClient(toy_behaviours(), seed=1, model_id="cmp-test")
        comparator = ScaffoldComparator(
            model=model, verifier=verifier, s_star=SStar(max_candidates=3),
            budget_calls=10, n_reps=10,
        )
        suite = toy_suite().by_id()
        result = comparator.compare_task(suite["toy/gcd"], record)
        assert set(result.realised_spend_ratio) == {"S0", "S_star"}
        assert result.realised_spend_ratio["S0"] == pytest.approx(1 / 10)


class TestLoadFrontierRecords:
    def test_round_trips_essential_fields(self, tmp_path, verifier):
        record = make_s0_record(tmp_path, "toy/gcd", n_max=30, verifier=verifier)
        path = tmp_path / "records.jsonl"
        path.write_text(record.to_json() + "\n", encoding="utf-8")
        loaded = load_frontier_records(path)
        assert "toy/gcd" in loaded
        r = loaded["toy/gcd"]
        assert r.n_samples == record.n_samples
        assert r.n_correct == record.n_correct
        assert r.species_counts == record.species_counts
