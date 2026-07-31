"""Tests for the ablation runner (brief section 9.2).

Validated against EvolvedScaffold + the example agents (safe to run today) as
a stand-in for the real S_evo-vs-ablated-S_evo comparison Phase 5 will run
once a Docker-capable host exists -- run_ablation itself is scaffold-agnostic.
"""

from __future__ import annotations

from cbs.ablation import run_ablation
from cbs.models.mock import MockModelClient
from cbs.sandbox import select_backend
from cbs.scaffolds.evolved import ArchiveEntry, EvolvedScaffold
from cbs.scaffolds.example_agents import blind_best_of_n_agent, feedback_repair_agent
from cbs.tasks import Verifier
from cbs.tasks.families.toy import toy_behaviours, toy_suite


def make_model(seed=0):
    return MockModelClient(toy_behaviours(), seed=seed, model_id="ablation-test")


def make_pair(agent_fn, ablated_op="execution_feedback"):
    entry = ArchiveEntry(variant_id="v1", generation=0, description=agent_fn.__name__)
    baseline = EvolvedScaffold(agent_fn, entry, name="baseline")
    ablated = EvolvedScaffold(agent_fn, entry, name="ablated", disabled_ops=frozenset({ablated_op}))
    return baseline, ablated


class TestRunAblation:
    def test_ablated_scaffold_never_shows_the_ablated_op(self):
        """The clearest, structural check: the op must not appear at all in
        the ablated summary's aggregated op_counts."""
        verifier = Verifier(select_backend("auto"))
        baseline, ablated = make_pair(feedback_repair_agent)
        suite = toy_suite().by_id()
        result = run_ablation(
            "execution_feedback", baseline, ablated, suite["toy/gcd"],
            make_model(), verifier, budget_calls=15, n_reps=20,
        )
        assert "execution_feedback" not in result.ablated.op_counts

    def test_solve_rates_are_valid_proportions(self):
        verifier = Verifier(select_backend("auto"))
        baseline, ablated = make_pair(feedback_repair_agent)
        suite = toy_suite().by_id()
        result = run_ablation(
            "execution_feedback", baseline, ablated, suite["toy/gcd"],
            make_model(), verifier, budget_calls=15, n_reps=20,
        )
        assert 0.0 <= result.baseline.p_hat <= 1.0
        assert 0.0 <= result.ablated.p_hat <= 1.0

    def test_content_blind_mock_shows_no_load_bearing_effect(self):
        """Expected result on the mock backend, matching D-20: the mock
        ignores prompt content, so a repair conditioned on error text behaves
        identically to a blind resample. Ablating conditioning should not move
        the solve rate on this backend -- if it did, that would indicate the
        mock is leaking prompt-dependence somewhere, which would be a real bug.
        """
        verifier = Verifier(select_backend("auto"))
        baseline, ablated = make_pair(feedback_repair_agent)
        suite = toy_suite().by_id()
        result = run_ablation(
            "execution_feedback", baseline, ablated, suite["toy/gcd"],
            make_model(), verifier, budget_calls=15, n_reps=60,
        )
        assert result.baseline.n_solved == result.ablated.n_solved

    def test_fraction_removed_is_zero_when_nothing_removed(self):
        verifier = Verifier(select_backend("auto"))
        baseline, ablated = make_pair(feedback_repair_agent)
        suite = toy_suite().by_id()
        result = run_ablation(
            "execution_feedback", baseline, ablated, suite["toy/gcd"],
            make_model(), verifier, budget_calls=15, n_reps=40,
        )
        if result.solves_removed == 0:
            assert result.fraction_removed == 0.0
            assert not result.load_bearing

    def test_beyond_frontier_task_never_solved_either_way(self):
        verifier = Verifier(select_backend("auto"))
        baseline, ablated = make_pair(blind_best_of_n_agent)
        suite = toy_suite().by_id()
        result = run_ablation(
            "execution_feedback", baseline, ablated, suite["toy/impossible_parity"],
            make_model(), verifier, budget_calls=10, n_reps=15,
        )
        assert result.baseline.n_solved == 0
        assert result.ablated.n_solved == 0

    def test_summary_line_is_human_readable(self):
        verifier = Verifier(select_backend("auto"))
        baseline, ablated = make_pair(feedback_repair_agent)
        suite = toy_suite().by_id()
        result = run_ablation(
            "execution_feedback", baseline, ablated, suite["toy/gcd"],
            make_model(), verifier, budget_calls=10, n_reps=10,
        )
        line = result.summary_line()
        assert "toy/gcd" in line and "execution_feedback" in line

    def test_as_dict_round_trips_key_fields(self):
        verifier = Verifier(select_backend("auto"))
        baseline, ablated = make_pair(feedback_repair_agent)
        suite = toy_suite().by_id()
        result = run_ablation(
            "execution_feedback", baseline, ablated, suite["toy/gcd"],
            make_model(), verifier, budget_calls=10, n_reps=10,
        )
        payload = result.as_dict()
        assert payload["ablated_op"] == "execution_feedback"
        assert payload["solves_removed"] == result.solves_removed
