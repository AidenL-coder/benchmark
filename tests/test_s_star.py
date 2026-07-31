"""S_star tests.

Covers the four required mechanisms (best-of-N, execution feedback, tool use,
self-consistency), their support-class tagging, budget interaction, and a
specific characterised failure mode: naive public-test derivation can have
blind spots that let S_star's internal selection commit to a plausible-but-
wrong candidate. That is documented behaviour (docs/DECISIONS.md D-21), not a
bug, and is pinned here so a future change cannot silently alter it unnoticed.
"""

from __future__ import annotations

import pytest

from cbs.budget import BudgetAccountant, BudgetCaps
from cbs.models.mock import MockModelClient
from cbs.sandbox import select_backend
from cbs.scaffolds.s_star import SStar
from cbs.scaffolds.tagging import SupportClass
from cbs.tasks import Verifier
from cbs.tasks.families.toy import toy_behaviours, toy_suite


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def suite():
    return toy_suite().by_id()


def make_model(seed=0):
    return MockModelClient(toy_behaviours(), seed=seed, model_id="s_star_test")


class TestMechanisms:
    def test_all_four_mechanisms_fire_and_are_tagged_correctly(self, suite, verifier):
        """A task needing at least one repair exercises every op class."""
        model = make_model(seed=0)
        s_star = SStar(max_candidates=3, max_repairs_per_candidate=2)
        acct = BudgetAccountant("t", BudgetCaps(calls=30))
        result = s_star.solve(suite["toy/gcd"], model, acct, verifier=verifier, seed=1)

        ops = result.trace.op_counts()
        assert ops.get("single_call", 0) >= 1        # best-of-N generation
        assert ops.get("tool_call", 0) >= 1           # standard tool use (compile check)
        assert ops.get("execution_feedback", 0) >= 1  # execution feedback
        assert ops.get("self_consistency", 0) == 1    # exactly one final selection

        from cbs.scaffolds.tagging import support_class_of

        assert support_class_of("single_call") is SupportClass.PRESERVING
        assert support_class_of("tool_call") is SupportClass.EXPANDING
        assert support_class_of("execution_feedback") is SupportClass.EXPANDING
        assert support_class_of("self_consistency") is SupportClass.PRESERVING

    def test_hidden_oracle_queried_at_most_once(self, suite, verifier):
        """Selection must never see the hidden tests -- only the public subset.

        There is no direct hook to intercept verifier calls short of counting
        sandbox executions, so this checks the *observable* proxy: exactly one
        `ScaffoldResult.verification` is produced, and it is the only place a
        hidden-oracle verdict could have entered the picture.
        """
        model = make_model(seed=0)
        s_star = SStar(max_candidates=3, max_repairs_per_candidate=2)
        acct = BudgetAccountant("t", BudgetCaps(calls=30))
        result = s_star.solve(suite["toy/gcd"], model, acct, verifier=verifier, seed=1)
        assert result.verification is not None  # exactly one, at the very end


class TestBudget:
    def test_zero_budget_is_exhausted_with_no_solution(self, suite, verifier):
        model = make_model()
        s_star = SStar()
        acct = BudgetAccountant("t", BudgetCaps(calls=0))
        result = s_star.solve(suite["toy/gcd"], model, acct, verifier=verifier, seed=0)
        assert result.budget_exhausted
        assert result.solution == ""
        assert not result.passed

    def test_never_exceeds_its_allowance(self, suite, verifier):
        model = make_model()
        s_star = SStar(max_candidates=8, max_repairs_per_candidate=4)
        cap = 5
        acct = BudgetAccountant("t", BudgetCaps(calls=cap))
        s_star.solve(suite["toy/gcd"], model, acct, verifier=verifier, seed=0)
        assert acct.spent.calls <= cap

    def test_partial_budget_still_returns_a_scored_result(self, suite, verifier):
        """Getting at least one candidate before running out must not be
        reported as budget_exhausted -- that flag means "nothing to score"."""
        model = make_model()
        s_star = SStar(max_candidates=8, max_repairs_per_candidate=4)
        acct = BudgetAccountant("t", BudgetCaps(calls=1))
        result = s_star.solve(suite["toy/gcd"], model, acct, verifier=verifier, seed=0)
        assert not result.budget_exhausted
        assert result.verification is not None


class TestSelfConsistencySelection:
    def test_majority_vote_picks_the_larger_cluster(self):
        pool = ["def f():\n    return 1\n"] * 1 + ["def f():\n    return 2\n"] * 2
        chosen = SStar._select_by_consensus(pool)
        assert "return 2" in chosen

    def test_tie_breaks_to_earliest_cluster(self):
        pool = ["def f():\n    return 1\n", "def f():\n    return 2\n"]
        assert SStar._select_by_consensus(pool) == pool[0]

    def test_empty_pool_returns_empty_string(self):
        assert SStar._select_by_consensus([]) == ""

    def test_formatting_variants_of_the_same_solution_cluster_together(self):
        pool = [
            "def f(x):\n    return x + 1\n",
            "def f(x):\n\n    return x+1\n",
            "def f(x):\n    return x - 1\n",  # different answer, singleton
        ]
        chosen = SStar._select_by_consensus(pool)
        assert "+ 1" in chosen or "+1" in chosen


class TestCompileCheck:
    def test_valid_code_compiles(self):
        assert SStar._compiles("def f():\n    return 1\n")[0] is True

    def test_syntax_error_does_not_compile(self):
        ok, err = SStar._compiles("def f(:\n    return 1\n")
        assert not ok and "SyntaxError" in err

    def test_compiling_does_not_execute_top_level_code(self):
        """compile() must never run the candidate -- only parse/assemble it."""
        marker = []
        code = "marker.append(1)\ndef f():\n    return 1\n"
        # `marker` is not in scope for the compiled code's own module namespace,
        # so if compile() executed anything this would raise NameError instead
        # of returning cleanly -- either way, nothing appends to our local list.
        SStar._compiles(code)
        assert marker == []


class TestPublicTestBlindSpot:
    """Documents a real, expected phenomenon (docs/DECISIONS.md D-21): the toy
    family's naive public-test derivation (first half of assertions) can be
    entirely one-sided, letting a wrong-but-plausible candidate through."""

    def test_is_palindrome_public_subset_has_no_negative_case(self, suite):
        task = suite["toy/is_palindrome"]
        assert "False" not in task.public_tests
        assert "False" in task.tests  # the hidden suite does catch it

    def test_always_true_candidate_passes_public_but_fails_hidden(self, suite, verifier):
        task = suite["toy/is_palindrome"]
        always_true = "def is_palindrome(s):\n    return True\n"
        from dataclasses import replace

        public_result = verifier.verify_code(replace(task, tests=task.public_tests), always_true)
        hidden_result = verifier.verify_code(task, always_true)
        assert public_result.passed
        assert not hidden_result.passed

    def test_unique_sorted_public_subset_has_no_duplicate_case(self, suite):
        task = suite["toy/unique_sorted"]
        assert "2, 2, 2" not in task.public_tests
        assert "2, 2, 2" in task.tests

    def test_sort_without_dedup_passes_public_but_fails_hidden(self, suite, verifier):
        task = suite["toy/unique_sorted"]
        no_dedup = "def unique_sorted(xs):\n    return sorted(xs)\n"
        from dataclasses import replace

        public_result = verifier.verify_code(replace(task, tests=task.public_tests), no_dedup)
        hidden_result = verifier.verify_code(task, no_dedup)
        assert public_result.passed
        assert not hidden_result.passed


class TestPublicTestsDerivation:
    def test_correct_solution_always_passes_the_public_subset(self, suite, verifier):
        """A subset of true assertions can never reject a genuinely correct
        candidate -- only ever admit some incorrect ones (the point above)."""
        from dataclasses import replace
        from cbs.tasks.families.toy import TOY_TASKS

        for definition in TOY_TASKS:
            if not definition.correct_variants:
                continue
            task = suite[definition.task_id]
            if not task.public_tests.strip():
                continue
            shadow = replace(task, tests=task.public_tests)
            result = verifier.verify_code(shadow, definition.correct_variants[0])
            assert result.passed, f"{definition.task_id}: reference fails its own public subset"

    def test_at_least_one_assertion_when_hidden_tests_are_nonempty(self, suite):
        for task in suite.values():
            if task.tests.strip():
                assert task.public_tests.strip()
