"""MBPP family tests.

Mirrors test_humaneval_family.py: `test_all_reference_solutions_pass` runs all
427 real, vendored reference solutions through the actual sandbox and
verifier. Marked slow for the same reason (427 subprocess sandboxes).
"""

from __future__ import annotations

import random

import pytest

from cbs.sandbox import select_backend
from cbs.tasks import Verifier
from cbs.tasks.families.mbpp import _extract_entry_point, mbpp_suite

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def suite():
    return mbpp_suite()


class TestSuiteLoading:
    def test_loads_all_427_problems(self, suite):
        assert len(suite) == 427

    def test_task_ids_are_unique_and_well_formed(self, suite):
        ids = [t.task_id for t in suite.tasks]
        assert len(ids) == len(set(ids))
        assert all(tid.startswith("Mbpp/") for tid in ids)

    def test_every_task_has_a_reference_solution(self, suite):
        for task in suite.tasks:
            assert task.reference_solution

    def test_prompt_includes_an_example_test(self, suite):
        """MBPP's own prompt gives no signature -- an example test must be
        shown, or the model has no way to know the expected function shape."""
        task = suite.by_id()["Mbpp/2"]
        assert "assert" in task.prompt

    def test_suite_hash_is_deterministic(self):
        assert mbpp_suite().suite_hash() == mbpp_suite().suite_hash()


class TestVerification:
    def test_all_reference_solutions_pass(self, suite, verifier):
        failures = []
        for task in suite.tasks:
            result = verifier.verify_code(task, task.reference_solution)
            if not result.passed:
                failures.append((task.task_id, result.reason))
        assert not failures, f"reference solutions that failed: {failures[:10]}"

    @pytest.mark.parametrize("index", [0, 50, 150, 300, 426])
    def test_a_deliberately_wrong_solution_is_rejected(self, suite, verifier, index):
        task = suite.tasks[index]
        wrong = f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
        assert not verifier.verify_code(task, wrong).passed

    def test_wrong_solutions_rejected_across_a_random_sample(self, suite, verifier):
        rng = random.Random(0)
        sample = rng.sample(suite.tasks, 25)
        for task in sample:
            wrong = f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
            result = verifier.verify_code(task, wrong)
            assert not result.passed, f"{task.task_id}: 'return None' should not pass"


class TestPublicTestDerivation:
    def test_every_task_gets_a_derivable_public_subset(self, suite):
        """Unlike HumanEval, MBPP's test_list is always flat top-level
        asserts (no loop-scoped setup, no check(candidate) wrapper), so every
        task should get a derivable, valid public subset."""
        without_public = [t.task_id for t in suite.tasks if not t.public_tests.strip()]
        assert not without_public

    def test_reference_solution_always_passes_its_own_public_subset(self, suite, verifier):
        from dataclasses import replace

        for task in suite.tasks:
            shadow = replace(task, tests=task.public_tests)
            result = verifier.verify_code(shadow, task.reference_solution)
            assert result.passed, f"{task.task_id}: reference fails its own public subset"

    def test_public_subset_includes_the_prompts_own_example(self, suite):
        """The example test shown in the prompt is always test_list[0], and
        the public subset is always test_list[:n>=1], so the model's own
        visible example is never excluded from what it can self-check."""
        task = suite.by_id()["Mbpp/2"]
        first_assert = "similar_elements((3, 4, 5, 6),(5, 7, 4, 10))"
        assert first_assert in task.prompt
        assert first_assert in task.public_tests


class TestEntryPointExtraction:
    def test_extracts_the_function_called_in_the_first_assertion(self):
        assert (
            _extract_entry_point("assert similar_elements((1,2),(3,4)) == set()")
            == "similar_elements"
        )

    def test_prefers_the_outer_call_over_a_call_in_the_expected_value(self):
        """The right-hand side of the assertion is often itself a call
        (`set(...)`, `sorted(...)`) -- extraction must find the function under
        test (the left-hand call), not whichever call it encounters first by
        accident of AST layout."""
        entry = _extract_entry_point("assert f(1, 2) == set((1, 2))")
        assert entry == "f"

    def test_unparseable_test_yields_empty_string(self):
        assert _extract_entry_point("not valid python (((") == ""

    def test_every_real_task_gets_a_nonempty_entry_point(self, suite):
        assert all(t.entry_point for t in suite.tasks)
