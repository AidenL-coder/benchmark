"""Transfer/reasoning family tests (D-17).

Small enough (10 tasks) that this is not marked slow, unlike the HumanEval/MBPP
suites -- every reference solution runs as part of the normal fast test pass.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from cbs.sandbox import select_backend
from cbs.tasks import Verifier
from cbs.tasks.families.transfer_reasoning import TRANSFER_TASKS, transfer_suite


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def suite():
    return transfer_suite()


class TestSuiteLoading:
    def test_loads_all_tasks(self, suite):
        assert len(suite) == len(TRANSFER_TASKS)
        assert len(suite) >= 8

    def test_task_ids_are_unique_and_namespaced(self, suite):
        ids = [t.task_id for t in suite.tasks]
        assert len(ids) == len(set(ids))
        assert all(tid.startswith("transfer/") for tid in ids)

    def test_family_field_is_set(self, suite):
        assert all(t.family == "transfer_reasoning" for t in suite.tasks)

    def test_suite_hash_is_deterministic(self):
        assert transfer_suite().suite_hash() == transfer_suite().suite_hash()


class TestVerification:
    def test_all_reference_solutions_pass(self, suite, verifier):
        failures = []
        for task in suite.tasks:
            result = verifier.verify_code(task, task.reference_solution)
            if not result.passed:
                failures.append((task.task_id, result.reason))
        assert not failures, f"reference solutions that failed: {failures}"

    def test_self_test_matches_verify_code(self, suite, verifier):
        for task in suite.tasks:
            assert verifier.self_test(task).passed, task.task_id

    def test_return_none_is_rejected_everywhere(self, suite, verifier):
        for task in suite.tasks:
            wrong = f"def {task.entry_point}(*a, **k):\n    return None\n"
            result = verifier.verify_code(task, wrong)
            assert not result.passed, f"{task.task_id}: return-None should not pass"


class TestPublicTestDerivation:
    def test_every_task_has_a_derivable_public_subset(self, suite):
        assert all(t.public_tests.strip() for t in suite.tasks)

    def test_reference_solution_always_passes_its_own_public_subset(self, suite, verifier):
        for task in suite.tasks:
            shadow = replace(task, tests=task.public_tests)
            result = verifier.verify_code(shadow, task.reference_solution)
            assert result.passed, f"{task.task_id}: reference fails its own public subset"

    def test_public_subset_is_a_strict_prefix_of_hidden_assertions(self, suite):
        for task in suite.tasks:
            public_lines = [l for l in task.public_tests.splitlines() if l.strip()]
            hidden_lines = [l for l in task.tests.splitlines() if l.strip()]
            assert public_lines == hidden_lines[: len(public_lines)]
            assert len(public_lines) <= len(hidden_lines)


class TestContentIsGenuinelyDistinctFromOtherFamilies:
    """Not a code-checkable property, but worth asserting the intent
    explicitly: this family exists to test transfer across problem
    *character*, and should read as math/logic, not general programming."""

    def test_reasoning_keywords_appear_across_the_set(self, suite):
        combined = " ".join(t.prompt.lower() for t in suite.tasks)
        reasoning_markers = [
            "triangle", "prime", "linear system", "nim", "compound interest",
            "josephus", "coins", "perfect number", "quadratic", "common multiple",
        ]
        hits = sum(1 for marker in reasoning_markers if marker in combined)
        assert hits >= 6
