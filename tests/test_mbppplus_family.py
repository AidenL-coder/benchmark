"""MBPP+ family tests.

Mirrors test_mbpp_family.py's structure. Unlike test_humanevalplus_family.py,
no new public-test derivation mechanism needed testing here -- MBPP+ reuses
plain MBPP's test_list-based extraction unchanged (see the family module's
docstring for why that turned out to be sufficient, unlike HumanEval+).

Two genuinely different bug classes were found validating this family (see
`KNOWN_BROKEN_TASK_IDS`'s docstring and docs/DECISIONS.md D-35): a
floating-point-tolerance gap in evalplus's own `is_floats` helper
(`Mbpp/590`), and three tasks (`Mbpp/737`/`787`/`794`) whose generated
`assertion()` function computes a result and never asserts it -- a
completely non-functional test that accepts any candidate. Both are pinned
here, not just the exclusion mechanism, so a future change to the vendored
data or the exclusion list is checked against the actual finding.

Requires numpy (`pip install -e .[dev]` or `.[evalplus]`); skipped entirely if
unavailable.
"""

from __future__ import annotations

import ast
import random
from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="MBPP+ hidden tests require numpy at runtime")

from cbs.sandbox import select_backend
from cbs.tasks import Verifier
from cbs.tasks.families.mbppplus import KNOWN_BROKEN_TASK_IDS, TIMEOUT_OVERRIDES, mbppplus_suite

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def suite():
    return mbppplus_suite()


class TestSuiteLoading:
    def test_loads_374_problems_excluding_the_four_known_broken(self, suite):
        assert len(suite) == 378 - len(KNOWN_BROKEN_TASK_IDS)

    def test_known_broken_tasks_are_excluded_by_default(self, suite):
        by_id = suite.by_id()
        for task_id in KNOWN_BROKEN_TASK_IDS:
            assert task_id not in by_id

    def test_exclusion_can_be_disabled_to_inspect_the_raw_data(self):
        raw = mbppplus_suite(exclude_known_broken=False, validate_public_tests=False)
        assert len(raw) == 378
        for task_id in KNOWN_BROKEN_TASK_IDS:
            assert task_id in raw.by_id()

    def test_task_ids_are_unique_and_well_formed(self, suite):
        ids = [t.task_id for t in suite.tasks]
        assert len(ids) == len(set(ids))
        assert all(tid.startswith("Mbpp/") for tid in ids)

    def test_every_task_has_a_nonempty_entry_point(self, suite):
        assert all(t.entry_point for t in suite.tasks)

    def test_suite_hash_is_deterministic(self):
        assert mbppplus_suite().suite_hash() == mbppplus_suite().suite_hash()

    def test_hidden_test_is_the_expanded_field_not_the_original_test_list(self, suite):
        """The whole point of the '+' upgrade: the hidden oracle is the larger,
        evalplus-generated test field, not the handful of original asserts."""
        task = suite.by_id()["Mbpp/2"]
        # The expanded test defines evalplus's helper machinery, absent from
        # plain MBPP's test_list-only hidden tests.
        assert "assertion" in task.tests
        assert "import numpy" in task.tests


class TestKnownBrokenTasksAreGenuinelyBroken:
    """Pins the actual findings, not just the exclusion mechanism."""

    def test_no_op_assertion_tasks_have_no_assert_statement_at_all(self):
        """Mbpp/737, 787, 794: assertion() computes exact_match and never
        asserts it. Confirmed by parsing every task in the raw (unfiltered)
        data and checking for an Assert node anywhere in assertion() -- a
        full scan, not a spot check, so this pins the complete set."""
        raw = mbppplus_suite(exclude_known_broken=False, validate_public_tests=False)
        no_op_tasks = []
        for task in raw.tasks:
            tree = ast.parse(task.tests)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "assertion":
                    if not any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                        no_op_tasks.append(task.task_id)
                    break
        assert set(no_op_tasks) == {"Mbpp/737", "Mbpp/787", "Mbpp/794"}

    def test_no_op_assertion_means_any_candidate_passes(self, verifier):
        """Concrete demonstration of the severity: an obviously wrong
        candidate (always returns None) passes because the test verifies
        nothing, not because the candidate is secretly correct."""
        raw = mbppplus_suite(exclude_known_broken=False, validate_public_tests=False)
        task = raw.by_id()["Mbpp/737"]
        wrong = f"def {task.entry_point}(*a, **k):\n    return None\n"
        assert verifier.verify_code(task, wrong).passed

    def test_float_tolerance_gap_task_reference_fails_exact_equality(self, verifier):
        """Mbpp/590: is_floats() does not recognise a tuple mixing a
        tuple-of-floats and a complex number as float-ish, so atol stays 0
        and ordinary cross-platform floating-point noise in cmath.polar's
        result fails exact tuple equality -- even for the reference
        solution itself."""
        raw = mbppplus_suite(exclude_known_broken=False, validate_public_tests=False)
        task = raw.by_id()["Mbpp/590"]
        result = verifier.verify_code(task, task.reference_solution)
        assert not result.passed


class TestTimeoutOverride:
    def test_mbpp_599_gets_a_longer_timeout(self, suite):
        task = suite.by_id()["Mbpp/599"]
        assert task.timeout_s == TIMEOUT_OVERRIDES["Mbpp/599"]
        assert task.timeout_s > 20.0

    def test_other_tasks_keep_the_default_timeout(self, suite):
        task = suite.by_id()["Mbpp/2"]
        assert task.timeout_s == 20.0


class TestVerification:
    def test_all_reference_solutions_pass(self, suite, verifier):
        failures = []
        for task in suite.tasks:
            result = verifier.verify_code(task, task.reference_solution)
            if not result.passed:
                failures.append((task.task_id, result.reason))
        assert not failures, f"reference solutions that failed: {failures[:10]}"

    @pytest.mark.parametrize("index", [0, 90, 180, 270, 373])
    def test_a_deliberately_wrong_solution_is_rejected(self, suite, verifier, index):
        task = suite.tasks[index]
        wrong = f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
        assert not verifier.verify_code(task, wrong).passed

    def test_wrong_solutions_rejected_across_a_random_sample(self, suite, verifier):
        rng = random.Random(0)
        sample = rng.sample(suite.tasks, 20)
        for task in sample:
            wrong = f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
            result = verifier.verify_code(task, wrong)
            assert not result.passed, f"{task.task_id}: 'return None' should not pass"


class TestPublicTestDerivation:
    def test_every_task_gets_a_derivable_public_subset(self, suite):
        without_public = [t.task_id for t in suite.tasks if not t.public_tests.strip()]
        assert not without_public

    def test_reference_solution_always_passes_its_own_public_subset(self, suite, verifier):
        for task in suite.tasks:
            shadow = replace(task, tests=task.public_tests)
            result = verifier.verify_code(shadow, task.reference_solution)
            assert result.passed, f"{task.task_id}: reference fails its own public subset"

    def test_public_subset_is_the_original_small_test_list_not_the_expanded_one(self, suite):
        """Confirms the public/hidden split: public stays small and simple
        (the original MBPP asserts), hidden is the big evalplus expansion."""
        task = suite.by_id()["Mbpp/2"]
        assert len(task.public_tests) < len(task.tests) / 10
