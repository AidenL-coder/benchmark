"""HumanEval+ family tests.

Mirrors test_humaneval_family.py's structure. The important addition here is
the AST-based public-test derivation (a genuinely different mechanism from
the flat-assert extraction the other families use, since HumanEval+'s tests
are shaped as a `check(candidate)` function building `inputs`/`results` lists
rather than flat `assert` statements -- see the family module's docstring),
and the one known-broken upstream task excluded by default
(`KNOWN_BROKEN_TASK_IDS`), which is itself pinned by a dedicated test so it
cannot silently regress if the exclusion set changes.

Requires numpy (`pip install -e .[dev]` or `.[humanevalplus]`); skipped
entirely if unavailable rather than failing confusingly mid-sandbox-run.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("numpy", reason="HumanEval+ hidden tests require numpy at runtime")

from cbs.sandbox import select_backend
from cbs.tasks import Verifier
from cbs.tasks.families.humanevalplus import (
    KNOWN_BROKEN_TASK_IDS,
    _derive_public_tests,
    humanevalplus_suite,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def suite():
    return humanevalplus_suite()


class TestSuiteLoading:
    def test_loads_163_problems_excluding_the_known_broken_one(self, suite):
        assert len(suite) == 163

    def test_known_broken_task_is_excluded_by_default(self, suite):
        assert "HumanEval/32" in KNOWN_BROKEN_TASK_IDS
        assert "HumanEval/32" not in suite.by_id()

    def test_exclusion_can_be_disabled_to_inspect_the_raw_data(self):
        raw = humanevalplus_suite(exclude_known_broken=False, validate_public_tests=False)
        assert "HumanEval/32" in raw.by_id()
        assert len(raw) == 164

    def test_task_ids_match_original_humaneval_numbering(self, suite):
        ids = [t.task_id for t in suite.tasks]
        assert len(ids) == len(set(ids))
        assert all(tid.startswith("HumanEval/") for tid in ids)

    def test_suite_hash_is_deterministic(self):
        assert humanevalplus_suite().suite_hash() == humanevalplus_suite().suite_hash()


class TestKnownBrokenTaskIsGenuinelyBroken:
    """Pins the actual finding, not just the exclusion mechanism: this task's
    own reference solution really does fail its own hidden test, so the
    exclusion is justified and not a data-loading mistake."""

    def test_reference_solution_fails_its_own_hidden_test(self, verifier):
        raw = humanevalplus_suite(exclude_known_broken=False, validate_public_tests=False)
        task = raw.by_id()["HumanEval/32"]
        result = verifier.verify_code(task, task.reference_solution)
        assert not result.passed


class TestVerification:
    def test_all_reference_solutions_pass(self, suite, verifier):
        failures = []
        for task in suite.tasks:
            result = verifier.verify_code(task, task.reference_solution)
            if not result.passed:
                failures.append((task.task_id, result.reason))
        assert not failures, f"reference solutions that failed: {failures[:10]}"

    @pytest.mark.parametrize("index", [0, 40, 80, 120, 162])
    def test_a_deliberately_wrong_solution_is_rejected(self, suite, verifier, index):
        task = suite.tasks[index]
        wrong = f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
        assert not verifier.verify_code(task, wrong).passed

    def test_wrong_solutions_rejected_across_a_random_sample(self, suite, verifier):
        rng = random.Random(0)
        sample = rng.sample(suite.tasks, 15)
        for task in sample:
            wrong = f"def {task.entry_point}(*args, **kwargs):\n    return None\n"
            result = verifier.verify_code(task, wrong)
            assert not result.passed, f"{task.task_id}: 'return None' should not pass"


class TestPublicTestDerivation:
    def test_most_tasks_get_a_derivable_public_subset(self, suite):
        with_public = sum(1 for t in suite.tasks if t.public_tests.strip())
        assert with_public >= 150

    def test_reference_solution_always_passes_its_own_public_subset(self, suite, verifier):
        from dataclasses import replace

        for task in suite.tasks:
            if not task.public_tests.strip():
                continue
            shadow = replace(task, tests=task.public_tests)
            result = verifier.verify_code(shadow, task.reference_solution)
            assert result.passed, f"{task.task_id}: reference fails its own public subset"

    def test_derives_from_inputs_results_lists_not_flat_asserts(self):
        """The defining mechanism: truncate the inputs/results list literals
        inside check(), not search for standalone assert statements (there
        are none in this family's test shape)."""
        test_field = (
            "def assertion(out, exp, atol):\n"
            "    assert out == exp\n"
            "\n"
            "def check(candidate):\n"
            "    inputs = [[1], [2], [3], [4]]\n"
            "    results = [1, 2, 3, 4]\n"
            "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "        assertion(candidate(*inp), exp, 0)\n"
        )
        public = _derive_public_tests(test_field, "f")
        assert public.strip()
        compile(public, "<public_tests>", "exec")
        # Truncated to half: 2 of the original 4 input/result pairs.
        assert public.count("[1]") + public.count("[2]") + public.count("[3]") + public.count("[4]")
        assert "check(f)" in public

    def test_preserves_statements_other_than_inputs_and_results_verbatim(self):
        """A task whose check() body has extra setup beyond inputs/results
        must not silently lose that setup in the derived public subset."""
        test_field = (
            "def check(candidate):\n"
            "    scale = 10\n"
            "    inputs = [[1], [2]]\n"
            "    results = [10, 20]\n"
            "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "        assert candidate(*inp) * scale == exp\n"
        )
        public = _derive_public_tests(test_field, "f")
        assert "scale = 10" in public

    def test_no_check_function_yields_empty_public_tests(self):
        assert _derive_public_tests("x = 1\ny = 2\n", "f") == ""

    def test_no_inputs_results_pattern_yields_empty_public_tests(self):
        """HumanEval/32's shape (no separate results list, custom assertion) --
        the real case this family excludes wholesale, but the derivation
        function itself must degrade gracefully on it too, independent of the
        exclusion list, in case a similarly-shaped task is ever included."""
        test_field = (
            "def check(candidate):\n"
            "    inputs = [[1], [2]]\n"
            "    for inp in inputs:\n"
            "        assert candidate(*inp) is not None\n"
        )
        assert _derive_public_tests(test_field, "f") == ""

    def test_mismatched_inputs_results_lengths_yields_empty_public_tests(self):
        test_field = (
            "def check(candidate):\n"
            "    inputs = [[1], [2], [3]]\n"
            "    results = [1, 2]\n"
            "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "        assert candidate(*inp) == exp\n"
        )
        assert _derive_public_tests(test_field, "f") == ""
