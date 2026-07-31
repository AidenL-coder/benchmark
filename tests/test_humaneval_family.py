"""HumanEval family tests.

The `test_all_canonical_solutions_pass` test is the important one: it runs all
164 real, vendored reference solutions through the actual sandbox and
verifier, which is a far stronger check on the verification pipeline than the
hand-written toy family alone -- real code has genuine syntax diversity,
multi-line bodies, imports, and edge cases no toy example anticipates. It is
marked slow (spawns 164 subprocess sandboxes) and runs in CI/full-suite passes,
not by default in a quick local loop.
"""

from __future__ import annotations

import random

import pytest

from cbs.sandbox import select_backend
from cbs.tasks import Verifier
from cbs.tasks.families.humaneval import _derive_public_tests, humaneval_suite

#: Multi-line list-literal assertion: the original regex-based extractor
#: truncated at the first newline, producing a syntactically broken subset.
_MULTILINE_ASSERT_TEST_FIELD = (
    "def check(candidate):\n"
    "    assert candidate('a b') == [\n"
    "        'a', 'b'\n"
    "    ]\n"
)

#: Loop-scoped setup: the assert alone references names never defined in
#: isolation. AST-correct extraction alone does not catch this -- only
#: validating against a known-correct reference solution does.
_LOOP_SCOPED_TEST_FIELD = (
    "def check(candidate):\n"
    "    for _ in range(5):\n"
    "        s = 'fixed'\n"
    "        encoded = s[::-1]\n"
    "        assert candidate(encoded) == s\n"
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def suite():
    return humaneval_suite()


class TestSuiteLoading:
    def test_loads_all_164_problems(self, suite):
        assert len(suite) == 164

    def test_task_ids_are_unique_and_well_formed(self, suite):
        ids = [t.task_id for t in suite.tasks]
        assert len(ids) == len(set(ids))
        assert all(tid.startswith("HumanEval/") for tid in ids)

    def test_every_task_has_a_reference_solution_and_entry_point(self, suite):
        for task in suite.tasks:
            assert task.reference_solution
            assert task.entry_point

    def test_suite_hash_is_deterministic(self):
        assert humaneval_suite().suite_hash() == humaneval_suite().suite_hash()


class TestVerification:
    def test_all_canonical_solutions_pass(self, suite, verifier):
        """The strong validation: real reference code through the real sandbox."""
        failures = []
        for task in suite.tasks:
            result = verifier.verify_code(task, task.reference_solution)
            if not result.passed:
                failures.append((task.task_id, result.reason))
        assert not failures, f"canonical solutions that failed: {failures[:10]}"

    def test_self_test_matches_verify_code(self, suite, verifier):
        task = suite.by_id()["HumanEval/0"]
        assert verifier.self_test(task).passed

    @pytest.mark.parametrize("index", [0, 10, 50, 100, 163])
    def test_a_deliberately_wrong_solution_is_rejected(self, suite, verifier, index):
        task = suite.tasks[index]
        wrong = task.prompt + "    return None\n"
        assert not verifier.verify_code(task, wrong).passed

    def test_wrong_solutions_rejected_across_a_random_sample(self, suite, verifier):
        rng = random.Random(0)
        sample = rng.sample(suite.tasks, 20)
        for task in sample:
            wrong = task.prompt + "    return None\n"
            result = verifier.verify_code(task, wrong)
            assert not result.passed, f"{task.task_id}: 'return None' should not pass"


class TestPublicTestDerivation:
    def test_most_tasks_get_a_derivable_public_subset(self, suite):
        with_public = sum(1 for t in suite.tasks if t.public_tests.strip())
        # A handful of problems use loops/helpers rather than flat asserts and
        # legitimately have none -- but the large majority should.
        assert with_public >= 150

    def test_reference_solution_always_passes_its_own_public_subset(self, suite, verifier):
        from dataclasses import replace

        for task in suite.tasks:
            if not task.public_tests.strip():
                continue
            shadow = replace(task, tests=task.public_tests)
            result = verifier.verify_code(shadow, task.reference_solution)
            assert result.passed, f"{task.task_id}: reference fails its own public subset"

    def test_derivation_is_a_strict_subset_of_hidden_assertions(self):
        test_field = (
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(2) == 2\n"
            "    assert candidate(3) == 3\n"
        )
        public = _derive_public_tests(test_field, "f")
        lines = [l for l in public.splitlines() if l.strip()]
        assert len(lines) == 2  # ceil(3/2)
        assert all("candidate" not in l for l in lines)
        assert all("f(" in l for l in lines)

    def test_no_assert_candidate_lines_yields_empty_public_tests(self):
        test_field = "def check(candidate):\n    for i in range(10):\n        pass\n"
        assert _derive_public_tests(test_field, "f") == ""

    def test_multiline_assertions_are_not_truncated(self):
        """Regression test: a line-anchored regex extractor truncated a
        multi-line list-literal assertion mid-expression, producing a
        syntactically broken public test. AST-based extraction re-emits the
        complete statement regardless of its original line count."""
        public = _derive_public_tests(_MULTILINE_ASSERT_TEST_FIELD, "f")
        assert public.strip()
        compile(public, "<public_tests>", "exec")  # must not raise SyntaxError
        assert "['a','b']" in public.replace(" ", "")

    def test_string_literal_containing_the_word_candidate_is_untouched(self):
        """Renaming happens on the AST (actual Name references only), not by
        text substitution -- a string literal that happens to contain
        'candidate' as a substring must survive unchanged."""
        test_field = (
            "def check(candidate):\n"
            "    assert candidate('a candidate string') == 'ok'\n"
        )
        public = _derive_public_tests(test_field, "f")
        assert "'a candidate string'" in public
        assert public.count("candidate") == 1  # only inside the string literal


class TestHumanEvalSuiteLoading:
    """The real regression this project hit: extraction can be AST-correct
    and still be unsafe to use, if the extracted assert depends on setup state
    that only exists inside its original loop. Only a small handful of real
    HumanEval problems have this shape; both known instances are pinned here
    by task id so a future change to the vendored data or the extractor is
    checked against them specifically, not just against the aggregate count.
    """

    def test_loop_scoped_setup_is_caught_by_validation_not_by_extraction_alone(
        self, verifier
    ):
        """AST extraction alone succeeds here (it's syntactically valid) --
        only running it against the reference solution reveals the NameError.
        """
        from dataclasses import replace

        from cbs.tasks.schema import Task

        public = _derive_public_tests(_LOOP_SCOPED_TEST_FIELD, "f")
        assert public.strip()  # extraction itself finds something

        task = Task(
            task_id="synthetic/loop_scoped",
            family="synthetic",
            prompt="def f(x):\n",
            tests=public,
        )
        result = verifier.verify_code(task, "def f(x):\n    return x\n")
        assert not result.passed  # confirms the hazard this test documents

    def test_known_loop_scoped_problems_end_up_with_no_public_tests(self, suite):
        """HumanEval/38 and HumanEval/50 both build randomised setup state
        across loop iterations before asserting; the safety-net validation in
        `humaneval_suite` must blank their public_tests rather than ship a
        subset that fails even the correct reference solution."""
        by_id = suite.by_id()
        for task_id in ("HumanEval/38", "HumanEval/50"):
            assert by_id[task_id].public_tests == ""

    def test_validation_can_be_disabled_revealing_the_unsafe_raw_extraction(self):
        """Confirms the escape hatch exists and that skipping validation
        surfaces the raw (unsafe) extraction for these two known cases --
        demonstrating why the default is True."""
        raw_suite = humaneval_suite(validate_public_tests=False)
        by_id = raw_suite.by_id()
        # Without validation, extraction alone still finds *something* for
        # these tasks (it's syntactically valid Python) -- validation is what
        # detects it doesn't actually run standalone.
        assert by_id["HumanEval/38"].public_tests.strip() != ""
