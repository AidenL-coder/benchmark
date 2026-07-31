"""Verifier, extraction and canonicalisation tests.

False positives are the failure mode that matters most: a verifier that passes
wrong code would manufacture apparent capability. Several tests below are
adversarial rather than typical -- code that exits zero without running the
tests, code that reassigns `print`, code that loops forever.
"""

from __future__ import annotations

import pytest

from cbs.sandbox import select_backend
from cbs.tasks import Verifier, canonicalize_solution, extract_code
from cbs.tasks.families.toy import TOY_TASKS, toy_suite
from cbs.tasks.schema import Task


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def simple_task() -> Task:
    return Task(
        task_id="unit/add",
        family="unit",
        prompt="write add(a, b)",
        tests="assert add(1, 2) == 3\nassert add(-1, 1) == 0\n",
        entry_point="add",
        reference_solution="def add(a, b):\n    return a + b\n",
        timeout_s=15.0,
    )


class TestExtraction:
    def test_plain_code_passthrough(self):
        assert extract_code("def f():\n    pass") == "def f():\n    pass"

    def test_fenced_block(self):
        text = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
        assert extract_code(text) == "def f():\n    return 1"

    def test_bare_fence(self):
        assert extract_code("```\nx = 1\n```") == "x = 1"

    def test_multiple_blocks_are_concatenated(self):
        text = "```python\nimport math\n```\nand\n```python\ndef f():\n    return math.pi\n```"
        result = extract_code(text)
        assert "import math" in result
        assert "def f()" in result

    def test_unterminated_fence_still_extracts(self):
        assert extract_code("```python\ndef f():\n    return 1") == "def f():\n    return 1"


class TestVerifier:
    def test_correct_solution_passes(self, verifier, simple_task):
        result = verifier.verify_code(simple_task, "def add(a, b):\n    return a + b\n")
        assert result.passed
        assert result.reason == "ok"

    def test_wrong_solution_fails(self, verifier, simple_task):
        result = verifier.verify_code(simple_task, "def add(a, b):\n    return a - b\n")
        assert not result.passed
        assert result.reason == "test_failed"

    def test_empty_candidate_fails(self, verifier, simple_task):
        assert verifier.verify_code(simple_task, "   ").reason == "empty_candidate"

    def test_syntax_error_fails(self, verifier, simple_task):
        result = verifier.verify_code(simple_task, "def add(a, b)\n    return a + b\n")
        assert not result.passed

    def test_early_exit_cannot_forge_a_pass(self, verifier, simple_task):
        """`sys.exit(0)` before the tests must not be scored as success.

        This is why success requires an explicit marker rather than exit code 0.
        """
        code = "import sys\nsys.exit(0)\ndef add(a, b):\n    return a + b\n"
        result = verifier.verify_code(simple_task, code)
        assert not result.passed
        assert result.reason == "no_success_marker"

    def test_reassigning_print_cannot_forge_the_marker(self, verifier, simple_task):
        """The marker goes out via os.write, so tampering with print is futile."""
        code = (
            "import builtins\n"
            "builtins.print = lambda *a, **k: None\n"
            "def add(a, b):\n    return a - b\n"
        )
        result = verifier.verify_code(simple_task, code)
        assert not result.passed

    def test_printing_the_marker_without_passing_tests_still_fails(
        self, verifier, simple_task
    ):
        """A candidate that prints the marker itself is caught by the assertions.

        The marker is necessary but the tests still have to run to completion:
        the assertion raises first, so the interpreter exits non-zero.
        """
        code = (
            "import os\n"
            f'os.write(1, b"{Verifier.SUCCESS_MARKER}\\n")\n'
            "def add(a, b):\n    return a - b\n"
        )
        result = verifier.verify_code(simple_task, code)
        assert not result.passed
        assert result.reason == "marker_but_nonzero_exit"

    def test_infinite_loop_times_out(self, verifier):
        task = Task(
            task_id="unit/loop",
            family="unit",
            prompt="",
            tests="assert f() == 1\n",
            timeout_s=5.0,
        )
        result = verifier.verify_code(task, "def f():\n    while True:\n        pass\n")
        assert not result.passed
        assert result.reason == "timeout"

    def test_self_test_of_reference(self, verifier, simple_task):
        assert verifier.self_test(simple_task).passed

    def test_self_test_without_reference(self, verifier):
        task = Task(task_id="u/x", family="u", prompt="", tests="assert True\n")
        assert verifier.self_test(task).reason == "no_reference_solution"


class TestToyGroundTruth:
    """The toy family's declared ground truth must actually hold.

    If a declared-correct variant does not pass, or a declared-incorrect variant
    does, then the Phase 2 validation is checking the estimator against a
    fiction.
    """

    @pytest.mark.parametrize("definition", TOY_TASKS, ids=lambda d: d.task_id)
    def test_declared_correct_variants_pass(self, verifier, definition):
        task = toy_suite().by_id()[definition.task_id]
        for i, code in enumerate(definition.correct_variants):
            result = verifier.verify_code(task, code)
            assert result.passed, (
                f"{definition.task_id} correct_variants[{i}] failed: {result.reason}"
            )

    @pytest.mark.parametrize("definition", TOY_TASKS, ids=lambda d: d.task_id)
    def test_declared_incorrect_variants_fail(self, verifier, definition):
        task = toy_suite().by_id()[definition.task_id]
        for i, code in enumerate(definition.incorrect_variants):
            result = verifier.verify_code(task, code)
            assert not result.passed, (
                f"{definition.task_id} incorrect_variants[{i}] wrongly passed"
            )

    @pytest.mark.parametrize("definition", TOY_TASKS, ids=lambda d: d.task_id)
    def test_variants_are_distinct_species(self, definition):
        """Canonicalisation must not merge declared-distinct solutions.

        A merge would deflate observed richness and make Chao1 look wrong when
        the estimator was fine.
        """
        keys = {canonicalize_solution(c).key for c in definition.correct_variants}
        assert len(keys) == len(definition.correct_variants)


class TestCanonicalization:
    def test_formatting_noise_collapses(self):
        a = "def f(x):\n    return x + 1\n"
        b = "def f(x):\n\n    # a comment\n    return x+1\n"
        assert canonicalize_solution(a).key == canonicalize_solution(b).key

    def test_variable_renaming_collapses(self):
        a = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"
        b = "def f(items):\n    acc = 0\n    for i in items:\n        acc += i\n    return acc\n"
        assert canonicalize_solution(a).key == canonicalize_solution(b).key

    def test_docstrings_are_stripped(self):
        a = "def f(x):\n    return x\n"
        b = 'def f(x):\n    """Return x."""\n    return x\n'
        assert canonicalize_solution(a).key == canonicalize_solution(b).key

    def test_genuinely_different_algorithms_stay_distinct(self):
        a = "def f(s):\n    return s == s[::-1]\n"
        b = "def f(s):\n    return s == ''.join(reversed(s))\n"
        assert canonicalize_solution(a).key != canonicalize_solution(b).key

    def test_different_modules_stay_distinct(self):
        """Imported names are protected, so math.sqrt != cmath.sqrt."""
        a = "import math\ndef f(n):\n    return math.sqrt(n)\n"
        b = "import cmath\ndef f(n):\n    return cmath.sqrt(n)\n"
        assert canonicalize_solution(a).key != canonicalize_solution(b).key

    def test_entry_point_name_is_significant(self):
        a = "def f(x):\n    return x\n"
        b = "def g(x):\n    return x\n"
        assert canonicalize_solution(a).key != canonicalize_solution(b).key

    def test_unparseable_falls_back_to_text(self):
        form = canonicalize_solution("def f(:\n  ???")
        assert form.method == "text"
        assert form.key

    def test_docstring_only_body_stays_valid(self):
        form = canonicalize_solution('def f():\n    """doc"""\n')
        assert form.method == "ast"
