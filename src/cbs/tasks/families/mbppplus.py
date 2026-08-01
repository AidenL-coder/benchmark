"""MBPP+ task family (brief section 8; upgrade from D-29's `mbpp`).

Vendored, not fetched at runtime -- see `data/vendored/mbppplus/ATTRIBUTION.md`
for provenance, license, and the `numpy` runtime dependency this family
introduces (declared as the `evalplus` optional extra in
`pyproject.toml`, shared with that family rather than duplicated).

Why this was materially easier than the HumanEval+ upgrade (D-34)
--------------------------------------------------------------------
HumanEval+ replaced its predecessor's flat `assert` statements with a
`check(candidate)` function wrapping `inputs`/`results` lists, which needed
genuinely new AST-based derivation logic (see `humanevalplus.py`'s module
docstring). MBPP+ turned out not to need that: each row retains the
*original*, small `test_list` (the same few flat asserts
`cbs.tasks.families.mbpp` already uses) **alongside** the new, much larger
`test` field. `public_tests` is therefore derived exactly as it is for plain
MBPP -- the first half of `test_list`, with `test_imports` prepended -- and
only the *hidden* oracle changes, to the expanded `test` field. No new
extraction logic was needed; the module docstring records the reasoning
mainly so a future reader isn't left assuming an omission where there wasn't
one.

The expanded `test` field also calls the candidate by its real entry-point
name directly (e.g. `similar_elements(*inp)`), not aliased to a generic
`candidate` the way HumanEval+'s does -- consistent with plain MBPP's own
convention, and is already fully self-contained (it has its own
`import numpy as np`), so unlike `mbpp.py`'s hidden-test construction,
`test_imports` does not need prepending to it.

Fetched via the HF datasets-server rows API, not the dataset's own parquet
file -- MBPP+ ships no plain JSONL, and reading parquet directly would have
meant adding `pandas`/`pyarrow` as a new dependency for a one-time fetch. The
rows API returns identical data as plain JSON; the vendored file here is that
JSON, paginated and concatenated, not a live dependency at import time.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace as _replace
from pathlib import Path

from cbs.tasks.schema import Task, TaskSuite
from cbs.tasks.verifier import Verifier

__all__ = [
    "mbppplus_suite",
    "DEFAULT_MBPPPLUS_PATH",
    "KNOWN_BROKEN_TASK_IDS",
    "TIMEOUT_OVERRIDES",
]

DEFAULT_MBPPPLUS_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "vendored" / "mbppplus" / "MbppPlus.jsonl"
)

#: Genuine upstream data bugs found by running this family's own reference
#: solutions against the vendored hidden tests, same discipline as
#: `humanevalplus.KNOWN_BROKEN_TASK_IDS` (see docs/DECISIONS.md D-35 for the
#: full investigation). Not introduced by this loader; not "fixed" by editing
#: the vendored data.
#:
#: `Mbpp/590` ("polar_rect"): the generated `assertion()` computes
#: `atol = 1e-6` only when `is_floats(exp)` is True, but `is_floats` does not
#: recognise a tuple containing a nested tuple-of-floats *and* a complex
#: number as "float-ish" -- so `atol` stays 0 and the check falls through to
#: exact tuple equality. `cmath.polar`'s result differs from the vendored
#: expected value in the last few significant digits (ordinary floating-point
#: non-reproducibility across platforms/library versions for a transcendental
#: function), and exact equality has no tolerance for that. A real gap in
#: evalplus's own `is_floats` helper, not something introduced here.
#:
#: `Mbpp/737`, `Mbpp/787`, `Mbpp/794` (all three regex-`re.search`-returning
#: functions): confirmed by parsing every task's `assertion()` function and
#: checking for an `Assert` node anywhere in it -- these three compute a local
#: `exact_match = exp == (out is not None)` and then **never assert it**. The
#: function silently returns `None` unconditionally: it verifies nothing, and
#: any candidate (including `lambda *a, **k: None`) passes trivially. A
#: full scan of all 378 tasks found exactly these three and no others.
KNOWN_BROKEN_TASK_IDS = frozenset({"Mbpp/590", "Mbpp/737", "Mbpp/787", "Mbpp/794"})

#: Not bugs: `Mbpp/599` ("sum_average") is legitimately slow, not wrong.
#: evalplus's stress-test inputs include numbers up to ~10^8, and the
#: reference solution computes `sum(range(1, number+1))` in pure Python
#: rather than the closed-form `number*(number+1)//2` -- measured at ~25.5s
#: for the full test at generous timeout, comfortably exceeding the family's
#: default 20s. Raised per-task rather than raising the default for all 378
#: tasks, which would slow down the whole suite to accommodate one outlier.
TIMEOUT_OVERRIDES: dict[str, float] = {"Mbpp/599": 45.0}


def _extract_entry_point(first_test: str) -> str:
    try:
        tree = ast.parse(first_test)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
    return ""


def _build_tests(imports: list[str], assertions: list[str]) -> str:
    lines = list(imports) + list(assertions)
    return "\n".join(lines) + "\n" if lines else ""


def _to_task(problem: dict, family: str) -> Task:
    test_list = problem.get("test_list") or []
    imports = problem.get("test_imports") or []
    entry_point = _extract_entry_point(test_list[0]) if test_list else ""

    example = f"\n\nYour code should satisfy this test:\n{test_list[0]}" if test_list else ""
    prompt = (
        f"{problem['prompt']}{example}\n\n"
        "Respond with only the function definition in a Python code block."
    )

    n_public = max(1, (len(test_list) + 1) // 2) if test_list else 0
    public_tests = _build_tests(imports, test_list[:n_public]) if n_public else ""

    task_id = f"Mbpp/{problem['task_id']}"
    return Task(
        task_id=task_id,
        family=family,
        prompt=prompt,
        tests=problem["test"],  # self-contained: own numpy import, no prepending needed
        public_tests=public_tests,
        entry_point=entry_point,
        reference_solution=problem["code"],
        # evalplus's expanded test cases run longer than the original; a
        # handful of tasks with large stress-test inputs need even more
        # (TIMEOUT_OVERRIDES, e.g. Mbpp/599).
        timeout_s=TIMEOUT_OVERRIDES.get(task_id, 20.0),
        metadata={
            "source": "evalplus/mbppplus",
            "variant": "evalplus-extended (supersedes plain MBPP, D-29)",
            "source_file": problem.get("source_file", ""),
            "requires": ["numpy"],
        },
    )


def _load_problems(path: Path) -> list[dict]:
    problems = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _validate_public_tests(tasks: list[Task], verifier: Verifier) -> list[Task]:
    """Blank a task's `public_tests` if its own reference solution fails it --
    same safety net as `humaneval.py`/`humanevalplus.py` (see those modules for
    the concrete bugs it has caught)."""
    out = []
    for task in tasks:
        if not task.public_tests.strip():
            out.append(task)
            continue
        shadow = _replace(task, tests=task.public_tests)
        result = verifier.verify_code(shadow, task.reference_solution)
        out.append(task if result.passed else _replace(task, public_tests=""))
    return out


def mbppplus_suite(
    path: Path | str = DEFAULT_MBPPPLUS_PATH,
    family: str = "mbppplus",
    validate_public_tests: bool = True,
    verifier: Verifier | None = None,
    exclude_known_broken: bool = True,
) -> TaskSuite:
    """Load the vendored MBPP+ family. Requires `numpy` importable in the
    verifying Python environment (see module docstring and
    `data/vendored/mbppplus/ATTRIBUTION.md`).

    `exclude_known_broken` drops `KNOWN_BROKEN_TASK_IDS` (default on); see
    that constant for exactly why each is excluded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"vendored MBPP+ data not found at {path}. See "
            "data/vendored/mbppplus/ATTRIBUTION.md for provenance."
        )
    problems = _load_problems(path)
    if exclude_known_broken:
        problems = [
            p for p in problems if f"Mbpp/{p['task_id']}" not in KNOWN_BROKEN_TASK_IDS
        ]
    tasks = [_to_task(p, family) for p in problems]

    if validate_public_tests:
        if verifier is None:
            from cbs.sandbox import select_backend

            verifier = Verifier(select_backend("auto"))
        tasks = _validate_public_tests(tasks, verifier)

    return TaskSuite(name=family, tasks=tasks)
