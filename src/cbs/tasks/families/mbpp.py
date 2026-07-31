"""MBPP task family (brief section 8, "Coding (primary)").

Vendored, not fetched at runtime -- see `data/vendored/mbpp/ATTRIBUTION.md` for
provenance, license, and the same two caveats as HumanEval (D-27): this is
plain MBPP, not the evalplus-extended MBPP+ the brief actually names, and it is
almost certainly present in the pretraining corpus of any web-scale frozen
model.

Format adaptation
------------------
MBPP's schema differs from HumanEval's in a way that matters for what counts
as "public":

*   `prompt` is a bare natural-language instruction ("Write a function to find
    the shared elements from the given two lists.") with **no function
    signature or example** -- unlike HumanEval's docstring-with-doctest style.
    A model given only that text has no way to know the expected function
    name or argument shape. Standard MBPP evaluation practice (and what every
    published harness does) is to show the model one test case alongside the
    instruction to fix the signature; `cbs` follows that convention and
    includes `test_list[0]` in `Task.prompt`.
*   Because that first test case is already shown to the model, it is
    unambiguously public. `public_tests` is derived the same
    first-half-of-assertions heuristic as the other families (D-18), applied
    to `test_list` -- so the prompt's own example test is always included in
    the derived public subset, never excluded from it.
*   `entry_point` is not given explicitly; it is inferred from the first `Call`
    node encountered walking the first test assertion's AST (the function the
    test actually calls). This is descriptive metadata only -- nothing in the
    verifier consumes `Task.entry_point` -- so the heuristic does not need to
    be perfect, only good enough to be informative.
*   `test_imports`, when present, is prepended to both `tests` and
    `public_tests` so a test relying on e.g. `math` or `heapq` does not fail
    with a `NameError` before the candidate is even exercised.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from cbs.tasks.schema import Task, TaskSuite

__all__ = ["mbpp_suite", "DEFAULT_MBPP_PATH"]

DEFAULT_MBPP_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "vendored" / "mbpp" / "sanitized-mbpp.json"
)


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

    return Task(
        task_id=f"Mbpp/{problem['task_id']}",
        family=family,
        prompt=prompt,
        tests=_build_tests(imports, test_list),
        public_tests=public_tests,
        entry_point=entry_point,
        reference_solution=problem["code"],
        timeout_s=15.0,
        metadata={
            "source": "google-research/mbpp (sanitized)",
            "variant": "original (not MBPP+)",
            "source_file": problem.get("source_file", ""),
        },
    )


def _load_problems(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mbpp_suite(path: Path | str = DEFAULT_MBPP_PATH, family: str = "mbpp") -> TaskSuite:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"vendored MBPP data not found at {path}. See "
            "data/vendored/mbpp/ATTRIBUTION.md for provenance."
        )
    problems = _load_problems(path)
    return TaskSuite(name=family, tasks=[_to_task(p, family) for p in problems])
