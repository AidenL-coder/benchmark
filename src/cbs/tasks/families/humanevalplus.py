"""HumanEval+ task family (brief section 8; upgrade from D-27's `humaneval`).

Vendored, not fetched at runtime -- see
`data/vendored/humanevalplus/ATTRIBUTION.md` for provenance, license, and the
`numpy` runtime dependency this family specifically introduces (declared as
the `evalplus` optional extra in `pyproject.toml`, not a core dependency).

Why `public_tests` needed new logic, not a reused loader
-----------------------------------------------------------
Original HumanEval's hidden tests (`cbs.tasks.families.humaneval`) are flat
`assert candidate(...) == expected` statements, so deriving a weaker public
subset is "take the first half of the assert statements" (D-18). HumanEval+'s
tests are structured completely differently -- every task's `test` field
defines helper functions (`is_floats`, `assertion`, using
`numpy.testing.assert_allclose` for float-tolerant comparison) and a
`check(candidate)` function of the shape::

    def check(candidate):
        inputs = [[...], [...], ...]
        results = [..., ..., ...]
        for i, (inp, exp) in enumerate(zip(inputs, results)):
            assertion(candidate(*inp), exp, 0)

There are no flat `assert` statements referencing `candidate` at all -- the
original family's extraction logic (`ast.Assert` nodes) finds nothing here,
and would silently degrade every task to a compile-only check if reused
unmodified. Instead, `_derive_public_tests` here locates the `inputs =` and
`results =` list-literal assignments inside `check` via AST, truncates both to
their first half (rounded up) *as list elements*, and reconstructs a
standalone test snippet: everything before `check` (imports, helper
functions) unchanged, then `check` unchanged except for the two truncated
assignments, re-serialised with `ast.unparse`. Any statement in `check` other
than those two assignments is preserved verbatim and in order -- this does
not assume the loop is written any particular way, only that `inputs` and
`results` are literal lists it can safely shorten.

If a task's test does not match this shape (no `check` function, or `inputs`/
`results` are not literal lists), `_derive_public_tests` returns `""` and the
task falls back to a compile-only check for `S_star`'s execution feedback --
the same honest-degradation policy as the other families, not a crash.

As with `humaneval.py`, every derived public subset is validated against its
own task's reference solution at load time (default on) and blanked out if
even that fails -- a broken public test that rejects a *correct* candidate is
worse than no public signal at all (see `humaneval.py`'s module docstring for
the two concrete bugs on the original family this same safety net caught).
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace as _replace
from pathlib import Path

from cbs.tasks.schema import Task, TaskSuite
from cbs.tasks.verifier import Verifier

__all__ = ["humanevalplus_suite", "DEFAULT_HUMANEVALPLUS_PATH", "KNOWN_BROKEN_TASK_IDS"]

DEFAULT_HUMANEVALPLUS_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "vendored"
    / "humanevalplus"
    / "HumanEvalPlus.jsonl"
)

#: Genuine upstream data bugs found by running this family's own reference
#: solutions against the vendored hidden tests (the same check every family
#: gets -- see the humaneval/mbpp modules for the two comparable bugs found
#: there). Not something introduced by this loader, and not "fixed" by
#: editing the vendored data -- the vendored copy stays a faithful reproduction
#: of evalplus's own file; broken tasks are excluded and documented instead.
#:
#: `HumanEval/32` ("find_zero"): the generated test asserts
#: `_poly(*candidate(*inp), inp) <= 0.0001`. `find_zero` returns a single
#: float root, and `*` on a scalar raises `TypeError: Value after * must be
#: an iterable, not float` -- the reference solution itself fails this
#: assertion, which would make ANY correct candidate on this task register as
#: a failure. Excluded rather than silently kept, which would corrupt the
#: frontier estimate specifically for this one task.
KNOWN_BROKEN_TASK_IDS = frozenset({"HumanEval/32"})


def _derive_public_tests(test_field: str, entry_point: str) -> str:
    try:
        tree = ast.parse(test_field)
    except SyntaxError:
        return ""

    check_index = None
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_index = i
            break
    if check_index is None:
        return ""
    check_fn = tree.body[check_index]

    def literal_list_assign(stmt: ast.stmt, name: str) -> ast.List | None:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
            and isinstance(stmt.value, ast.List)
        ):
            return stmt.value
        return None

    inputs_list = results_list = None
    for stmt in check_fn.body:
        inputs_list = inputs_list or literal_list_assign(stmt, "inputs")
        results_list = results_list or literal_list_assign(stmt, "results")

    if inputs_list is None or results_list is None:
        return ""
    if len(inputs_list.elts) != len(results_list.elts) or not inputs_list.elts:
        return ""

    n_total = len(inputs_list.elts)
    n_public = max(1, (n_total + 1) // 2)

    new_body = []
    for stmt in check_fn.body:
        if literal_list_assign(stmt, "inputs") is not None:
            new_body.append(
                ast.Assign(
                    targets=stmt.targets,
                    value=ast.List(elts=stmt.value.elts[:n_public], ctx=ast.Load()),
                )
            )
        elif literal_list_assign(stmt, "results") is not None:
            new_body.append(
                ast.Assign(
                    targets=stmt.targets,
                    value=ast.List(elts=stmt.value.elts[:n_public], ctx=ast.Load()),
                )
            )
        else:
            new_body.append(stmt)  # preserve everything else verbatim, in order

    new_check = ast.FunctionDef(
        name="check",
        args=check_fn.args,
        body=new_body,
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    new_module = ast.Module(body=tree.body[:check_index] + [new_check], type_ignores=[])
    ast.fix_missing_locations(new_module)

    try:
        source = ast.unparse(new_module)
    except (AttributeError, RecursionError, ValueError):  # pragma: no cover
        return ""
    return f"{source}\n\ncheck({entry_point})\n"


def _load_problems(path: Path) -> list[dict]:
    problems = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _to_task(problem: dict, family: str) -> Task:
    entry_point = problem["entry_point"]
    prompt = problem["prompt"]
    reference_solution = prompt + problem["canonical_solution"]
    tests = f"{problem['test']}\n\ncheck({entry_point})\n"
    return Task(
        task_id=problem["task_id"],
        family=family,
        prompt=prompt,
        tests=tests,
        public_tests=_derive_public_tests(problem["test"], entry_point),
        entry_point=entry_point,
        reference_solution=reference_solution,
        timeout_s=20.0,  # evalplus's expanded test cases run longer than original
        metadata={
            "source": "evalplus/humanevalplus",
            "variant": "evalplus-extended (supersedes original HumanEval, D-27)",
            "requires": ["numpy"],
        },
    )


def _validate_public_tests(tasks: list[Task], verifier: Verifier) -> list[Task]:
    """Blank a task's `public_tests` if its own reference solution fails it --
    identical safety net to `humaneval.py`'s (see that module for why)."""
    out = []
    for task in tasks:
        if not task.public_tests.strip():
            out.append(task)
            continue
        shadow = _replace(task, tests=task.public_tests)
        result = verifier.verify_code(shadow, task.reference_solution)
        out.append(task if result.passed else _replace(task, public_tests=""))
    return out


def humanevalplus_suite(
    path: Path | str = DEFAULT_HUMANEVALPLUS_PATH,
    family: str = "humanevalplus",
    validate_public_tests: bool = True,
    verifier: Verifier | None = None,
    exclude_known_broken: bool = True,
) -> TaskSuite:
    """Load the vendored HumanEval+ family. Requires `numpy` importable in the
    verifying Python environment (see module docstring and
    `data/vendored/humanevalplus/ATTRIBUTION.md`).

    `exclude_known_broken` drops `KNOWN_BROKEN_TASK_IDS` (default on). Set to
    `False` only to inspect the raw vendored data as-is, e.g. to re-verify the
    exclusion is still warranted after an upstream update.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"vendored HumanEval+ data not found at {path}. See "
            "data/vendored/humanevalplus/ATTRIBUTION.md for provenance."
        )
    problems = _load_problems(path)
    if exclude_known_broken:
        problems = [p for p in problems if p["task_id"] not in KNOWN_BROKEN_TASK_IDS]
    tasks = [_to_task(p, family) for p in problems]

    if validate_public_tests:
        if verifier is None:
            from cbs.sandbox import select_backend

            verifier = Verifier(select_backend("auto"))
        tasks = _validate_public_tests(tasks, verifier)

    return TaskSuite(name=family, tasks=tasks)
