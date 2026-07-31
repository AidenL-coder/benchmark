"""HumanEval task family (brief section 8, "Coding (primary)").

Vendored, not fetched at runtime -- see `data/vendored/humaneval/ATTRIBUTION.md`
for provenance, license, and two caveats that matter before this family is used
for anything beyond instrument validation: it is the **original** HumanEval,
not the evalplus-extended HumanEval+ the brief actually names (D-27), and it is
almost certainly present in the pretraining corpus of any web-scale frozen
model (contamination; brief section 8).

Format adaptation
------------------
HumanEval's own schema doesn't match `cbs.tasks.schema.Task` directly:

*   `prompt` is a signature + docstring with no body -- used as-is, since that
    is exactly what a model is meant to complete.
*   `canonical_solution` is a body fragment, valid only concatenated onto
    `prompt` -- `reference_solution` does that concatenation once here so the
    verifier's self-test always runs the complete function, not a fragment.
*   `test` defines a `check(candidate)` function; it is not runnable on its
    own the way `cbs`'s hidden-test convention expects (a script that asserts
    directly against the entry point). A trailing `check(entry_point)` call is
    appended so the same verifier convention as every other family applies
    unmodified.
*   `public_tests` is derived the same way as the toy family (D-18): the first
    half (rounded up) of the `assert` statements inside `test` that reference
    `candidate`, rewritten to call the entry point directly. Extraction is
    AST-based, not line-based: several HumanEval assertions span multiple
    source lines (a list literal continuing on the next line, say), and a
    regex anchored to `$` in `MULTILINE` mode silently truncates those mid-
    expression, producing syntactically broken "public tests" that fail to
    parse. Parsing with `ast` and re-emitting each complete statement with
    `ast.unparse` cannot truncate mid-expression the way line-oriented text
    matching can.

    A second, subtler failure survives even AST-correct extraction: a handful
    of problems' assertions live inside a loop that builds randomised setup
    state across iterations (`for _ in range(100): s = random_string();
    encoded = encode(s); assert candidate(encoded) == s`), so the extracted
    assert alone references names (`encoded`, `s`) that were never defined --
    syntactically valid on its own, but a `NameError` at runtime, which is a
    broken public test, not merely a missing one. A broken public test that
    even a *correct* candidate fails is actively harmful (worse than no public
    signal at all): it would make `S_star`'s execution-feedback loop discard
    genuinely correct candidates in an endless repair loop. `humaneval_suite`
    therefore validates every derived `public_tests` against the task's own
    known-correct `reference_solution` at load time and blanks it out (falling
    back to a compile-only check) if even the reference fails it -- caught two
    such cases (`HumanEval/38`, `HumanEval/50`) during development, both of
    exactly this random-loop-setup shape.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace as _replace
from pathlib import Path

from cbs.tasks.schema import Task, TaskSuite
from cbs.tasks.verifier import Verifier

__all__ = ["humaneval_suite", "DEFAULT_HUMANEVAL_PATH"]

DEFAULT_HUMANEVAL_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "vendored" / "humaneval" / "HumanEval.jsonl"
)


class _RenameCandidate(ast.NodeTransformer):
    """Renames every `candidate` Name reference to the real entry point.

    Operating on the AST rather than the source text also means a `candidate`
    substring that happens to appear inside a string literal is never touched
    -- only an actual name reference is renamed.
    """

    def __init__(self, entry_point: str):
        self.entry_point = entry_point

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "candidate":
            node.id = self.entry_point
        return node


def _iter_asserts_in_source_order(node: ast.AST):
    """Depth-first, left-to-right walk yielding `Assert` nodes.

    `ast.walk` is breadth-first and does not guarantee statement order;
    `ast.iter_child_nodes` yields each node's children in field-definition
    order (source order for a statement list), so recursing through it
    directly gives a true, guaranteed source-order traversal.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Assert):
            yield child
        yield from _iter_asserts_in_source_order(child)


def _derive_public_tests(test_field: str, entry_point: str) -> str:
    try:
        tree = ast.parse(test_field)
    except SyntaxError:
        return ""

    candidate_asserts = [
        node
        for node in _iter_asserts_in_source_order(tree)
        if any(isinstance(n, ast.Name) and n.id == "candidate" for n in ast.walk(node))
    ]
    if not candidate_asserts:
        return ""

    n_public = max(1, (len(candidate_asserts) + 1) // 2)
    renamer = _RenameCandidate(entry_point)
    lines = []
    for node in candidate_asserts[:n_public]:
        renamed = renamer.visit(node)
        ast.fix_missing_locations(renamed)
        lines.append(ast.unparse(renamed))
    return "\n".join(lines) + "\n"


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
        timeout_s=15.0,
        metadata={"source": "openai/human-eval", "variant": "original (not HumanEval+)"},
    )


def _validate_public_tests(tasks: list[Task], verifier: Verifier) -> list[Task]:
    """Blank a task's `public_tests` if its own reference solution fails it.

    See module docstring: extraction can produce a public subset that is
    syntactically valid but references names never defined in isolation
    (loop-scoped setup state). A public test the *correct* answer fails is
    worse than none -- it would make execution-feedback loops discard correct
    candidates -- so this is checked, not assumed, at load time.
    """
    out = []
    for task in tasks:
        if not task.public_tests.strip():
            out.append(task)
            continue
        shadow = _replace(task, tests=task.public_tests)
        result = verifier.verify_code(shadow, task.reference_solution)
        out.append(task if result.passed else _replace(task, public_tests=""))
    return out


def humaneval_suite(
    path: Path | str = DEFAULT_HUMANEVAL_PATH,
    family: str = "humaneval",
    validate_public_tests: bool = True,
    verifier: Verifier | None = None,
) -> TaskSuite:
    """Load the vendored HumanEval family.

    `validate_public_tests` runs each task's own reference solution against
    its derived public subset and blanks the subset out if even that fails
    (see module docstring) -- on by default because a silently broken public
    test is actively harmful, not merely uninformative. It costs one sandboxed
    execution per task with a non-empty subset (a couple of seconds total);
    set to `False` only if that cost matters more than the guarantee, e.g. in
    a context that has already validated this vendored copy once.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"vendored HumanEval data not found at {path}. See "
            "data/vendored/humaneval/ATTRIBUTION.md for provenance."
        )
    problems = _load_problems(path)
    tasks = [_to_task(p, family) for p in problems]

    if validate_public_tests:
        if verifier is None:
            from cbs.sandbox import select_backend

            verifier = Verifier(select_backend("auto"))
        tasks = _validate_public_tests(tasks, verifier)

    return TaskSuite(name=family, tasks=tasks)
