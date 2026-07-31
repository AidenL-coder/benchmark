"""Canonicalisation of correct solutions into "species".

The frontier protocol (brief section 3.2) records not just `p_hat(x)` but the
*set of distinct correct solutions*, which is the sample the Good-Turing and
Chao1 estimators consume. Those estimators are counts-of-counts based, so what
counts as "the same solution" directly determines the unseen-mass and richness
estimates. Getting this wrong biases the headline numbers:

*   too strict (raw string equality) inflates species richness, because
    whitespace and variable-name jitter each become a new species, which in turn
    inflates the singleton count `f1` and so inflates estimated unseen mass;
*   too loose (semantic equivalence) is undecidable in general.

The chosen middle ground is AST normalisation: parse, strip docstrings and
comments, alpha-rename local identifiers in order of first appearance, and
re-emit. That collapses formatting and naming noise while keeping genuinely
different algorithms apart. Unparseable code falls back to whitespace-normalised
text, and the record says which path was taken.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import re
from dataclasses import dataclass

__all__ = ["CanonicalForm", "canonicalize_solution", "species_key"]

# Names we must never rename: renaming these would change behaviour.
_BUILTIN_NAMES = frozenset(dir(builtins))


def _imported_names(tree: ast.AST) -> set[str]:
    """Names bound by import statements.

    These must stay fixed. If `math` were alpha-renamed to `v0`, then
    `math.sqrt(n)` and `cmath.sqrt(n)` would canonicalise identically and two
    genuinely different solutions would be merged into one species, deflating
    richness.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


@dataclass(frozen=True)
class CanonicalForm:
    text: str
    method: str  # "ast" | "text"
    key: str     # sha256 of `text`, the species identifier

    def as_dict(self) -> dict:
        return {"method": self.method, "key": self.key}


class _Normalizer(ast.NodeTransformer):
    """Alpha-renames locally bound names and drops docstrings."""

    def __init__(self, protected: frozenset[str]):
        self.protected = protected
        self._mapping: dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name in self.protected or name.startswith("__"):
            return name
        if name not in self._mapping:
            self._mapping[name] = f"v{len(self._mapping)}"
        return self._mapping[name]

    # -- identifier sites -------------------------------------------------
    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._rename(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._rename(node.arg)
        node.annotation = None  # annotations are not semantics here
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # The entry-point name is part of the interface; renaming it would make
        # two solutions with different entry points look identical.
        self._strip_docstring(node)
        node.returns = None
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self._strip_docstring(node)
        return self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._strip_docstring(node)
        return self.generic_visit(node)

    @staticmethod
    def _strip_docstring(node) -> None:
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            # Keep the body non-empty if the docstring was the only statement.
            node.body = body[1:] or [ast.Pass()]


def _normalize_text(code: str) -> str:
    """Fallback: collapse whitespace and strip comments and blank lines."""
    lines = []
    for raw in code.splitlines():
        line = re.sub(r"(?<!['\"])#.*$", "", raw).rstrip()
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def canonicalize_solution(code: str) -> CanonicalForm:
    """Reduce `code` to a canonical form identifying its species."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        text = _normalize_text(code)
        return CanonicalForm(
            text=text,
            method="text",
            key=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    protected = _BUILTIN_NAMES | _imported_names(tree)
    try:
        normalized = _Normalizer(protected).visit(tree)
        ast.fix_missing_locations(normalized)
        text = ast.unparse(normalized)
        method = "ast"
    except (AttributeError, RecursionError, ValueError):  # pragma: no cover
        text = _normalize_text(code)
        method = "text"

    return CanonicalForm(
        text=text,
        method=method,
        key=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def species_key(code: str) -> str:
    """Shorthand for `canonicalize_solution(code).key`."""
    return canonicalize_solution(code).key
