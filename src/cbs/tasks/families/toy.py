"""Toy Python task family with fully known ground truth.

Purpose (brief section 7, Phase 2 DoD): "on a toy task with known ground-truth
solve rate, estimator recovers it within CI". That demands a family where both

*   the true solve probability `p(x)`, and
*   the true number of distinct correct solutions (species richness),

are known exactly. Here they are known by construction: the mock model draws from
the declared `correct_variants` / `incorrect_variants` pools at a declared rate.

The variants are *real source code* run through the *real verifier*, so
validating the estimator also validates extraction, sandboxing, verification and
canonicalisation. `tests/test_toy_family.py` asserts that every declared correct
variant genuinely passes and every declared incorrect variant genuinely fails --
without that check the "ground truth" would be an unverified assumption, and a
canonicaliser bug could silently merge two variants into one species and make the
richness estimate look wrong when the estimator was fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cbs.models.mock import MockTaskBehaviour
from cbs.tasks.schema import Task, TaskSuite

__all__ = ["ToyTaskDef", "TOY_TASKS", "toy_suite", "toy_behaviours"]


@dataclass(frozen=True)
class ToyTaskDef:
    """A toy task plus the ground-truth behaviour of the mock model on it."""

    task_id: str
    prompt: str
    entry_point: str
    tests: str
    correct_variants: list[str]
    incorrect_variants: list[str]
    #: True per-sample solve probability under the mock model.
    p_correct: float = 0.5
    #: Relative weights over `correct_variants`. Skewed weights produce
    #: singletons, which is the regime where Good-Turing is informative.
    species_weights: list[float] | None = None
    metadata: dict = field(default_factory=dict)

    def to_task(self, family: str = "toy") -> Task:
        return Task(
            task_id=self.task_id,
            family=family,
            prompt=self.prompt,
            tests=self.tests,
            entry_point=self.entry_point,
            reference_solution=self.correct_variants[0],
            timeout_s=10.0,
            metadata={
                **self.metadata,
                "true_p_correct": self.p_correct,
                "true_species_count": len(self.correct_variants),
            },
        )

    def to_behaviour(self, p_correct: float | None = None) -> MockTaskBehaviour:
        return MockTaskBehaviour(
            p_correct=self.p_correct if p_correct is None else p_correct,
            correct_variants=list(self.correct_variants),
            incorrect_variants=list(self.incorrect_variants),
            species_weights=(
                list(self.species_weights) if self.species_weights else None
            ),
        )


def _prompt(signature: str, description: str) -> str:
    return (
        f"Write a Python function `{signature}` that {description}\n"
        "Respond with only the function definition in a Python code block."
    )


TOY_TASKS: list[ToyTaskDef] = [
    ToyTaskDef(
        task_id="toy/sum_list",
        prompt=_prompt("sum_list(xs)", "returns the sum of a list of integers."),
        entry_point="sum_list",
        tests=(
            "assert sum_list([]) == 0\n"
            "assert sum_list([1]) == 1\n"
            "assert sum_list([1, 2, 3]) == 6\n"
            "assert sum_list([-1, 1]) == 0\n"
            "assert sum_list([10, -3, 5, 8]) == 20\n"
        ),
        correct_variants=[
            "def sum_list(xs):\n    return sum(xs)\n",
            "def sum_list(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n",
            "import functools\n\ndef sum_list(xs):\n    return functools.reduce(lambda a, b: a + b, xs, 0)\n",
            "def sum_list(xs):\n    if not xs:\n        return 0\n    return xs[0] + sum_list(xs[1:])\n",
        ],
        incorrect_variants=[
            "def sum_list(xs):\n    return len(xs)\n",
            "def sum_list(xs):\n    return sum(xs[1:])\n",
            "def sum_list(xs):\n    return max(xs) if xs else 0\n",
        ],
        p_correct=0.60,
        species_weights=[8.0, 4.0, 1.5, 1.0],
    ),
    ToyTaskDef(
        task_id="toy/is_palindrome",
        prompt=_prompt(
            "is_palindrome(s)", "returns True if the string s reads the same backwards."
        ),
        entry_point="is_palindrome",
        tests=(
            "assert is_palindrome('') is True or is_palindrome('') == True\n"
            "assert is_palindrome('a') == True\n"
            "assert is_palindrome('racecar') == True\n"
            "assert is_palindrome('abba') == True\n"
            "assert is_palindrome('ab') == False\n"
            "assert is_palindrome('abc') == False\n"
            "assert is_palindrome('abca') == False\n"
        ),
        correct_variants=[
            "def is_palindrome(s):\n    return s == s[::-1]\n",
            "def is_palindrome(s):\n    i, j = 0, len(s) - 1\n    while i < j:\n        if s[i] != s[j]:\n            return False\n        i += 1\n        j -= 1\n    return True\n",
            "def is_palindrome(s):\n    return s == ''.join(reversed(s))\n",
            "def is_palindrome(s):\n    return list(s) == list(s)[::-1]\n",
        ],
        incorrect_variants=[
            "def is_palindrome(s):\n    return True\n",
            "def is_palindrome(s):\n    return sorted(s) == sorted(s[::-1])\n",
            "def is_palindrome(s):\n    return s[0] == s[-1] if s else True\n",
        ],
        p_correct=0.45,
        species_weights=[10.0, 2.0, 2.0, 1.0],
    ),
    ToyTaskDef(
        task_id="toy/count_vowels",
        prompt=_prompt(
            "count_vowels(s)",
            "returns the number of vowels (a, e, i, o, u) in the lowercase string s.",
        ),
        entry_point="count_vowels",
        tests=(
            "assert count_vowels('') == 0\n"
            "assert count_vowels('xyz') == 0\n"
            "assert count_vowels('aeiou') == 5\n"
            "assert count_vowels('hello world') == 3\n"
            "assert count_vowels('rhythm') == 0\n"
            "assert count_vowels('queue') == 4\n"
        ),
        correct_variants=[
            "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
            "def count_vowels(s):\n    n = 0\n    for c in s:\n        if c in ('a', 'e', 'i', 'o', 'u'):\n            n += 1\n    return n\n",
            "import re\n\ndef count_vowels(s):\n    return len(re.findall(r'[aeiou]', s))\n",
            "def count_vowels(s):\n    return len([c for c in s if c in set('aeiou')])\n",
        ],
        incorrect_variants=[
            "def count_vowels(s):\n    return sum(1 for c in s if c not in 'aeiou')\n",
            "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeio')\n",
            "def count_vowels(s):\n    return len(s)\n",
        ],
        p_correct=0.35,
        species_weights=[6.0, 3.0, 1.0, 1.0],
    ),
    ToyTaskDef(
        task_id="toy/gcd",
        prompt=_prompt(
            "gcd(a, b)",
            "returns the greatest common divisor of two positive integers.",
        ),
        entry_point="gcd",
        tests=(
            "assert gcd(1, 1) == 1\n"
            "assert gcd(12, 18) == 6\n"
            "assert gcd(18, 12) == 6\n"
            "assert gcd(7, 13) == 1\n"
            "assert gcd(100, 75) == 25\n"
            "assert gcd(9, 9) == 9\n"
        ),
        correct_variants=[
            "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
            "def gcd(a, b):\n    if b == 0:\n        return a\n    return gcd(b, a % b)\n",
            "import math\n\ndef gcd(a, b):\n    return math.gcd(a, b)\n",
        ],
        incorrect_variants=[
            "def gcd(a, b):\n    return min(a, b)\n",
            "def gcd(a, b):\n    return 1\n",
            "def gcd(a, b):\n    return a * b\n",
        ],
        p_correct=0.25,
        species_weights=[5.0, 2.0, 1.0],
    ),
    ToyTaskDef(
        task_id="toy/unique_sorted",
        prompt=_prompt(
            "unique_sorted(xs)",
            "returns the distinct elements of the list xs in ascending order.",
        ),
        entry_point="unique_sorted",
        tests=(
            "assert unique_sorted([]) == []\n"
            "assert unique_sorted([1]) == [1]\n"
            "assert unique_sorted([3, 1, 2]) == [1, 2, 3]\n"
            "assert unique_sorted([2, 2, 2]) == [2]\n"
            "assert unique_sorted([5, 3, 5, 1, 3]) == [1, 3, 5]\n"
        ),
        correct_variants=[
            "def unique_sorted(xs):\n    return sorted(set(xs))\n",
            "def unique_sorted(xs):\n    out = []\n    for x in sorted(xs):\n        if not out or out[-1] != x:\n            out.append(x)\n    return out\n",
            "def unique_sorted(xs):\n    return sorted(dict.fromkeys(xs))\n",
        ],
        incorrect_variants=[
            "def unique_sorted(xs):\n    return sorted(xs)\n",
            "def unique_sorted(xs):\n    return list(set(xs))[:1]\n",
            "def unique_sorted(xs):\n    return xs\n",
        ],
        p_correct=0.50,
        species_weights=[7.0, 2.0, 1.0],
    ),
    # A deliberately hard task: p_correct = 0 makes it a known beyond-frontier
    # case, so the crossing test's negative branch is exercisable end to end.
    ToyTaskDef(
        task_id="toy/impossible_parity",
        prompt=_prompt(
            "weird_parity(n)",
            "returns 'even' for even n and 'odd' for odd n, but 'zero' for n == 0.",
        ),
        entry_point="weird_parity",
        tests=(
            "assert weird_parity(0) == 'zero'\n"
            "assert weird_parity(1) == 'odd'\n"
            "assert weird_parity(2) == 'even'\n"
            "assert weird_parity(-3) == 'odd'\n"
            "assert weird_parity(-4) == 'even'\n"
        ),
        correct_variants=[
            "def weird_parity(n):\n    if n == 0:\n        return 'zero'\n    return 'even' if n % 2 == 0 else 'odd'\n",
        ],
        incorrect_variants=[
            "def weird_parity(n):\n    return 'even' if n % 2 == 0 else 'odd'\n",
            "def weird_parity(n):\n    return 'zero'\n",
        ],
        p_correct=0.0,
        metadata={"designed_beyond_frontier": True},
    ),
]


def toy_suite(family: str = "toy") -> TaskSuite:
    return TaskSuite(name=family, tasks=[d.to_task(family) for d in TOY_TASKS])


def toy_behaviours(
    p_overrides: dict[str, float] | None = None,
) -> dict[str, MockTaskBehaviour]:
    """Mock behaviours matching `toy_suite()`, keyed by task id."""
    p_overrides = p_overrides or {}
    return {
        d.task_id: d.to_behaviour(p_overrides.get(d.task_id)) for d in TOY_TASKS
    }


def toy_defs_by_id() -> dict[str, ToyTaskDef]:
    return {d.task_id: d for d in TOY_TASKS}
