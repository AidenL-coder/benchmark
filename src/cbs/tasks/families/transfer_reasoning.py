"""Transfer/reasoning task family (brief section 8, D-17).

The proposal and brief both require a "transfer (reasoning)" family: "a
disjoint family the loop never optimises on... e.g., a math/logic set with
checkable answers", used only to test RQ4 -- whether an evolved scaffold's
gains generalise to a distribution it was never evolved against, or evaporate
outside the training distribution (brief section 5.2, `cbs.tasks.splits`'s
`transfer_families` mechanism already routes a whole family to the transfer
split rather than splitting it by ratio, exactly for this reason).

What makes a family genuinely "transfer" here is *content*, not verification
mechanics: every task below is still "write a Python function", verified
identically to `toy`/`humaneval`/`mbpp` (no new verification mode needed), but
each is a math/logic/combinatorial-reasoning problem -- number theory, basic
algebra, game-theoretic optimal play, combinatorial counting -- deliberately
unlike the general-purpose programming idioms `humaneval`/`mbpp` are built
from. A scaffold that generalises here is generalising across problem
*character*, not just across a held-out slice of the same distribution.

Hand-authored, not vendored: no small, permissively-licensed, code-checkable
reasoning benchmark was as immediately available as HumanEval/MBPP were, and
the set only needs to be a genuinely distinct, checkable distribution, not
large -- brief section 8 asks for "one set", not exhaustive coverage. Every
reference solution is verified against the real sandbox exactly like the other
real families (`tests/test_transfer_reasoning_family.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cbs.tasks.schema import Task, TaskSuite

__all__ = ["TRANSFER_TASKS", "TransferTaskDef", "transfer_suite"]


@dataclass(frozen=True)
class TransferTaskDef:
    task_id: str
    prompt: str
    entry_point: str
    tests: str
    reference_solution: str
    metadata: dict = field(default_factory=dict)

    def to_task(self, family: str = "transfer_reasoning") -> Task:
        return Task(
            task_id=self.task_id,
            family=family,
            prompt=self.prompt,
            tests=self.tests,
            public_tests=_derive_public_tests(self.tests),
            entry_point=self.entry_point,
            reference_solution=self.reference_solution,
            timeout_s=10.0,
            metadata={**self.metadata, "source": "hand-authored (cbs)"},
        )


def _derive_public_tests(tests: str) -> str:
    """First half (rounded up) of the flat assertion lines -- same convention
    as D-18/toy.py. Safe here because every task's tests below are simple,
    single-line, self-contained assertions by construction (no loop-scoped
    setup, no multi-line literals), unlike some real-world HumanEval cases."""
    lines = [l for l in tests.splitlines() if l.strip()]
    n_public = max(1, (len(lines) + 1) // 2)
    return "\n".join(lines[:n_public]) + "\n"


def _prompt(signature: str, description: str) -> str:
    return (
        f"Write a Python function `{signature}` that {description}\n"
        "Respond with only the function definition in a Python code block."
    )


TRANSFER_TASKS: list[TransferTaskDef] = [
    TransferTaskDef(
        task_id="transfer/compound_interest",
        prompt=_prompt(
            "compound_interest(principal, rate_pct, years)",
            "returns the final amount after compounding annually at rate_pct "
            "percent for the given whole number of years, rounded to 2 decimal places.",
        ),
        entry_point="compound_interest",
        tests=(
            "assert compound_interest(1000, 0, 5) == 1000.0\n"
            "assert compound_interest(1000, 10, 1) == 1100.0\n"
            "assert compound_interest(1000, 10, 2) == 1210.0\n"
            "assert compound_interest(2000, 5, 3) == 2315.25\n"
            "assert compound_interest(500, 100, 1) == 1000.0\n"
        ),
        reference_solution=(
            "def compound_interest(principal, rate_pct, years):\n"
            "    return round(principal * (1 + rate_pct / 100) ** years, 2)\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/triangle_validity",
        prompt=_prompt(
            "is_valid_triangle(a, b, c)",
            "returns True if three positive side lengths a, b, c can form a "
            "triangle (the triangle inequality holds strictly), else False.",
        ),
        entry_point="is_valid_triangle",
        tests=(
            "assert is_valid_triangle(3, 4, 5) == True\n"
            "assert is_valid_triangle(1, 1, 3) == False\n"
            "assert is_valid_triangle(2, 2, 2) == True\n"
            "assert is_valid_triangle(1, 2, 3) == False\n"
            "assert is_valid_triangle(5, 5, 9.9) == True\n"
            "assert is_valid_triangle(0, 1, 1) == False\n"
        ),
        reference_solution=(
            "def is_valid_triangle(a, b, c):\n"
            "    if a <= 0 or b <= 0 or c <= 0:\n"
            "        return False\n"
            "    return a + b > c and a + c > b and b + c > a\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/min_coins",
        prompt=_prompt(
            "min_coins(amount, denominations)",
            "returns the minimum number of coins from the given list of "
            "denominations needed to make exactly amount, or -1 if it cannot "
            "be made exactly. Assume unlimited coins of each denomination.",
        ),
        entry_point="min_coins",
        tests=(
            "assert min_coins(0, [1, 2, 5]) == 0\n"
            "assert min_coins(11, [1, 2, 5]) == 3\n"
            "assert min_coins(6, [1, 3, 4]) == 2\n"
            "assert min_coins(7, [2, 4]) == -1\n"
            "assert min_coins(27, [1, 5, 10, 25]) == 3\n"
        ),
        reference_solution=(
            "def min_coins(amount, denominations):\n"
            "    INF = float('inf')\n"
            "    best = [0] + [INF] * amount\n"
            "    for total in range(1, amount + 1):\n"
            "        for coin in denominations:\n"
            "            if coin <= total and best[total - coin] + 1 < best[total]:\n"
            "                best[total] = best[total - coin] + 1\n"
            "    return -1 if best[amount] == INF else best[amount]\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/perfect_number",
        prompt=_prompt(
            "is_perfect_number(n)",
            "returns True if the positive integer n is a perfect number "
            "(equal to the sum of its proper divisors), else False.",
        ),
        entry_point="is_perfect_number",
        tests=(
            "assert is_perfect_number(6) == True\n"
            "assert is_perfect_number(28) == True\n"
            "assert is_perfect_number(12) == False\n"
            "assert is_perfect_number(1) == False\n"
            "assert is_perfect_number(496) == True\n"
        ),
        reference_solution=(
            "def is_perfect_number(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    return sum(d for d in range(1, n) if n % d == 0) == n\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/lcm_of_list",
        prompt=_prompt(
            "lcm_of_list(numbers)",
            "returns the least common multiple of a non-empty list of "
            "positive integers.",
        ),
        entry_point="lcm_of_list",
        tests=(
            "assert lcm_of_list([4, 6]) == 12\n"
            "assert lcm_of_list([2, 3, 5]) == 30\n"
            "assert lcm_of_list([7]) == 7\n"
            "assert lcm_of_list([6, 8, 12]) == 24\n"
            "assert lcm_of_list([1, 1, 1]) == 1\n"
        ),
        reference_solution=(
            "import math\n\n"
            "def lcm_of_list(numbers):\n"
            "    result = numbers[0]\n"
            "    for n in numbers[1:]:\n"
            "        result = result * n // math.gcd(result, n)\n"
            "    return result\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/quadratic_roots",
        prompt=_prompt(
            "quadratic_roots(a, b, c)",
            "returns a sorted tuple of the real roots of a*x**2 + b*x + c = 0 "
            "rounded to 4 decimal places, or None if there are no real roots. "
            "Assume a != 0.",
        ),
        entry_point="quadratic_roots",
        tests=(
            "assert quadratic_roots(1, -3, 2) == (1.0, 2.0)\n"
            "assert quadratic_roots(1, 0, -4) == (-2.0, 2.0)\n"
            "assert quadratic_roots(1, 2, 1) == (-1.0, -1.0)\n"
            "assert quadratic_roots(1, 0, 1) is None\n"
            "assert quadratic_roots(2, -8, 8) == (2.0, 2.0)\n"
        ),
        reference_solution=(
            "def quadratic_roots(a, b, c):\n"
            "    disc = b * b - 4 * a * c\n"
            "    if disc < 0:\n"
            "        return None\n"
            "    sqrt_disc = disc ** 0.5\n"
            "    r1 = (-b - sqrt_disc) / (2 * a)\n"
            "    r2 = (-b + sqrt_disc) / (2 * a)\n"
            "    return tuple(sorted((round(r1, 4), round(r2, 4))))\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/nim_winner",
        prompt=_prompt(
            "nim_winner(piles)",
            "returns 'first' if the first player wins a normal-play Nim game "
            "with the given list of pile sizes under optimal play from both "
            "sides, else 'second'.",
        ),
        entry_point="nim_winner",
        tests=(
            "assert nim_winner([1, 1]) == 'second'\n"
            "assert nim_winner([1, 2, 3]) == 'second'\n"
            "assert nim_winner([1, 2, 4]) == 'first'\n"
            "assert nim_winner([0]) == 'second'\n"
            "assert nim_winner([5]) == 'first'\n"
        ),
        reference_solution=(
            "def nim_winner(piles):\n"
            "    xor_sum = 0\n"
            "    for pile in piles:\n"
            "        xor_sum ^= pile\n"
            "    return 'first' if xor_sum != 0 else 'second'\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/distinct_prime_factors",
        prompt=_prompt(
            "count_distinct_prime_factors(n)",
            "returns the number of distinct prime factors of the positive "
            "integer n (n > 1).",
        ),
        entry_point="count_distinct_prime_factors",
        tests=(
            "assert count_distinct_prime_factors(2) == 1\n"
            "assert count_distinct_prime_factors(12) == 2\n"
            "assert count_distinct_prime_factors(30) == 3\n"
            "assert count_distinct_prime_factors(97) == 1\n"
            "assert count_distinct_prime_factors(60) == 3\n"
        ),
        reference_solution=(
            "def count_distinct_prime_factors(n):\n"
            "    count = 0\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "        if n % d == 0:\n"
            "            count += 1\n"
            "            while n % d == 0:\n"
            "                n //= d\n"
            "        d += 1\n"
            "    if n > 1:\n"
            "        count += 1\n"
            "    return count\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/linear_system_2x2",
        prompt=_prompt(
            "solve_linear_system(a1, b1, c1, a2, b2, c2)",
            "solves the 2x2 linear system a1*x + b1*y = c1, a2*x + b2*y = c2 "
            "and returns (x, y) rounded to 4 decimal places, or None if the "
            "system has no unique solution.",
        ),
        entry_point="solve_linear_system",
        tests=(
            "assert solve_linear_system(1, 1, 5, 1, -1, 1) == (3.0, 2.0)\n"
            "assert solve_linear_system(2, 0, 4, 0, 3, 9) == (2.0, 3.0)\n"
            "assert solve_linear_system(1, 1, 2, 2, 2, 4) is None\n"
            "assert solve_linear_system(1, 0, 7, 0, 1, -3) == (7.0, -3.0)\n"
        ),
        reference_solution=(
            "def solve_linear_system(a1, b1, c1, a2, b2, c2):\n"
            "    det = a1 * b2 - a2 * b1\n"
            "    if det == 0:\n"
            "        return None\n"
            "    x = (c1 * b2 - c2 * b1) / det\n"
            "    y = (a1 * c2 - a2 * c1) / det\n"
            "    return (round(x, 4), round(y, 4))\n"
        ),
    ),
    TransferTaskDef(
        task_id="transfer/josephus_survivor",
        prompt=_prompt(
            "josephus_survivor(n, k)",
            "returns the 1-indexed position of the sole survivor in the "
            "Josephus problem with n people standing in a circle, eliminating "
            "every k-th person going around, starting the count at person 1.",
        ),
        entry_point="josephus_survivor",
        tests=(
            "assert josephus_survivor(1, 3) == 1\n"
            "assert josephus_survivor(5, 2) == 3\n"
            "assert josephus_survivor(7, 3) == 4\n"
            "assert josephus_survivor(6, 5) == 1\n"
        ),
        reference_solution=(
            "def josephus_survivor(n, k):\n"
            "    survivor = 0\n"
            "    for i in range(2, n + 1):\n"
            "        survivor = (survivor + k) % i\n"
            "    return survivor + 1\n"
        ),
    ),
]


def transfer_suite(family: str = "transfer_reasoning") -> TaskSuite:
    return TaskSuite(name=family, tasks=[d.to_task(family) for d in TRANSFER_TASKS])
