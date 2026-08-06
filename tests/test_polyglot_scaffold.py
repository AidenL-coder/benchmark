"""Tests for `cbs.scaffolds.polyglot_scaffold` (D-42).

Entirely synthetic -- no Docker, no real HGM checkout, no network -- same
style as `tests/test_swebench_scaffold.py`: `agent_fn` is a hand-built fake
with known, controllable behaviour, so `S0Polyglot`'s own logic (budget
charging, trace passthrough, result shape) is tested in isolation from
anything Docker/HGM-specific.
"""

from __future__ import annotations

from cbs.budget import BudgetAccountant, BudgetCaps, Usage
from cbs.scaffolds.polyglot_scaffold import PolyglotRunResult, S0Polyglot
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.polyglot import PolyglotInstance


def make_instance(**overrides) -> PolyglotInstance:
    defaults = dict(
        instance_id="cpp__fake-exercise",
        language="cpp",
        problem_statement="Implement the thing.",
        raw={"instance_id": "cpp__fake-exercise", "language": "cpp"},
    )
    defaults.update(overrides)
    return PolyglotInstance(**defaults)


def make_accountant(caps: BudgetCaps | None = None) -> BudgetAccountant:
    return BudgetAccountant("test", caps=caps)


def trajectory_trace(had_tool_call: bool = True) -> OperationTrace:
    t = OperationTrace()
    t.record_instant("single_call", intercepted=True, had_tool_calls=had_tool_call)
    if had_tool_call:
        t.record_instant("tool_call", intercepted=True)
    return t


class TestS0Polyglot:
    def test_returns_the_agent_functions_solution_verbatim(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="diff --git a/foo.cpp b/foo.cpp\n+fix\n",
                eval_result="resolved",
                trace=trajectory_trace(),
                usage=Usage(calls=1, prompt_tokens=100, completion_tokens=50),
            )

        result = S0Polyglot().solve(make_instance(), agent_fn, make_accountant())
        assert result.solution == "diff --git a/foo.cpp b/foo.cpp\n+fix\n"
        assert result.usage == Usage(calls=1, prompt_tokens=100, completion_tokens=50)

    def test_passed_reflects_resolved_eval_result(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="d", eval_result="resolved", trace=OperationTrace(), usage=Usage()
            )

        result = S0Polyglot().solve(make_instance(), agent_fn, make_accountant())
        assert result.passed is True
        assert result.eval_result == "resolved"

    def test_passed_false_for_unresolved(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="d", eval_result="unresolved", trace=OperationTrace(), usage=Usage()
            )

        result = S0Polyglot().solve(make_instance(), agent_fn, make_accountant())
        assert result.passed is False
        assert result.eval_result == "unresolved"

    def test_passed_false_for_empty_patch(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="", eval_result="empty_patch", trace=OperationTrace(), usage=Usage()
            )

        result = S0Polyglot().solve(make_instance(), agent_fn, make_accountant())
        assert result.passed is False
        assert result.eval_result == "empty_patch"

    def test_trajectory_trace_is_passed_through(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="d", eval_result="unresolved", trace=trajectory_trace(), usage=Usage()
            )

        result = S0Polyglot().solve(make_instance(), agent_fn, make_accountant())
        assert result.trace.op_counts() == {"single_call": 1, "tool_call": 1}

    def test_agent_error_is_reported(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="",
                eval_result="",
                trace=OperationTrace(),
                usage=Usage(),
                error="container crashed",
            )

        result = S0Polyglot().solve(make_instance(), agent_fn, make_accountant())
        assert result.error == "container crashed"
        assert result.passed is False

    def test_budget_exceeded_is_reported_not_raised(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="d",
                eval_result="resolved",
                trace=OperationTrace(),
                usage=Usage(calls=1, prompt_tokens=1000),
            )

        tight = make_accountant(BudgetCaps(prompt_tokens=10))
        result = S0Polyglot().solve(make_instance(), agent_fn, tight)
        assert result.budget_exhausted is True
        # The trajectory already ran and already produced a real solution --
        # post-hoc charging surfaces the overrun, it doesn't discard what
        # was already produced (same discipline as S0SweBench/SStarSweBench).
        assert result.solution == "d"
        assert result.passed is True

    def test_instance_id_and_seed_recorded(self):
        def agent_fn(instance, problem_statement):
            return PolyglotRunResult(
                solution="d", eval_result="resolved", trace=OperationTrace(), usage=Usage()
            )

        result = S0Polyglot().solve(
            make_instance(instance_id="go__other"), agent_fn, make_accountant(), seed=7
        )
        assert result.instance_id == "go__other"
        assert result.metadata == {"seed": 7}

    def test_agent_fn_receives_the_instances_problem_statement(self):
        received = []

        def agent_fn(instance, problem_statement):
            received.append(problem_statement)
            return PolyglotRunResult(
                solution="d", eval_result="resolved", trace=OperationTrace(), usage=Usage()
            )

        instance = make_instance(problem_statement="Solve the specific thing.")
        S0Polyglot().solve(instance, agent_fn, make_accountant())
        assert received == ["Solve the specific thing."]

    def test_config_fingerprint(self):
        assert S0Polyglot().config_fingerprint() == {"name": "S0_polyglot"}
