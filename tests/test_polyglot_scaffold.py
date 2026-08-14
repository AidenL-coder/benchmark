"""Tests for `cbs.scaffolds.polyglot_scaffold` (D-42).

Entirely synthetic -- no Docker, no real HGM checkout, no network -- same
style as `tests/test_swebench_scaffold.py`: `agent_fn` is a hand-built fake
with known, controllable behaviour, so `S0Polyglot`'s own logic (budget
charging, trace passthrough, result shape) is tested in isolation from
anything Docker/HGM-specific.
"""

from __future__ import annotations

from cbs.budget import BudgetAccountant, BudgetCaps, Usage
from cbs.scaffolds.polyglot_scaffold import (
    PolyglotRunResult,
    S0Polyglot,
    SStarPolyglotBestOfN,
)
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


class TestSStarPolyglotBestOfN:
    """Elicitation control (D-47). The critical property under test is that
    selection is oracle-blind: a passing candidate must NOT be preferred
    just because it passed."""

    @staticmethod
    def _runs_agent(sequence):
        """agent_fn returning a scripted list of (diff, eval_result)."""
        it = iter(sequence)

        def agent_fn(instance, problem_statement):
            diff, ev = next(it)
            return PolyglotRunResult(
                solution=diff,
                eval_result=ev,
                trace=OperationTrace(),
                usage=Usage(calls=1, prompt_tokens=10, completion_tokens=10),
            )

        return agent_fn

    def test_selection_is_oracle_blind_majority_wins_over_the_passing_one(self):
        # Two identical failing diffs outvote a single passing one. An
        # oracle-peeking scaffold would pick "good"; this must not.
        agent_fn = self._runs_agent(
            [("bad", "unresolved"), ("bad", "unresolved"), ("good", "resolved")]
        )
        r = SStarPolyglotBestOfN(n_candidates=3).solve(
            make_instance(), agent_fn, make_accountant()
        )
        assert r.solution == "bad"
        assert r.passed is False, "selection must not prefer a candidate for passing"

    def test_pass_at_n_records_the_upper_bound_separately(self):
        agent_fn = self._runs_agent(
            [("bad", "unresolved"), ("bad", "unresolved"), ("good", "resolved")]
        )
        r = SStarPolyglotBestOfN(n_candidates=3).solve(
            make_instance(), agent_fn, make_accountant()
        )
        # scaffold failed, but the frozen model did reach a solution in N tries
        assert r.passed is False
        assert r.metadata["pass_at_n"] is True
        assert r.metadata["n_passing_candidates"] == 1

    def test_consensus_picks_the_passing_diff_when_it_is_the_majority(self):
        agent_fn = self._runs_agent(
            [("good", "resolved"), ("good", "resolved"), ("bad", "unresolved")]
        )
        r = SStarPolyglotBestOfN(n_candidates=3).solve(
            make_instance(), agent_fn, make_accountant()
        )
        assert r.solution == "good"
        assert r.passed is True
        assert r.metadata["pass_at_n"] is True

    def test_usage_sums_across_all_candidates(self):
        agent_fn = self._runs_agent([("a", "unresolved")] * 3)
        acc = make_accountant()
        r = SStarPolyglotBestOfN(n_candidates=3).solve(make_instance(), agent_fn, acc)
        assert r.usage.calls == 3
        assert r.usage.completion_tokens == 30

    def test_self_consistency_operation_is_traced(self):
        agent_fn = self._runs_agent([("a", "unresolved")] * 2)
        r = SStarPolyglotBestOfN(n_candidates=2).solve(
            make_instance(), agent_fn, make_accountant()
        )
        assert r.trace.op_counts().get("self_consistency") == 1

    def test_all_empty_diffs_degrades_without_crashing(self):
        agent_fn = self._runs_agent([("", "empty_patch")] * 2)
        r = SStarPolyglotBestOfN(n_candidates=2).solve(
            make_instance(), agent_fn, make_accountant()
        )
        assert r.solution == ""
        assert r.passed is False
        assert r.metadata["pass_at_n"] is False

    def test_config_fingerprint_records_n(self):
        assert SStarPolyglotBestOfN(n_candidates=5).config_fingerprint() == {
            "name": "S_star_polyglot_bestofn",
            "n_candidates": 5,
        }
