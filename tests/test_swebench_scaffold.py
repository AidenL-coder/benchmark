"""Tests for `cbs.scaffolds.swebench_scaffold` (D-40).

Entirely synthetic -- no Docker, no real fork, no network -- the same style
`tests/test_evolved.py` uses for `EvolvedScaffold`: `agent_fn`/`verify_fn`
are hand-built fakes with known, controllable behaviour, so the scaffold
logic (best-of-N, repair-on-feedback, self-consistency, budget charging,
trace merging) can be tested in isolation from anything Docker/HGM-specific.
"""

from __future__ import annotations

import pytest

from cbs.budget import BudgetAccountant, BudgetCaps, Usage
from cbs.scaffolds.swebench_scaffold import (
    S0SweBench,
    SStarSweBench,
    SweBenchAttempt,
    SweBenchVerifyResult,
)
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.swebench import SweBenchInstance


def make_instance(**overrides) -> SweBenchInstance:
    defaults = dict(
        instance_id="fake__repo-1",
        repo="fake/repo",
        base_commit="a" * 40,
        problem_statement="Fix the bug in foo().",
        fail_to_pass=("tests/test_foo.py::test_bug_fixed",),
        pass_to_pass=("tests/test_foo.py::test_existing_behaviour",),
        patch="diff --git a/foo.py b/foo.py\n+gold fix\n",
        test_patch="diff --git a/tests/test_foo.py b/tests/test_foo.py\n+new test\n",
    )
    defaults.update(overrides)
    return SweBenchInstance(**defaults)


def make_accountant(caps: BudgetCaps | None = None) -> BudgetAccountant:
    return BudgetAccountant("test", caps=caps)


def trajectory_trace(had_tool_call: bool = True) -> OperationTrace:
    """A small, realistic sub-trace, as `fork_bridge.reconstruct_trace_from_events`
    would produce for one real trajectory."""
    t = OperationTrace()
    t.record_instant("single_call", intercepted=True, had_tool_calls=had_tool_call)
    if had_tool_call:
        t.record_instant("tool_call", intercepted=True)
    return t


class TestS0SweBench:
    def test_returns_the_agent_functions_diff_verbatim(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(
                diff="diff --git a/foo.py b/foo.py\n+fix\n",
                trace=trajectory_trace(),
                usage=Usage(calls=3, prompt_tokens=100, completion_tokens=50),
            )

        result = S0SweBench().solve(make_instance(), agent_fn, make_accountant())
        assert result.solution == "diff --git a/foo.py b/foo.py\n+fix\n"
        assert result.usage == Usage(calls=3, prompt_tokens=100, completion_tokens=50)

    def test_trajectory_trace_is_merged_into_the_result_trace(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff="d", trace=trajectory_trace(), usage=Usage())

        result = S0SweBench().solve(make_instance(), agent_fn, make_accountant())
        assert result.trace.op_counts() == {"single_call": 1, "tool_call": 1}

    def test_calls_verify_fn_exactly_once_with_fail_to_pass_only(self):
        calls = []

        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff="d", trace=OperationTrace(), usage=Usage())

        def verify_fn(instance, diff, tests):
            calls.append((diff, tests))
            return SweBenchVerifyResult(passed=True)

        instance = make_instance()
        result = S0SweBench().solve(instance, agent_fn, make_accountant(), verify_fn)
        assert len(calls) == 1
        assert calls[0] == ("d", instance.fail_to_pass)
        assert result.verification.passed is True

    def test_no_verify_fn_means_no_verification(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff="d", trace=OperationTrace(), usage=Usage())

        result = S0SweBench().solve(make_instance(), agent_fn, make_accountant())
        assert result.verification is None

    def test_agent_error_prevents_verification_but_is_reported(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(
                diff="", trace=OperationTrace(), usage=Usage(), error="container crashed"
            )

        calls = []

        def verify_fn(instance, diff, tests):
            calls.append(diff)
            return SweBenchVerifyResult(passed=False)

        result = S0SweBench().solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        assert result.error == "container crashed"
        assert calls == []

    def test_budget_exceeded_is_reported_not_raised(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(
                diff="d", trace=OperationTrace(), usage=Usage(calls=1, prompt_tokens=1000)
            )

        tight = make_accountant(BudgetCaps(prompt_tokens=10))
        result = S0SweBench().solve(make_instance(), agent_fn, tight)
        assert result.budget_exhausted is True
        # The trajectory already ran (real cost already incurred) -- the
        # solution it produced is still reported, not thrown away.
        assert result.solution == "d"


class TestSStarSweBenchBestOfNAndConsensus:
    def test_single_candidate_that_passes_public_is_taken(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff="good-diff", trace=OperationTrace(), usage=Usage(calls=1))

        def verify_fn(instance, diff, tests):
            return SweBenchVerifyResult(passed=True)

        scaffold = SStarSweBench(max_candidates=4, stop_on_first_public_pass=True)
        result = scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        assert result.solution == "good-diff"
        assert result.metadata["n_candidates"] == 1

    def test_majority_vote_among_public_passing_candidates(self):
        diffs = iter(["diff-A", "diff-B", "diff-A", "diff-A"])

        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff=next(diffs), trace=OperationTrace(), usage=Usage(calls=1))

        def verify_fn(instance, diff, tests):
            return SweBenchVerifyResult(passed=True)  # everything passes public

        scaffold = SStarSweBench(
            max_candidates=4, max_repairs_per_candidate=0, stop_on_first_public_pass=False
        )
        result = scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        assert result.solution == "diff-A"  # 3 of 4 votes

    def test_falls_back_to_all_candidates_if_none_pass_public(self):
        diffs = iter(["diff-A", "diff-B", "diff-A"])

        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff=next(diffs), trace=OperationTrace(), usage=Usage(calls=1))

        def verify_fn(instance, diff, tests):
            return SweBenchVerifyResult(passed=False, failed_tests=("t1",))

        scaffold = SStarSweBench(
            max_candidates=3, max_repairs_per_candidate=0, stop_on_first_public_pass=False
        )
        result = scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        assert result.solution == "diff-A"  # still majority, from the full pool

    def test_exactly_one_fail_to_pass_query_on_the_final_choice(self):
        fail_to_pass_calls = []

        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff="d", trace=OperationTrace(), usage=Usage(calls=1))

        def verify_fn(instance, diff, tests):
            if tests == instance.fail_to_pass:
                fail_to_pass_calls.append(diff)
            return SweBenchVerifyResult(passed=True)

        scaffold = SStarSweBench(max_candidates=3, stop_on_first_public_pass=True)
        scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        assert len(fail_to_pass_calls) == 1


class TestSStarSweBenchExecutionFeedback:
    def test_a_failing_candidate_triggers_a_repair_attempt_with_feedback_in_the_prompt(self):
        seen_prompts = []

        def agent_fn(instance, problem_statement):
            seen_prompts.append(problem_statement)
            if len(seen_prompts) == 1:
                return SweBenchAttempt(diff="broken-diff", trace=OperationTrace(), usage=Usage(calls=1))
            return SweBenchAttempt(diff="fixed-diff", trace=OperationTrace(), usage=Usage(calls=1))

        def verify_fn(instance, diff, tests):
            if diff == "broken-diff":
                return SweBenchVerifyResult(passed=False, raw_output="AssertionError: boom")
            return SweBenchVerifyResult(passed=True)

        scaffold = SStarSweBench(max_candidates=1, max_repairs_per_candidate=1)
        result = scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)

        assert result.solution == "fixed-diff"
        assert len(seen_prompts) == 2
        # The repair prompt must actually carry the failure text forward --
        # a real repair loop, not a blind retry.
        assert "broken-diff" in seen_prompts[1]
        assert "AssertionError: boom" in seen_prompts[1]

    def test_repairs_are_capped_by_max_repairs_per_candidate(self):
        call_count = [0]

        def agent_fn(instance, problem_statement):
            call_count[0] += 1
            return SweBenchAttempt(diff=f"diff-{call_count[0]}", trace=OperationTrace(), usage=Usage(calls=1))

        def verify_fn(instance, diff, tests):
            return SweBenchVerifyResult(passed=False, raw_output="always fails")

        scaffold = SStarSweBench(max_candidates=1, max_repairs_per_candidate=2)
        scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        # 1 initial attempt + 2 repairs = 3 calls for this one candidate
        assert call_count[0] == 3

    def test_no_public_signal_takes_the_first_attempt_without_feedback(self):
        call_count = [0]

        def agent_fn(instance, problem_statement):
            call_count[0] += 1
            return SweBenchAttempt(diff="only-attempt", trace=OperationTrace(), usage=Usage(calls=1))

        verify_calls = []

        def verify_fn(instance, diff, tests):
            verify_calls.append(tests)
            return SweBenchVerifyResult(passed=True)

        instance = make_instance(pass_to_pass=())
        scaffold = SStarSweBench(max_candidates=1, max_repairs_per_candidate=2)
        result = scaffold.solve(instance, agent_fn, make_accountant(), verify_fn)

        assert call_count[0] == 1  # no repair attempts without a public signal
        assert result.solution == "only-attempt"
        # verify_fn is still called once at the end, for FAIL_TO_PASS
        assert verify_calls == [instance.fail_to_pass]

    def test_execution_feedback_and_self_consistency_are_tagged_in_the_trace(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff="d", trace=OperationTrace(), usage=Usage(calls=1))

        def verify_fn(instance, diff, tests):
            return SweBenchVerifyResult(passed=True)

        scaffold = SStarSweBench(max_candidates=1)
        result = scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        assert "execution_feedback" in result.trace.op_counts()
        assert "self_consistency" in result.trace.op_counts()
        assert result.trace.used_expanding  # execution_feedback is support-expanding


class TestSStarSweBenchBudgetAndFailure:
    def test_agent_trajectory_error_stops_that_candidate_without_crashing(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(diff="", trace=OperationTrace(), usage=Usage(calls=1), error="boom")

        def verify_fn(instance, diff, tests):
            raise AssertionError("must not be called on an errored trajectory")

        scaffold = SStarSweBench(max_candidates=1, max_repairs_per_candidate=1)
        result = scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        assert result.solution == ""
        assert result.budget_exhausted is False
        assert "boom" in result.error

    def test_budget_exceeded_on_the_first_attempt_still_keeps_its_real_diff(self):
        """Unlike s_star.py's pre-check (which can skip a call entirely
        before spending anything), this module charges post-hoc: the
        trajectory has already run and already produced a real diff by the
        time the overrun is detected, so it is kept as a genuine candidate
        rather than discarded -- see S0SweBench's identical rule."""

        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(
                diff="d", trace=OperationTrace(), usage=Usage(calls=1, prompt_tokens=1000)
            )

        def verify_fn(instance, diff, tests):
            return SweBenchVerifyResult(passed=True)

        tight = make_accountant(BudgetCaps(prompt_tokens=10))
        scaffold = SStarSweBench(max_candidates=3)
        result = scaffold.solve(make_instance(), agent_fn, tight, verify_fn)
        assert result.budget_exhausted is True
        assert result.solution == "d"

    def test_agent_fn_that_never_produces_a_diff_reports_no_candidates(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(
                diff="", trace=OperationTrace(), usage=Usage(calls=1, prompt_tokens=1000)
            )

        def verify_fn(instance, diff, tests):
            raise AssertionError("must not be called with no diff to verify")

        tight = make_accountant(BudgetCaps(prompt_tokens=10))
        scaffold = SStarSweBench(max_candidates=3)
        result = scaffold.solve(make_instance(), agent_fn, tight, verify_fn)
        assert result.solution == ""
        assert result.error == "budget exhausted before any candidate was generated"

    def test_usage_accumulates_across_every_trajectory_attempted(self):
        def agent_fn(instance, problem_statement):
            return SweBenchAttempt(
                diff="d", trace=OperationTrace(), usage=Usage(calls=1, prompt_tokens=10, completion_tokens=5)
            )

        def verify_fn(instance, diff, tests):
            return SweBenchVerifyResult(passed=False, raw_output="fail")

        scaffold = SStarSweBench(max_candidates=2, max_repairs_per_candidate=1, stop_on_first_public_pass=False)
        result = scaffold.solve(make_instance(), agent_fn, make_accountant(), verify_fn)
        # 2 candidates * (1 initial + 1 repair) = 4 trajectory calls
        assert result.usage == Usage(calls=4, prompt_tokens=40, completion_tokens=20)
