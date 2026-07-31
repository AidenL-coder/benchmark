"""Tests for the S_evo integration layer (interception, classification, and
EvolvedScaffold's handling of untrusted agent code).

The central property under test is that support-class tagging does not depend
on the agent function's cooperation: it is derived entirely from observing
what the agent actually did (which model prompts it sent, whether an earlier
verifier failure's text reappears in a later prompt), never from a tag the
agent claims for itself. See docs/DECISIONS.md D-24.
"""

from __future__ import annotations

import pytest

from cbs.budget import BudgetAccountant, BudgetCaps, BudgetExceeded
from cbs.models.mock import MockModelClient
from cbs.sandbox import select_backend
from cbs.scaffolds.evolved import ArchiveEntry, EvolvedScaffold, InterceptionSession
from cbs.scaffolds.example_agents import (
    blind_best_of_n_agent,
    crashing_agent,
    feedback_repair_agent,
    single_call_agent,
)
from cbs.scaffolds.tagging import SupportClass
from cbs.tasks import Verifier
from cbs.tasks.families.toy import toy_behaviours, toy_suite


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


@pytest.fixture(scope="module")
def suite():
    return toy_suite().by_id()


def make_model(seed=0):
    return MockModelClient(toy_behaviours(), seed=seed, model_id="evolved-test")


def make_scaffold(agent_fn, disabled_ops=frozenset(), variant_id="v1"):
    entry = ArchiveEntry(variant_id=variant_id, generation=0, description=agent_fn.__name__)
    return EvolvedScaffold(agent_fn, entry, disabled_ops=disabled_ops)


class TestClassificationByBehaviour:
    def test_single_call_agent_produces_only_single_call(self, suite, verifier):
        scaffold = make_scaffold(single_call_agent)
        acct = BudgetAccountant("t", BudgetCaps(calls=10))
        result = scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=1)
        assert result.trace.op_counts() == {"single_call": 1}
        assert not result.trace.used_expanding

    def test_blind_selection_is_classified_preserving_not_expanding(self, suite, verifier):
        """No later prompt references an earlier failure's error text, so this
        must classify as test_guided_selection, not execution_feedback --
        proof the classifier does not default to 'every verifier use is
        expanding'."""
        scaffold = make_scaffold(blind_best_of_n_agent)
        acct = BudgetAccountant("t", BudgetCaps(calls=20))
        result = scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=1)
        ops = result.trace.op_counts()
        assert ops.get("test_guided_selection", 0) >= 1
        assert "execution_feedback" not in ops
        assert not result.trace.used_expanding

    def test_real_conditioning_is_classified_expanding(self, suite, verifier):
        """The agent builds its next prompt out of the previous failure's
        error text -- this must be caught as execution_feedback."""
        scaffold = make_scaffold(feedback_repair_agent)
        # Seed chosen so at least one repair is needed on this task/model.
        found_a_repair_case = False
        for seed in range(20):
            acct = BudgetAccountant("t", BudgetCaps(calls=20))
            result = scaffold.solve(
                suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=seed
            )
            if "execution_feedback" in result.trace.op_counts():
                found_a_repair_case = True
                assert result.trace.used_expanding
                break
        assert found_a_repair_case, "no seed produced a repair case to check"

    def test_model_calls_are_always_tagged_single_call_regardless_of_prompt(
        self, suite, verifier
    ):
        """Even a repair-conditioned generation is itself still a sample from
        M -- it is the VERIFIER event that may be expanding, never the model
        call, which is always support-preserving on its own."""
        scaffold = make_scaffold(feedback_repair_agent)
        acct = BudgetAccountant("t", BudgetCaps(calls=20))
        result = scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=3)
        for record in result.trace.records:
            if record.name == "single_call":
                assert record.support_class is SupportClass.PRESERVING


class TestUntrustedAgentSafety:
    def test_crashing_agent_does_not_propagate(self, suite, verifier):
        scaffold = make_scaffold(crashing_agent)
        acct = BudgetAccountant("t", BudgetCaps(calls=10))
        result = scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=0)
        assert not result.passed
        assert "RuntimeError" in (result.error or "")
        assert not result.budget_exhausted  # it crashed, it wasn't starved of budget

    def test_agent_cannot_exceed_its_allowance(self, suite, verifier):
        scaffold = make_scaffold(blind_best_of_n_agent)
        cap = 3
        acct = BudgetAccountant("t", BudgetCaps(calls=cap))
        scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=0)
        assert acct.spent.calls <= cap

    def test_zero_budget_is_reported_as_exhausted(self, suite, verifier):
        scaffold = make_scaffold(single_call_agent)
        acct = BudgetAccountant("t", BudgetCaps(calls=0))
        result = scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=0)
        assert result.budget_exhausted
        assert result.solution == ""

    def test_missing_verifier_is_rejected(self, suite):
        scaffold = make_scaffold(single_call_agent)
        acct = BudgetAccountant("t", BudgetCaps(calls=10))
        with pytest.raises(ValueError, match="requires a verifier"):
            scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=None, seed=0)

    def test_hidden_oracle_queried_at_most_once(self, suite, verifier):
        scaffold = make_scaffold(blind_best_of_n_agent)
        acct = BudgetAccountant("t", BudgetCaps(calls=20))
        result = scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=1)
        assert result.verification is not None  # exactly one, at the very end


class TestAblationInterception:
    def test_disabled_execution_feedback_withholds_error_text_from_the_agent(
        self, suite, verifier
    ):
        scaffold = make_scaffold(
            feedback_repair_agent, disabled_ops=frozenset({"execution_feedback"})
        )
        acct = BudgetAccountant("t", BudgetCaps(calls=20))
        result = scaffold.solve(suite["toy/gcd"], make_model(), acct, verifier=verifier, seed=3)
        assert "execution_feedback" not in result.trace.op_counts()
        assert not result.trace.used_expanding

    def test_ablation_does_not_affect_pass_fail_visibility(self, suite, verifier):
        """Selection must still work under ablation -- only the error DETAIL
        is withheld, not the boolean pass/fail a selector needs."""
        scaffold = make_scaffold(
            blind_best_of_n_agent, disabled_ops=frozenset({"execution_feedback"})
        )
        acct = BudgetAccountant("t", BudgetCaps(calls=20))
        result = scaffold.solve(suite["toy/sum_list"], make_model(), acct, verifier=verifier, seed=0)
        assert result.verification is not None

    def test_session_bookkeeping_records_the_real_result_even_when_withheld(self):
        """The session's own records (for after-the-fact analysis) must reflect
        what actually happened, even though the agent itself was denied it."""
        from cbs.models.base import CompletionRequest

        model = make_model()
        verifier = Verifier(select_backend("auto"))
        session = InterceptionSession(model, verifier, disabled_ops=frozenset({"execution_feedback"}))
        task = toy_suite().by_id()["toy/gcd"]
        acct = BudgetAccountant("t", BudgetCaps(calls=5))

        completion = session.model_client.complete(
            CompletionRequest(prompt=task.prompt, temperature=0.8, seed=0, meta={"task_id": task.task_id}),
            acct,
        )
        from cbs.tasks.verifier import extract_code

        result_seen_by_agent = session.verifier.verify_code(task, extract_code(completion.text))
        if not result_seen_by_agent.passed:
            assert result_seen_by_agent.exec_result is None
            assert result_seen_by_agent.reason == "ablated: error detail withheld"
            # But the session's internal event still has the real error text.
            assert session._verifier_events[-1].error_text != ""


class TestInterceptionSessionDirectly:
    def test_finalize_preserves_call_order(self, suite, verifier):
        from cbs.models.base import CompletionRequest
        from cbs.scaffolds.tagging import OperationTrace

        model = make_model()
        session = InterceptionSession(model, verifier)
        task = suite["toy/gcd"]
        acct = BudgetAccountant("t", BudgetCaps(calls=10))

        for i in range(3):
            session.model_client.complete(
                CompletionRequest(prompt=task.prompt, seed=i, meta={"task_id": task.task_id}),
                acct,
            )
        trace = OperationTrace()
        session.finalize(trace)
        assert [r.name for r in trace.records] == ["single_call"] * 3
        assert [r.seq for r in trace.records] == [0, 1, 2]

    def test_total_usage_accumulates_across_calls(self, suite):
        from cbs.models.base import CompletionRequest

        model = make_model()
        verifier_obj = Verifier(select_backend("auto"))
        session = InterceptionSession(model, verifier_obj)
        task = suite["toy/gcd"]
        acct = BudgetAccountant("t", BudgetCaps(calls=10))
        for i in range(3):
            session.model_client.complete(
                CompletionRequest(prompt=task.prompt, seed=i, meta={"task_id": task.task_id}),
                acct,
            )
        assert session.total_usage.calls == 3
        assert session.total_usage.calls == acct.spent.calls
