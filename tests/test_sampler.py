"""Sampler, resume, splits and record tests.

Resume correctness matters as much as the statistics here: on a preemptible
runtime the frontier run *will* be interrupted, and a resume that pooled samples
drawn under different conditions, or double-counted a partial write, would
corrupt the estimate rather than fail loudly.
"""

from __future__ import annotations

import json

import pytest

from cbs.budget import BudgetAccountant, BudgetCaps
from cbs.frontier.records import build_record
from cbs.frontier.sampler import DEFAULT_SCHEDULE, FrontierSampler, TemperatureSchedule
from cbs.models.mock import MockModelClient, MockTaskBehaviour
from cbs.sandbox import select_backend
from cbs.tasks import Verifier
from cbs.tasks.families.toy import toy_behaviours, toy_suite
from cbs.tasks.schema import Split
from cbs.tasks.splits import SplitRatios, assign_splits, verify_manifest, write_manifest


@pytest.fixture(scope="module")
def verifier() -> Verifier:
    return Verifier(select_backend("auto"))


def make_sampler(tmp_path, seed=0, schedule=DEFAULT_SCHEDULE, verifier=None):
    model = MockModelClient(
        behaviours=toy_behaviours(), seed=seed, model_id="mock-test"
    )
    return FrontierSampler(
        model=model,
        verifier=verifier,
        schedule=schedule,
        output_dir=tmp_path,
        flush_every=1,
    )


class TestTemperatureSchedule:
    def test_allocation_length_is_exact(self):
        for n in (1, 7, 10, 33, 100, 1000):
            assert len(DEFAULT_SCHEDULE.allocate(n)) == n

    def test_allocation_is_deterministic(self):
        assert DEFAULT_SCHEDULE.allocate(50) == DEFAULT_SCHEDULE.allocate(50)

    def test_allocation_is_prefix_stable(self):
        """Widening the budget must not change earlier samples' temperatures.

        Regression test. A block allocation made sample i's temperature depend on
        N_max, so resuming a shard at a larger budget retroactively rewrote the
        schedule of samples already drawn.
        """
        full = DEFAULT_SCHEDULE.allocate(500)
        for m in (1, 2, 7, 15, 40, 99, 250):
            assert DEFAULT_SCHEDULE.allocate(m) == full[:m]

    def test_truncated_prefix_still_spans_the_schedule(self):
        """A preempted run must not end up sampling one temperature only."""
        prefix = DEFAULT_SCHEDULE.allocate(20)
        assert len(set(prefix)) >= 4

    def test_allocation_respects_fractions(self):
        schedule = TemperatureSchedule(((0.0, 0.5), (1.0, 0.5)))
        allocated = schedule.allocate(100)
        assert allocated.count(0.0) == 50
        assert allocated.count(1.0) == 50

    def test_proportions_converge_to_requested_fractions(self):
        allocated = DEFAULT_SCHEDULE.allocate(10_000)
        for temperature, fraction in DEFAULT_SCHEDULE.stages:
            assert allocated.count(temperature) / 10_000 == pytest.approx(
                fraction, abs=0.01
            )

    def test_temperature_at_matches_allocate(self):
        full = DEFAULT_SCHEDULE.allocate(30)
        assert [DEFAULT_SCHEDULE.temperature_at(i) for i in range(30)] == full

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError):
            TemperatureSchedule(((0.0, 0.3), (1.0, 0.3)))

    def test_empty_schedule_rejected(self):
        with pytest.raises(ValueError):
            TemperatureSchedule(())

    def test_zero_allocation(self):
        assert DEFAULT_SCHEDULE.allocate(0) == []


class TestSampling:
    def test_produces_record_with_expected_sample_count(self, tmp_path, verifier):
        sampler = make_sampler(tmp_path, verifier=verifier)
        task = toy_suite().by_id()["toy/sum_list"]
        record = sampler.estimate_task(task, 30, BudgetAccountant("t"), resume=False)
        assert record.n_samples == 30
        assert record.n_correct > 0
        assert record.model_id == "mock-test"

    def test_zero_probability_task_is_beyond_frontier(self, tmp_path, verifier):
        sampler = make_sampler(tmp_path, verifier=verifier)
        task = toy_suite().by_id()["toy/impossible_parity"]
        record = sampler.estimate_task(task, 25, BudgetAccountant("t"), resume=False)
        assert record.n_correct == 0
        assert record.beyond_frontier
        # Never claims impossibility -- only a positive bound at this budget.
        assert record.p_upper_bound > 0
        assert "not reached at compute" in record.as_dict()["beyond_frontier_qualifier"]

    def test_beyond_frontier_task_has_no_diversity_estimates(self, tmp_path, verifier):
        sampler = make_sampler(tmp_path, verifier=verifier)
        task = toy_suite().by_id()["toy/impossible_parity"]
        record = sampler.estimate_task(task, 15, BudgetAccountant("t"), resume=False)
        assert record.good_turing_unseen_mass is None
        assert record.chao1 is None

    def test_writes_samples_and_solutions_shards(self, tmp_path, verifier):
        sampler = make_sampler(tmp_path, verifier=verifier)
        task = toy_suite().by_id()["toy/sum_list"]
        sampler.estimate_task(task, 20, BudgetAccountant("t"), resume=False)
        paths = sampler.shard_paths(task)
        assert paths.samples.exists() and paths.meta.exists()
        rows = [json.loads(l) for l in paths.samples.read_text().splitlines() if l.strip()]
        assert len(rows) == 20
        solutions = [
            json.loads(l) for l in paths.solutions.read_text().splitlines() if l.strip()
        ]
        assert len(solutions) == len({r["species_key"] for r in rows if r["passed"]})


class TestResume:
    def test_resume_tops_up_to_n_max(self, tmp_path, verifier):
        task = toy_suite().by_id()["toy/sum_list"]
        first = make_sampler(tmp_path, verifier=verifier)
        r1 = first.estimate_task(task, 20, BudgetAccountant("a"), resume=False)
        assert r1.n_samples == 20

        second = make_sampler(tmp_path, verifier=verifier)
        r2 = second.estimate_task(task, 50, BudgetAccountant("b"), resume=True)
        assert r2.n_samples == 50

    def test_resume_does_not_duplicate_completed_samples(self, tmp_path, verifier):
        task = toy_suite().by_id()["toy/sum_list"]
        sampler = make_sampler(tmp_path, verifier=verifier)
        sampler.estimate_task(task, 20, BudgetAccountant("a"), resume=False)
        again = make_sampler(tmp_path, verifier=verifier)
        record = again.estimate_task(task, 20, BudgetAccountant("b"), resume=True)
        assert record.n_samples == 20

    def test_resume_preserves_species_counts(self, tmp_path, verifier):
        task = toy_suite().by_id()["toy/sum_list"]
        whole = make_sampler(tmp_path / "whole", verifier=verifier)
        expected = whole.estimate_task(task, 40, BudgetAccountant("w"), resume=False)

        part = make_sampler(tmp_path / "split", verifier=verifier)
        part.estimate_task(task, 15, BudgetAccountant("p1"), resume=False)
        rest = make_sampler(tmp_path / "split", verifier=verifier)
        resumed = rest.estimate_task(task, 40, BudgetAccountant("p2"), resume=True)

        # Same seed and same per-index temperature => identical draws either way.
        assert resumed.n_correct == expected.n_correct
        assert resumed.species_counts == expected.species_counts

    def test_changed_conditions_refuse_to_pool(self, tmp_path, verifier):
        """Samples drawn under a different schedule must not join one estimate."""
        task = toy_suite().by_id()["toy/sum_list"]
        first = make_sampler(tmp_path, verifier=verifier)
        first.estimate_task(task, 10, BudgetAccountant("a"), resume=False)

        other_schedule = TemperatureSchedule(((0.9, 1.0),))
        second = make_sampler(tmp_path, schedule=other_schedule, verifier=verifier)
        with pytest.raises(RuntimeError, match="different sampling conditions"):
            second.estimate_task(task, 20, BudgetAccountant("b"), resume=True)

    def test_truncated_final_line_is_tolerated(self, tmp_path, verifier):
        """A process killed mid-write leaves a partial line; it must not crash."""
        task = toy_suite().by_id()["toy/sum_list"]
        sampler = make_sampler(tmp_path, verifier=verifier)
        sampler.estimate_task(task, 10, BudgetAccountant("a"), resume=False)

        paths = sampler.shard_paths(task)
        with paths.samples.open("a", encoding="utf-8") as fh:
            fh.write('{"i": 10, "t": 0.8, "pas')  # torn write

        resumed = make_sampler(tmp_path, verifier=verifier)
        record = resumed.estimate_task(task, 20, BudgetAccountant("b"), resume=True)
        assert record.n_samples >= 20


class TestBudgetInteraction:
    def test_budget_cap_truncates_without_raising(self, tmp_path, verifier):
        task = toy_suite().by_id()["toy/sum_list"]
        sampler = make_sampler(tmp_path, verifier=verifier)
        accountant = BudgetAccountant("tight", BudgetCaps(calls=7))
        record = sampler.estimate_task(task, 100, accountant, resume=False)
        assert record.n_samples == 7
        assert record.n_budget_exhausted > 0
        assert record.metadata["truncated"] is True

    def test_truncated_record_reports_its_actual_budget(self, tmp_path, verifier):
        task = toy_suite().by_id()["toy/sum_list"]
        sampler = make_sampler(tmp_path, verifier=verifier)
        record = sampler.estimate_task(
            task, 100, BudgetAccountant("t", BudgetCaps(calls=5)), resume=False
        )
        # The bound must reflect samples actually drawn, not the requested N_max.
        assert record.rule_of_three_bound == pytest.approx(3 / 5)


class TestRecords:
    def test_rejects_more_correct_than_samples(self):
        with pytest.raises(ValueError, match="exceeds"):
            build_record(
                task_id="x",
                model_id="m",
                split=None,
                family="f",
                n_samples=5,
                species_counts={"a": 10},
                temperature_schedule=[],
                scaffold_fingerprint={},
                usage=__import__("cbs.budget", fromlist=["Usage"]).Usage(),
                verifier_backend="subprocess",
                verifier_is_security_boundary=False,
            )

    def test_record_round_trips_to_json(self, tmp_path, verifier):
        sampler = make_sampler(tmp_path, verifier=verifier)
        task = toy_suite().by_id()["toy/gcd"]
        record = sampler.estimate_task(task, 12, BudgetAccountant("t"), resume=False)
        payload = json.loads(record.to_json())
        assert payload["task_id"] == "toy/gcd"
        assert payload["n_samples"] == 12
        assert "beyond_frontier_qualifier" in payload


class TestSplits:
    def test_assignment_is_deterministic(self):
        suite = toy_suite()
        ratios = SplitRatios(train=0.5, held_out=0.5)
        a = assign_splits(suite, ratios, salt="s")
        b = assign_splits(suite, ratios, salt="s")
        assert [t.split for t in a.tasks] == [t.split for t in b.tasks]

    def test_salt_changes_assignment(self):
        suite = toy_suite()
        ratios = SplitRatios(train=0.5, held_out=0.5)
        a = assign_splits(suite, ratios, salt="one")
        b = assign_splits(suite, ratios, salt="two")
        assert a.suite_hash() != b.suite_hash()

    def test_transfer_family_is_never_split_by_ratio(self):
        """A different distribution must go wholly to transfer, or RQ4 is void."""
        suite = toy_suite()
        assigned = assign_splits(
            suite, SplitRatios(train=1.0, held_out=0.0), transfer_families={"toy"}
        )
        assert all(t.split is Split.TRANSFER for t in assigned.tasks)

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError):
            SplitRatios(train=0.5, held_out=0.9)

    def test_manifest_freeze_and_verify(self, tmp_path):
        suite = assign_splits(toy_suite(), SplitRatios(train=0.5, held_out=0.5))
        path = tmp_path / "splits.json"
        write_manifest(suite, SplitRatios(train=0.5, held_out=0.5), path)
        assert verify_manifest(suite, path)["ok"]

    def test_manifest_detects_content_drift(self, tmp_path):
        from dataclasses import replace

        suite = assign_splits(toy_suite(), SplitRatios(train=0.5, held_out=0.5))
        path = tmp_path / "splits.json"
        write_manifest(suite, SplitRatios(train=0.5, held_out=0.5), path)

        mutated_tasks = list(suite.tasks)
        mutated_tasks[0] = replace(mutated_tasks[0], prompt="TAMPERED")
        mutated = type(suite)(name=suite.name, tasks=mutated_tasks)

        result = verify_manifest(mutated, path)
        assert not result["ok"]
        assert any("content changed" in p for p in result["problems"])

    def test_refuses_to_overwrite_a_different_frozen_split(self, tmp_path):
        from dataclasses import replace

        ratios = SplitRatios(train=0.5, held_out=0.5)
        suite = assign_splits(toy_suite(), ratios)
        path = tmp_path / "splits.json"
        write_manifest(suite, ratios, path)

        mutated_tasks = list(suite.tasks)
        mutated_tasks[0] = replace(mutated_tasks[0], prompt="DIFFERENT")
        mutated = type(suite)(name=suite.name, tasks=mutated_tasks)
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            write_manifest(mutated, ratios, path)


class TestMockModel:
    def test_is_deterministic_across_instances(self):
        b = {"t": MockTaskBehaviour(0.5, ["def f(): return 1\n"], ["def f(): return 2\n"])}
        from cbs.models.base import CompletionRequest

        m1 = MockModelClient(b, seed=7)
        m2 = MockModelClient(b, seed=7)
        req = CompletionRequest(prompt="p", temperature=0.8, seed=3, meta={"task_id": "t"})
        a1 = m1.complete(req, BudgetAccountant("x")).text
        a2 = m2.complete(req, BudgetAccountant("y")).text
        assert a1 == a2

    def test_unregistered_task_raises(self):
        from cbs.models.base import CompletionRequest

        model = MockModelClient({}, seed=0)
        with pytest.raises(KeyError):
            model.complete(
                CompletionRequest(prompt="p", meta={"task_id": "nope"}),
                BudgetAccountant("x"),
            )

    def test_charges_the_accountant(self):
        from cbs.models.base import CompletionRequest

        b = {"t": MockTaskBehaviour(1.0, ["def f(): return 1\n"])}
        model = MockModelClient(b, seed=0)
        acct = BudgetAccountant("x")
        model.complete(CompletionRequest(prompt="hello", meta={"task_id": "t"}), acct)
        assert acct.spent.calls == 1
        assert acct.spent.total_tokens > 0

    def test_behaviour_validation(self):
        with pytest.raises(ValueError):
            MockTaskBehaviour(p_correct=1.5, correct_variants=["x"])
        with pytest.raises(ValueError):
            MockTaskBehaviour(p_correct=0.5, correct_variants=[], incorrect_variants=["y"])
        with pytest.raises(ValueError):
            MockTaskBehaviour(p_correct=0.5, correct_variants=["x"], incorrect_variants=[])
