"""Tests for the validation harness itself.

The validation harness is what certifies the estimators, so its pass/fail logic
needs testing too -- a gate that cannot fail is not a gate.
"""

from __future__ import annotations

import pytest

from cbs.frontier.validate import (
    CoverageValidation,
    PipelineValidation,
    TaskValidation,
    binomial_tail_ge,
    validate_ci_coverage,
    validate_verifier_agreement,
)


def _task(recovered: bool) -> TaskValidation:
    return TaskValidation(
        task_id="t",
        true_p=0.5,
        n_samples=100,
        n_correct=50,
        p_hat=0.5,
        ci_low=0.4,
        ci_high=0.6,
        p_recovered=recovered,
        true_species=1,
        observed_species=1,
        chao1_estimate=1.0,
        chao1_low=1.0,
        chao1_high=1.0,
        species_recovered=True,
        beyond_frontier=False,
    )


class TestBinomialTail:
    def test_known_values(self):
        assert binomial_tail_ge(0, 6, 0.05) == 1.0
        assert binomial_tail_ge(7, 6, 0.05) == 0.0
        assert binomial_tail_ge(1, 6, 0.05) == pytest.approx(1 - 0.95**6, abs=1e-9)

    def test_decreasing_in_k(self):
        values = [binomial_tail_ge(k, 20, 0.05) for k in range(21)]
        assert all(a >= b for a, b in zip(values, values[1:]))


class TestPipelineCriterion:
    def test_all_recovered_passes(self):
        v = PipelineValidation(results=[_task(True)] * 6)
        assert v.ok and v.n_missed == 0

    def test_one_miss_in_six_passes(self):
        """A 95% interval is supposed to miss occasionally.

        Requiring 6/6 would fail ~26% of the time on a healthy pipeline.
        """
        v = PipelineValidation(results=[_task(True)] * 5 + [_task(False)])
        assert v.ok
        assert v.miss_p_value == pytest.approx(1 - 0.95**6, abs=1e-9)

    def test_many_misses_fail(self):
        """A gross plumbing fault produces simultaneous misses and must fail."""
        v = PipelineValidation(results=[_task(False)] * 4 + [_task(True)] * 2)
        assert not v.ok

    def test_all_missed_fails(self):
        v = PipelineValidation(results=[_task(False)] * 6)
        assert not v.ok
        assert v.miss_p_value < 1e-6

    def test_report_states_the_test(self):
        v = PipelineValidation(results=[_task(True)] * 6, n_samples=300)
        assert "correctly calibrated" in v.report()


class TestCoverageValidation:
    def test_conservative_coverage_passes(self):
        v = CoverageValidation(
            n_replicates=100,
            n_samples=100,
            cases=[{"true_p": 0.5, "coverage": 0.99, "nominal": 0.95,
                    "mc_tolerance": 0.03, "mean_width": 0.1}],
        )
        assert v.ok

    def test_under_coverage_fails(self):
        v = CoverageValidation(
            n_replicates=100,
            n_samples=100,
            cases=[{"true_p": 0.5, "coverage": 0.70, "nominal": 0.95,
                    "mc_tolerance": 0.03, "mean_width": 0.1}],
        )
        assert not v.ok

    @pytest.mark.slow
    def test_real_coverage_meets_nominal(self):
        result = validate_ci_coverage(
            true_ps=(0.0, 0.05, 0.5), n_samples=200, n_replicates=200
        )
        assert result.ok
        for case in result.cases:
            assert case["coverage"] >= 0.90


class TestVerifierAgreement:
    def test_no_false_positives_or_negatives(self):
        """The verifier must agree exactly with the mock's known labels.

        A false positive would inflate p_hat, shrink the beyond-frontier set,
        and under S_evo could manufacture an apparent crossing outright.
        """
        result = validate_verifier_agreement(n_per_task=25)
        assert result["false_positives"] == 0, result["disagreements"]
        assert result["false_negatives"] == 0, result["disagreements"]
        assert result["ok"]
        assert result["n_checked"] > 0
