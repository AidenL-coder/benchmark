"""Estimator tests against published reference values.

Brief section 11 singles out the Good-Turing/Chao estimators for unit testing.
Reference intervals below are the standard published Clopper-Pearson values;
they are hard-coded rather than computed so a regression in our own beta
implementation cannot silently move the expected answer too.
"""

from __future__ import annotations

import math

import pytest

from cbs.frontier.estimators import (
    chao1,
    clopper_pearson,
    counts_of_counts,
    good_turing_unseen_mass,
    pass_at_k,
    regularized_incomplete_beta,
    rule_of_three,
    sample_coverage,
    solution_rarefaction,
    success_probability_curve,
    zero_success_upper_bound,
)

TOL = 5e-5


class TestIncompleteBeta:
    @pytest.mark.parametrize(
        "a,b,x,expected",
        [(2, 3, 0.5, 0.6875), (1, 1, 0.5, 0.5), (2, 2, 0.3, 0.216), (5, 5, 0.5, 0.5)],
    )
    def test_known_values(self, a, b, x, expected):
        assert regularized_incomplete_beta(a, b, x) == pytest.approx(expected, abs=TOL)

    def test_boundaries(self):
        assert regularized_incomplete_beta(2, 3, 0.0) == 0.0
        assert regularized_incomplete_beta(2, 3, 1.0) == 1.0

    def test_monotone(self):
        values = [regularized_incomplete_beta(3, 4, x / 50) for x in range(51)]
        assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))


class TestClopperPearson:
    @pytest.mark.parametrize(
        "k,n,low,high",
        [
            (2, 10, 0.0252124, 0.5560955),
            (0, 10, 0.0, 0.3084971),
            (5, 10, 0.1870860, 0.8129140),
            (10, 10, 0.6915029, 1.0),
            (1, 100, 0.0002532, 0.0544572),
        ],
    )
    def test_reference_intervals(self, k, n, low, high):
        ci = clopper_pearson(k, n)
        assert ci.low == pytest.approx(low, abs=TOL)
        assert ci.high == pytest.approx(high, abs=TOL)

    def test_zero_successes_has_positive_upper_bound(self):
        """A beyond-frontier task must never be recorded as p == 0 exactly."""
        ci = clopper_pearson(0, 10_000)
        assert ci.low == 0.0
        assert ci.high > 0.0

    def test_contains_point_estimate(self):
        ci = clopper_pearson(37, 200)
        assert ci.contains(ci.point)

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            clopper_pearson(11, 10)

    def test_zero_trials_is_maximally_uninformative(self):
        ci = clopper_pearson(0, 0)
        assert (ci.low, ci.high) == (0.0, 1.0)
        assert math.isnan(ci.point)


class TestZeroSuccessBounds:
    def test_rule_of_three(self):
        assert rule_of_three(1000) == pytest.approx(0.003)
        assert rule_of_three(0) == 1.0

    def test_exact_bound_agrees_with_clopper_pearson(self):
        """The one-sided exact bound is the 90%-two-sided CP upper limit."""
        for n in (10, 100, 1000):
            assert zero_success_upper_bound(n, 0.95) == pytest.approx(
                clopper_pearson(0, n, 0.90).high, abs=1e-9
            )

    def test_rule_of_three_approximates_exact(self):
        """`3/n` overstates the exact bound slightly, converging as n grows.

        The exact bound is `1 - 0.05**(1/n) ~ 2.9957/n`, so the relative gap is
        about 1.7% at n=100 and shrinks from there. Both are reported on every
        record precisely because they are not identical.
        """
        for n in (100, 1000, 10000):
            assert rule_of_three(n) == pytest.approx(
                zero_success_upper_bound(n), rel=0.02
            )
            assert rule_of_three(n) >= zero_success_upper_bound(n)


class TestPassAtK:
    def test_known_values(self):
        assert pass_at_k(10, 5, 1) == pytest.approx(0.5)
        assert pass_at_k(10, 5, 10) == pytest.approx(1.0)
        assert pass_at_k(10, 2, 2) == pytest.approx(1 - 28 / 45)

    def test_zero_successes_is_zero_at_any_k(self):
        assert pass_at_k(100, 0, 1) == 0.0
        assert pass_at_k(100, 0, 100) == 0.0

    def test_monotone_in_k(self):
        values = [pass_at_k(100, 3, k) for k in range(1, 101)]
        assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))

    def test_pass_at_1_equals_empirical_rate(self):
        assert pass_at_k(1000, 137, 1) == pytest.approx(0.137)

    def test_large_budget_is_numerically_stable(self):
        """n ~ 10^4 is the study's target budget; must not overflow."""
        value = pass_at_k(10_000, 1, 10_000)
        assert value == pytest.approx(1.0)

    @pytest.mark.parametrize("n,c,k", [(10, 5, 0), (10, 5, 11), (10, 11, 1)])
    def test_rejects_invalid(self, n, c, k):
        with pytest.raises(ValueError):
            pass_at_k(n, c, k)


class TestGoodTuring:
    def test_counts_of_counts(self):
        assert counts_of_counts({"a": 1, "b": 1, "c": 2}) == {1: 2, 2: 1}

    def test_unseen_mass(self):
        counts = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 5}
        assert good_turing_unseen_mass(counts) == pytest.approx(0.3)
        assert sample_coverage(counts) == pytest.approx(0.7)

    def test_empty_sample_is_none_not_zero(self):
        """The critical distinction: no data is not the same as full coverage.

        Returning 0.0 here would read as "the solution distribution is fully
        covered" for exactly the zero-success tasks where nothing was observed.
        """
        assert good_turing_unseen_mass({}) is None
        assert sample_coverage({}) is None

    def test_all_singletons_means_maximal_unseen_mass(self):
        assert good_turing_unseen_mass({f"s{i}": 1 for i in range(10)}) == 1.0

    def test_no_singletons_means_full_coverage(self):
        assert good_turing_unseen_mass({"a": 5, "b": 7}) == 0.0


class TestChao1:
    def test_bias_corrected_known_value(self):
        counts = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 5}
        est = chao1(counts, bias_corrected=True)
        assert est.f1 == 3 and est.f2 == 1
        assert est.estimate == pytest.approx(6.5)

    def test_classic_known_value(self):
        counts = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 5}
        est = chao1(counts, bias_corrected=False)
        assert est.estimate == pytest.approx(9.5)

    def test_no_doubletons_does_not_divide_by_zero(self):
        est = chao1({"a": 1, "b": 1, "c": 3})
        assert est is not None
        assert math.isfinite(est.estimate)
        assert est.method == "chao1-bias-corrected"

    def test_empty_sample_is_none(self):
        assert chao1({}) is None

    def test_estimate_never_below_observed(self):
        for counts in ({"a": 5}, {"a": 1, "b": 1}, {"a": 3, "b": 3, "c": 1}):
            est = chao1(counts)
            assert est.estimate >= est.observed - 1e-9
            assert est.low >= est.observed - 1e-9

    def test_saturated_sample_estimates_no_unseen_species(self):
        """No singletons means no evidence of unseen species."""
        est = chao1({"a": 10, "b": 10, "c": 10})
        assert est.estimated_unseen == pytest.approx(0.0)


class TestCurves:
    def test_rarefaction_endpoints_and_monotonicity(self):
        counts = {"a": 1, "b": 1, "c": 1, "d": 2, "e": 5}
        curve = solution_rarefaction(counts, [1, 2, 5, 10, 20])
        assert curve[1] == pytest.approx(1.0)
        assert curve[10] == pytest.approx(5.0)
        assert curve[20] == pytest.approx(5.0)
        keys = [1, 2, 5, 10]
        assert all(
            curve[a] <= curve[b] + 1e-9 for a, b in zip(keys, keys[1:])
        )

    def test_success_probability_curve_matches_pass_at_k(self):
        curve = success_probability_curve(1000, 1, [1, 10, 100, 1000])
        assert curve[1] == pytest.approx(0.001)
        assert curve[1000] == pytest.approx(1.0)

    def test_curve_drops_budgets_beyond_n(self):
        assert 2000 not in success_probability_curve(1000, 5, [500, 2000])
