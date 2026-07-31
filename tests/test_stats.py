"""Tests for cbs.stats: bootstrap CIs and Benjamini-Hochberg correction.

The BH test cases are hand-computed (see comments) rather than checked against
another implementation, matching this project's convention (D-07) of
validating estimators against worked reference values.
"""

from __future__ import annotations

import pytest

from cbs.stats import benjamini_hochberg, bootstrap_ci


class TestBootstrapCI:
    def test_point_estimate_is_the_statistic_on_original_data(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_ci(values, n_resamples=500, seed=0)
        assert ci.point == pytest.approx(3.0)

    def test_ci_brackets_the_point_estimate(self):
        values = [1.0, 2.0, 2.5, 3.0, 10.0, 1.5, 2.2]
        ci = bootstrap_ci(values, n_resamples=2000, seed=1)
        assert ci.low <= ci.point <= ci.high

    def test_constant_values_give_a_degenerate_interval(self):
        ci = bootstrap_ci([5.0, 5.0, 5.0, 5.0], n_resamples=500, seed=0)
        assert ci.low == pytest.approx(5.0)
        assert ci.high == pytest.approx(5.0)

    def test_wider_confidence_gives_a_wider_interval(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        narrow = bootstrap_ci(values, n_resamples=3000, confidence=0.80, seed=0)
        wide = bootstrap_ci(values, n_resamples=3000, confidence=0.99, seed=0)
        assert (wide.high - wide.low) >= (narrow.high - narrow.low)

    def test_deterministic_given_a_seed(self):
        values = [1.0, 7.0, 3.0, 9.0, 2.0]
        a = bootstrap_ci(values, n_resamples=500, seed=42)
        b = bootstrap_ci(values, n_resamples=500, seed=42)
        assert (a.low, a.point, a.high) == (b.low, b.point, b.high)

    def test_empty_sample_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_custom_statistic(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        ci = bootstrap_ci(values, statistic=max, n_resamples=1000, seed=0)
        assert ci.point == 100.0
        assert ci.high == 100.0  # max of a resample can never exceed the true max

    def test_invalid_confidence_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, 2.0], confidence=1.5)


class TestBenjaminiHochberg:
    """Reference case, hand-computed:

    p = [0.01, 0.04, 0.03, 0.005, 0.20, 0.15, 0.09, 0.001, 0.30, 0.02], alpha=0.05
    Sorted ascending: 0.001 0.005 0.01 0.02 0.03 0.04 0.09 0.15 0.20 0.30
    BH thresholds (rank/10)*0.05: .005 .01 .015 .02 .025 .03 .035 .04 .045 .05
    p_(k) <= threshold_(k)?        T    T    T   T    F    F    F    F    F    F
    Largest satisfying rank = 4 -> reject sorted ranks 1-4:
    {0.001, 0.005, 0.01, 0.02} -> original indices {7, 3, 0, 9}.
    """

    P = [0.01, 0.04, 0.03, 0.005, 0.20, 0.15, 0.09, 0.001, 0.30, 0.02]
    REJECTED_INDICES = {0, 3, 7, 9}

    def test_matches_hand_computed_rejections(self):
        result = benjamini_hochberg(self.P, alpha=0.05)
        rejected_indices = {i for i, r in enumerate(result.rejected) if r}
        assert rejected_indices == self.REJECTED_INDICES
        assert result.n_rejected == 4

    def test_adjusted_p_values_match_hand_computation(self):
        # Hand-computed q-values (see module test docstring derivation);
        # original-index -> expected q.
        expected = {
            7: 0.01,
            3: 0.025,
            0: 0.0333,
            9: 0.05,
            2: 0.06,
            1: 0.0667,
            6: 0.1286,
            5: 0.1875,
            4: 0.2222,
            8: 0.30,
        }
        result = benjamini_hochberg(self.P, alpha=0.05)
        for idx, expected_q in expected.items():
            assert result.adjusted_p[idx] == pytest.approx(expected_q, abs=2e-4)

    def test_adjusted_p_values_are_monotone_nondecreasing_in_sorted_order(self):
        result = benjamini_hochberg(self.P, alpha=0.05)
        order = sorted(range(len(self.P)), key=lambda i: self.P[i])
        sorted_q = [result.adjusted_p[i] for i in order]
        assert all(a <= b + 1e-9 for a, b in zip(sorted_q, sorted_q[1:]))

    def test_all_significant_p_values_rejects_everything(self):
        result = benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.05)
        assert all(result.rejected)

    def test_all_insignificant_p_values_rejects_nothing(self):
        result = benjamini_hochberg([0.9, 0.8, 0.99], alpha=0.05)
        assert not any(result.rejected)
        assert result.n_rejected == 0

    def test_empty_input(self):
        result = benjamini_hochberg([], alpha=0.05)
        assert result.rejected == []
        assert result.n_rejected == 0

    def test_single_p_value_below_alpha_is_rejected(self):
        assert benjamini_hochberg([0.01], alpha=0.05).rejected == [True]

    def test_single_p_value_above_alpha_is_not_rejected(self):
        assert benjamini_hochberg([0.5], alpha=0.05).rejected == [False]

    def test_more_comparisons_makes_the_same_p_value_harder_to_survive(self):
        """The multiple-comparison correction this exists for: p=0.03 alone
        survives at alpha=0.05, but padded out with many null results (large
        p-values) among many comparisons, it may not."""
        alone = benjamini_hochberg([0.03], alpha=0.05)
        padded = benjamini_hochberg([0.03] + [0.9] * 19, alpha=0.05)
        assert alone.rejected[0] is True
        assert padded.rejected[0] is False

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValueError):
            benjamini_hochberg([0.1, 0.2], alpha=1.5)
