"""Tests for interpretation-matrix placement (brief section 7 / 9.3).

Each test targets exactly one of the four matrix rows via the inputs that row
is defined by, so a regression in the placement logic points at a specific
cell rather than "something about interpretation broke".
"""

from __future__ import annotations

from cbs.interpretation import InterpretationRow, place_in_interpretation_matrix


class TestFourRows:
    def test_bounded_search(self):
        """~0 crossing rate, S_evo not meaningfully better than S_star."""
        placement = place_in_interpretation_matrix(
            crossing_rate=0.0,
            s_evo_ci_low=0.40,
            s_star_ci_high=0.45,  # overlaps -- not a real "beats"
            transfer_retention=None,
            overfitting_gap=None,
        )
        assert placement.row is InterpretationRow.BOUNDED_SEARCH

    def test_superhuman_elicitation(self):
        """~0 crossing rate, but S_evo's CI clears S_star's CI entirely."""
        placement = place_in_interpretation_matrix(
            crossing_rate=0.0,
            s_evo_ci_low=0.70,
            s_star_ci_high=0.50,
            transfer_retention=0.9,
            overfitting_gap=0.05,
        )
        assert placement.row is InterpretationRow.SUPERHUMAN_ELICITATION

    def test_genuine_expansion(self):
        """Crossing rate > 0, S_evo beats S_star, transfer retained, low overfit gap."""
        placement = place_in_interpretation_matrix(
            crossing_rate=0.15,
            s_evo_ci_low=0.70,
            s_star_ci_high=0.50,
            transfer_retention=0.8,
            overfitting_gap=0.05,
        )
        assert placement.row is InterpretationRow.GENUINE_EXPANSION

    def test_illusory_expansion_via_vanishing_transfer(self):
        """Crossing rate > 0 but the gain does not survive on transfer tasks."""
        placement = place_in_interpretation_matrix(
            crossing_rate=0.15,
            s_evo_ci_low=0.70,
            s_star_ci_high=0.50,
            transfer_retention=0.05,  # essentially vanished
            overfitting_gap=0.05,
        )
        assert placement.row is InterpretationRow.ILLUSORY_EXPANSION

    def test_illusory_expansion_via_large_overfitting_gap(self):
        """Crossing rate > 0 but a large train-held_out gap indicates overfitting."""
        placement = place_in_interpretation_matrix(
            crossing_rate=0.15,
            s_evo_ci_low=0.70,
            s_star_ci_high=0.50,
            transfer_retention=0.8,
            overfitting_gap=0.5,  # large
        )
        assert placement.row is InterpretationRow.ILLUSORY_EXPANSION

    def test_illusory_expansion_when_s_evo_does_not_beat_s_star_despite_crossings(self):
        """Even with crossings, if S_evo isn't meaningfully ahead of S_star,
        the crossings are not evidence of a generally superior mechanism."""
        placement = place_in_interpretation_matrix(
            crossing_rate=0.1,
            s_evo_ci_low=0.40,
            s_star_ci_high=0.45,
            transfer_retention=0.9,
            overfitting_gap=0.05,
        )
        assert placement.row is InterpretationRow.ILLUSORY_EXPANSION


class TestThresholdsAreExplicitNotHardcoded:
    def test_custom_crossing_epsilon_changes_the_near_zero_boundary(self):
        placement = place_in_interpretation_matrix(
            crossing_rate=0.02,
            s_evo_ci_low=0.40,
            s_star_ci_high=0.45,
            transfer_retention=None,
            overfitting_gap=None,
            crossing_rate_epsilon=0.05,  # 0.02 now counts as "~0"
        )
        assert placement.row is InterpretationRow.BOUNDED_SEARCH

    def test_custom_transfer_threshold_changes_genuine_vs_illusory(self):
        common = dict(
            crossing_rate=0.1, s_evo_ci_low=0.70, s_star_ci_high=0.50, overfitting_gap=0.05
        )
        strict = place_in_interpretation_matrix(
            **common, transfer_retention=0.6, transfer_retention_high=0.9
        )
        lenient = place_in_interpretation_matrix(
            **common, transfer_retention=0.6, transfer_retention_high=0.3
        )
        assert strict.row is InterpretationRow.ILLUSORY_EXPANSION
        assert lenient.row is InterpretationRow.GENUINE_EXPANSION

    def test_thresholds_are_recorded_on_the_result(self):
        placement = place_in_interpretation_matrix(
            crossing_rate=0.0,
            s_evo_ci_low=0.4,
            s_star_ci_high=0.4,
            transfer_retention=None,
            overfitting_gap=None,
            crossing_rate_epsilon=0.02,
            transfer_retention_high=0.6,
            overfitting_gap_high=0.4,
        )
        assert placement.thresholds == {
            "crossing_rate_epsilon": 0.02,
            "transfer_retention_high": 0.6,
            "overfitting_gap_high": 0.4,
        }


class TestReporting:
    def test_every_row_has_a_nonempty_description(self):
        cases = {
            InterpretationRow.BOUNDED_SEARCH: dict(
                crossing_rate=0.0, s_evo_ci_low=0.4, s_star_ci_high=0.4,
                transfer_retention=None, overfitting_gap=None,
            ),
            InterpretationRow.SUPERHUMAN_ELICITATION: dict(
                crossing_rate=0.0, s_evo_ci_low=0.7, s_star_ci_high=0.4,
                transfer_retention=0.9, overfitting_gap=0.05,
            ),
            InterpretationRow.GENUINE_EXPANSION: dict(
                crossing_rate=0.1, s_evo_ci_low=0.7, s_star_ci_high=0.4,
                transfer_retention=0.9, overfitting_gap=0.05,
            ),
            InterpretationRow.ILLUSORY_EXPANSION: dict(
                crossing_rate=0.1, s_evo_ci_low=0.7, s_star_ci_high=0.4,
                transfer_retention=0.05, overfitting_gap=0.05,
            ),
        }
        for row, kwargs in cases.items():
            placement = place_in_interpretation_matrix(**kwargs)
            assert placement.row is row, f"expected {row}, got {placement.row}"
            assert placement.description.strip()

    def test_as_dict_round_trips(self):
        placement = place_in_interpretation_matrix(
            crossing_rate=0.1,
            s_evo_ci_low=0.7,
            s_star_ci_high=0.4,
            transfer_retention=0.9,
            overfitting_gap=0.05,
        )
        payload = placement.as_dict()
        assert payload["row"] == placement.row.value
        assert payload["crossing_rate"] == 0.1
