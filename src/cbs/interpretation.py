"""Interpretation-matrix placement (brief section 7 / 9.3).

The matrix maps three summary quantities to one of four conclusions:

| Crossing rate | `S_evo` vs `S_star` @matched | Transfer | Conclusion |
|---|---|---|---|
| ~0 | ~ | -- | Bounded search (deflationary) |
| ~0 | `S_evo` > | high | Superhuman elicitation, still within frontier |
| > 0 | `S_evo` > | high; crossings need expanding ops | Genuine expansion |
| > 0 | ~ | vanish on transfer / large overfit gap | Illusory expansion |

Placing a result requires three judgment calls the brief deliberately leaves
as thresholds to fix (`docs/DECISIONS.md` D-14): what counts as "~0" crossing
rate, what counts as `S_evo` being meaningfully ">" `S_star`, and what counts
as "high" transfer retention. `place_in_interpretation_matrix` takes them as
explicit parameters rather than hard-coding a guess, so the placement is
mechanical once those are pre-registered, and is never an argument to have
after seeing results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["InterpretationRow", "InterpretationPlacement", "place_in_interpretation_matrix"]


class InterpretationRow(str, Enum):
    BOUNDED_SEARCH = "bounded_search"
    SUPERHUMAN_ELICITATION = "superhuman_elicitation"
    GENUINE_EXPANSION = "genuine_expansion"
    ILLUSORY_EXPANSION = "illusory_expansion"


_DESCRIPTIONS = {
    InterpretationRow.BOUNDED_SEARCH: (
        "Bounded search. Self-improvement rediscovers known generic "
        "scaffolding; no expansion. (Deflationary.)"
    ),
    InterpretationRow.SUPERHUMAN_ELICITATION: (
        "Evolution finds better elicitation than experts, but stays within "
        "the frontier. ('Self-improvement = superhuman scaffold engineering, "
        "not new capability.')"
    ),
    InterpretationRow.GENUINE_EXPANSION: (
        "Genuine expansion via the identified mechanism(s). (Constructive.)"
    ),
    InterpretationRow.ILLUSORY_EXPANSION: (
        "Illusory expansion. Crossings are eval-specific overfitting, not "
        "capability. (Deflationary + a cautionary methodology contribution.)"
    ),
}


@dataclass
class InterpretationPlacement:
    row: InterpretationRow
    description: str
    #: The raw inputs the placement was computed from, for auditability.
    crossing_rate: float
    s_evo_beats_s_star: bool
    transfer_retention: float | None
    overfitting_gap: float | None
    thresholds: dict

    def as_dict(self) -> dict:
        return {
            "row": self.row.value,
            "description": self.description,
            "crossing_rate": self.crossing_rate,
            "s_evo_beats_s_star": self.s_evo_beats_s_star,
            "transfer_retention": self.transfer_retention,
            "overfitting_gap": self.overfitting_gap,
            "thresholds": self.thresholds,
        }


def place_in_interpretation_matrix(
    crossing_rate: float,
    s_evo_ci_low: float,
    s_star_ci_high: float,
    transfer_retention: float | None,
    overfitting_gap: float | None,
    *,
    crossing_rate_epsilon: float = 0.0,
    transfer_retention_high: float = 0.5,
    overfitting_gap_high: float = 0.3,
) -> InterpretationPlacement:
    """Place one result in the interpretation matrix.

    `s_evo_ci_low` / `s_star_ci_high`: `S_evo` counts as beating `S_star`
    ("`S_evo` >") only when their confidence intervals do not overlap
    (`s_evo_ci_low > s_star_ci_high`) -- a raw point-estimate difference would
    call noise a result. Passing a CI's own point estimate for both bounds
    (i.e. `s_evo_ci_low = s_evo_point`, `s_star_ci_high = s_star_point`)
    degrades this to a simple point comparison, if that is deliberately wanted.

    `crossing_rate_epsilon`: crossing rates at or below this count as "~0".
    Defaults to `0.0` (exactly zero); brief section 6 implies this should
    account for multiple-comparison correction (`cbs.stats.benjamini_hochberg`)
    having already been applied to the underlying per-task crossing claims
    before this function sees the rate, so that a rate here is a rate of
    *significant, corrected* crossings, not raw ones.

    `transfer_retention_high` / `overfitting_gap_high`: thresholds for "high"
    retention and "large" overfitting gap respectively. Both are placeholders
    for the pre-registered values `docs/DECISIONS.md` D-14 still needs to fix;
    passed explicitly, never hard-coded, so a caller cannot use this function
    without consciously choosing them.
    """
    thresholds = {
        "crossing_rate_epsilon": crossing_rate_epsilon,
        "transfer_retention_high": transfer_retention_high,
        "overfitting_gap_high": overfitting_gap_high,
    }
    s_evo_beats_s_star = s_evo_ci_low > s_star_ci_high
    near_zero_crossing = crossing_rate <= crossing_rate_epsilon

    if near_zero_crossing:
        row = (
            InterpretationRow.SUPERHUMAN_ELICITATION
            if s_evo_beats_s_star
            else InterpretationRow.BOUNDED_SEARCH
        )
    else:
        transfer_is_high = (
            transfer_retention is not None and transfer_retention >= transfer_retention_high
        )
        overfitting_is_large = (
            overfitting_gap is not None and overfitting_gap >= overfitting_gap_high
        )
        if s_evo_beats_s_star and transfer_is_high and not overfitting_is_large:
            row = InterpretationRow.GENUINE_EXPANSION
        else:
            row = InterpretationRow.ILLUSORY_EXPANSION

    return InterpretationPlacement(
        row=row,
        description=_DESCRIPTIONS[row],
        crossing_rate=crossing_rate,
        s_evo_beats_s_star=s_evo_beats_s_star,
        transfer_retention=transfer_retention,
        overfitting_gap=overfitting_gap,
        thresholds=thresholds,
    )
