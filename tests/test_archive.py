"""Tests for archive persistence and the overfitting/transfer/hard-coding
analyses (brief section 9.1-9.2)."""

from __future__ import annotations

from cbs.archive import (
    ArchiveRecord,
    hard_coding_heuristic,
    load_archive,
    overfitting_gap,
    persist_archive,
    transfer_retention,
)
from cbs.scaffolds.evolved import ArchiveEntry


def make_record(train, held_out=None, transfer=None, variant_id="v1"):
    return ArchiveRecord(
        entry=ArchiveEntry(variant_id=variant_id, generation=0),
        train_solve_rate=train,
        held_out_solve_rate=held_out,
        transfer_solve_rate=transfer,
    )


class TestOverfittingGap:
    def test_computes_the_difference(self):
        record = make_record(train=0.9, held_out=0.5)
        assert overfitting_gap(record) == 0.4 or abs(overfitting_gap(record) - 0.4) < 1e-9

    def test_none_when_never_evaluated_on_held_out(self):
        """Distinct from a gap of 0.0, which would claim 'no overfitting'
        about a variant that was simply never checked."""
        record = make_record(train=0.9, held_out=None)
        assert overfitting_gap(record) is None

    def test_zero_gap_is_a_real_value_not_none(self):
        record = make_record(train=0.5, held_out=0.5)
        assert overfitting_gap(record) == 0.0


class TestTransferRetention:
    def test_full_retention_is_one(self):
        record = make_record(train=0.9, held_out=0.6, transfer=0.5)
        # held_out gain = 0.6 - 0.4 = 0.2; transfer gain = 0.5 - 0.3 = 0.2
        retention = transfer_retention(record, s0_held_out_rate=0.4, s0_transfer_rate=0.3)
        assert abs(retention - 1.0) < 1e-9

    def test_zero_retention_when_transfer_gain_vanishes(self):
        record = make_record(train=0.9, held_out=0.6, transfer=0.3)
        retention = transfer_retention(record, s0_held_out_rate=0.4, s0_transfer_rate=0.3)
        assert abs(retention - 0.0) < 1e-9

    def test_none_when_transfer_never_measured(self):
        record = make_record(train=0.9, held_out=0.6, transfer=None)
        assert transfer_retention(record, s0_held_out_rate=0.4, s0_transfer_rate=0.3) is None

    def test_none_when_held_out_gain_is_zero(self):
        """Division would be meaningless -- there is no gain for a transfer
        gain to retain a fraction of."""
        record = make_record(train=0.9, held_out=0.4, transfer=0.5)
        retention = transfer_retention(record, s0_held_out_rate=0.4, s0_transfer_rate=0.3)
        assert retention is None


class TestHardCodingHeuristic:
    def test_flags_a_verbatim_literal(self):
        source = "def solve(x):\n    if x == 987654:\n        return 'special'\n"
        hits = hard_coding_heuristic(source, ["987654", "irrelevant"])
        assert hits == ["987654"]

    def test_ignores_short_literals(self):
        """Below the length threshold, a match is more likely coincidence
        (a small integer, 'True', empty string) than genuine hard-coding."""
        source = "def solve(x):\n    return 1\n"
        assert hard_coding_heuristic(source, ["1"]) == []

    def test_clean_source_flags_nothing(self):
        source = "def solve(x):\n    return x * 2\n"
        assert hard_coding_heuristic(source, ["987654321", "some_specific_output"]) == []

    def test_empty_literal_list_flags_nothing(self):
        assert hard_coding_heuristic("def f(): return 1", []) == []


class TestPersistence:
    def test_round_trip(self, tmp_path):
        records = [
            make_record(train=0.8, held_out=0.6, variant_id="a"),
            make_record(train=0.5, held_out=None, variant_id="b"),
        ]
        path = tmp_path / "archive.jsonl"
        persist_archive(records, path)
        loaded = load_archive(path)
        assert len(loaded) == 2
        assert loaded[0].entry.variant_id == "a"
        assert loaded[0].held_out_solve_rate == 0.6
        assert loaded[1].held_out_solve_rate is None

    def test_persist_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "archive.jsonl"
        persist_archive([make_record(train=0.5)], path)
        assert path.exists()
