"""Support-class tagging tests.

The study's central falsifiable claim is stated in terms of this partition
(brief section 3.1), so the invariants that keep it honest are tested directly:
unknown operations must fail loudly, and classifications must not drift.
"""

from __future__ import annotations

import pytest

from cbs.scaffolds.tagging import (
    OperationTrace,
    SupportClass,
    UnregisteredOperation,
    contested_operations,
    definition_of,
    register_operation,
    registry_snapshot,
    support_class_of,
)


class TestRegistry:
    def test_core_classifications(self):
        assert support_class_of("single_call") is SupportClass.PRESERVING
        assert support_class_of("best_of_n") is SupportClass.PRESERVING
        assert support_class_of("execution_feedback") is SupportClass.EXPANDING
        assert support_class_of("decomposition") is SupportClass.EXPANDING

    def test_test_guided_selection_is_preserving(self):
        """Selection by verifier only chooses among M's own samples.

        Classifying it as expanding would make the frontier definition circular,
        since the frontier is itself estimated by best-of-N with a verifier
        (brief section 3.2).
        """
        assert support_class_of("test_guided_selection") is SupportClass.PRESERVING

    def test_unknown_operation_raises(self):
        """An evolved scaffold must not be able to emit an untagged operation."""
        with pytest.raises(UnregisteredOperation):
            support_class_of("some_operation_evolution_invented")

    def test_every_entry_has_a_rationale(self):
        for name, entry in registry_snapshot().items():
            assert entry["rationale"].strip(), f"{name} has no rationale"

    def test_reclassification_is_refused(self):
        with pytest.raises(ValueError, match="refusing to reclassify"):
            register_operation("single_call", SupportClass.EXPANDING, "nope")

    def test_reregistration_with_same_class_is_idempotent(self):
        register_operation("best_of_n", SupportClass.PRESERVING, "same as before")
        assert support_class_of("best_of_n") is SupportClass.PRESERVING

    def test_new_operation_requires_rationale(self):
        with pytest.raises(ValueError, match="requires a rationale"):
            register_operation("brand_new_op_no_rationale", SupportClass.PRESERVING)

    def test_contested_operations_are_flagged(self):
        contested = contested_operations()
        assert "temperature_schedule" in contested
        assert "execution_feedback" not in contested
        assert definition_of("temperature_schedule").contested is True


class TestOperationTrace:
    def test_records_in_order(self):
        trace = OperationTrace()
        with trace.record("single_call"):
            pass
        with trace.record("format_extract"):
            pass
        assert [r.name for r in trace.records] == ["single_call", "format_extract"]
        assert [r.seq for r in trace.records] == [0, 1]

    def test_preserving_only_trace(self):
        trace = OperationTrace()
        with trace.record("best_of_n"):
            pass
        assert not trace.used_expanding
        assert trace.classes_used == {SupportClass.PRESERVING}
        assert trace.expanding_ops == set()

    def test_expanding_detected(self):
        trace = OperationTrace()
        with trace.record("best_of_n"):
            pass
        with trace.record("execution_feedback"):
            pass
        assert trace.used_expanding
        assert trace.expanding_ops == {"execution_feedback"}

    def test_unregistered_operation_raises_before_body_runs(self):
        """Resolution happens on entry, so nothing executes untagged."""
        trace = OperationTrace()
        ran = False
        with pytest.raises(UnregisteredOperation):
            with trace.record("mystery_op"):
                ran = True  # pragma: no cover
        assert not ran
        assert trace.records == []

    def test_record_survives_exception_in_body(self):
        trace = OperationTrace()
        with pytest.raises(RuntimeError):
            with trace.record("single_call"):
                raise RuntimeError("boom")
        assert len(trace.records) == 1

    def test_op_counts(self):
        trace = OperationTrace()
        for _ in range(3):
            with trace.record("single_call"):
                pass
        assert trace.op_counts() == {"single_call": 3}

    def test_contested_ops_surfaced(self):
        trace = OperationTrace()
        with trace.record("temperature_schedule"):
            pass
        assert trace.contested_ops_used == {"temperature_schedule"}

    def test_as_dict_carries_attribution_fields(self):
        trace = OperationTrace()
        with trace.record("retrieval"):
            pass
        payload = trace.as_dict()
        assert payload["used_expanding"] is True
        assert payload["expanding_ops"] == ["retrieval"]
        assert payload["n_ops"] == 1
