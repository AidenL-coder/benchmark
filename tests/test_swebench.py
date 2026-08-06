"""SWE-bench Verified instance-loading tests (D-40).

Marked `slow`: loading the real dataset needs network access to the
HuggingFace Hub (cached after the first run, but not something to do on
every fast test pass). Unlike `humanevalplus`/`mbppplus`, there is no
vendored copy to fall back to -- see `cbs.tasks.swebench`'s module
docstring for why vendoring doesn't make sense for this family.

Deliberately does not validate that the reference `patch` actually resolves
each instance under the real Docker/SWE-bench harness -- that is D-40's
verification-wiring piece (reusing `swe_bench/harness.py` +
`report.py:make_report`/`run_evals`), not this loader's job. What is
validated here is that the real dataset's fields are parsed into
`SweBenchInstance` correctly and deterministically.
"""

from __future__ import annotations

import pytest

from cbs.tasks.swebench import SweBenchInstance, load_swebench_verified

pytestmark = pytest.mark.slow

#: A fixed, small, known-good instance to smoke-test field parsing against,
#: without paying to materialise and iterate the full 500-instance dataset
#: in every test.
_CANARY_INSTANCE_ID = "astropy__astropy-12907"


@pytest.fixture(scope="module")
def canary_suite():
    return load_swebench_verified(instance_ids=[_CANARY_INSTANCE_ID])


@pytest.fixture(scope="module")
def full_suite():
    return load_swebench_verified()


class TestFieldParsing:
    """Confirmed against the real dataset before writing this loader (not
    assumed from documentation): FAIL_TO_PASS/PASS_TO_PASS arrive as
    JSON-encoded strings, not native lists, despite printing like one."""

    def test_loads_exactly_the_requested_instance(self, canary_suite):
        assert len(canary_suite) == 1
        inst = canary_suite.instances[0]
        assert isinstance(inst, SweBenchInstance)
        assert inst.instance_id == _CANARY_INSTANCE_ID

    def test_fail_to_pass_and_pass_to_pass_are_parsed_tuples_of_str(self, canary_suite):
        inst = canary_suite.instances[0]
        assert isinstance(inst.fail_to_pass, tuple)
        assert isinstance(inst.pass_to_pass, tuple)
        assert len(inst.fail_to_pass) >= 1
        assert len(inst.pass_to_pass) >= 1
        assert all(isinstance(t, str) for t in inst.fail_to_pass)
        assert all(isinstance(t, str) for t in inst.pass_to_pass)
        # A real test node id, not a bare name -- these get passed straight
        # to pytest by the real harness, so the shape matters.
        assert "::" in inst.fail_to_pass[0]

    def test_fail_to_pass_and_pass_to_pass_are_disjoint(self, canary_suite):
        # The hidden oracle and the public execution-feedback signal must
        # not overlap, or running "public" tests would leak the answer.
        inst = canary_suite.instances[0]
        assert set(inst.fail_to_pass).isdisjoint(set(inst.pass_to_pass))

    def test_core_fields_are_nonempty(self, canary_suite):
        inst = canary_suite.instances[0]
        assert inst.repo == "astropy/astropy"
        assert len(inst.base_commit) == 40  # a real git commit SHA
        assert inst.problem_statement.strip()
        assert inst.patch.strip()
        assert inst.test_patch.strip()


class TestHashing:
    def test_content_hash_is_deterministic(self, canary_suite):
        inst = canary_suite.instances[0]
        assert inst.content_hash() == inst.content_hash()

    def test_content_hash_differs_for_different_instances(self, full_suite):
        by_id = full_suite.by_id()
        first_two = list(by_id.values())[:2]
        assert first_two[0].content_hash() != first_two[1].content_hash()

    def test_suite_hash_is_deterministic_and_order_independent(self):
        a = load_swebench_verified(
            instance_ids=["astropy__astropy-12907", "astropy__astropy-14182"]
        )
        b = load_swebench_verified(
            instance_ids=["astropy__astropy-14182", "astropy__astropy-12907"]
        )
        assert a.suite_hash() == b.suite_hash()


class TestFullSuite:
    def test_loads_all_500_verified_instances(self, full_suite):
        assert len(full_suite) == 500

    def test_instance_ids_are_unique(self, full_suite):
        ids = [inst.instance_id for inst in full_suite.instances]
        assert len(ids) == len(set(ids))

    def test_every_instance_has_a_grading_oracle(self, full_suite):
        # Every real instance must have at least one FAIL_TO_PASS test --
        # an instance with none would be unscoreable by definition.
        assert all(len(inst.fail_to_pass) >= 1 for inst in full_suite.instances)
