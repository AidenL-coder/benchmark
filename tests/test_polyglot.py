"""Tests for `cbs.tasks.polyglot` (D-42).

Uses a small synthetic fixture matching the real entry shape (confirmed
against a real HGM checkout's `polyglot_benchmark_metadata.json` -- see
docs/DECISIONS.md D-42 for the one-time real-data validation, which cannot
run as a local pytest since the 225-entry, ~6MB metadata file lives only
inside a real HGM checkout, not anywhere network-fetchable or reasonable to
vendor -- unlike `cbs.tasks.swebench`, which loads a real, small public
HuggingFace dataset directly)."""

import json

import pytest

from cbs.tasks.polyglot import PolyglotInstance, PolyglotSuite, load_polyglot_benchmark


def _fixture_entry(instance_id="cpp__all-your-base", language="cpp"):
    return {
        "instance_id": instance_id,
        "language": language,
        "task_name": "all-your-base",
        "files": {
            "solution": ["all_your_base.cpp", "all_your_base.h"],
            "test": ["all_your_base_test.cpp"],
            "example": [".meta/example.cpp", ".meta/example.h"],
        },
        "problem_statement": "Convert a number from one base to another.",
        "base_commit": "7fb37bc7216855a4110bd8d89b567ca28a5912e0",
        "test_commit": "9a984a13c8fe828f391f375f4176e005a49425f7",
        "reference_tests": "TEST_CASE(...) { ... }",
        "test_patch": "diff --git a/all_your_base_test.cpp ...",
        "reference_answers": "std::vector<unsigned int> convert(...) { ... }",
        "repo": "polyglot/polyglot-benchmark/cpp/exercises/practice/all-your-base",
    }


class TestPolyglotInstance:
    def test_fields_populated_from_raw_entry(self):
        entry = _fixture_entry()
        instance = PolyglotInstance(
            instance_id=entry["instance_id"],
            language=entry["language"],
            problem_statement=entry["problem_statement"],
            raw=entry,
        )
        assert instance.instance_id == "cpp__all-your-base"
        assert instance.language == "cpp"
        assert instance.raw["base_commit"] == entry["base_commit"]
        assert instance.raw["test_commit"] == entry["test_commit"]
        # `raw` is the complete original dict, not a re-derived subset --
        # every field process_entry might need is still there unmodified.
        assert instance.raw == entry

    def test_content_hash_stable_across_equal_instances(self):
        entry = _fixture_entry()
        a = PolyglotInstance("x", "cpp", "p", raw=dict(entry))
        b = PolyglotInstance("x", "cpp", "p", raw=dict(entry))
        assert a.content_hash() == b.content_hash()

    def test_content_hash_changes_with_raw_content(self):
        entry = _fixture_entry()
        a = PolyglotInstance("x", "cpp", "p", raw=dict(entry))
        mutated = dict(entry)
        mutated["base_commit"] = "different_commit"
        b = PolyglotInstance("x", "cpp", "p", raw=mutated)
        assert a.content_hash() != b.content_hash()

    def test_content_hash_changes_with_problem_statement(self):
        entry = _fixture_entry()
        a = PolyglotInstance("x", "cpp", "problem A", raw=dict(entry))
        b = PolyglotInstance("x", "cpp", "problem B", raw=dict(entry))
        assert a.content_hash() != b.content_hash()

    def test_content_hash_unaffected_by_dict_key_order(self):
        entry = _fixture_entry()
        reordered = dict(reversed(list(entry.items())))
        a = PolyglotInstance("x", "cpp", "p", raw=entry)
        b = PolyglotInstance("x", "cpp", "p", raw=reordered)
        assert a.content_hash() == b.content_hash()


class TestPolyglotSuite:
    def test_len_and_iteration(self):
        instances = tuple(
            PolyglotInstance(f"id{i}", "cpp", "p", raw={}) for i in range(3)
        )
        suite = PolyglotSuite(name="s", instances=instances)
        assert len(suite) == 3
        assert list(suite) == list(instances)

    def test_by_id(self):
        instances = (
            PolyglotInstance("a", "cpp", "p", raw={}),
            PolyglotInstance("b", "go", "p", raw={}),
        )
        suite = PolyglotSuite(name="s", instances=instances)
        by_id = suite.by_id()
        assert by_id["a"].language == "cpp"
        assert by_id["b"].language == "go"

    def test_suite_hash_independent_of_instance_order(self):
        instances = (
            PolyglotInstance("a", "cpp", "p", raw={"k": 1}),
            PolyglotInstance("b", "go", "p", raw={"k": 2}),
        )
        suite1 = PolyglotSuite(name="s", instances=instances)
        suite2 = PolyglotSuite(name="s", instances=tuple(reversed(instances)))
        assert suite1.suite_hash() == suite2.suite_hash()

    def test_suite_hash_changes_if_any_instance_changes(self):
        instances_a = (PolyglotInstance("a", "cpp", "p", raw={"k": 1}),)
        instances_b = (PolyglotInstance("a", "cpp", "p", raw={"k": 2}),)
        suite_a = PolyglotSuite(name="s", instances=instances_a)
        suite_b = PolyglotSuite(name="s", instances=instances_b)
        assert suite_a.suite_hash() != suite_b.suite_hash()


class TestLoadPolyglotBenchmark:
    def test_loads_all_entries_by_default(self, tmp_path):
        entries = [_fixture_entry("cpp__a"), _fixture_entry("go__b", language="go")]
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps(entries), encoding="utf-8")

        suite = load_polyglot_benchmark(metadata_path)
        assert suite.name == "polyglot_benchmark"
        assert len(suite) == 2
        ids = {inst.instance_id for inst in suite}
        assert ids == {"cpp__a", "go__b"}

    def test_filters_by_instance_ids(self, tmp_path):
        entries = [
            _fixture_entry("cpp__a"),
            _fixture_entry("go__b", language="go"),
            _fixture_entry("java__c", language="java"),
        ]
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps(entries), encoding="utf-8")

        suite = load_polyglot_benchmark(metadata_path, instance_ids=["go__b"])
        assert len(suite) == 1
        assert suite.instances[0].instance_id == "go__b"

    def test_raw_dict_preserved_unmodified_for_real_glue_to_consume(self, tmp_path):
        entry = _fixture_entry()
        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps([entry]), encoding="utf-8")

        suite = load_polyglot_benchmark(metadata_path)
        loaded = suite.instances[0]
        # process_entry(entry, ...) consumes the raw dict directly and
        # unmodified -- confirm round-tripping through JSON didn't drop or
        # rename anything real glue code would need.
        assert loaded.raw == entry

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_polyglot_benchmark(tmp_path / "does_not_exist.json")
