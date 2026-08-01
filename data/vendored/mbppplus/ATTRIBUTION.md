# MBPP+

Vendored from `evalplus/mbppplus` (HuggingFace dataset,
https://huggingface.co/datasets/evalplus/mbppplus), as of 2026-08-01 (378
problems -- the sanitized subset evalplus builds on, same task IDs as
`cbs.tasks.families.mbpp`, some of the 427 originals dropped by evalplus's own
curation).

**License:** Apache License 2.0, Copyright the EvalPlus authors.

**Fetched via the HF datasets-server rows API** (`datasets-server.huggingface.co/rows`,
paginated at 100 rows/request), not the dataset's own parquet file. The
dataset ships only a parquet file (no plain JSONL, unlike HumanEval+), and the
rows API returns the same data as plain JSON without needing `pandas`/
`pyarrow` as a new project dependency.

**This is the evalplus upgrade** referenced in `docs/DECISIONS.md` D-29:
plain MBPP's original test suites under-specify correctness; MBPP+ adds
substantially more test cases per problem via mutation testing, the same
motivation as HumanEval+ (D-27/D-34). Superseding `mbpp`
(`cbs.tasks.families.mbpp`) as the family to use for any real capability
claim, not just instrument validation.

**Simpler to integrate than HumanEval+ turned out to be** (see
`docs/DECISIONS.md` D-35): each row retains the *original*, small `test_list`
(the same few flat asserts plain MBPP already used) alongside the new,
expanded `test` field -- so `public_tests` derivation reuses the exact same
mechanism as `cbs.tasks.families.mbpp` unchanged, no new AST logic needed the
way HumanEval+ required. The expanded `test` field also calls the candidate
by its real entry-point name directly (not aliased to `candidate`, unlike
HumanEval+), matching plain MBPP's convention.

**Requires `numpy` at verification time**, same as HumanEval+ -- the expanded
`test` field's `assertion()` helper uses `numpy.allclose` for tolerant
comparison. Declared under the same `evalplus` optional extra in
`pyproject.toml` (kept one name rather than adding a near-identical second
extra, since both point at the same single dependency).

**Likely pretraining contamination**, same as plain MBPP -- see
`data/vendored/mbpp/ATTRIBUTION.md`.
