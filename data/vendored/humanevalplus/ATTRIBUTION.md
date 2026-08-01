# HumanEval+

Vendored from `evalplus/humanevalplus` (HuggingFace dataset,
https://huggingface.co/datasets/evalplus/humanevalplus), `test.jsonl`, as of
2026-08-01 (164 problems, same task IDs as original HumanEval).

**License:** Apache License 2.0, Copyright the EvalPlus authors.

**This is the evalplus-extended upgrade** referenced throughout
`docs/DECISIONS.md` D-27: HumanEval's original hidden tests are known to
under-specify correctness; HumanEval+ adds substantially more test cases per
problem (often hundreds, generated via mutation testing) specifically to
catch wrong solutions the original test suite let through. Superseding the
`humaneval` family (`cbs.tasks.families.humaneval`) as the family to use for
any real capability claim, not just instrument validation -- see
`docs/DECISIONS.md` D-34.

**Requires `numpy` at verification time.** Unlike original HumanEval, every
task's hidden test defines an `assertion(out, exp, atol)` helper that uses
`numpy.testing.assert_allclose` for floating-point-tolerant comparison, and
imports `numpy` directly. This is a real, new runtime dependency for this
family specifically -- declared as the `evalplus` optional extra in
`pyproject.toml`, not a core dependency of `cbs`.

**Test structure differs from original HumanEval**, which matters for how
`public_tests` is derived (see `cbs.tasks.families.humanevalplus` module
docstring): tests are not flat `assert candidate(...) == expected` statements
but a `check(candidate)` function that builds `inputs`/`results` lists and
loops over them calling the `assertion()` helper. The original family's
line-based/assert-based extraction finds nothing here; a new AST-based
approach specific to this structure is used instead.

**Likely pretraining contamination**, same as original HumanEval (same 164
problems, same task IDs) -- see `data/vendored/humaneval/ATTRIBUTION.md`.

**One task excluded: `HumanEval/32`.** Its generated hidden test asserts
`_poly(*candidate(*inp), inp) <= 0.0001`, which unpacks `find_zero`'s scalar
float return value with `*` as if it were an iterable -- a `TypeError` that
fires against the task's own reference solution, not just a wrong candidate.
This is a genuine bug in the upstream vendored test file (found by running
this project's own reference-solution verification, the same check every
family gets), not something introduced by this loader, and not "fixed" here
by editing the vendored data. Excluded by default
(`cbs.tasks.families.humanevalplus.KNOWN_BROKEN_TASK_IDS`); load with
`exclude_known_broken=False` to inspect the raw file as-is.
