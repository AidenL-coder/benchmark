# MBPP (sanitized)

Vendored from `google-research/google-research` (https://github.com/google-research/google-research),
`mbpp/sanitized-mbpp.json`, as of 2026-07-30 (427 problems -- the
human-verified subset, not the full ~974-problem MBPP set).

**License:** Apache License 2.0, Copyright Google LLC.

**This is plain MBPP, not MBPP+.** As with HumanEval (see
`data/vendored/humaneval/ATTRIBUTION.md`, `docs/DECISIONS.md` D-27), the
`evalplus` project extends MBPP with substantially more test cases per problem
because the original `test_list` fields under-specify correctness. Move to
MBPP+ before this family backs any real capability claim, not just instrument
validation.

**Likely pretraining contamination**, for the same reason as HumanEval: MBPP
is a widely reproduced benchmark and should be assumed present in the
pretraining corpus of any web-scale frozen model. Brief section 8's
contamination probes and "prefer newer or perturbed variants" guidance apply
before treating a solve here as evidence about capability rather than
memorisation.

**Format note:** unlike HumanEval, MBPP's `prompt` is a bare natural-language
instruction with no function signature. Following standard MBPP evaluation
convention, `cbs.tasks.families.mbpp` includes the first hidden test case in
the prompt shown to the model (otherwise the model has no way to know the
expected function name or argument shape) -- see the module docstring for how
this interacts with `public_tests`.
