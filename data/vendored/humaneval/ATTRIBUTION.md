# HumanEval

Vendored from `openai/human-eval` (https://github.com/openai/human-eval),
`data/HumanEval.jsonl.gz`, commit as of 2026-07-30, decompressed as
`HumanEval.jsonl` (164 problems).

**License:** MIT, Copyright (c) OpenAI.

**This is the original HumanEval, not HumanEval+.** The `evalplus` project
(`evalplus/evalplus`) extends this benchmark with substantially more test
cases per problem specifically because the original test suites are known to
under-specify correctness -- some wrong solutions slip through the original
`test` field's assertions. See `docs/DECISIONS.md` D-27 for how this bears on
verifier reliability (brief section 7's audit requirement) and why this
project should move to HumanEval+ before drawing any real capability
conclusion from this family, rather than only for instrument validation.

**Likely pretraining contamination.** HumanEval is one of the most widely
reproduced benchmarks in ML; assume it appears, verbatim or near-verbatim, in
the pretraining corpus of any web-scale frozen model this project evaluates.
It is included here to validate the measurement instrument against
real-world-shaped tasks (genuine syntax diversity, doctest-derived prompts,
non-trivial reference solutions) -- not as a contamination-clean source of
frontier estimates. Brief section 8's contamination probes and "prefer newer
or perturbed variants" guidance apply before this family is used for any
actual capability claim.
