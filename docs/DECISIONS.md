# Decision log

Every `[DECISION]` from the project brief (§13), plus choices made while building
that a reviewer could reasonably challenge. Each has a default already
implemented, so nothing is blocked; overturning one is a config change unless
noted.

Status: **P** proposed (implemented as default, open to change) · **C** confirmed
by the human · **D** deferred until the phase that needs it.

---

## D-01 — Frozen-model backend · **C**

**Brief §13.1.** Default was local vLLM.

**Decision:** provider-agnostic. All model access goes through
`cbs.models.base.ModelClient`; backends are `mock` (deterministic, known ground
truth) and `openai_compat` (vLLM, Ollama, or hosted — all speak the same wire
format). The backend is a config key, not a code path.

**Why:** the development machine has no CUDA GPU, so vLLM cannot run locally, but
the *instrument* does not need a model to be built or validated. Deferring the
backend choice costs nothing and avoids rework when the real runs move to Colab
or a rented GPU. Swapping backends is `model.backend` + `model.base_url`.

**Consequence:** `OpenAICompatConfig` refuses to construct a non-`local` endpoint
without per-token pricing set, so a hosted backend cannot silently spend money
outside the accountant.

---

## D-02 — Sandbox policy · **C**

**Brief §10** makes Docker mandatory. Docker is unavailable on the dev machine
and *cannot* exist on Colab (the runtime is already a container; nesting is not
supported).

**Decision:** pluggable `SandboxBackend` with two implementations.
`DockerSandbox` (network `none`, read-only rootfs, `cap-drop ALL`,
`no-new-privileges`, memory/PID limits, non-root user) is a real security
boundary. `SubprocessSandbox` (fresh temp cwd, scrubbed env, wall-clock timeout,
POSIX rlimits where available) is **not**. `select_backend(...)` prefers Docker,
and callers that run genuinely untrusted code pass
`require_security_boundary=True`, which refuses the subprocess backend unless
`allow_insecure_fallback` is explicitly set in config.

**Why:** a Docker-only implementation would block Phase 1 and Phase 2 entirely on
the available hardware, and silently downgrading would be worse — every
`ExecResult` and every `FrontierRecord` therefore carries
`is_security_boundary`, so no result can claim an isolation property it did not
have.

**Standing constraint:** frontier sampling of a small frozen model on curated
coding tasks is acceptable on the subprocess backend. **Running `S_evo`'s
self-modifying scaffolds is not.** Phase 4 requires Docker.

---

## D-03 — Compute plan and where the real runs happen · **P**

**Decision:** build and validate on CPU against the mock backend; run real
frontier estimation on Colab (or a rented 24 GB GPU) via vLLM.

**Sizing.** Brief §5.4 asks for `N_max` in 10³–10⁴ over 100–300 tasks on ≥2
models. At the top of that range this is ~6M generations. Recommended starting
point is **N_max = 1000 over ~100 tasks per model**, ≈200k generations,
≈5–6 T4-hours. Rule of three then bounds a zero-success task at `p < 0.003`,
which is inside the brief's stated range and is a defensible
"beyond the practical frontier at N_max=10³" claim.

Scale `N_max` up only for the model/split where the crossing analysis actually
turns on it, rather than uniformly.

---

## D-04 — `test_guided_selection` is support-**preserving** · **P**

Initially classified as support-expanding, which was wrong.

**Decision:** best-of-N with a verifier as selector is support-**preserving**.

**Why:** the emitted solution is always one the model actually sampled, so it
cannot have ~zero probability under `M`. More decisively, the frontier is itself
*defined* as N-sample best-of-N with a verifier (brief §3.2) — classifying the
selector as expanding would make the baseline estimator an expanding scaffold and
the definition circular.

The oracle signal does come from outside `M`. What matters for the partition is
whether an operation can emit something `M` would never produce, and selection
cannot.

---

## D-05 — `prompt_rewrite` split into two operations · **P**

**Decision:** replaced with `prompt_template` (preserving, flagged contested) and
`adaptive_prompt_rewrite` (expanding).

**Why:** "rewriting the prompt" conflates two different things. A fixed
human-authored template applied identically to every task shifts `M`'s
conditional but does so equally for every system being compared. Rewriting
conditioned on the outcome of prior attempts routes verifier/execution output
back into the context, which is self-refinement and genuinely expanding.

Registry entries carry a `contested` flag for classifications a reviewer could
argue with (`temperature_schedule`, `prompt_template`). Any crossing that depends
on a contested operation must be reported with that caveat attached.

---

## D-06 — Unknown scaffold operations fail loudly · **P**

**Decision:** `support_class_of()` raises `UnregisteredOperation` for any
untagged operation; registering a new one requires a written rationale.

**Why:** `S_evo` rewrites its own scaffold and will invent operations. Defaulting
unknown ops to *preserving* would let genuine expansion masquerade as bounded
search; defaulting to *expanding* would do the reverse. Both silently corrupt the
study's central claim. Failing the run until a human classifies the operation is
the only option that cannot bias the result.

---

## D-07 — Statistics implemented from scratch, no scipy · **P**

**Decision:** Clopper-Pearson (via a hand-rolled regularised incomplete beta),
rule of three, unbiased pass@k, Good-Turing, Chao1 with log-normal CI, and
rarefaction are implemented in `cbs.frontier.estimators` with no third-party
dependency.

**Why:** the measurement core stays installable and testable anywhere, including
a bare Colab runtime. Every function is unit-tested against published reference
values (`tests/test_estimators.py`) rather than against another implementation.

---

## D-08 — Good-Turing measures diversity, not solvability · **P**

**Decision:** `good_turing_unseen_mass()` and `chao1()` return `None` — not
`0.0` — for a task with zero observed successes.

**Why:** these estimators answer "how much of the *correct-solution*
distribution have we not seen?", computed from the sample of correct solutions.
For a zero-success task there is no such sample. Returning `0.0` would render as
"the solution distribution is fully covered" for exactly the beyond-frontier
tasks where nothing was observed — precisely backwards. Solvability bounds for
those tasks come from Clopper-Pearson and the rule of three instead.

---

## D-09 — Temperature schedule is prefix-stable · **P**

**Decision:** `TemperatureSchedule.allocate(n)` uses D'Hondt apportionment, so
`allocate(n)[:m] == allocate(m)` for all `m ≤ n`.

**Why:** the original block allocation made sample `i`'s temperature depend on
`N_max`. Two failures followed, both specific to a preemptible runtime:
resuming a shard at a larger budget retroactively rewrote the temperatures of
samples already drawn (pooling two schedules into one estimate), and a run
truncated by preemption or a budget cap would have sampled *only* the lowest
temperatures, biasing the estimate toward a schedule nobody chose.

**Note:** `p_hat(x)` is the solve probability under **the mixture**, not at any
single temperature. The schedule is recorded on every record and folded into the
resume key, so estimates drawn under different schedules cannot be pooled.

---

## D-10 — Validation passes on a binomial tail test, not perfection · **P**

**Decision:** `PipelineValidation.ok` fails only when the number of CI misses is
implausible under correct calibration (`P(≥ k misses) < 0.01`), rather than
requiring every task to be recovered.

**Why:** a 95% interval is supposed to miss ~1 task in 20. Over 6 tasks,
`P(≥1 miss) ≈ 26%`, so an all-must-pass gate fails a quarter of the time on a
healthy pipeline and trains you to ignore it. Gross faults produce many
simultaneous misses and are still caught. Calibration itself is checked at high
power by `validate_ci_coverage`, which measured 0.944 empirical coverage over 90
task-seeds against a nominal 0.95.

---

## D-11 — Verifier agreement is a permanent validation gate · **P**

**Decision:** `validate_verifier_agreement()` compares the verifier's verdict
against the mock's known label on every sample and requires exact agreement.

**Why:** false positives are the failure mode that would manufacture an apparent
frontier-crossing out of nothing, and the brief (§7, Phase 1) requires bounding
them. Because the mock knows which variant pool each sample came from, every
sample is a labelled example, making this far stronger than the manual audit of a
sample that the brief asks for. Current measurement: 0 false positives and 0
false negatives in 360 labelled samples.

The manual audit is still required for *real* benchmark families, where no ground
truth label exists.

---

## D-18 — Execution feedback reads a separate, weaker `public_tests` · **P**

**Decision:** `Task` gained a `public_tests` field, mechanically derived for the
toy family as the first half of `tests`' assertion lines (rounded up, min 1).
`S_star`'s execution-feedback loop runs candidates against `public_tests` only;
`tests` (the hidden oracle) is queried exactly once, at the very end, to score
the final choice — identical to `S0`.

**Why:** a real deployed coding agent does not see the held-out grading suite
while it works. If execution feedback read the same hidden tests used for
scoring, an "expanding" operation could walk straight to a passing answer by
querying the answer key repeatedly, and any resulting frontier-crossing would be
an artefact of oracle access, not of the operation. Keeping the channels
separate is what makes an `S_star` elicitation gain mean something.

---

## D-19 — `S_star`'s internal selection never queries the hidden oracle · **P**

**Decision:** final candidate selection is majority vote by canonical form
among public-test-passing candidates (falling back to a vote over all
candidates if none passed), not `test_guided_selection` against `tests`.

**Why:** `test_guided_selection` is correctly classified support-preserving
(D-04) and would have been a legitimate design — but it would make `S_star`
unrealistic. A real strong baseline does not get to try candidates against the
grading suite and keep the first one that happens to pass. Restricting
selection to public signal plus consensus means any elicitation gain `S_star`
shows is earned the way a real scaffold would earn it, not manufactured by
oracle access unavailable to a real deployment.

---

## D-20 — the mock backend cannot demonstrate a genuine feedback benefit · **P**

**Observation, not a decision to make, but one worth stating plainly.**
`MockModelClient._generate` reads only `(task_id, temperature, seed/index)` —
never prompt content (deliberately: this is what keeps its ground truth exactly
known, see D-07/D-08). So on the mock, every repair call in `S_star`'s loop is
statistically an independent fresh draw at the same true `p(x)`, indistinguishable
from just drawing another top-level candidate. Execution feedback cannot help
the mock, structurally, no matter how good the mechanism is.

Measured consequence: on `configs/compare_toy_mock.yaml`, `S_star`'s solve rate
is *not* reliably above `S0`'s at matched compute — mean elicitation gain
≈ −0.11 across the toy suite (see D-21 for why). **This is the expected,
correct result on this backend**, not a failure of `S_star`. The comparator
validates the *harness* here — that solve rates, CIs, and matched-compute
bookkeeping are computed correctly — not whether execution feedback helps,
which requires a real model that reads its own error messages (Phase 3 real
run, deferred pending GPU access per D-03).

---

## D-21 — naive public-test derivation has real blind spots · **P**

**Finding**, pinned by `tests/test_s_star.py::TestPublicTestBlindSpot`.

On `toy/is_palindrome`, the mechanical "first half of assertions" split (D-18)
happens to put all four `True`-expected examples in `public_tests` and all
three `False`-expected examples in the hidden-only remainder. The declared
incorrect variant `def is_palindrome(s): return True` therefore **passes the
public subset** and is only caught by the hidden oracle. Same mechanism on
`toy/unique_sorted`: the duplicate-collapsing case (`[2,2,2] -> [2]`) is hidden-
only, so `def unique_sorted(xs): return sorted(xs)` (no dedup) passes public and
fails hidden.

Consequence: on exactly these two tasks, `S_star`'s "prefer a public-passing
candidate" rule can commit to a plausible-but-wrong answer that a `S0`-style
best-of-N-with-full-oracle-selection would never have been fooled by (`S0`
never sees an intermediate verdict; it is only ever scored once, same as
`S_star`'s final check, but with no internal step that could be misled).

**This is left as-is, not "fixed" by hand-curating a more representative public
subset.** A hand-picked subset would hide a genuine, real-world phenomenon —
partial test coverage misleading an agent's self-selection — behind an
artificially thorough example set that real benchmarks (a single doctest
example, a handful of visible unit tests) usually do not provide either. It is
exactly the kind of failure the study's overfitting/verifier-reliability
concerns (proposal §8, brief §7) are about, and it is more useful documented
than papered over.

---

## D-22 — `S0`'s side of a matched-compute comparison must be read from an
oversampled frontier record, never sampled at exactly the comparison budget · **P**

**Bug found and fixed**, in `cbs.cli.cmd_compare`.

`pass_at_k(n, c, k)` is *exactly* 1.0 whenever `k == n` and `c > 0`: "did any of
all n samples succeed" is trivially yes if one did. Sampling `S0` at
`N_max == B` (the comparison budget) and reading `pass@B` off it therefore
degenerates to an uninformative point mass, discarding the entire reason the
unbiased pass@k estimator exists — to give a low-variance read of "solve rate at
budget `k`" from a **larger** sample `n > k` (Chen et al. 2021).

**Fix:** `cmd_compare` samples `S0` at `N_max = budget_calls * s0_oversample_factor`
(default 8x) and reads `pass@B` off that curve. `tests/test_compare.py` pins
both the correct (oversampled, non-degenerate CI) and the degenerate (`n == k`)
cases so this cannot silently regress.

**Side finding, worth keeping in mind when choosing `N_max`/`B` for real runs:**
for any task with `p(x)` not tiny, `S0`'s `pass@B` saturates near 1.0 for even
modest `B` (12 independent tries already means ~1 − 0.4¹² ≈ 99.9999% at
`p = 0.6`). The elicitation-gain metric is therefore only informative either
near the practical frontier (small `p(x)`, where resampling alone does not save
you either) or as an efficiency comparison (mean calls actually spent, which
`ComparisonRecord` reports alongside the rate). Reassuringly, that low-`p`
regime is exactly where the real study's beyond-frontier tasks live, so this is
not a limitation of the design — it is a reason to expect the comparison to be
most informative precisely on the tasks the study cares about most.

---

## D-23 — nowhere in current infrastructure can safely run a real `S_evo` · **D**

**Finding, surfaced rather than worked around.** Brief §10 makes Docker
mandatory for executing model-generated code, and D-02 already established
that reasoning applies double to `S_evo`: its blast radius is self-modifying
*orchestration* code with live network access to a model endpoint, run
repeatedly across generations, not a one-shot candidate solution. Neither
environment this project currently has access to satisfies that:

- the Windows dev machine has no Docker installed;
- Colab's runtime is already a container and **cannot nest Docker** — this
  isn't a missing package, it's a platform constraint.

So there is currently no host in this project's plan that can safely run a
forked DGM-style loop. This blocks *executing* Phase 4 for real, not building
its measurement layer (§5.2 — which is what this session built: D-24 through
D-26 below, all validated without trusting any self-modifying code).

**What resolves it:** a rented Linux instance with both a GPU and Docker
(a RunPod/Lambda/vast.ai box, e.g.) — which also happens to satisfy D-03's GPU
requirement for real vLLM serving in the same environment. This converges two
open needs into one provisioning step, worth doing together rather than
separately. Needs a decision (and likely a budget cap, D-16) before Phase 4 can
actually run.

---

## D-24 — `S_evo` tagging is enforced by interception, not self-report · **P**

**Decision:** `S0`/`S_star` tag their own operations at the call site
(`OperationTrace.record`), which is fine because we author their control flow
and can trust it. `S_evo`'s agent code cannot be trusted the same way: it is
produced by an optimisation process whose objective is "solve more benchmark
tasks", not "report its own operations honestly", and an agent that discovered
it could omit tagging an expanding operation to look more like bounded search
would have no incentive not to.

`cbs.scaffolds.evolved.InterceptionSession` instead sits between the untrusted
agent function and the only two channels it can act through — the frozen model
(`InterceptingModelClient`) and the verifier (`InterceptingVerifier`) — and
classifies every call from what was actually observed, after the fact
(`OperationTrace.record_instant`, a new primitive added alongside the existing
call-site `record()`).

**The one ambiguity interception cannot resolve by construction:** a verifier
call's pass/fail result might be used only to *select* among already-generated
candidates (`test_guided_selection`, preserving) or to *condition the next
generation* on the failure (`execution_feedback`, expanding) — both look
identical as "a sandboxed run happened". Resolved with behavioural evidence: if
a later prompt to the model contains a substantial fragment (≥24 chars) of an
earlier verification's error text, that verifier call classifies as feedback;
otherwise as selection. A blind default in either direction would be worse —
always-expanding makes the study's central claim unfalsifiable (everything
"depends on expansion" trivially, since nearly every agent runs *some*
verification), and always-preserving would systematically hide real expansion.
Validated in `tests/test_evolved.py` against two synthetic agents built
specifically to probe each side: one that verifies candidates but never
references a failure's error text in a later prompt (correctly classified
`test_guided_selection`), and one that explicitly builds its repair prompt out
of the prior error (correctly classified `execution_feedback`).

---

## D-25 — ablation of an untrusted scaffold blocks the information channel,
not the agent's own code · **P**

**Decision:** `InterceptionSession(disabled_ops=...)` withholds error detail
from what is *returned to the agent* when `"execution_feedback"` is disabled —
the real result (with its real error text) is still what gets recorded for the
session's own bookkeeping, but the agent genuinely never receives the text, so
it cannot embed it in a later prompt no matter what its own code tries to do.

**Why:** for `S0`/`S_star` (our own code), "ablate an operation" can just mean
"construct it with a different argument" (e.g. `SStar(max_repairs_per_candidate=0)`).
That option does not exist for an evolved scaffold whose source we do not
control and should not trust to honor a config flag. Blocking the channel
itself is the only ablation that is causally valid regardless of what the
agent's code does with what it's given. Verified in
`tests/test_evolved.py::TestAblationInterception` and exercised end-to-end by
`cbs.ablation.run_ablation`.

---

## D-26 — `S_star`'s own execution-feedback tag is call-site, not causal · **P**

**Known simplification, left as-is.** `S_star` (code we author) tags its
public-test check as `execution_feedback` at the call site every time it runs,
even in a configuration (`max_repairs_per_candidate=0`) where the loop always
terminates after that first check and no repair — and therefore no actual
conditioning — ever follows. Functionally, that lone check is then equivalent
to `test_guided_selection`, not `execution_feedback`, and is inconsistent with
the more careful, evidence-based classification `InterceptionSession` applies
to `S_evo` (D-24).

**Why left as-is:** the discrepancy only affects `S_star`'s own hand-authored
tagging, which we already fully trust and inspect directly (unlike `S_evo`'s),
and the static tag still correctly describes the *operation's intent* (deciding
whether to repair) even in a configuration that happens never to exercise it.
Fixing it would mean either duplicating `InterceptionSession`'s post-hoc
classification inside `S_star` or accepting the mismatch; duplicating it for
code whose honesty is not in question is not worth the complexity. Noted here
so a reader of an `S_star` ablation result is not confused by
`execution_feedback` appearing in a trace where no repair could possibly have
occurred.

---

## D-27 — vendored the original HumanEval, not HumanEval+ · **P**

**Decision:** `cbs.tasks.families.humaneval` vendors `openai/human-eval`'s 164
problems (MIT-licensed; `data/vendored/humaneval/ATTRIBUTION.md`), not the
evalplus-extended HumanEval+ the brief actually names (§8).

**Why vendor at all, and why this one first:** validating the pipeline against
*real* code (real syntax diversity, multi-line bodies, imports, genuine
reference solutions) is a materially stronger check than the hand-written toy
family alone — confirmed by running: **all 164 canonical solutions pass the
real sandbox and verifier**, and a random sample of "return None" wrong-answer
probes are all correctly rejected. Original HumanEval is the easiest
permissively-licensed, single-file, no-dependency source to vendor and was the
fastest path to that validation; evalplus's data requires its own
package/release artifact and is the natural next upgrade, not a blocker on
getting *a* real family in now.

**This validation immediately paid for itself — two real extraction bugs found
and fixed, neither of which the toy family (hand-written, single-line
assertions by construction) could ever have surfaced:**

1. `public_tests` was first derived with a `MULTILINE`-anchored regex over
   `assert candidate(...)` lines, mirroring the toy family's approach (D-18).
   Several HumanEval assertions span multiple source lines (a list literal
   continuing on the next line); the regex silently truncated those
   mid-expression, producing a subset that failed to even *parse*. Caught by
   the test that checks a reference solution passes its own derived public
   subset. Fixed by extracting complete statements via `ast` (parse, find
   `Assert` nodes referencing `candidate`, re-emit with `ast.unparse`) instead
   of matching source text — a statement re-serialized from its own AST cannot
   be truncated mid-expression.
2. Two problems (`HumanEval/38`, `HumanEval/50`) build randomised setup state
   across loop iterations before asserting (`for _ in range(100): s = ...;
   encoded = encode(s); assert candidate(encoded) == s`). Extracting the
   `assert` alone is syntactically valid AST but references names never bound
   in isolation — a `NameError` at runtime, not a legitimate test result. AST
   correctness alone could not catch this; only running the derived subset
   against the task's own known-correct reference solution could. So
   `humaneval_suite()` now validates every derived `public_tests` against its
   reference solution at load time (default on) and blanks it out if even the
   reference fails — because a public test that rejects a *correct* candidate
   is actively harmful (it would make `S_star`'s execution-feedback loop
   discard correct answers), not merely uninformative like an empty one.

Both are pinned as regression tests in `tests/test_humaneval_family.py` by
task id and by synthetic minimal reproductions, not only by the aggregate
"most tasks get a public subset" count. With the fix, 161/164 problems have a
validated public subset; the remaining 3 (2 loop-scoped, 1 with no flat
asserts at all) fall back to a compile-only check for execution feedback — an
honest degradation, not a silent one.

**Why this matters before any real capability claim (not just instrument
validation):** two caveats, both already known and stated in the vendored
data's own attribution rather than left implicit —

1. The original test suites are known to under-specify correctness (this is
   evalplus's entire premise); a scaffold could pass HumanEval's hidden tests
   on a subtly wrong solution in a way HumanEval+ would catch. Move to
   HumanEval+ before this family backs any frontier-crossing claim.
2. HumanEval is one of the most reproduced benchmarks in ML — assume any
   web-scale frozen model has seen it during pretraining. Brief §8's
   contamination probes and "prefer newer or perturbed variants" guidance
   apply before treating a solve on this family as evidence about capability
   rather than about memorisation.

Both are exactly the caveats brief §7/§8 asks to be handled, stated here so
they travel with the family rather than being rediscovered later.

---

## D-28 — crossing determination, interpretation placement, and the
statistical plan built as pure logic ahead of having data to run them on · **P**

**Decision:** built `cbs.crossing` (the four-part frontier-crossing
determination, brief §3.3), `cbs.interpretation` (mechanical placement into
the §7/§9.3 interpretation matrix), and `cbs.stats` (bootstrap CIs and
Benjamini-Hochberg multiple-comparison correction, both required by §6/§9.4
and not built before now) — all as logic over objects Phases 2-4 already
produce (`FrontierRecord`, `ScaffoldRunSummary`, `AblationResult`), not as
scripts that assume a real `S_evo` run exists.

**Why now, given D-23 blocks actually running the crossing test for real:**
none of this needs a GPU, Docker, or a forked loop — it needs the *shape* of
the data those things will eventually produce, which is already fixed by
Phases 2-4's schemas. Building and testing the combination logic now means
Phase 5 is "point this at real records" rather than "design and debug this
from scratch under time pressure once GPU access finally arrives."

**Two things this deliberately does NOT do**, because they are pre-registration
decisions (D-14), not implementation details:

- `evaluate_crossing`'s `k`/`K` reliability threshold is a required parameter,
  never a default — there is no principled value to guess at without
  pre-registering one.
- `place_in_interpretation_matrix`'s three judgment-call thresholds
  (near-zero crossing rate, "high" transfer retention, "large" overfitting
  gap) are all explicit parameters with documented meanings, not hard-coded
  guesses. Calling the function forces a conscious choice; it cannot be used
  by accident with unexamined defaults.

**One design choice worth flagging:** "`S_evo` beats `S_star`" is decided by
non-overlapping confidence intervals (`s_evo_ci_low > s_star_ci_high`), not a
raw point-estimate comparison — a point-estimate-only comparison would call
noise a result. This is stricter than a simple difference-of-means test and
is intentional given how costly a wrong "genuine expansion" claim would be.

---

## D-29 — vendored MBPP (sanitized), with the model shown one test case
per the standard evaluation convention · **P**

**Decision:** `cbs.tasks.families.mbpp` vendors the 427-problem sanitized
subset of `google-research/mbpp` (Apache 2.0;
`data/vendored/mbpp/ATTRIBUTION.md`) — plain MBPP, not the evalplus-extended
MBPP+ the brief names, same posture and same caveats as HumanEval (D-27):
under-specified original test suites, near-certain pretraining contamination.

**Why the prompt includes an example test.** Unlike HumanEval, MBPP's own
`prompt` field is bare natural language ("Write a function to find the shared
elements from the given two lists.") with no function signature — a model
given only that text has no way to know the expected function name or
argument shape. Every published MBPP harness addresses this by showing one
test case alongside the instruction; `cbs` follows that convention
(`test_list[0]` is included in `Task.prompt`) rather than inventing a
non-standard variant that would make results incomparable to the literature.
Because that example is already shown to the model, it is unambiguously public
— `public_tests` (the same first-half heuristic as D-18) always includes it.

**Validation, mirroring D-27:** all 427 reference solutions pass the real
sandbox and verifier, all 427 get a valid derived public subset (every
reference solution passes its own), and a random sample of "return None"
probes are all correctly rejected. Cleaner than HumanEval's extraction: MBPP's
`test_list` is always flat, single-line, self-contained assertions — none of
HumanEval's multi-line-literal or loop-scoped-setup failure modes (D-27)
applied here, so no comparable extraction bugs were found. `entry_point` (not
given explicitly by the dataset) is inferred from the first `Call` node found
walking the first test assertion's AST — descriptive metadata only, since
nothing in the verifier consumes `Task.entry_point`, so the heuristic does not
need to be perfect.

---

## D-30 — transfer/reasoning family (D-17) hand-authored, not vendored · **P**

**Decision:** `cbs.tasks.families.transfer_reasoning` is 10 hand-authored
math/logic/combinatorics tasks (compound interest, triangle validity, coin-
change DP, perfect numbers, LCM, quadratic roots, Nim optimal play, prime
factorisation, 2x2 linear systems, the Josephus problem) — verified through
the exact same mechanism as every other family (write a function, execute
against tests in the sandbox), so no new verification mode was needed.

**Why hand-authored rather than vendored:** no small, permissively-licensed,
directly code-checkable reasoning benchmark was as immediately available as
HumanEval/MBPP were. Brief §8 asks for "one set" with checkable answers, not
exhaustive coverage, so hand-authoring 10 solid problems was the faster,
equally valid path to satisfying D-17 now rather than blocking on finding and
adapting a third external dataset.

**What makes this "transfer" rather than just more of the same:** content, not
mechanics. Every task is deliberately unlike the general-purpose programming
idioms HumanEval/MBPP are built from — number theory, game-theoretic optimal
play, basic algebra — so that generalisation here is genuinely about problem
*character* transferring, not about a held-out slice of the same distribution.
Frozen entirely into the `transfer` split (`data/splits/transfer_reasoning.json`,
train=0/held_out=0/transfer=1.0 — since it is a standalone suite, this routes
every task to `transfer` without needing `assign_splits`'s
`transfer_families` override, which exists for mixing multiple families in
one suite rather than for a family that is transfer-only by construction).

**Validated the same way as the other real families**: all 10 reference
solutions pass the real sandbox, all 10 get a valid public subset (every
reference solution passes its own — verified before committing, exactly as
for HumanEval/MBPP), and "return None" is correctly rejected on every task.
One hand-computed test value (`min_coins(27, [1, 5, 10, 25])`) was initially
wrong (asserted 4, the true minimum is 3: 25+1+1) — caught by re-deriving it
by hand a second time before running the suite, a reminder that hand-authored
ground truth needs the same scrutiny as vendored data, not less.

---

## D-12 — recommend `metauto-ai/HGM` over `jennyzzt/dgm`, pending confirmation · **P (research finding, not yet user-confirmed)**

**Researched** (GitHub API metadata + direct source inspection, both repos'
`README.md`, `coding_agent.py`, `llm.py`, `llm_withtools.py`, top-level file
listings, `tools/`, `swe_bench/`). Not yet acted on — forking a repo is a
bigger commitment than most decisions in this log, so this is presented as a
recommendation for the human to confirm, not a default silently adopted.

### What both repos actually are, structurally

`metauto-ai/HGM`'s own README states it plainly: "built upon the code from the
Darwin-Gödel Machine." Confirmed by comparing top-level file listings —
`coding_agent.py`, `coding_agent_polyglot.py`, `llm.py`, `llm_withtools.py`,
`self_improve_step.py`, `swe_bench/`, `polyglot/`, `tools/`, `prompts/`,
`utils/` are present, essentially unchanged, in **both**. HGM replaces DGM's
`DGM_outer.py` (flat archive, score-proportional/best/random parent selection)
with `hgm.py` + `hgm_utils.py` + `tree.py` — a clade/subtree-based archive that
estimates the promise of entire self-modification subtrees before expanding
them, which is the paper's actual contribution (ICLR 2026 oral). Everything
else — how the agent calls the frozen model, what tools it has, what a "task"
looks like — is the **same code** in both.

That last point matters most for this project: **the fork choice barely
affects integration cost, because the two forks share the exact same
integration surface.** It mainly affects *which specific evolutionary/
selection mechanism is under study* (DGM's simpler heuristics vs. HGM's
clade-based promise estimation) — a more interesting and more current test
case, not a harder integration.

### Two structural facts that apply to *either* fork, and change this
project's scope

1. **The agent is hardcoded to Claude/OpenAI's hosted APIs, not a generic
   OpenAI-compatible endpoint.** `llm.py`'s `create_client(model)` dispatches
   on model-name string prefixes (`claude-`, `gpt-`, `o1-`/`o3-`,
   `deepseek-`, one `openrouter` case) with no existing branch for an
   arbitrary local `base_url`. The DeepSeek/OpenRouter branches already show
   the exact pattern needed (`openai.OpenAI(api_key=..., base_url=...)`), so
   adding a `local/<model>` branch that points at a local vLLM server is a
   small, well-scoped patch (roughly 20-30 lines across `llm.py` and
   `coding_agent.py`'s hardcoded `self.code_model = CLAUDE_MODEL`) — not a
   rewrite, and consistent with "fork and add the measurement layer on top,"
   not "rebuild the loop." One real wrinkle:
   `llm_withtools.get_response_withtools`'s tool-calling dispatch has explicit
   branches for Claude and o3-style responses, plus a generic fallback that
   parses a literal `<tool_use>` tag out of a plain-text response for "any
   other LLM" — usable, but means tool-call reliability for a small open
   model depends on that model reliably emitting the expected tag format,
   which is worth validating early rather than assuming.
2. **The agent operates on real git repositories via SWE-bench/Polyglot, not
   on atomic function-completion tasks.** `AgenticSystem` takes a
   `git_tempdir` + `problem_statement` + `base_commit`, and solves by editing
   files and producing a diff (via `bash`/`edit` tools — that is the entire
   tool surface, `tools/bash.py` and `tools/edit.py`). This does not map
   directly onto `cbs.tasks.schema.Task` (prompt + assert-based tests), which
   is what `toy`/`humaneval`/`mbpp`/`transfer_reasoning` all are. Two ways to
   reconcile this, not yet chosen:
   - **(a)** Adapt: wrap each `cbs` task as a minimal one-file git repo (a
     single stub file + its hidden test file as the "problem"), bridge
     `AgenticSystem`'s diff output through `EvolvedScaffold`'s
     `AgentFunction` interface, and verify the result with `cbs`'s own
     verifier regardless of what the loop's internal SWE-bench-style check
     says (keeping `cbs` the authoritative scorer, consistent with how `S0`
     and `S_star` are scored). Plausible, but nontrivial glue code.
   - **(b)** Don't adapt: let the fork evolve against its own native
     SWE-bench Verified / Polyglot substrate unmodified, and treat *that* as
     `cbs`'s primary coding family for any run that includes `S_evo`, with
     `humaneval`/`mbpp`/`transfer_reasoning` remaining useful for cheap
     `S0`/`S_star` frontier estimation and calibration but not something
     `S_evo` is measured against. This is simpler and more faithful to each
     paper's published methodology, but means **SWE-bench Verified (D-13) is
     not an optional "nice to have" family — it is the substrate this
     project's fork choice actually commits to**, and it is a substantially
     heavier lift than `humaneval`/`mbpp` were: SWE-bench's own harness runs
     a full per-instance Docker container per problem, coupling this
     directly to D-23, not just to "a Docker sandbox for candidate code."

Not resolving (a) vs (b) here — that is itself a design decision worth its own
discussion once a host exists, not something to guess at now.

### Comparison

| | `jennyzzt/dgm` | `metauto-ai/HGM` |
|---|---|---|
| License | Apache-2.0 | Apache-2.0 |
| Stars | 2204 | 405 |
| Last push (as of 2026-08-01) | 2025-08-13 — ~1 year stale | 2026-02-07 — actively maintained |
| Open issues | 26 | 5 |
| Venue | arXiv 2505.22954 | ICLR 2026 oral (arXiv 2510.21614) |
| Core agent/model/task code | original | same code, inherited from DGM |
| Selection mechanism | flat archive; score-proportional / best / random parent choice | clade/subtree promise estimation (the paper's actual contribution) |

### Recommendation

**`metauto-ai/HGM`**, over the brief's stated default (`jennyzzt/dgm`), for
three reasons: it is materially more recently maintained (DGM's own repo has
had no commits in a year); it studies a more sophisticated and more current
selection mechanism, which is a more interesting test case for whether
self-improvement crosses the frontier; and — critically — it costs **nothing
extra** to integrate relative to DGM, since the integration-relevant code
(model calling, task representation, tool surface) is identical between them.
Keep `jennyzzt/dgm` available as the literature-baseline reference the brief
asks for (§5.1: "a possible second `S_evo` variant"), not discarded.

**This is a recommendation, not a decision** — confirm before forking
anything (D-12 in the table below stays open until then), and (a) vs (b)
above needs a decision alongside it once a host is being provisioned.

---

## D-32 — elaborated preregistration thresholds with full reasoning,
recommendation still awaiting sign-off · **P (recommendations, not yet
user-confirmed)**

`docs/preregistration.md` §3's `[TO FIX]` thresholds previously carried a bare
"proposed X" with no justification. Replaced with full reasoning per
threshold — see that file's "Reasoning behind each recommendation"
subsection for the complete argument. Two changed from their original
one-line proposal after actually thinking through the trade-off, not just
formatting the same numbers more verbosely:

- **Reliability `k/K`**: 3/5 → **6/10** (same 60% proportion, more
  resolution). The crossing/reliability check runs only on the small,
  already-filtered beyond-frontier subset, not the full held-out set `N_max`
  sampling covers — so `K` is comparatively cheap to raise here, and doing so
  tightens the CI on the highest-stakes claim in the study (a confirmed
  crossing) for near-zero extra cost.
- **"Large" overfitting gap**: 0.3 → **0.2**. A 30-point train/held-out gap is
  a permissive bar that would let a meaningfully overfit result still land in
  "genuine expansion" rather than "illusory expansion." Deliberately biased
  this specific threshold conservative (easier to trigger "illusory
  expansion" on a smaller gap): a real discovery should survive a stricter
  overfitting check, and a false one is exactly what a stricter check exists
  to catch. Asymmetric treatment relative to the transfer-retention threshold
  (left at 0.5, not tightened) is deliberate — retention is the *constructive*
  claim the study is built to be able to make truthfully and shouldn't be
  made arbitrarily hard to reach, whereas the overfitting gap is a
  *disqualifying* check where the conservative direction is the safe one.

**Not acted on beyond updating the recommendation** — `[TO FIX]` markers stay
in place in `preregistration.md` until the user actually signs off. These are
judgment calls about acceptable risk in a scientific claim, which is squarely
the researcher's call to make, not something to finalize on their behalf.

---

## D-33 — LiveCodeBench is a materially bigger lift than HumanEval/MBPP were;
scoped, not implemented · **D (deliberately, after real investigation)**

**Investigated** (HF API metadata + range-fetched samples of the actual data,
not just the paper/README) before attempting to vendor it the way
HumanEval/MBPP were. The conclusion: it doesn't fit `cbs`'s existing
assert-based `Task`/`Verifier` model the way every family so far has, and
retrofitting it properly is a cross-cutting change, not a new loader file.

**What the data actually looks like**, confirmed by fetching real rows from
`livecodebench/code_generation_lite` (HF dataset, CC-licensed, not gated):

- **Size**: six release-version files, ~130 MB to ~1.25 GB *each* (≈4.5 GB
  total) — three to four orders of magnitude larger than HumanEval (214 KB)
  or MBPP (255 KB). Committing this into the repo the way those were vendored
  is not appropriate; it would need to be fetched/cached at run time into the
  already-gitignored `data/raw/`, with only a small curated slice (brief §8
  explicitly says "a LiveCodeBench slice", not the whole set) vendored small
  and hashed.
- **At least three distinct test-execution conventions**, not one:
  - `testtype: "stdin"` (AtCoder/Codeforces problems) — the candidate is a
    **full program** reading stdin and writing stdout; correctness is
    string/whitespace comparison of captured output, not a return value.
  - `testtype: "functional"` (LeetCode problems) — `starter_code` gives a
    `class Solution:` method stub; `input`/`output` are JSON-string-encoded
    representations of call arguments and the expected return value that need
    parsing (not raw text) before a driver can instantiate the class,
    reflectively call the right method, and compare.
  - `private_test_cases` is base64+zlib-compressed (unlike `public_test_cases`,
    which is a plain JSON string) — deliberate, per the dataset's own
    "contamination-free" design goal, and needs decoding before use regardless
    of which execution convention applies.
- **`contest_date`** is present per problem and matters specifically for this
  family: LiveCodeBench's whole premise is temporal — later releases contain
  problems from contests after a model's pretraining cutoff, which is the
  actual contamination-avoidance mechanism brief §8 asks this family to
  provide. Picking *which* release/date-range to slice is therefore a real
  methodological choice, not an implementation detail, and should probably be
  made close to when a specific frozen model's pretraining cutoff is known.

**Why this doesn't fit as a same-shape addition to `humaneval.py`/`mbpp.py`**:
the "stdin" convention has no return-value/entry-point to call at all (the
"candidate" is a whole program, not a function) and cannot be verified by
`cbs.tasks.verifier.Verifier`'s current approach (concatenate candidate +
assert-based tests into one script) without a fundamentally different
execution model — running the candidate as its own subprocess per test case,
piping input, capturing output. That needs `cbs.sandbox.ExecRequest` to grow
stdin support (currently explicitly `stdin=subprocess.DEVNULL` in
`SubprocessSandbox`, to prevent hangs) in **both** sandbox backends, plus a
parallel verification path outside `Verifier`'s marker-based pass/fail logic
(a marker doesn't make sense when correctness is "does captured stdout match,"
not "did assertions complete"). And because several other pieces of the
codebase implicitly assume "one function to call" — `S_star`'s public-test
repair loop, `cbs.tasks.canonicalize`'s AST-based renaming (assumes a single
function definition, not an arbitrary program or a class method) — a new I/O-
judge task type would ripple into those too, not stay contained to a loader.

**Decision: not implemented this session.** The "functional" (LeetCode)
subset is closer to the existing model and might be tractable without the
stdin/stdout extension, but confirming that needs a design pass, not another
range-fetch. Flagging this clearly rather than either rushing a half-verified
verification mode (the exact kind of shortcut this project has avoided
everywhere else) or silently deprioritising it without explanation.

---

## D-34 — vendored HumanEval+, a genuinely new (not reused) public-test
derivation, and a real upstream bug found and excluded · **P**

**Decision:** `cbs.tasks.families.humanevalplus` vendors `evalplus/humanevalplus`
(Apache 2.0; `data/vendored/humanevalplus/ATTRIBUTION.md`) — the evalplus
upgrade to HumanEval that D-27 flagged as needed before this family backs any
real capability claim (the original's hidden tests under-specify
correctness). Adds `numpy` as a new runtime dependency (the `humanevalplus`
optional extra; every other family needs nothing beyond core `cbs`), since
every task's hidden test uses `numpy.testing.assert_allclose` for
floating-point-tolerant comparison.

**Why this was not a drop-in swap of a file path in the existing loader.**
HumanEval+'s tests are structured completely differently from original
HumanEval's: instead of flat `assert candidate(...) == expected` statements,
every task defines a `check(candidate)` function that builds `inputs`/
`results` list literals and loops over them calling an `assertion()` helper.
The original family's extraction (`ast.Assert` nodes referencing `candidate`)
finds **nothing** here — reusing it unmodified would have silently degraded
every task's `public_tests` to empty (a compile-only check), with no error to
notice. `_derive_public_tests` here instead locates the `inputs`/`results`
list-literal assignments via AST, truncates *the list elements* to their
first half, and preserves every other statement in `check` verbatim and in
order — deliberately not assuming a specific loop shape, only that `inputs`/
`results` are literal lists safe to shorten.

**A genuine upstream data bug found and excluded, not fixed.**
`HumanEval/32` ("find_zero")'s generated hidden test asserts
`_poly(*candidate(*inp), inp) <= 0.0001`. `find_zero` returns a single float
root; `*` on a scalar raises `TypeError`. The task's *own reference solution*
fails this assertion — confirmed directly, not inferred — which would make
any correct candidate register as a failure on this specific task. This is
evalplus's test-generation pipeline mis-templating the assertion for a
problem shape (`list -> scalar`) unlike most HumanEval problems, not
something this loader introduced. Excluded via `KNOWN_BROKEN_TASK_IDS`
(default on, documented, reversible with `exclude_known_broken=False`) rather
than either silently keeping a task that would corrupt its own frontier
estimate, or editing the vendored copy — the vendored data stays a faithful,
unmodified reproduction of evalplus's file; the exclusion lives in `cbs`'s
loader, not in the data.

**Validated the same way as every other real family**: all 163 (of 164,
1 excluded) reference solutions pass the real sandbox; the derived public
subset is checked against its own task's reference solution and blanked out
on failure (same safety net as D-27); a sample of "return None" probes are
all correctly rejected.

---

## D-35 — vendored MBPP+ (via the HF datasets-server API, no new dependency),
materially easier than HumanEval+, and three more genuine upstream bugs found
and excluded · **P**

**Decision:** `cbs.tasks.families.mbppplus` vendors `evalplus/mbppplus` (378
problems, Apache 2.0; `data/vendored/mbppplus/ATTRIBUTION.md`) — the D-29
upgrade, same motivation as HumanEval+ (D-27/D-34): the original's test
suites under-specify correctness.

**Fetched differently than every other family so far.** MBPP+ ships no plain
JSONL, only a parquet file — reading it directly would have meant adding
`pandas`/`pyarrow` as a new project dependency for a one-time data fetch.
Used the HF *datasets-server* rows API instead (`datasets-server.huggingface.co/rows`,
paginated at 100 rows/request, 4 requests total), which returns the identical
data as plain JSON. The vendored file is that JSON, concatenated — not a live
dependency at import time.

**Materially easier to integrate than HumanEval+ turned out to be, confirmed
before assuming it**: each row retains the *original* small `test_list`
(the same handful of flat asserts plain `mbpp` already uses) alongside the
new, much larger `test` field. `public_tests` derivation therefore reuses
`cbs.tasks.families.mbpp`'s existing mechanism completely unchanged — no new
AST logic was needed the way HumanEval+'s `inputs`/`results`-inside-`check()`
shape required (D-34). The expanded hidden `test` field also calls the
candidate by its real entry-point name directly, matching plain MBPP's own
convention (HumanEval+ aliases to a generic `candidate` instead).

**Three more genuine upstream bugs found by the same validation discipline as
every other family** — running every reference solution against the real
sandbox, not assuming vendored "official" data is correct by default:

1. **`Mbpp/590` ("polar_rect")** — evalplus's `is_floats()` helper does not
   recognise a tuple mixing a tuple-of-floats *and* a complex number as
   "float-ish," so the tolerance (`atol`) it would otherwise apply stays `0`
   and the check falls through to exact tuple equality. `cmath.polar`'s
   result differs from the vendored expected value in the last few
   significant digits — ordinary cross-platform floating-point
   non-reproducibility for a transcendental function, not a logic error —
   and exact equality has no tolerance for that. A real, narrow gap in
   evalplus's own tolerance-detection helper.
2. **`Mbpp/737`, `Mbpp/787`, `Mbpp/794`** (all three wrap `re.search`,
   returning `Match | None`) — confirmed by parsing **every** task's
   `assertion()` function and checking for an `ast.Assert` node anywhere in
   it, a full scan rather than trusting the handful a random sample happened
   to catch: these three compute `exact_match = exp == (out is not None)`
   and then **never assert it**. The function silently returns `None`
   unconditionally — it verifies nothing, and a candidate that always
   returns `None` regardless of input passes trivially. This is a more
   severe defect than a precision gap: the test doesn't just occasionally
   misfire, it never checks anything at all. Confirmed empirically (a
   `return None` stub passes) before concluding it, not inferred from
   reading the source alone.

All four excluded via `KNOWN_BROKEN_TASK_IDS` (default on, documented,
reversible with `exclude_known_broken=False`), same posture as D-34 — the
vendored copy stays a faithful, unmodified reproduction of evalplus's file.

**One more finding that is not a bug**: `Mbpp/599` ("sum_average") timed out
at the family's default 20s — not because it's wrong, but because its
reference solution computes `sum(range(1, number+1))` in pure Python against
evalplus stress-test inputs up to ~10⁸, measured at ~25.5s for the full test
at a generous timeout. Fixed with a per-task `TIMEOUT_OVERRIDES` entry (45s)
rather than raising the default for all 378 tasks, which would slow the whole
suite down to accommodate one outlier.

**Validated the same way as every other real family**: all 374 (of 378, 4
excluded) reference solutions pass the real sandbox; every derived public
subset passes its own task's reference solution; a sample of "return None"
probes are all correctly rejected outside the three now-excluded tasks.

---

## Still open

| # | Decision | Status | Needed by |
|---|---|---|---|
| D-12 | Primary `S_evo` fork — **researched, `metauto-ai/HGM` recommended over the brief's stated default**, awaiting confirmation (see write-up above) | **D** | Phase 4 execution |
| D-13 | LiveCodeBench (investigated, real scope now known — needs sandbox stdin support + a new I/O-judge verification path, not just a loader; D-33) and SWE-bench Verified remain open. HumanEval→HumanEval+ (D-34) and MBPP→MBPP+ (D-35) are **done**. **note:** whichever fork is chosen likely makes SWE-bench Verified load-bearing, not optional (D-12) | **D** | before real capability claims |
| D-31 | Adapt `cbs`'s simple function-completion tasks to DGM/HGM's git-repo-based agent (D-12's option (a)), vs. letting `S_evo` evolve natively against SWE-bench Verified/Polyglot and reserving `humaneval`/`mbpp`/`transfer_reasoning` for `S0`/`S_star` only (option (b)) | **D** | Phase 4 execution, alongside D-12 |
| D-14 | Exact `N_max`, the `k`/`K` reliability threshold, and the interpretation matrix's three placement thresholds — **fully-reasoned recommendations now in `preregistration.md` §3 (D-32), awaiting sign-off**, not bare placeholders | **D** | before Phase 5 |
| D-15 | Whether to add a third model family | **D** | Phase 3 (done otherwise) |
| D-16 | Per-phase budget caps in dollars / GPU-hours | **D** | before first paid run |
| D-17 | ~~Which reasoning set is the transfer family~~ — resolved: `transfer_reasoning`, D-30 | **C** | — |
| D-23 | Provision a Linux host with both GPU and Docker | **D** | Phase 4 execution (blocking) |

D-14 in particular must be fixed in `preregistration.md` **before** the full run,
not after seeing results. D-23 is the practical blocker on actually *running*
Phase 4/5 — its measurement layer is built and tested (D-24 through D-26), but
has nothing safe to run against without it.
