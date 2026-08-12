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

## D-12 — `metauto-ai/HGM` as primary `S_evo`, `facebookresearch/HyperAgents` as a confirmed second variant, `jennyzzt/dgm` kept as literature baseline only · **C (confirmed by user, 2026-08-04)**

**Researched** (GitHub API metadata + direct source inspection, both repos'
`README.md`, `coding_agent.py`, `llm.py`, `llm_withtools.py`, top-level file
listings, `tools/`, `swe_bench/`), then **confirmed by the user, 2026-08-04**
(see the update near the end of this entry). Not yet acted on beyond that —
forking a repo is a bigger commitment than most decisions in this log, and
the actual fork/clone is real engineering work still to do, not something
a confirmation completes by itself.

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

**Update, 2026-08-01 — (b) scoped concretely by shallow-cloning `metauto-ai/HGM`
and reading the actual source**, rather than continuing to guess at its shape.
Findings that change the cost estimate:

- HGM ships **complete, working harnesses for both benchmarks already** —
  `swe_bench/harness.py` and `polyglot/harness.py`, plus a `Dockerfile` at
  repo root. Nothing needs to be built to get SWE-bench Verified or Polyglot
  running against `S_evo`; `evaluate_agent.py --split Verified` is a real,
  already-supported invocation (the `--split` arg is passed straight to
  `load_dataset(f'princeton-nlp/SWE-bench_{split}')`).
- **Both interception points D-24's `InterceptionSession` needs already exist
  as single, well-defined choke points** — confirmed by reading the code, not
  inferred: every model call in the loop funnels through
  `llm.py:get_response_from_llm`/`create_client` (the exact function D-12
  already identified as needing the ~20-30 line local-endpoint patch), and
  every verification funnels through `hgm_utils.eval_agent` →
  `swe_bench.harness.harness(...)` (Docker-per-instance, one call site). This
  is exactly the shape `InterceptingModelClient`/`InterceptingVerifier` were
  designed to wrap — no redesign of D-24/D-25 needed, just pointing them at
  a different pair of call sites than the synthetic `example_agents.py` ones
  they were validated against.
- **What (b) actually costs, concretely, is now:** (i) the `llm.py` local-
  endpoint patch (already scoped, ~20-30 lines); (ii) a thin wrapper that
  monkeypatches/subclasses HGM's `create_client`/`eval_agent` call sites to
  route through `cbs`'s interceptors instead of calling straight through;
  (iii) a `cbs` task-family-shaped **wrapper**, not a reimplementation, around
  SWE-bench Verified for `S0`/`S_star`'s own frontier estimation, so all three
  scaffolds are measured against the same substrate — likely reusing HGM's own
  `swe_bench/harness.py` directly rather than porting SWE-bench Verified into
  `cbs.tasks.schema.Task` from scratch, since re-deriving a from-scratch
  assert-based verifier for SWE-bench (a git-diff-and-run-tests benchmark) is
  exactly the kind of reinvention D-33 already decided against for
  LiveCodeBench.
- **Recommendation, given the best-paper ambition (§ this session):** (b),
  not (a). It's both more faithful to each fork's published methodology (a
  reviewer-legible strength, per the earlier NeurIPS discussion) and, now that
  it's actually been scoped, *not* the heavier of the two options — (a) still
  requires inventing a translation layer between two task representations
  that were never designed to correspond; (b) reuses working harness code
  directly. The one real cost (b) still has that (a) wouldn't: Docker-per-
  instance verification is unavoidable and couples this tightly to D-23 —
  but D-23 is already a hard requirement for running `S_evo` at all (D-23),
  so (b) adds no *new* infrastructure dependency, just uses the one already
  required more thoroughly.

**Still not a final decision** — recorded here as a concrete scoping so the
choice is "confirm (b)" rather than "figure out what (b) even costs," which
is what this update resolves.

### Comparison

**Updated 2026-08-01 with a third candidate, `facebookresearch/HyperAgents`,
surfaced by the D-36 literature check — it is DGM's own original author
(`jennyzzt`) extending DGM herself, alongside Jeff Clune and Jakob Foerster,
so it needed weighing on the same footing as HGM, not left as a side note.**

| | `jennyzzt/dgm` | `metauto-ai/HGM` | `facebookresearch/HyperAgents` |
|---|---|---|---|
| License | Apache-2.0 | Apache-2.0 | **CC BY-NC-SA 4.0 — non-commercial, share-alike** |
| Stars | 2204 | 405 | 2655 |
| Last push (as of 2026-08-01) | 2025-08-13 — ~1 year stale | 2026-02-07 | 2026-07-31 — updated yesterday |
| Open issues | 26 | 5 | 29 |
| Venue | arXiv 2505.22954 | ICLR 2026 oral (arXiv 2510.21614) | arXiv 2603.19461 (not yet at a named venue) |
| Core agent/model/task code | original | same code, inherited from DGM | same lineage, restructured so the meta-level modification procedure is itself editable |
| Selection mechanism | flat archive; score-proportional / best / random parent choice | clade/subtree promise estimation (the paper's actual contribution) | "hyperagents" — task agent + meta agent fused into one self-referential editable program; meta-level procedure evolves too |
| Stated own limitation | — | — | authors themselves note "evaluation protocols remain fixed" — i.e. still no elicitation-vs-expansion test, same gap this project fills |

**The license difference is the one new fact that actually matters for a
decision, not just novelty tracking.** `HyperAgents` is CC BY-NC-SA 4.0, not
Apache-2.0 — academic research use is fine, but: (a) any redistributed
derivative touching its code must carry the same non-commercial/share-alike
terms, which would force part of this project's own codebase to inherit that
license if `EvolvedScaffold` code merges with it rather than calling it as a
separate process; (b) CC licenses are designed for creative works, not
software, and are a known source of ambiguity when applied to code (unclear
how "share-alike" interacts with e.g. a paper's supplementary-material
release). Neither DGM nor HGM has this complication.

### Recommendation

Still **`metauto-ai/HGM`** as the primary fork — the reasoning stands
(actively maintained, more interesting selection mechanism, zero extra
integration cost vs. DGM since the integration-relevant code is identical
across all three). **`HyperAgents` is a strong second `S_evo` variant to run
in addition, not instead of HGM**, if the best-paper ambition holds:its
"meta-level procedure is itself editable" framing is a genuinely different
selection mechanism, its own authors already admit the elicitation-vs-
expansion gap this project targets, and running the crossing test across two
independently-implemented mechanisms (HGM's clade/subtree selection vs.
HyperAgents' self-referential meta-editing) is a materially stronger result
than one implementation's idiosyncrasy — provided the license's non-commercial
term is acceptable (it should be, for a research paper) and any HyperAgents-
touching code is kept in a clearly separate, CC-BY-NC-SA-licensed module
rather than merged into the rest of `cbs`. Keep `jennyzzt/dgm` available too,
per the brief's own request for a literature-baseline reference (§5.1).

**Confirmed by the user, 2026-08-04.** Scope going forward: `metauto-ai/HGM`
is the primary `S_evo`; `facebookresearch/HyperAgents` runs as a second,
independently-implemented `S_evo` variant (its code kept in its own
CC-BY-NC-SA-licensed module, never merged into the rest of `cbs`, per the
license reasoning above); `jennyzzt/dgm` is kept available only as the
brief's requested literature-baseline reference (§5.1), not run as a third
`S_evo`. D-31 (option (b), native SWE-bench Verified/Polyglot, not option
(a)'s task-adaptation layer) was confirmed in the same round. Neither fork
has actually been cloned into this repo yet — that's real engineering work
for once a host is re-provisioned (D-37's Lambda instance was terminated),
not something this confirmation completes on its own.

---

## D-32 — preregistration thresholds locked, signed off by the user ·
**C (confirmed by user, 2026-08-04)**

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

**Update — signed off by the user, 2026-08-04.** All ten rows in §3's table
are now locked in `preregistration.md` (status changed from `[TO FIX]` to
**locked**, including the two that changed value: `k/K` = 6/10, "large"
overfitting gap = 0.2; the rest kept their original recommended values with
fuller reasoning attached). Per this project's own preregistration
discipline (§0 of `preregistration.md`), **these values must not be revisited
after seeing any real Phase 4/5 results** — that is the entire point of
fixing them now, while genuinely blind to outcomes. `preregistration.md` §4
(models and task families) remains open, but that is unbuilt engineering
work (the SWE-bench Verified/Polyglot integration, now scoped concretely by
D-31's confirmation), not an undecided judgment call — the two are
deliberately kept distinct in the file so "done" isn't overclaimed.

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

## D-36 — novelty check against current self-improvement/elicitation
literature, prompted by a stated NeurIPS/best-paper ambition · **P (partial
check, not exhaustive — see caveats)**

**Why this exists:** the user stated the actual goal is not just "a working
instrument" but the strongest possible paper, explicitly aiming at NeurIPS and
naming a best-paper ambition. That raises the bar on novelty diligence enough
to warrant checking the current literature *before* committing to the
expensive infra path (D-23), rather than discovering an overlap after paying
for GPU-hours. This was a web-search + abstract/full-text-fetch pass, not a
systematic citation-graph review — treat it as a first pass that materially
changes confidence, not a final clearance.

**Method:** searched for recent (2025-2026) work on self-improving coding
agents and on capability-elicitation-gap measurement, then read abstracts (and,
where fetchable, full text via the ar5iv HTML mirror — raw arXiv PDFs failed
to parse as text through `WebFetch`) for the closest seven hits.

**What was checked and where each one sits relative to this project:**

1. **SICA** (Robeyns, Szummer, Aitchison — arXiv 2504.15228, submitted as a
   NeurIPS 2025 preprint). Structurally the closest match: a frozen model
   wrapped by a self-editing scaffold, gains from 17%→53% on a SWE-bench
   Verified subset. Confirmed by full-text read: **no frozen-model baseline
   is independently estimated** (no repeated sampling / best-of-N / pass@k for
   the base model alone), **no elicitation-vs-genuine-capability distinction is
   tested or argued**, no preregistration, no contamination discussion. It
   reports a performance delta and calls it self-improvement; it does not ask
   or answer whether that delta reflects a wider reachable frontier. This is
   the paper this project's contribution most needs to differentiate itself
   from in a related-work section, and the gap is real: this project's whole
   instrument is built to answer the question SICA's own methodology cannot.
2. **Hyperagents / DGM-Hyperagents** (Zhang, Zhao, Foerster, Clune, et al. —
   arXiv 2603.19461, `facebookresearch/Hyperagents`, open source). **This is
   DGM's own original author (`jennyzzt`) extending DGM herself**, plus Jeff
   Clune (open-endedness) and Jakob Foerster — i.e., the source lab is already
   moving past plain DGM. Makes the meta-level modification procedure itself
   editable ("hyperagents"), evaluated across coding/paper-review/robotics/math
   grading, 5 seeds, frozen FM confirmed explicitly in-text. Full-text read:
   **same gap as SICA** — no elicitation-vs-expansion test, no independent
   frontier estimate of the frozen model, explicitly lists "evaluation
   protocols remain fixed" as a stated limitation (i.e., they know their own
   evaluation methodology doesn't do this). **Directly relevant to D-12**: the
   fork landscape now has a third real option (DGM → HGM → Hyperagents), and
   this needs folding into that decision, not just novelty tracking (see
   updated "Still open" row below).
3. **The Red Queen Gödel Machine** (Iacob et al., Cambridge/NVIDIA-affiliated,
   arXiv 2606.26294, "preliminary preprint"). Different core question entirely
   — co-evolving the *evaluator*/utility function under non-stationary
   objectives (paper review, Olympiad grading), not capability attribution.
   Low direct overlap; full PDF text wasn't machine-readable so this rests on
   the abstract only.
4. **Meta-Agent Challenge** (Lu et al., Ant Research, arXiv 2606.04455). A
   benchmark for meta-agents building agent artifacts from scratch against a
   held-out test set, focused on reward-hacking robustness and whether
   meta-agents match human-engineered baselines. Adjacent territory
   ("empirical proxy for evaluating recursive self-improvement") but a
   different specific question; no elicitation-vs-expansion framing found.
5. **DemoEvolve** (Che et al., arXiv 2605.24539) — demonstration-bootstrapped
   harness evolution in sparse-feedback long-horizon game environments (Liar's
   Dice, Balatro). Different domain, low overlap.
6. **SIA** (Hebbar et al., arXiv 2605.27276) — explicitly updates *both*
   harness and model weights, bridging what it calls the "harness-update" and
   "test-time-training" schools. Not a frozen-model system at all, so outside
   this project's premise, but useful as a related-work citation for *why*
   this project deliberately keeps `M` frozen (isolating scaffold effects
   requires it; SIA is the demonstration of what you lose — clean
   attribution — by not doing so).
7. **"Scaffold Effects on GAIA: A Controlled Comparison"** (Starace, arXiv
   2606.08529) — not a self-improvement paper at all, but the closest match
   found for this project's *methodological* lineage: a preregistered,
   controlled comparison of three **fixed** scaffolds across five frontier
   models, explicitly quantifying the "elicitation gap" (up to 28 points from
   scaffold choice alone) and explicitly rejecting its own preregistered
   hypothesis that more-capable models are less scaffold-sensitive. This
   confirms the elicitation-gap framing, preregistration discipline, and
   controlled-comparison rigor this project already uses are the current bar
   in this space, not overkill — solo-authored, not yet at a named venue,
   submitted one week before this check was run. Its scaffolds are static;
   this project's is an *evolving* scaffold. That's the structural difference.
   Also worth checking: METR's and Apollo Research's elicitation-gap /
   time-horizon work (found via search, not yet read in full) is the
   established source of the "elicitation is a lower bound" framing and the
   best-of-N-style methodology this project's frontier estimator extends —
   strong citable prior art for *why* the statistical machinery (Clopper-
   Pearson, Good-Turing, Chao1) is the right tool, not a competing claim.

**Net read:** the specific intersection this project occupies — apply
elicitation-gap-style frontier estimation (established in the evals/safety
literature) to the question of whether an *evolving* (DGM-style, not static)
scaffold crosses that frontier, with a causally-verified (interception-based,
not self-reported) attribution of which operations are support-preserving vs
support-expanding, under preregistration — does not appear to be already
published as of this check. Every self-improvement paper found reports
performance deltas without isolating the base model's independent ceiling;
every elicitation-gap paper found studies static scaffolds, not an evolving
one. That gap looks real, not assumed.

**Caveats, stated plainly rather than buried (original pass):**
- This was 7 papers via search + abstract/full-text pass, not a systematic
  citation-graph search (no Semantic Scholar backward/forward citation check
  on DGM/HGM/SICA, no direct search of NeurIPS/ICML/ICLR 2026 accepted-papers
  lists, no check of the METR/Apollo elicitation literature's full text).
- Full PDF text extraction failed for two papers (RQGM, and the initial
  attempt on SICA) via direct `WebFetch`; the ar5iv HTML mirror worked and
  should be preferred over raw arXiv PDF links for any future check like this.

**Update — proper citation-graph pass done (Semantic Scholar API, not
keyword search), plus full-text read of METR/Apollo, per the caveats
above.** Full write-up now lives in
`docs/self-improving-agents-proposal.md` §10 rather than duplicated here;
summary of what changed:

- Pulled the actual citing-paper lists for DGM (arXiv:2505.22954, 100
  results), HGM (arXiv:2510.21614, 21 results), and SICA (arXiv:2504.15228,
  33 results) directly from Semantic Scholar's API — a structurally
  different, more complete method than keyword search, and it found
  something keyword search had missed entirely.
- **The most substantive find of either pass**: a 3-paper preregistered
  cluster (İşcan et al. — arXiv 2607.26117, 2606.31511, 2607.12962) doing
  placebo-controlled decomposition of execution-feedback self-repair vs.
  blind resampling in frozen small code models (0.5B-7B), with real
  statistical rigor (Holm-corrected McNemar tests, a priori power analysis,
  hashed seed namespaces). Read in full, not abstracts. Their finding:
  feedback's value over a content-free placebo is statistically
  indistinguishable from zero at these scales. This is the closest any
  paper found has come to this project's actual RQ2/RQ3 comparison — but it
  studies a **fixed, static retry loop** (one paper explicitly cites DGM and
  frames itself as *contrasting* with "the broader agenda of agents that
  learn from their own histories" — the authors' own scoping, not an
  inference), and **has no frontier-estimation or frontier-crossing
  concept** (their "dead" unit is zero-passes-in-8-cached-samples, explicitly
  caveated as not a claim about the true model distribution). Both gaps are
  exactly where this project's contribution sits. Genuinely useful beyond
  novelty-clearing too: their result is a strong prior for what to expect
  from this project's own `S_star` (if execution feedback shows ~no gain
  there either, that corroborates rather than surprises), making a genuine
  `S_evo` crossing via a support-expanding operation a fixed retry loop never
  had access to a materially sharper contrast to draw in the write-up.
- METR (arXiv 2503.14499) and Apollo Research (arXiv 2502.15850) now read
  in full: confirmed neither has a formal elicitation-vs-expansion
  framework or frontier-crossing concept — METR's "elicitation" is a
  scaffold-selection step, not a decomposed, measured quantity; its CIs
  bound the trend slope, not any individual capability estimate. Strong
  citable grounding for the general framing, not a competing methodology.
- New "still open" item this pass surfaced rather than closed: a forward-
  citation check on the İşcan cluster itself (it's recent enough — mid-2026 —
  that little has had time to cite it yet, but that will change), and a
  check of whatever's been posted since this pass was run.

**Update — forward-citation check on the İşcan cluster done** (Semantic
Scholar API, one query per arXiv ID: 2607.26117, 2606.31511, 2607.12962).
Two of the three papers have zero forward citations as of this check.
The third (2606.31511) has exactly one: a survey, "Recursive
Self-Improvement in AI: From Bounded Self-Refinement to Autonomous
Research Loops" (arXiv 2607.07663, Chen/Wang/Qu), which taxonomizes 1,250
self-improvement papers (2024-2026) along two axes (what's improved;
degree of loop closure) rather than reporting new experiments. Read its
full text (ar5iv HTML, not just the abstract) specifically checking
whether it already stakes out this project's intersection: it names DGM
explicitly as an example of open-ended RSI ("maintaining an open-ended
archive of self-modifications validated against coding benchmarks") and
separately discusses elicitation as amplifying latent capability rather
than creating it, citing the same general elicitation-gap literature this
project already leans on — but it never combines the two into a measured
claim. No mention of "frontier crossing," "capability frontier," or any
empirical test of whether a DGM-style evolving scaffold's gains survive
against the base model's own ceiling; it's explicitly a taxonomy/survey,
not an empirical instrument. Net effect: the forward-citation check
surfaces one paper that confirms this project's two building blocks (DGM,
elicitation-gap framing) are both on other researchers' radar as of
mid-2026, but nobody has yet joined them into this project's actual
question. Gap still holds; nothing here narrows it.

**A real NeurIPS-quality related-work section still needs one more pass
before submission** (systematic recency check closer to submission date —
forward citations checked now, but this field moves fast enough that a
paper posted the week before submission could still land in the gap).
Treat this entry as "confidence materially improved three times now, not
cleared" — same discipline as before, not a license to stop checking.

**Update, 2026-08-06 — an informal recency sweep (plain web search, not the
Semantic Scholar citation-graph method above; treat as a lighter-weight
check than the passes above, not a replacement for the "one more
systematic pass before submission" this entry already calls for) surfaced
two new papers in the gap, both close enough to read carefully before
submission, neither read in full yet — abstracts/summaries only so far:**

- **EvoAgentBench** (Gao et al., arXiv 2607.05202, submitted 2026-07-06) —
  benchmarks agent self-evolution via "Ability Transfer": extracts
  trace-grounded Abilities from agent executions, canonicalizes them, and
  builds domain-specific Ability Graphs linking tasks with procedural
  overlap, across four domains (web research, algorithmic reasoning,
  software engineering, knowledge work). On the surface this is the
  closest-sounding title found in any pass so far — worth a full-text read,
  not just the abstract, before trusting the following read: its stated
  question (does curated ability content transfer across model families
  under an automatic method) is about *transfer efficiency of accumulated
  procedures*, not this project's question (does the frozen model's own
  reachable frontier move, decomposed from elicitation/overfitting, under
  causally-verified attribution). No mention found of an independent
  frozen-baseline ceiling estimate, matched-compute elicitation control, or
  interception-based (vs. self-reported) operation tagging in what's been
  read so far.
- **SEA-Eval** (arXiv 2604.08988, v1 2026-04-10, v3 2026-05-24) — "the
  first benchmark designed specifically for evaluating Self-Evolving
  Agents," an "Evolutionary Flywheel" architecture, using success rate and
  token consumption as primary metrics across sequential task streams, with
  a headline finding that token cost varies up to 31.2× between frameworks
  at matched success rate. This is adjacent on efficiency/cost-of-evolution
  grounds, not on the elicitation-vs-expansion attribution question — no
  mention found (abstract/summary level only) of frontier estimation,
  preregistration, or causal ablation of specific operations.
- Both are recent enough (April and July 2026) that neither could have
  appeared in the Semantic Scholar citation-graph pass above, which is
  exactly the kind of gap a field-moving-fast recency check is supposed to
  catch. **Neither, on this pass, appears to close this project's specific
  gap** — both are about different axes of "self-evolving agent
  evaluation" (transfer efficiency; cost-of-evolution dynamics) than
  elicitation-vs-genuine-expansion attribution under a frozen model — but
  this is a lighter-weight check than the standard this entry otherwise
  holds itself to, and both deserve a full-text read (not just
  search-summary text) before the related-work section is finalized, ideally
  as part of the "one more systematic pass" already called for above, not
  instead of it.
- Also confirmed, useful for D-12's own framing as much as novelty tracking:
  **HGM is not just published but an ICLR 2026 *oral*** (top tier within an
  already-selective venue), and DGM itself is an ICLR 2026 conference paper
  — both now confirmed accepted, not just arXiv preprints, which raises
  the bar for how sharply this project's related-work section needs to
  differentiate from HGM's own claims specifically, since it is the
  best-credentialed, most likely-to-be-known-by-reviewers prior work in
  this exact space.

**Update, 2026-08-10 — HGM's baseline structure verified against its own
full text, not inferred.** Because the workshop paper (D-45) makes a claim
about what prior work does *not* control for, the closest and most
load-bearing case was checked directly rather than left as a general
assertion. Read the full HTML of arXiv 2510.21614 specifically looking for
(a) an independent frozen-backbone ceiling estimate via repeated sampling /
pass@k / best-of-N, (b) a fixed hand-authored scaffold control at matched
compute, (c) any explicit elicitation-vs-expansion distinction. Result:
none of the three is present. HGM's evaluation compares **three
evolutionary methods** (SICA, DGM, HGM) from an identical initial agent;
its only human-engineered reference is a leaderboard comparison against
SWE-agent, which uses a different backbone at an unmatched compute budget.
So there is no condition anywhere in that paper where the frozen backbone
is measured *without* an evolving scaffold. The gain is real; what it is a
gain *over* is another moving scaffold, not a characterised model ceiling.
This is now stated in the workshop paper's related-work section as a
specific, checkable claim rather than a general one — which is both more
useful to a reviewer and more falsifiable if wrong.

---

## D-37 — D-23 resolved: real Docker+GPU host provisioned, HGM+vLLM+Docker
confirmed working end-to-end · **C**

**Host:** Lambda Labs on-demand instance (chosen over RunPod after discovering
RunPod's shared multi-tenant Pods explicitly disable privileged/nested-Docker
mode for security — confirmed by the user, not just documentation; RunPod's
Bare Metal tier is sales-only, not self-serve, so not a viable fast path
either). Lambda gives a real VM, not a shared container, so Docker works
normally with no privileged-mode fight. 1x A10 (24GB VRAM), Ubuntu 22.04.5 +
Lambda Stack, CUDA 12.8, persistent NFS filesystem mounted at
`/lambda/nfs/cbs-project` for anything that must survive instance termination
(the instance's local disk does not survive termination; the persistent
filesystem does, at its own separate ~$0.20/GiB/month regardless of whether
an instance is attached).

**What's confirmed working, in order:**
1. Docker with GPU passthrough (`docker run --gpus all` sees the A10).
2. vLLM 0.26.0 serving `Qwen/Qwen2.5-Coder-7B-Instruct` via its OpenAI-
   compatible endpoint (`--gpu-memory-utilization 0.85 --max-model-len
   16384`, comfortably inside 24GB) — confirmed with a real generation, not
   just a health check.
3. **HGM already supports a local vLLM endpoint with no code changes** —
   `llm.py`'s `create_client` has an `elif "vllm" in model.lower()` branch
   (`base_url=f"http://{model[11:]}:8000/v1"`) in both the top-level `llm.py`
   *and* the copy bundled with `best_agent/`. Model strings of the form
   `vllm-model:<host>` route correctly (`model[11:]` slices off exactly the
   11-character `"vllm-model:"` prefix). **This corrects D-12's original
   estimate of a ~20-30 line patch to `llm.py` — that estimate was made
   without cloning the repo and reading current source; HGM had already
   added this itself.** `config.yaml`'s `self_improve_llm`/`downstream_llm`/
   `diagnose_llm` all point at `vllm-model:localhost` — preserving the
   single-frozen-model requirement, since HGM's own default already used one
   model for all three roles.
4. A real end-to-end run, on the *second* attempt (see the correction
   directly below — the first attempt's apparent success was wrong, and
   catching that mattered): HGM built a Docker image for `default_agent`,
   ran it in a container against one Polyglot task, the containerized agent
   called the locally-served model, and the model produced a real,
   substantive response (670 completion tokens of genuine reasoning about
   the task, `tool_calls=None` — no crash, no connection error, no
   malformed-response error). The result was copied back to the host and a
   real evaluation report was written (1 submitted, 1 completed, 0 resolved —
   an empty patch, because the model chose to respond conversationally
   rather than invoke a tool on this particular trial, not because anything
   in the pipeline broke). **This is the brief's Phase 0 DoD** ("vLLM serves
   a frozen model; the forked loop runs one tiny task end-to-end") **met for
   real.** Whether a baseline agent reliably invokes tools/solves tasks is a
   question about agent behavior — squarely what the actual study measures —
   not something to keep "fixing" at the infrastructure level.

**A claimed success that was actually wrong, caught only because it was
checked rather than trusted — worth recording prominently, not quietly
overwritten, because catching it is the same discipline this project
already applies everywhere else (README/CLAUDE.md: "validate before building
on top of something").** The *first* smoke test attempt looked like a clean
success: it completed without crashing, produced a real evaluation report,
and an in-container log line read `"Using vllm API with model
vllm-model:localhost."`. That log line only proves the client was
*configured* to attempt the call — not that the call *succeeded* — and this
was initially misreported as confirmed end-to-end success on that basis
alone. Checking the actual agent output file (not just the summary
statistics) revealed the real content: `Error in get_response_withtools:
Connection error.`, repeated five times. The model was never actually
reached; the "empty patch" result was a masked total failure, not a benign
non-solve. Two real, separate bugs were hiding behind that one misleading
log line:

- **Docker's default bridge network does not let a container reach the
  host's `localhost`.** Confirmed empirically (a throwaway container hitting
  `http://localhost:8000/health` timed out — `HTTP_000`) before assuming a
  fix, then confirmed again after fixing it (same test against a
  `--network host` container returned `HTTP_200`). HGM's own container
  creation calls (`polyglot/docker_build.py` and, identically, the vendored
  `swe_bench/SWE-bench/swebench/harness/docker_build.py`) set no network mode
  at all, so containers got Docker's isolated default bridge network, in
  which the container's own `localhost` is not the host's. **Fix:** added
  `network_mode="host"` to both `client.containers.create(...)` calls
  (originals preserved as `.orig` alongside). This is a deliberate,
  documented relaxation of *network* isolation specifically to let the
  container reach a locally-served model — it does not weaken the
  filesystem/process isolation Docker provides for the untrusted candidate
  code itself, which is D-23's actual safety requirement. Worth revisiting
  if a future host runs genuinely untrusted network-facing services
  alongside this project, which the current single-user research instance
  does not.
- **vLLM needs explicit flags for OpenAI-style tool calling.** Once the
  network was fixed, the next real (non-connection-error) response was an
  HTTP 400: `"auto" tool choice requires --enable-auto-tool-choice and
  --tool-call-parser to be set`. HGM's agent code (`llm_withtools.py`) hard-
  requires a populated `response.choices[0].message.tool_calls`, so this is
  not optional. **Fix:** relaunched vLLM with `--enable-auto-tool-choice
  --tool-call-parser hermes` — confirmed correct by checking the model's own
  tokenizer chat template (fetched from its Hugging Face repo), which
  defines Qwen2.5's native tool-call format as `<tool_call>{"name": ...,
  "arguments": ...}</tool_call>` — the same tag convention `hermes` expects.
  One operational trap hit while restarting: killing the vLLM API server
  process (`pkill -f vllm.entrypoints.openai.api_server`) did **not** free
  the GPU — its child engine process (a separate PID, named
  `VLLM::EngineCore`, matching neither that pattern nor a `ps aux | grep
  vllm`-friendly name) kept holding all 19.7GB VRAM, and the restarted server
  failed with a GPU-memory error until that child was found via
  `nvidia-smi --query-compute-apps` and killed by explicit PID.

**Three further, separate real problems hit and fixed along the way, worth
recording so a future session doesn't rediscover them the hard way:**

1. **Never put a Python venv inside a directory a DGM-style fork treats as
   its own source tree.** HGM's `copy_src_files` (used to snapshot each
   archive node, `source_dir="."`  by default) copies *everything* not
   excluded by `.dockerignore`. A venv created inside the HGM repo directory
   is not excluded (HGM's own `.dockerignore` has no reason to expect one —
   its own convention is conda, living outside the repo entirely), so it gets
   swept into every node snapshot: one early attempt tried to copy over
   50,000 files / 7.6GB (all of vLLM/torch/CUDA's source trees) before being
   caught and killed. Fix: create the venv as a *sibling* directory
   (`/lambda/nfs/cbs-project/hgm_venv`, not `hgm/venv`), not inside the
   forked repo. This generalizes to whichever fork is ultimately used, not
   just HGM.
2. **The editable SWE-bench install has to be redone in whichever venv is
   actually active.** `pip install -e swe_bench/SWE-bench` was run once in
   the (wrong-location) venv before it was discovered and rebuilt elsewhere;
   forgetting to redo it in the new venv produced a `ModuleNotFoundError:
   swebench` crash inside `polyglot/test_spec.py` (Polyglot's own harness
   imports `swebench.harness.utils` — Polyglot depends on the SWE-bench
   *package*, not just its own benchmark data, confirming D-31's read that
   these two benchmarks share real infrastructure, not just superficial
   similarity).
3. **The initial baseline evaluation's task count is not controlled by
   `--max_task_evals`.** That flag only bounds the outer self-improvement
   loop's iteration count. The *initial* baseline agent is always evaluated
   against a fixed task list — for Polyglot, hardcoded in `hgm.py`'s `main()`
   as `medium.json + small.json` (50 + 10 = 60 tasks), not a CLI-configurable
   subset. A first "smoke test" attempt with `--max_task_evals 1` silently
   tried to Docker-evaluate the baseline agent against all 60 tasks
   sequentially (`--max_workers 1`) before the loop could even start —
   nothing was wrong, it just wasn't going to finish in a smoke-test-sized
   window. Fix used here: temporarily point `polyglot/subsets/medium.json`
   at a 1-task list (and `small.json` at an empty list), run, then restore
   the real files from backups — reversible, no permanent change to the real
   subset definitions. **For the real run, this hardcoding is the thing to
   budget: the "initial" baseline always costs 60 real Docker-evaluated
   Polyglot tasks (or the SWE-bench equivalent) up front, before any
   evolution happens, and that cost is fixed regardless of `--max_task_evals`
   or `N_max` tuning elsewhere.**

**One loose end, not yet resolved:** after the single-task evaluation
completed successfully, the outer loop crashed with `ValueError: attempt to
get argmax of an empty sequence` in `TS_sample`/`expand()` — HGM's
node-selection logic choosing which archive entry to expand next. This is
downstream of everything Phase 0's DoD requires and most likely an artifact
of deliberately shrinking the task list to 1 for the smoke test (the
promise-estimation statistics plausibly need more than one data point) rather
than a real bug — but it has **not** been reproduced or root-caused against
the real 60-task subset, and shouldn't be assumed fixed until it is. Flag for
whoever runs the first real (non-smoke-test) evolution step.

**Not yet done:** the actual measurement-layer bridge — routing HGM's model
calls and verification calls through `cbs.scaffolds.evolved.InterceptionSession`
so operations get tagged. This is real, non-trivial new code, not a
configuration change: `EvolvedScaffold.solve()`'s current `AgentFunction`
contract assumes one call per task, but HGM runs a long-lived archive search
evaluating batches of tasks per node — the bridge has to monkeypatch HGM's
own call sites (`llm.py`'s client functions; the SWE-bench/Polyglot harness
call in `hgm_utils.eval_agent`) directly and reconstruct one `OperationTrace`
per task afterward, rather than slotting an HGM agent into the existing
`AgentFunction` shape unchanged.

**Update, 2026-08-05 — instance re-provisioned after the user terminated the
original one; confirms the persistent-filesystem/local-disk split worked
exactly as documented, plus one new real setup gotcha on the fresh instance.**
New Lambda instance, same region-appropriate A10 GPU. Re-verified in order:
SSH access, `nvidia-smi` (A10 present, idle), Docker (same `permission
denied` on `docker.sock` as the *first* instance — `ubuntu` not yet in the
`docker` group on a fresh instance either; `sudo usermod -aG docker ubuntu`
fixes it every time, evidently not something Lambda Stack images do by
default), then `docker run --gpus all` (works). The persistent filesystem
(`/lambda/nfs/cbs-project`) reattached with everything from the prior
session intact — `hgm/` (network_mode="host" patches still in place,
confirmed via `.orig` backups sitting next to the patched files), `hgm_venv/`
(Python 3.10.12, vLLM 0.26.0), `cbs_pkg` — exactly as D-37's original
"local disk does not survive termination; the persistent filesystem does"
claim predicted. Docker's own image store, being local-disk state, was
correctly gone (`docker images` showed nothing but `hello-world` after a
fresh pull) — expected, not a problem, since HGM rebuilds its per-task agent
image on demand anyway.

**New gotcha, not seen on the first instance (or not noticed there — unclear
which):** relaunching vLLM failed with `FileNotFoundError: [Errno 2] No such
file or directory: 'ninja'` during KV-cache initialization (a JIT-compile
step vLLM's engine core runs internally). `ninja` **was** installed — `pip
show ninja` in `hgm_venv` confirmed 1.13.0, and the executable existed at
`hgm_venv/bin/ninja` — but that directory was never on `PATH`, because vLLM
was launched by calling `hgm_venv/bin/python -m vllm...` directly rather
than through an activated venv (`source hgm_venv/bin/activate`), and the
engine-core subprocess that shells out to `ninja` by bare name inherits the
system `PATH`, not the venv's. Fixed by prepending
`PATH=/lambda/nfs/cbs-project/hgm_venv/bin:$PATH` to the launch command
itself rather than relying on venv activation semantics that don't survive
into a subprocess tree spawned via SSH. Worth remembering for any future
re-provisioning: **never assume a venv's own console-script executables are
on `PATH` just because its `python` binary was invoked directly** — this
class of bug is invisible until something inside the process tree shells
out to a bare command name rather than importing a Python module.

**A second new gotcha, this one specific to reusing the persistent
filesystem for a task that was already run before termination:** re-running
`hgm_run_task_with_interception.py` against `python__dominoes` (already
tested in the prior session) failed at the Docker image-build step with
`PermissionError: [Errno 13] Permission denied` on paths under
`hgm/logs/build_images/instances/pb.eval.x86_64.python__dominoes__latest/…`
— specifically on copied `.git/objects/…` files carrying git's normal
read-only mode (`-r--r--r--`) from the prior session, dated before
termination. The image-build step tries to overwrite files at those same
paths on a rebuild, and overwriting a read-only file via a plain open-for-
write fails regardless of directory permissions (removing and recreating it
would not). Confirmed by checking the actual file mode
(`-r--r--r-- 1 ubuntu ubuntu … .git/objects/d7/1e6a80…`), not just inferred
from the error text. **Only affects re-running a task that was already
built once before an instance termination** — a fresh task (`python__two-
bucket`, never run before) built and ran cleanly on the same instance.
Fixed by deleting the stale `hgm/logs/build_images/instances/` directory
tree entirely (disposable build-log/context scratch space, not source data
or results — confirmed by reading what actually lives there before
deleting anything) and re-running; `python__dominoes` then completed
cleanly with the expected single support-preserving call, matching every
other real attempt so far. Worth remembering for any future re-provisioning
that repeats a previously-tested task: **clear `hgm/logs/build_images/
instances/` first**, or expect this exact permission error.

---

## D-38 — `cbs.scaffolds.fork_bridge`: network-layer interception for HGM,
validated against 7 real tasks, three real bugs found and fixed · **P**

**This corrects D-12/D-37's framing, not just extends it.** D-12 concluded
"no redesign of D-24/D-25 needed, just pointing them at a different pair of
call sites," and D-37 still described the remaining work as needing to
"monkeypatch HGM's own call sites (`llm.py`'s client functions...) directly."
Both assumed in-process interception. That assumption doesn't hold for this
fork's actual execution model — the agent runs as a separate OS process in a
container, so there is nothing in-process to monkeypatch — and the mechanism
built here (a network-layer reverse proxy) is not a variant of what either
D-12 or D-37 anticipated.

**What this is:** `EvolvedScaffold`/`InterceptionSession` (D-24/D-25) assume
the untrusted agent is an in-process Python callable. HGM's agent code runs
as a genuinely separate OS process inside a Docker container
(`container.exec_run(...)`), so there is no in-process object to wrap.
`cbs.scaffolds.fork_bridge` is the mechanism for that different situation:

- `ModelCallProxy` — a real HTTP reverse proxy sitting between the
  containerized agent and the actual model server, recording every
  `/v1/chat/completions` request/response pair it forwards. This is the only
  channel available for interception here, confirmed empirically (D-37): the
  container reaches the model over real HTTP, not an in-process call.
- `reconstruct_trace_from_events` — a pure function turning one task's
  captured conversation into a `cbs.scaffolds.tagging.OperationTrace`.
  Deliberately does not reuse `InterceptionSession._was_conditioned_on`: that
  heuristic resolves an ambiguity (blind selection vs. real conditioning)
  that doesn't arise here, because HGM's `coding_agent.py` runs one
  continuously-growing tool-use conversation per task, not N independent
  generate-then-select branches (confirmed by reading `llm_withtools.py`).
  Every tool round-trip is unambiguously `tool_call` (support-expanding);
  the existing registry entry already covers it without needing a new one.
- Verification (does the final patch solve the task) is computed by ordinary
  host-side Python (`polyglot.harness.process_entry`), not inside the
  container, so it needs no proxying — confirmed by reading the function in
  full: it runs the actual hidden test suite and returns `eval_result`
  directly (`"resolved"` = passed), so this bridge calls it directly rather
  than needing any separate verifier hook.

**A tool-count correction to D-12, found while writing this entry, worth
recording as an example of exactly the kind of mistake this project's own
"verify, don't assume" discipline exists to catch.** An earlier draft of
this entry stated HGM's tool surface as four tools (`bash`, `python_executor`,
`file_editor`, `ast_editor`), sourced from reading `best_agent/tools/*.py`
during an earlier exploratory pass. Checked again while finalizing this
entry: the agent actually executed in every validated run below copies its
tools from the **top-level** `hgm/tools/` directory into the container
(confirmed directly from the real run logs: `"Copying
measured_default_agent/src/tools to container at /hgm/tools"`), and that
directory has exactly **two** tools — `bash.py` and `edit.py` — matching
D-12's original "entire tool surface" claim exactly. `best_agent/tools/` is
a separate, unused-in-these-runs agent bundle with a larger toolset; citing
it was simply a wrong file to have checked, not a real discrepancy with
D-12. `edit.py` (view/create/edit files, no execution) is functionally the
same category as the `file_editor`/`ast_editor` tools the wrong draft
described, so the classification conclusion (both `bash` and `edit` map to
the existing `tool_call` registry entry, no new entry needed) is unaffected
— only the specific file names and count were wrong.

**Built test-first, validated in stages, not deployed blind:**
1. `ModelCallProxy` + `reconstruct_trace_from_events` unit-tested locally
   against a real (not mocked) backend HTTP server and hand-built
   OpenAI-shaped conversation histories (`tests/test_fork_bridge.py`, 14
   tests) — including the deliberately tricky case of a tool being
   *requested* but never actually run, which must NOT count as `tool_call`.
2. Deployed alongside HGM (`scripts/hgm_run_task_with_interception.py`,
   NOT part of the `cbs` package since it imports HGM's own modules
   directly) and run against real tasks, checked against ground truth by
   hand, not just trusted because the pieces were unit-tested.

**Three real bugs found running it for real, not three imagined ones:**

1. **A module-level global, not a function parameter.** `process_entry`
   takes no model-string argument at all — `hgm.py`'s own `main()` sets the
   actual model string as `polyglot.harness.llm` (a module global) before
   calling into the harness. My first real run skipped this, so it defaulted
   to `""`, `create_client` fell through to the OpenRouter branch, and the
   run produced `n_proxy_events: 0` with an empty patch — a silent, total
   failure to reach the model at all, not a benign non-solve. Caught by
   reading the actual log line (`"Using OpenRouter API with model ."`), not
   by trusting the JSON summary. Fixed by setting the global explicitly
   before calling `process_entry`, with the distinction from
   `model_name_or_path` (a label, not a model string) documented directly in
   the script's `--llm-model-string` help text so it isn't rediscovered.
2. **`process_entry` does not build Docker environment images itself** —
   that setup lives in `harness()`'s own batch orchestration
   (`build_env_images(...)`, called once before its `ThreadPoolExecutor`
   loop), which calling `process_entry` directly bypasses. First hit as
   `Error building image ...: Environment image ... not found` for a task
   whose image hadn't been built in an earlier session. Fixed by mirroring
   `harness()`'s own setup step in the script (calling `build_env_images`
   before the per-task loop), not working around the missing image.
3. **A real race condition in `ModelCallProxy` itself**, caught via the
   project's own re-verification discipline, not luck: a docstring-only edit
   triggered an unrelated intermittent test failure a few minutes after
   writing (and, ironically, right after documenting the very same race as
   "never observed in practice" in that docstring). The handler recorded a
   call *after* writing the response to the client, so a caller reading
   `.events` immediately after its own request returned could race ahead of
   the record — genuinely observed, not hypothetical. Fixed by recording
   before responding, which removes the race structurally rather than
   relying on timing to avoid it; confirmed by running the previously-flaky
   test 10 times consecutively (all clean) before trusting it, and by
   deploying the same fix to the remote instance's copy of the file (which
   is not live-synced from this repo — a separate `pip install -e` copy that
   would otherwise have kept the same latent bug). Independently
   re-confirmed still present and correctly deployed on the remote instance
   during review of this entry.

**Validated against 7 real Polyglot tasks** (`javascript__queen-attack`,
`java__sgf-parsing`, `javascript__robot-name`, `python__dominoes`,
`go__dominoes`, `cpp__all-your-base`, plus one earlier repeat) against the
`default_agent` baseline. One result cross-checked by hand against the raw
`.md` transcript: the bridge's classification (`single_call`,
`had_tool_calls: false`) matched the actual raw model response
byte-for-byte, not just a plausible-looking summary.

**Not yet validated: a real tool-call round trip.** All 7 real tasks so far
show the model making exactly one plain generation and stopping — never
invoking `bash`/`edit`, despite `tool_choice="auto"` and the tools being
correctly exposed (confirmed: `get_response_withtools`'s non-OpenAI branch —
the one `vllm-model:...` strings route through — correctly calls
`client.chat.completions.create(..., tools=tools, tool_choice="auto")`, and
`check_for_tool_use` correctly reads `response.choices[0].message.tool_calls`,
exactly matching vLLM's actual response shape; no code-path mismatch). This
looks like a real, plausible behavioural characteristic of a 7B baseline
model given only `tool_choice="auto"` (smaller models are well known to be
less reliable at spontaneous tool use), not a bug — but it has not been
confirmed by seeing a real positive case, only inferred by ruling out the
alternative explanations. The `tool_call` classification path itself remains
proven only against synthetic conversations, not real ones, until one
actually occurs.

**Operational note, not a correctness issue:** the validation runs above
left real (low-cost) clutter on the remote instance — ~27GB of Docker
images and ~20GB of build cache (mostly reclaimable, includes six leftover
per-task Polyglot eval images never pruned), three stopped `hello-world`
test containers, and a scatter of ad-hoc log files at the top level of
`/lambda/nfs/cbs-project/`. None of it is costing active compute (idle load,
no running containers, disk at 6% of 1.4TB), but it's worth a
`docker system prune` + log cleanup pass before the instance is handed to
the next session.

**Second review round found and fixed two real critical bugs, not just style
issues.** A four-dimension review (correctness / concurrency-and-resource-
safety / test-quality / integration-consistency), each finding independently
re-verified by a separate skeptic pass before being trusted, raised 34
findings; 30 confirmed, 3 overstated, 1 refuted. Two were rated critical and
both were real:

1. **`reset()`/`stop()` didn't wait for in-flight handler threads.** A
   straggling response to task N's last call — mid-flight between "backend
   responded" and "recorded" — could land in task N+1's event log after
   `reset()` had already run, silently contaminating the wrong task's trace.
   This directly violates the "one task at a time" serialization the whole
   design depends on (see above). **Fixed:** an in-flight counter
   (`_inflight`, guarded by the same lock as `_events`) that `reset()` now
   waits to hit zero before clearing, raising loudly (not proceeding
   silently) if a handler is stuck past a timeout — a corrupted trace should
   never pass quietly.
2. **`urlopen`'s exception handling only covered `HTTPError`/`URLError`.**
   A backend that was merely slow past the timeout, or that reset the
   connection mid-response, could raise other exception types that
   propagated uncaught — silently dropping that call from `.events` entirely,
   with no trace it was ever attempted. **Fixed:** widened to catch any
   exception during the backend call, always producing a synthesized error
   response rather than a silent gap.

A third, moderate-severity issue compounds with both: failed/errored calls
(from either bug above, or a genuine backend error) were being recorded and
reconstructed exactly like a successful generation, inflating `single_call`
counts with calls that never actually sampled from `M`. **Fixed:**
`ProxiedCall` now carries the HTTP `status`, and `reconstruct_trace_from_events`
only counts a `single_call` (and only advances its message-count bookkeeping)
for a genuine 2xx response with a real `choices[0].message` — a failed
attempt's message-history growth is correctly not double-counted on retry,
while any tool-role message already present in a failed attempt's *request*
is still correctly scanned (the tool genuinely ran regardless of whether that
follow-up generation then failed). Also fixed: no read timeout on the
client-facing socket (a stalled client could block a handler thread forever,
compounding with bug #1 above), and the `start()` double-start guard being a
check-then-act race without a lock (rewritten to build the server object
outside the lock, then atomically check-and-set, closing the redundant
socket if it lost the race).

12 new tests added covering all of the above (`tests/test_fork_bridge.py`,
now 26 tests total) — including a genuinely unreachable backend (not
mocked), a monkeypatched arbitrary exception to prove the broadened catch
actually works and not just `URLError`, a simulated stuck in-flight handler
to prove `reset()` actually raises rather than a hypothetical, and the
retry-after-failure sequencing. Full suite re-run 8 times consecutively with
no flakiness before trusting the concurrency-related fixes. Deployed to the
remote instance's copy and its import verified there, not just locally.

**Explicitly not fixed, by engineering judgment given this module's actual
deployment context (a single-user research instance, sequential one-task-
at-a-time usage, no adversarial network exposure, no streaming used by this
fork):** binding `0.0.0.0` instead of loopback; not forwarding arbitrary
request headers (e.g. `Authorization`) to the backend; the undocumented
assumption that responses are never SSE-streamed; a few redundant/weak test
assertions the review flagged as style issues. None of these affect the
correctness of any measurement this instrument produces; revisit if this
module is ever reused somewhere with a different threat model.

**Update, 2026-08-05 — the one remaining validation gap is closed: a real
tool-call round trip observed and correctly classified.** Every real attempt
before this (9 across two sessions/instances) showed the baseline agent
making one plain generation and stopping — plausible real 7B-model behavior
under `tool_choice="auto"`, but it meant `reconstruct_trace_from_events`'s
`tool_call`/support-expanding path had only ever been exercised against
synthetic conversations, never a real one. Rather than keep sampling
randomly and hoping, forced the issue directly: made an **isolated copy** of
the frozen agent variant (`hgm/toolcheck_agent_src/`, copied from
`measured_default_agent/src/`, never the canonical copy itself) and changed
`llm_withtools.py`'s two hardcoded `tool_choice="auto"` call sites to
`tool_choice="required"` — a deliberate, clearly-scoped validation
modification, not a change to any measurement pipeline or real experimental
data; kept in its own directory specifically so it can't be confused with
the real frozen baseline later.

Ran against `python__bowling` (a fresh task). Result: a real, substantive
64-round tool-using trajectory — `n_ops: 128` (64 `single_call` + 64
paired `tool_call`), `used_expanding: true`, `classes_used:
["support-expanding", "support-preserving"]`, `expanding_ops: ["tool_call"]`
— every field the crossing-determination logic (`cbs.crossing`) actually
reads, populated by a real model for the first time, not a hand-authored
test fixture. The produced patch was genuine, substantive code (a full
`BowlingGame` implementation, not a stub), passing the large majority of
the hidden test suite and failing exactly one edge case (a specific
exception-raising condition) — `eval_result: "unresolved"`, not
`"empty_patch"` like every prior real run, confirming this wasn't a
degenerate or garbled trajectory. `n_proxy_events: 68` vs. 64
`single_call` ops in the final trace is a small, plausible gap (a few
raw proxy events likely failed/errored and were correctly excluded by the
status-aware trace reconstruction from the second review round above,
consistent with what that fix is *for* — not independently confirmed
call-by-call, but consistent with everything else observed).

**This validates the classification logic end-to-end against real data for
the first time**, not just a bigger sample size — `used_expanding: true` had
never been produced by a real run before this, meaning `cbs.crossing`'s
core support-expanding-ablation branch had literally never been exercised
outside synthetic tests until now. The forced-`tool_choice` trial is
**validation-only and must not be conflated with real measurement data** —
`S0`/`S_star`/`S_evo` all need `tool_choice="auto"` (or whatever the fork's
own default is) to measure genuine, unforced agent behavior; this trial
exists purely to prove the instrument reads a real tool-call trajectory
correctly when one occurs; it says nothing about how often Qwen2.5-Coder-7B
would choose to use a tool unprompted on its own, which remains an open,
unresolved question about baseline agent behavior under this model — not
something this trial was designed to answer, and not something to mistake
it for having answered.

**Small real gap found and closed, next session (2026-08-06)**:
`scripts/hgm_run_task_with_interception.py` captured each task's trace via
`reconstruct_trace_from_events` but never turned the same proxy events into
a chargeable `Usage` via `fork_bridge.usage_from_events` — the exact
function D-40 built to close this — leaving this script's own output
without real token accounting even after that gap was closed elsewhere.
Fixed by wiring `usage_from_events` into the per-task record; also caught
that the remote instance's copy of this script had drifted (still an older
pre-D-38 docstring) and had never been re-synced after that validation
pass, fixed by re-copying the current local version wholesale rather than
patching around the drift. Re-validated against `javascript__queen-attack`
(the same task first used to hand-verify the trace-reconstruction logic):
real output now includes `usage: {"calls": 1, "prompt_tokens": 1239,
"completion_tokens": 652}`, sourced from the same proxied events already
used for the trace — same no-tool-use, single-generation pattern as every
other real run this session, `eval_result: "empty_patch"`, confirming the
fix adds real accounting without disturbing anything already working.
Also fixed, same pass: `cbs.scaffolds.evolved`'s module docstring still
claimed "there is currently nowhere in this project's infrastructure that
can safely run real self-modifying scaffold code" — true when written,
false since D-23/D-37/D-38 — corrected to point at `fork_bridge` as the
mechanism that actually exists now for a Docker-hosted fork, while noting
`EvolvedScaffold` itself remains the right (and still unused) adapter for
a hypothetical in-process `AgentFunction`, a different shape of integration
than any of HGM/HyperAgents.

---

## D-39 — HyperAgents integration cost, checked against real source now that
D-12 is confirmed: one real patch needed (not zero), Docker networking
already solved (unlike HGM) · **P**

D-12's confirmation was signed off before either fork was actually cloned
into anything — the comparison table's "same lineage" framing for
`HyperAgents` was accurate at the structural level (task/meta-agent split,
same DGM ancestry) but hadn't been checked at the code level the way HGM
was in D-37. With no host available, shallow-cloned both `metauto-ai/HGM`
and `facebookresearch/HyperAgents` locally (`git ls-remote`/`git clone
--depth 1` both work fine from this machine — no Docker, but plain GitHub
access is not blocked) to check that assumption before it turns into a
surprise once a host exists again. It only holds partially.

**Model-call chokepoint: still singular, as assumed.** Both `task_agent.py`
and `meta_agent.py` route every LLM call through
`agent/llm_withtools.py` → `agent/llm.py:get_response_from_llm` — same
shape as HGM's `llm.py:get_response_from_llm`. `fork_bridge.py`'s
network-layer `ModelCallProxy` approach (D-38) intercepts at the HTTP layer,
not by patching this function, so this should still work unmodified in
principle for HyperAgents too, provided the outbound call is a plain HTTP
request the proxy can sit in front of (see below — it is, via `litellm`).

**Local-endpoint patch: HyperAgents needs one; HGM, corrected in D-37,
needs none.** HGM's `create_client` already has a native
`elif "vllm" in model.lower()` branch. HyperAgents' `get_response_from_llm`
is a **different implementation**, not shared code — it calls
`litellm.completion(model=model, messages=..., ...)` with no `api_base`
handling anywhere and no vLLM-aware branch at all; `litellm` dispatches
purely off the `model` string's provider prefix (`openai/`, `anthropic/`,
`gemini/`, etc.). Pointing this at a local vLLM server needs a small, real
patch — passing `api_base=<url>` and a `hosted_vllm/<model>`-style prefix
into `completion_kwargs` when a designated local-model string is used
(litellm's own documented mechanism for self-hosted OpenAI-compatible
servers). Estimated at roughly the same size as the patch D-12 originally
(and wrongly) estimated for HGM before finding HGM didn't need it — on the
order of 10-20 lines, not a redesign, but a real cost this project should
not assume away for the second `S_evo` variant just because it didn't apply
to the first.

**Correction, 2026-08-05 — the claim below turned out to be wrong for the
container that actually matters, caught only once the polyglot domain was
actually exercised, not by re-reading the source more carefully.** Original
claim: "Docker networking already solved in HyperAgents, unlike HGM" — based
on `HyperAgents/utils/docker_utils.py` already setting `network_mode="host"`
in its `client.containers.run(**run_kwargs)` path. **That check looked at
the wrong file.** `utils/docker_utils.py`'s `network_mode="host"` is set on
its own separate container-launch path (and, at one other call site in the
same file, is only used for the *image build* step, `client.images.build
(..., network_mode="host", ...)` — a different concern from runtime
container networking entirely). The container that actually matters —
where `domains/polyglot/harness.py:process_entry` copies `task_agent.py`
into and runs it, exactly analogous to HGM's own agent container — is built
by `domains/polyglot/docker_build.py:build_container`'s
`client.containers.create(...)` call, which had **no** `network_mode` set
at all, defaulting to Docker's bridge network — the exact same
D-37 bug, unfixed. Patched it the same way (`network_mode="host"` added to
that `containers.create()` call, `.orig` backup kept alongside, same
convention as HGM's own patches). **HyperAgents does need this fix after
all — the cost-reduction claim below is retracted; net cost is now the
litellm patch plus this one, roughly matching HGM's own D-37 lift, not
less than it.**

**A loose end worth flagging, not yet resolved either way**: HGM's own
`swe_bench/harness.py:harness()` loads the plain `princeton-nlp/SWE-bench`
dataset and filters by `instance_id`, while `hgm_utils.py:sample_child`
(used during actual self-improvement) loads `princeton-nlp/SWE-bench_Verified`
directly — two different dataset objects feeding the same pipeline.
`evaluate_agent.py --split Verified` resolves this by pre-filtering
`test_task_list` to Verified's instance IDs before either path runs, and
since Verified is a curated subset of the same underlying instances (not a
separately-authored dataset), filtering the full dataset by Verified's IDs
should yield equivalent problem statements — but this has not been verified
empirically the way every other family's data has been in this project
(D-27/D-34/D-35's standing discipline). When `cbs`'s own SWE-bench Verified
wrapper for `S0`/`S_star` is actually built, it should load
`princeton-nlp/SWE-bench_Verified` directly rather than replicate HGM's
two-dataset indirection, and the equivalence claim above should be spot-
checked against a few real instances rather than assumed.

**Also confirmed while in there**: the actual grading step (resolved vs.
unresolved) is not inside `swe_bench/harness.py` at all — `harness()` only
produces a candidate `model_patch` per instance. Grading happens in
`swe_bench/report.py:make_report` → `run_evals`, which shells out to
`swe_bench/SWE-bench/swebench/harness/run_evaluation.py` (the official,
vendored SWE-bench package's own evaluation harness, applying the patch and
running the real test suite in a per-instance Docker image). This is a more
precise call chain than D-31's original "every verification funnels through
`hgm_utils.eval_agent` → `swe_bench.harness.harness(...)`" — that framing
undercounted a step. Doesn't change D-31's conclusion (reuse the harness
rather than reimplement), but the `cbs` wrapper should call `make_report`/
`run_evals`, not just `harness()`, to get an actual resolved/unresolved
verdict, not just a produced patch.

**Net effect on D-12/D-31 (as of the initial source-only pass): no change to
the recommendation, a refinement to its cost estimate.** HGM still costs
nothing extra to point at a local model; both still fit `fork_bridge.py`'s
existing interception design without redesigning it. (The original version
of this paragraph also claimed HyperAgents "saves the Docker-network fix
HGM needed" — retracted per the correction above; see the 2026-08-05 update
below for what actually held up once this was executed for real, not just
read.) None of this had been tested yet at this point — there was no Docker
on this machine to run either fork's container path — so this was
source-verified but execution-unverified, same epistemic status as D-12's
original scoping before D-37 actually ran anything for real.

**Update, 2026-08-05 — the patch is written and empirically verified (real
GPU host, real vLLM, no Docker involved yet).** Cloned `HyperAgents` onto
the persistent filesystem (`/lambda/nfs/cbs-project/HyperAgents/`) and
installed a **separate, minimal venv** (`hyperagents_venv/`, just `litellm`,
`backoff`, `requests`, `python-dotenv`) rather than installing into the
already-working `hgm_venv` — `HyperAgents/requirements.txt` pulls in heavy,
irrelevant robotics/RL dependencies (Genesis, minihack, gymnasium) for its
other domains, and there was no reason to risk the working HGM environment
for packages this patch doesn't need.

Patched `agent/llm.py:get_response_from_llm` with a `"vllm-model:<host>"`
branch, mirroring HGM's own convention (D-37) — but **empirically, not by
assumption**, this needed more than copying HGM's pattern:
- Directly testing a hardcoded model string against vLLM 0.26.0
  (`"model": "vllm-model:localhost"`) got a real `404 NotFoundError`. Reading
  HGM's own `llm_withtools.py` (line 62,
  `client.chat.completions.create(model=client.models.list().data[0].id, …)`)
  showed why HGM's real runs never hit this: the placeholder string is only
  used to pick the client/`base_url`, never sent as the literal `model=`
  field — the real served model name is resolved dynamically via `/v1/models`
  at call time. The HyperAgents patch replicates this same dynamic
  resolution (`requests.get(f"{api_base}/models")` →
  `data[0]["id"]` → `model = f"hosted_vllm/{real_model_id}"`, litellm's own
  provider prefix for a self-hosted OpenAI-compatible endpoint).
- **Confirmed HyperAgents needs no separate tool-calling integration.**
  Unlike HGM (native `tools=`/`tool_choice=` for Claude/OpenAI, prompt-based
  `<tool_use>` parsing only as a fallback for "other" models),
  `agent/llm_withtools.py:chat_with_agent` **always** uses prompt-based tool
  invocation regardless of model — tool descriptions go in the system
  prompt, tool use is parsed back out of plain text
  (`check_for_tool_uses`). So the single `get_response_from_llm` patch is
  the complete integration surface; nothing else needed touching.
- **A second real bug found only by actually calling it**: `chat_with_agent`
  always calls `get_response_from_llm` with no `max_tokens` override, so the
  library's `MAX_TOKENS = 16384` default (sized for frontier models with far
  larger context windows) got sent as-is — against this deployment's
  `--max-model-len 16384`, that left **zero** room for any input, and a
  bare "what is 2+2?" failed with `ContextWindowExceededError`. Fixed by
  capping `max_tokens` to 4096 inside the vLLM branch specifically
  (matching `hgm/llm.py`'s own `MAX_OUTPUT_TOKENS = 4096` for this exact
  deployment) rather than touching `chat_with_agent`'s signature — keeps the
  fix contained to the same single integration point.

**Verified working, end to end, outside Docker**: a direct
`get_response_from_llm("Say OK...", model="vllm-model:localhost")` call
returned a real `"OK"`; `chat_with_agent("What is 2+2?...", model=
"vllm-model:localhost", tools_available=[])` — the actual higher-level
function `task_agent.py` calls — returned a real, correct `"4"` after the
`max_tokens` fix. **Not yet tested**: a full Docker-based end-to-end task
run (HyperAgents' own harness/Docker orchestration hasn't been located or
exercised yet — `run_task_agent.py` is the CLI-shaped entrypoint,
directly analogous to HGM's `coding_agent.py`, but the Docker-image-build
and per-task-container layer around it hasn't been identified the way
`swe_bench/harness.py`/`polyglot/harness.py` were for HGM). That is now the
concrete remaining piece of work for HyperAgents, not "figure out if it can
talk to the model at all" — that part is done and real.

The patched `agent/llm.py` lives only on the remote instance's clone and in
this project's own scratch space, not in this repo — consistent with
keeping HyperAgents' CC-BY-NC-SA-licensed code clearly separate from `cbs`
itself (D-12).

**Update, 2026-08-05 (second instance) — full end-to-end HyperAgents task
run now succeeds for real, after root-causing and fixing three more real
bugs.** Instance was re-provisioned after a shutdown/restart; picked up
where the above left off and pushed all the way to a real, complete
polyglot task run (`python__bowling`) through HyperAgents' own harness.
Found `domains/polyglot/harness.py` — directly analogous to HGM's own
`polyglot/harness.py` (same `build_container`/`process_entry` shape).
Copied HGM's already-prepared `polyglot-benchmark/` data across (same
underlying exercism dataset both projects use) and generated
HyperAgents' own `polyglot_benchmark_metadata.json` via its
`prepare_polyglot_dataset.py` (run as `python -m
domains.polyglot.prepare_polyglot_dataset` — running the bare script path
instead shadows the top-level `utils/` package with `domains/polyglot/
utils.py`, a plain module, producing a confusing "'utils' is not a
package" error that has nothing to do with the actual dependency).

**Real bug #1 (correction of this entry's own earlier claim) — Docker
networking.** Detailed above: `domains/polyglot/docker_build.py:
build_container`'s `client.containers.create(...)` had no `network_mode`
set at all, unlike the *different* container path in `utils/
docker_utils.py` that was mistakenly checked before. Fixed with
`network_mode="host"`, `.orig` backup kept.

**Real bug #2 — hardcoded model.** `process_entry`'s `cmd` list hardcoded
`"--model", "o3-mini"`, silently ignoring the `model_name_or_path`
parameter passed into the very same function. Fixed to use it.

**Real bug #3 — missing runtime dependency inside the container.**
`utils/git_utils.py` needs `GitPython` (`import git`), not covered by the
minimal `requirements.txt` swapped in to avoid installing HyperAgents' full
dependency list (Genesis, minihack, gymnasium, balrog — needed only for
its other, unrelated domains) inside a lightweight polyglot task
container. Decision: **keep the trimmed `requirements.txt` as the
standing state for this deployment**, not a temporary swap to revert —
the full one is never actually needed for the coding/polyglot domain this
project uses, and installing Genesis et al. inside every task container
would be pure waste. Original backed up as `requirements.txt.orig`.

**Real bug #4, the significant one — a genuine, fully root-caused ~600s
hang, not a flaky slowdown.** The first full harness attempt (after fixes
#1–#3) hit `Script failed with exit code 124` at exactly the configured
`timeout` value (confirmed three times: 600s→timed out at 600s, then
600s→600s again, then a diagnostic-only reduction to 90s→timed out at
exactly 90s) — a strong signature of something genuinely stuck, not slow
generation, since real generation time doesn't track an arbitrarily chosen
cap that precisely. Root-caused by elimination, not guesswork, using a
sequence of cheap, targeted repros instead of repeating the full 600s wait
each time:
1. A **direct diagnostic** replicating `process_entry`'s container-build/
   copy/pip-install/agent-run steps manually, sequentially, completed in
   under 75 seconds with a real captured transcript — ruling out the
   Docker image, the copied files, and the agent code itself as the
   problem.
2. Wrapping the *exact same* diagnostic steps in a `ThreadPoolExecutor`
   (matching how `harness()` actually invokes `process_entry`) also
   succeeded quickly — ruling out threading.
3. Passing the exact `environment={"ANTHROPIC_API_KEY": None, ...}` dict
   the real harness passes (unset on this instance) also succeeded —
   ruling out the `None`-valued env vars.
4. Re-reading `harness()`'s own body (not just `process_entry`, which was
   the only part checked before) found it: `process_evaluation` computes
   `model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"` —
   e.g. `"vllm-model:localhost_0"` — for output-directory labeling, and
   this suffixed string is what actually reaches `process_entry` and then
   (via fix #2) `--model`. Confirmed directly and cheaply (no Docker
   needed): `get_response_from_llm(msg, model="vllm-model:localhost_0")`
   hangs indefinitely on its own, because the patch's host-parsing logic
   turns it into the unresolvable hostname `localhost_0`, `requests`
   raises a `ConnectionError` (a `RequestException`), and `backoff`
   silently retries that exact exception type for the full configured
   `max_time` before giving up — explaining every observed symptom
   (empty output, exact-timeout-tracking, three-for-three reproducibility).

**Fixed at the one correct place**: `domains/polyglot/harness.py`'s `cmd`
construction now strips a trailing `_<digits>` eval-index suffix
specifically for `vllm-model:` strings before passing to `--model`
(`re.sub(r"_\d+$", "", cli_model)`) — leaving `model_name_or_path` itself
unchanged everywhere else it's used (the result-dict labels), since only
the literal CLI argument needs to be a real, resolvable host string.

**Result, confirmed real and fast after all four fixes**: `"Running the
agent"` → `"Container output:"` in ~4 seconds (not the prior 90s/600s
timeouts), harness output
`{'completed_instances': 1, 'empty_patch_instances': 1, ...}` —
`"Successfully processed entry python__bowling for eval 0"`. An empty
patch here means the model chose not to edit files on this particular
trial, exactly the same benign, non-error outcome already established for
HGM's own baseline runs (D-37) — not a sign anything is still broken.

**Net effect**: HyperAgents' polyglot harness is now genuinely
execution-verified end-to-end against this project's real vLLM
deployment, not just source-read or unit-tested in isolation — the same
epistemic bar D-37 established for HGM. All patches (litellm routing +
`max_tokens` cap, `network_mode`, `--model` wiring, the eval-index-suffix
fix) live only on the remote instance's clone plus this project's own
scratch space, never in this repo, consistent with the license-separation
note above.

---

## D-40 — `S0`/`S_star` on SWE-bench Verified: concretely scoped, not yet
built · **P (design worked out and partly verified, awaiting confirmation)**

**The question this closes the scoping on**: D-31 decided `S_evo` evolves
natively against HGM's own SWE-bench Verified/Polyglot harnesses, and both
`preregistration.md` §4 and this project's own design intent require
`S0`/`S_star` to *also* be measured on SWE-bench Verified, on the same
substrate, so the primary `S_evo`-vs-`S_star` comparison is apples-to-apples.
What was missing was a concrete answer to *how* — `cbs.scaffolds.s0.S0`/
`s_star.SStar` are built entirely around `cbs.tasks.schema.Task` (a prompt
string, a single candidate code string, assert-based `tests`/`public_tests`),
and a SWE-bench Verified instance is a git repository at a commit plus a
natural-language problem statement, solved by producing a diff — not
remotely the same shape, for the same reason D-33 found LiveCodeBench didn't
fit as "just another loader." This entry works out the actual mapping by
reading the real code on both sides (`cbs`'s scaffolds/budget, and HGM's
`AgenticSystem`/`chat_with_agent`, already exercised extensively this
session), not by guessing, and by checking one empirical claim against the
real SWE-bench Verified dataset before relying on it.

### The core generalization: one `chat_with_agent` call is the atomic unit

`S0.solve()` for `humaneval`/`mbpp` is "one call to `M`, verbatim, plus
trivial extraction" — the frontier is *defined* relative to that unit (see
`s0.py`'s own docstring). For a real repository-editing task, a single raw
completion cannot produce a working diff at all — the repo doesn't fit in
one prompt, and there is no way to inspect/edit files without some form of
tool use. HGM's own `AgenticSystem.forward()` (and HyperAgents'
`TaskAgent.forward()`, confirmed identical in shape this session) already
solves exactly this problem with a single call to `chat_with_agent(...)`,
which internally may drive many tool-use rounds but is invoked, and
returns, exactly once per attempt. **That one call is the natural
generalization of "one call to `M`"** for this task shape: `S0`-for-SWE-
bench = one `chat_with_agent` trajectory, whatever diff results, submitted
as-is, no retries, no selection — the same "no retries, no repair, no
selection" spirit `s0.py` already states, just with a richer atomic unit.

### Budget accounting already generalizes — confirmed by reading the code,
not assumed

`cbs.budget.BudgetAccountant`/`Usage` charge and cap purely on
`calls`/`prompt_tokens`/`completion_tokens`/`usd` — nothing in `budget.py`
assumes a call maps to a `cbs.tasks.schema.Task`, or that a "task attempt"
is exactly one call. `MatchedComputeHarness.allowance_for(system, task_id)`
hands out a `BudgetAccountant` per (system, task) pair and `MatchReport`
compares realised `total_tokens`/`calls` after the fact — this works
identically whether the attempt behind it was one `S0` completion or a
whole multi-round `chat_with_agent` trajectory, *as long as every
underlying raw model call gets charged to that accountant*. That charging
is the one real gap, and it's a small one, not a redesign: `fork_bridge.
ModelCallProxy` already intercepts every raw request/response pair
(`ProxiedCall.response_body`), and a real captured response this session
(`bridge_test_output5/python__two-bucket.md`) already carries a standard
`usage: {prompt_tokens, completion_tokens, total_tokens}` field on every
successful call — so turning an intercepted call into a `Usage` charge is a
few lines added where `reconstruct_trace_from_events` already walks
`events`, not new interception machinery.

### `S_star`'s four mechanisms, mapped

- **Best-of-N** — N independent `chat_with_agent` trajectories (fresh git
  checkout each time), directly analogous to N independent completions.
- **Execution feedback — a real, verified mapping exists, not a made-up
  one.** `s_star.py`'s `_run_public_tests` runs candidates against
  `Task.public_tests`, a deliberately-authored weaker subset, never the
  hidden `tests`. SWE-bench Verified has no *authored* equivalent — but it
  does have a **structurally equivalent existing field**, confirmed by
  querying the real dataset directly (`princeton-nlp/SWE-bench_Verified`,
  not inferred from documentation): every instance carries `FAIL_TO_PASS`
  (the specific regression test(s) that must go from failing to passing —
  the actual grading oracle, i.e. this task family's `tests`) and
  `PASS_TO_PASS` (the repository's own pre-existing tests, which must keep
  passing — legitimately runnable by a real engineer working on the bug,
  and does not reveal whether the target bug is actually fixed, since it
  contains no test *of* that bug). `PASS_TO_PASS` is this family's
  `public_tests`: running it and feeding a failure back is genuine execution
  feedback with no oracle leakage, by the same reasoning `Task.public_tests`
  already documents.
- **Standard tool use** — already free: `chat_with_agent`'s bash/edit tools
  (or HyperAgents' prompt-parsed tool loop) *are* the tool use; nothing
  extra to add, unlike `S_star`'s bolt-on compile check for atomic-function
  tasks.
- **Self-consistency** — majority vote over the N produced diffs (by literal
  diff text or a normalized form), same idea as `_select_by_consensus`,
  applied to patches instead of canonicalized function bodies.
- The existing rule **"the hidden oracle is queried exactly once, on the
  final chosen candidate"** carries over unchanged: `FAIL_TO_PASS` gets
  checked only once, after selection — `PASS_TO_PASS` is the only thing
  intermediate attempts may query.

### Verification: reuse the real harness, per D-31/D-39, not reimplement

Same conclusion D-31/D-39 already reached for `S_evo`: call HGM's own
`swe_bench/harness.py` (produces the candidate patch — already exercised
this session for the polyglot family, same shape for SWE-bench) followed by
`swe_bench/report.py:make_report` → `run_evals` (the real resolved/
unresolved verdict, via the vendored SWE-bench package's own
`run_evaluation.py`) — not a new verifier. Dataset should be loaded as
`princeton-nlp/SWE-bench_Verified` directly, not replicated through HGM's
own `SWE-bench`-then-filter-by-Verified-IDs indirection (D-39's flagged
loose end).

### What this is not: a small addition, honestly sized

Every task family added so far (`humaneval`+, `mbpp`+, `transfer_reasoning`)
was "teach `cbs.tasks` to load a new dataset shape." This is materially
different: `S0`/`S_star`'s `solve()` currently calls `model.complete(request,
accountant)` and returns; for SWE-bench Verified they would need to
orchestrate real Docker containers directly (build/copy/exec, exactly the
steps `fork_bridge`'s interception script already does for `S_evo`), which
is a new capability for these two scaffolds, not a parameter change. Concrete
remaining pieces, roughly in dependency order:

1. ~~A non-`Task` SWE-bench Verified instance representation~~ — **built**:
   `cbs.tasks.swebench.SweBenchInstance`/`SweBenchSuite`/
   `load_swebench_verified`, loading `princeton-nlp/SWE-bench_Verified`
   directly (not HGM's two-dataset indirection). Validated against the real
   dataset, not just unit-tested against fixtures (`tests/test_swebench.py`,
   10 tests) — including a real, non-obvious catch: `FAIL_TO_PASS`/
   `PASS_TO_PASS` arrive as JSON-*encoded strings*, not native lists,
   despite printing like one; assuming the print output would have shipped
   a real parsing bug. New `swebench` optional extra (`datasets>=2.14`) in
   `pyproject.toml`, matching the `evalplus` extra's pattern.
2. `S0`/`S_star`-flavored driver code — **the pure scaffold logic is now
   built and tested**: `cbs.scaffolds.swebench_scaffold.S0SweBench`/
   `SStarSweBench`, following the exact same *injected-function* pattern
   `cbs.scaffolds.evolved.EvolvedScaffold` already established for agent
   code this project doesn't implement itself (`SweBenchAgentFunction`/
   `SweBenchVerifyFunction`), so best-of-N, the PASS_TO_PASS-feedback
   repair loop, self-consistency, budget charging, and trace merging are
   all tested with synthetic fakes (`tests/test_swebench_scaffold.py`, 18
   tests) — no Docker needed to validate this half, exactly the same
   `test_evolved.py` precedent. One real design gap surfaced and documented
   in the module itself, not glossed over: unlike `s_star.py`'s pre-check
   (`accountant.can_afford(...)` before spending), a Docker-run trajectory's
   real cost isn't known until it has already run, so `accountant.charge()`
   here is unavoidably *post-hoc* — it can still stop further candidates
   from starting, but cannot prevent one already-running trajectory from
   overshooting. **What's still not built**: the actual
   `SweBenchAgentFunction`/`SweBenchVerifyFunction` implementations that
   call into HGM's real `swe_bench.harness`/Docker machinery — this is the
   piece that actually needs a live host and is still the expensive,
   hard-to-reverse, execution-unverified part.
3. ~~Usage extraction added to `ModelCallProxy`/a sibling~~ — **built**:
   `cbs.scaffolds.fork_bridge.usage_from_events`, summing real
   `prompt_tokens`/`completion_tokens` out of already-captured
   `ProxiedCall`s, applying the same success-filter rule
   `reconstruct_trace_from_events` already uses (a failed call attempt is
   not a sample from `M` and must not inflate charged compute). 6 new
   tests in `tests/test_fork_bridge.py` (now 32 total).
4. Wiring verification through `make_report`/`run_evals` for a real
   resolved/unresolved verdict per attempt (mid-run, for feedback) and per
   final choice (for scoring) — **not yet built**.

Total: 414 tests passing project-wide as of this update (up from 380).

**Recommendation, updated**: the user explicitly delegated this class of
implementation-scope decision ("use your judgement for these decisions
always") rather than wanting a confirmation gate before each piece, so
piece 2's pure-logic half was built rather than left waiting on a
confirmation pass. What's left (pieces 2's Docker-glue half, and piece 4)
is still genuinely comparable in weight to D-31's own original two-option
framing in terms of engineering effort and real infrastructure risk — the
part that actually needs a live host, produces execution-unverified code
until it runs against one, and could still surface a design problem the
synthetic tests above can't catch. Proceeding on it is a judgement call,
not a rubber stamp: pieces 1 and 3 were safe to build unconditionally
because they were useful regardless of how the bigger design landed; piece
2's Docker-glue half is not similarly safe from being wrong until it is
actually exercised against a real instance.

**Update, same session — the real infrastructure-cost question behind that
caution is now checked, not just assumed, and it came back favorable.**
One real concern with SWE-bench Verified specifically (unlike Polyglot,
whose tiny per-exercise repos build in ~2-3 minutes): HGM's vendored
`swebench` package (2.1.0) has no pre-built-image-pull support at all
(confirmed by reading `docker_build.py` directly — no `namespace`/registry
logic present), so every environment image is built from scratch, and real
SWE-bench repos (astropy, django, etc.) have real dependency stacks. Rather
than guess at how expensive that actually is, built one for real:
`princeton-nlp/SWE-bench_Verified`'s `astropy__astropy-12907` (a `conda
create` + ~20-package scientific/test-tooling `pip install`) via
`swebench.harness.docker_build.build_env_images` directly against the real
Docker daemon. **Result: 2 minutes 9 seconds, base image included** — not
the 20-30+ minutes that would have made per-instance validation a real
constraint on this project's own iteration speed. This meaningfully lowers
the risk profile of piece 2's remaining Docker-glue half; the eval script
itself was also read in full at this point (`TestSpec.eval_script`) and
confirmed to activate the right conda env, apply `test_patch`, and run
`pytest -rA` against the affected test file, reverting the test file's
checkout afterward — a real, inspectable, faithful harness, not a black box.

Also clarified while reading it: `PASS_TO_PASS` tests are **pre-existing**
tests untouched by `test_patch` (only `FAIL_TO_PASS` depends on it), so a
mid-run `verify_fn` for `SStarSweBench`'s execution feedback can run
`pytest` against `pass_to_pass` node IDs directly, on the model's diff
alone, without ever applying `test_patch` or touching anything
`FAIL_TO_PASS`-related — a cleaner, more clearly oracle-safe path than
reusing the official combined eval script (which runs both sets together)
and trying to filter its output afterward. The final `FAIL_TO_PASS` check
does need `test_patch` applied, mirroring the official harness exactly.

**Still not built**: the actual `SweBenchAgentFunction` (run `coding_agent.py`
in a container, extract the diff) and `SweBenchVerifyFunction` (apply a diff,
run targeted tests, report pass/fail) implementations, and per-test-node
output parsing (a first pass can reasonably start with a coarser
pytest-exit-code signal rather than full per-node parsing, and be refined
later). This is a real, substantial piece of new engineering in its own
right — writing it well, following this session's own "test cheaply, expect
real bugs, verify before trusting" discipline, deserves its own dedicated
pass rather than being rushed to a close inside an already-long session.
Deliberately stopping here having de-risked the one open cost question,
not because the remaining work turned out to be small.

**Update, same session — the user explicitly said to keep working
autonomously, so the Docker-glue piece was written and validated for real,
not left at the scoping stage.** `scripts/swebench_glue.py` implements
`real_agent_fn`/`real_verify_fn` by driving HGM's actual
`swebench.harness.docker_build`/`test_spec` functions directly (the same
ones `swe_bench/harness.py:process_entry` itself calls) — not
reimplementing them. Validated incrementally, each piece checked before
trusting the next, exactly this project's own established discipline:

1. **`real_verify_fn` against the instance's own gold `patch`** (mirroring
   this project's `reference_solution` validation discipline used
   everywhere else — D-27/D-34/D-35): `PASS_TO_PASS` with the gold patch →
   13/13 passed; `FAIL_TO_PASS` with the gold patch → both tests passed;
   `FAIL_TO_PASS` with an **empty diff** (negative control) → correctly
   failed. All three as expected — the verifier distinguishes a real fix
   from no fix, not just "always reports pass."
2. **`real_agent_fn` against the real vLLM model** — one real trajectory,
   real token usage recorded (`Usage(calls=1, prompt_tokens=1109,
   completion_tokens=654)`), real trace (`single_call`, no tool use —
   consistent with every other real run this session under
   `tool_choice="auto"`), no crash.
3. **`S0SweBench.solve()` fully end-to-end**, both functions together: the
   model's attempt didn't actually edit the target source file (same
   no-tool-use pattern), and the pipeline **correctly determined this** —
   `verification.passed == False`, both `FAIL_TO_PASS` tests genuinely
   failed. Budget accounting, tagging, and verification all flow through
   correctly and agree with ground truth (an unresolved instance really is
   unresolved).

**Three more real bugs found and fixed along the way, each caught by
actually running it, not by re-reading the design:**
- `make_test_spec` needs a `hints_text` dict key (confirmed "Unused" in
  swebench's own source) that `SweBenchInstance` doesn't carry — fixed with
  an empty-string placeholder in the entry-dict adapter, not by adding an
  unused field to `cbs.tasks.swebench`.
- **A real, substantive correction to this entry's own earlier claim**:
  "PASS_TO_PASS tests are pre-existing and never need `test_patch`" was
  wrong. Confirmed directly: collecting `test_separable.py` at `base_commit`
  found 11 tests without `test_patch`, 15 with it — some `PASS_TO_PASS`
  entries are themselves new parametrized cases `test_patch` introduces
  that simply already pass pre-fix, not pre-existing tests at all. The real
  rule: PASS_TO_PASS means "passes regardless of the fix, *given*
  `test_patch` is applied," not "doesn't need `test_patch`." `test_patch`
  is now always applied in `real_verify_fn`; oracle-safety is preserved
  because `test_patch` is still never applied inside the *agent's own*
  container, only in this separate verify-only one, and only a filtered
  pass/fail signal for the requested test IDs is ever surfaced back.
- A missing `python -m pip install -r requirements.txt` step inside the
  agent's container (present in HGM's own `process_entry`, simply forgotten
  when writing this simplified glue) — produced an immediate, real
  `ModuleNotFoundError: No module named 'anthropic'` the first time this
  ran for real.
- (Not a bug in this code, but a real gotcha worth recording: bash treats
  `test_separable[compound_model0-result0]`-style test node IDs as glob
  patterns and silently mangles them unless the shell command sets `-f`
  first — caught by an unexplained "collected 11 items" / "no tests ran"
  result rather than a clean error.)

**What remains a known, stated simplification, not a silent gap**: pass/
fail is read off pytest's exit code, not per-test-node parsing of `-rA`
output — fine for validating the pipeline end-to-end, worth revisiting if
a real run needs to distinguish "some but not all requested tests broke"
from "all of them did." The agent's returned diff also carries incidental
environment-setup noise (`pyproject.toml`, from the eval script's own `pip
install -e .[test]`) alongside any real changes — confirmed this is
inherited from HGM's own harness (same `diff_versus_commit` call, same
eval script), not something this glue introduces, and harmless for scoring
since a real fix's genuine hunks are still present and still what
verification actually checks.

**Net result**: D-40 is now a complete, source-verified *and*
execution-verified measurement path from a real SWE-bench Verified
instance through a real vLLM-driven agent trajectory to a real, correct
resolved/unresolved verdict — for `S0`.

**Update, same session — `SStarSweBench` run for real too, immediately
after `S0SweBench`'s success.** `max_candidates=2, max_repairs_per_
candidate=1, stop_on_first_public_pass=False` against the same instance
(`astropy__astropy-12907`, environment image already cached from the `S0`
run). Real result: `usage=Usage(calls=2, prompt_tokens=2218,
completion_tokens=2032)`, `metadata={'n_candidates': 2,
'n_public_passing': 2}`, `trace.op_counts() == {'single_call': 2,
'execution_feedback': 2, 'self_consistency': 1}`, final verification
`passed=False`. Both real trajectories independently passed their
`PASS_TO_PASS` mid-run check (unsurprising — neither actually touched the
relevant source, so nothing they did could have broken a pre-existing
test), self-consistency ran across both, and the `FAIL_TO_PASS` check
correctly confirmed the instance is still unresolved, agreeing with `S0`'s
own verdict. Every mechanism — best-of-N, execution feedback, self-
consistency, budget summation across multiple real trajectories — is now
confirmed working against real infrastructure, not just synthetic tests.

**One honest gap stated plainly, not glossed over**: because both
candidates passed their `PASS_TO_PASS` check on the first attempt, the
*repair* branch (a real trajectory's diff genuinely failing `PASS_TO_PASS`,
triggering a real augmented-prompt second trajectory) was not exercised in
this real run. Assessed as low incremental risk rather than left silently
untested: the repair-prompt construction is pure Python already covered by
a synthetic unit test (`test_a_failing_candidate_triggers_a_repair_
attempt_with_feedback_in_the_prompt`), and re-invoking `agent_fn` with a
different `problem_statement` string is exactly the same code path as the
initial call — `agent_fn` has no notion of "first attempt" vs "repair," it
just runs whatever string it's given. Still, an explicit remaining item,
not something to claim as covered.

**Update, same session — gap closed for real, not left as a paper argument.**
Rather than wait for the live 7B model to naturally produce a diff that
fails `pass_to_pass` (never observed in ~10+ real runs this session — see
D-39/above, the model consistently makes zero or ineffective edits under
`tool_choice="auto"`), the repair branch was exercised deterministically: a
hand-constructed, deliberately-broken diff for `astropy__astropy-12907`
(`if transform.n_inputs == 1 and ...:` replaced with `if True:` in
`separable.py`, one line) as the first-attempt "diff," with the instance's
own gold `patch` substituted on any repair call, wired through a synthetic
`agent_fn` but the **real** `real_verify_fn` (real Docker, real `git apply`,
real `pytest`) — no part of the verification/repair machinery itself was
mocked, only the model call.

First, the bad diff itself needed validating as actually broken, not just
assumed broken: a hand-written unified diff hit `error: corrupt patch at
line 11` from `git apply` on the first attempt (a hunk-header/escaping
mistake from constructing it as an embedded Python string with an escaped
`"""` context line) — fixed by writing the diff as a plain heredoc file
instead of a Python string literal, and by anchoring the hunk on an
unambiguous code line rather than the docstring. The corrected diff applied
cleanly and, run through `real_verify_fn` against `instance.pass_to_pass`
directly, produced exactly the expected real result: **6 failed, 7 passed**
(`compound_model0/2/3/5/7/8` fail; `compound_model1/4`, `test_custom_model_
separable`, etc. pass) — confirming both that the diff is genuinely broken
and that `real_verify_fn` correctly distinguishes broken-fix failures on
`PASS_TO_PASS`, not just "always reports pass" (the same validation
discipline as the gold-patch/empty-diff checks above, now extended to a
partial-failure case).

With the bad diff confirmed broken, a full `SStarSweBench.solve()` run
(`max_candidates=1, max_repairs_per_candidate=1, stop_on_first_public_
pass=True`) against a synthetic `agent_fn` (call #1 → bad diff, call #2 →
`instance.patch`) and the real `verify_fn` produced, entirely for real:
`trace.op_counts() == {'execution_feedback': 2, 'adaptive_prompt_rewrite':
1, 'self_consistency': 1}` — confirming, for the first time this session,
that `adaptive_prompt_rewrite` (the actual repair step, distinct from
`execution_feedback`'s per-attempt check) fires. The repair prompt's
`problem_statement` was confirmed at runtime to actually contain the
real-feedback marker text ("Your previous attempt produced this patch...")
on the second call, i.e. the augmented prompt genuinely carried the first
attempt's diff and the real pytest failure output forward, not a stub.
The second trajectory's diff (gold patch) then passed `pass_to_pass` for
real, `self_consistency` selected it as the sole passing candidate
(`n_candidates: 1, n_public_passing: 1`), and the final hidden-oracle
`FAIL_TO_PASS` check came back **`passed: True`, `failed_tests: ()`** — a
genuinely resolved verdict, end to end, through real Docker containers at
every step (three separate verify-container runs: the standalone bad-diff
check, the in-loop repair check, and the final `FAIL_TO_PASS` check).

This is now a complete, real demonstration of every `SStarSweBench`
mechanism — best-of-N, execution feedback, adaptive repair, self-consistency,
and hidden-oracle verification — firing correctly against real Docker
infrastructure, ending in a correct resolved verdict when the "repair" is
actually correct. The only thing still synthetic is the model call itself
(substituted with a fixed diff instead of a live trajectory); every piece of
`cbs`'s own scaffold and verification logic ran unmodified and real. D-40's
stated gap is closed.

---

## D-41 — D-37's `TS_sample` argmax-on-empty crash: root-caused for real
(not just "probably a shrinkage artifact"), and fixed · **C**

D-37 left one loose end: after a deliberately-shrunk 1-task smoke test's
baseline evaluation completed, HGM's outer loop crashed with `ValueError:
attempt to get argmax of an empty sequence` in `TS_sample`/`expand()`,
flagged as "most likely an artifact of shrinking the task list to 1... but
not reproduced or root-caused against the real 60-task subset." Reproducing
it at real 60-task scale would mean actually running HGM's real baseline
evaluation and then its node-selection logic — which is functionally the
first step of starting a real self-improvement loop, the one action this
project has been explicit should get a dedicated scoping conversation, not
get slipped into via a diagnostic. So this was resolved the other way:
by reading HGM's actual source (`hgm.py`) directly, rather than by spending
real GPU/Docker time.

**Root cause, confirmed by reading the code, not inferred:** `expand()`
(`hgm.py:383`) filters the archive to `nodes = [n for n in hgm_utils.nodes.
values() if np.isfinite(n.mean_utility) and n.mean_utility > 0]`, then
calls `TS_sample(decendant_evals)` — which computes `np.argmax(np.random.
beta(alphas, betas))` — and indexes `nodes` with the result. `Node.
mean_utility` (`tree.py:61`) is `np.inf` if `num_evals == 0`, else `sum(
utility_measures) / num_evals`. In the 1-task smoke test, the root node was
evaluated once, failed (`utility_measures = [0]`), giving `mean_utility ==
0.0` — which fails the `> 0` filter, leaving `nodes = []`. `TS_sample([])`
then computes `np.random.beta([], [])` (a valid empty array, no error) and
`np.argmax(<empty array>)` — which is exactly where `ValueError: attempt to
get argmax of an empty sequence` comes from. Confirmed by direct code
reading, not by running it again.

**This is the crucial correction to D-37's own framing**: the crash's real
trigger is not "too few tasks" — it is "every node the archive has ever
evaluated has mean_utility exactly 0", i.e. *nothing in the archive has
solved anything yet*. Task count is only relevant insofar as more real,
varied tasks make it somewhat less likely that literally zero get solved by
chance — it does not make the underlying code path safe. Given this
project's own repeated, real, empirical finding this session — the frozen
Qwen2.5-Coder-7B baseline essentially never produces a working fix under
`tool_choice="auto"` (D-38/D-39/D-40: ~10+ real SWE-bench/Polyglot attempts,
not one genuine resolution) — there is a real, live risk that the real
60-task baseline evaluation also yields `mean_utility == 0` for the root
node, in which case this exact crash *would* recur at real scale, unchanged
from the 1-task case. This was not a smoke-test-only artifact; it was an
unhandled degenerate case in HGM's own algorithm that this project's
specific choice of frozen baseline makes a real, non-hypothetical risk.

**Fixed, not just documented** — a minimal, targeted patch to `expand()` on
the remote instance's HGM checkout (`/lambda/nfs/cbs-project/hgm/hgm.py`,
`.orig` backup kept, same discipline as every other real fork patch this
project has made): if the `mean_utility > 0` filter leaves `nodes` empty,
fall back to every evaluated (`isfinite`) node regardless of utility, rather
than crashing. This changes nothing about the normal case (≥1 node with
positive utility) — the fallback branch is never reached there — and only
changes behavior in the exact case that previously crashed the whole loop.

**Validated cheaply, not assumed correct** — `scripts/
verify_hgm_ts_sample_fix.py` replicates both the original and patched
`expand()` logic against a fake node registry (pure Python/numpy, no HGM
checkout, no Docker, no GPU, no real model call) and confirms, mechanically:
(1) the original logic really does crash on the exact 1-task-smoke-test
scenario; (2) it also crashes on a 60-task-scale analogue (three nodes, each
with 60/60 failures) — direct evidence this is not fixed merely by having
more tasks; (3) the patched logic does not crash in either case; (4) in a
normal case where one node has positive mean_utility, the patched logic
selects identically to the original across 200 trials with the same seed —
the fallback path is provably inert when it isn't needed.

**What this does and doesn't resolve**: D-37's crash is now root-caused and
fixed at the code level, and validated as fixed against both the exact
scenario that produced it and its real-scale analogue — nothing further is
needed here before a real loop runs. It does *not* mean the real 60-task
baseline evaluation has actually been run, or that this project has
observed what the archive's early state looks like for real — that
observation still only happens the first time the real loop is actually
started, which remains a deliberate, separate decision.

---

## D-42 — `S0` on the Polyglot benchmark: built and real-execution-verified,
completing Phase 3's coverage of D-31's *other* confirmed native `S_evo`
substrate · **C**

D-40 built and validated `S0`/`S_star` for SWE-bench Verified, one of the
two task families D-31 confirmed `S_evo` evolves natively against. The
other, Polyglot, had only ever been exercised through the ad-hoc
`hgm_run_task_with_interception.py` validation script (D-38) — real, but
not wired into `cbs`'s own `Scaffold`-shaped measurement layer the way
SWE-bench Verified now is. This closes that gap for `S0`.

**Real data investigated before designing, not assumed**: HGM's own
`polyglot/polyglot_benchmark_metadata.json` (225 real entries, 6 languages:
cpp 26, go 39, java 47, javascript 49, python 34, rust 30) and
`polyglot/harness.py:process_entry` were both read directly. Two structural
findings shaped the design:

1. **No `Task.tests`/`public_tests`-style split at the data level** — each
   entry has exactly one `files["test"]` file, not separate visible/hidden
   sets. Oracle safety instead comes from *when* the real test content
   becomes visible: the agent's own container is checked out at
   `base_commit` (no `test_commit` applied), and `process_entry` only
   reveals the true grading tests by `git reset --hard {test_commit}`
   *after* the agent has already produced its patch and stopped. Several
   languages' real eval commands additionally compile-gate most test cases
   behind a flag that real grading *does* set (confirmed in
   `polyglot/constants.py`, e.g. C++'s `cmake -DEXERCISM_RUN_ALL_TESTS=1`),
   so `process_entry`'s `eval_result` already reflects the full hidden
   suite, not an always-visible smoke subset — nothing extra needed here to
   get a true hidden-oracle check.
2. **`process_entry` is atomic — one call runs the agent *and* grades it**,
   unlike SWE-bench Verified's harness, which had a natural seam
   (`docker_build`/`test_spec`) D-40 used to split a separate agent-only
   container from a separate verify-only one. Polyglot's container does
   both, sequentially, itself. This means only **one** injected function is
   needed here (`PolyglotAgentFunction`), not the agent/verify pair
   `swebench_scaffold.py` has — and, more importantly, means
   `SStarSweBench`-style execution feedback (checking a candidate against
   something before the hidden oracle) has **no ready-made hook** for
   Polyglot the way `PASS_TO_PASS` gave SWE-bench Verified one. Building
   one would need the same kind of container-splitting surgery D-40 did —
   real, unbuilt, separate engineering, not attempted here. **`SStarPolyglot`
   does not exist yet, deliberately** — a half-built repair loop with no real
   feedback signal would be worse than an honestly-scoped `S0`-only family.

**Built**: `cbs.tasks.polyglot.PolyglotInstance`/`PolyglotSuite`/
`load_polyglot_benchmark` (mirrors `SweBenchInstance`'s frozen-hashed-split
discipline, but — unlike SWE-bench Verified's public HuggingFace dataset —
takes an explicit metadata-file path, since this data has no stable
`cbs`-independent host; it lives only inside a real HGM checkout's vendored
`polyglot-benchmark/` submodule). Keeps the *entire* original entry dict as
`raw` rather than re-deriving named fields and reconstructing them for
`process_entry` later, specifically to avoid the class of bug D-40 hit
repeatedly translating `SweBenchInstance` back into swebench's expected
shape (missing `hints_text`, JSON-encoded list fields, ...) — there is
nothing to mistranslate if the original dict is simply kept whole.
13 tests (`tests/test_polyglot.py`), synthetic fixtures only (the real
metadata file is ~6MB and lives only in a real HGM checkout, not vendored,
same reasoning as why SWE-bench Verified isn't vendored either).

`cbs.scaffolds.polyglot_scaffold.PolyglotRunResult`/`PolyglotResult`/
`S0Polyglot` (10 tests, `tests/test_polyglot_scaffold.py`, synthetic fakes,
same style as `test_swebench_scaffold.py`) — one trajectory, no retries,
budget charged post-hoc (same honest limitation as `S0SweBench`, a
Docker-run trajectory's cost isn't known until it has run).

`scripts/polyglot_glue.py` — `real_agent_fn` calls HGM's own
`polyglot.harness.process_entry` directly (not reimplemented), wrapped with
`fork_bridge.ModelCallProxy` for real tagging/usage, plus the two real
gotchas D-38 already found and documented (the `polyglot_harness.llm`
module-level global, and `process_entry` not building environment images
itself — `build_env_images` called first, mirroring `harness()`'s own
setup).

**Validated against real infrastructure, not just synthetic tests**:

1. `load_polyglot_benchmark` against the real, complete 225-entry metadata
   file: all entries parse, `suite_hash` computes deterministically, and —
   a real, previously-undocumented finding — filtering to HGM's own
   hardcoded 60-task baseline subset (`medium.json + small.json`) actually
   yields only **59 unique instances**, not 60: `python__dominoes` appears
   in *both* `medium.json` and `small.json`, confirmed directly (`medium.
   count("python__dominoes") == 1`, `small.count("python__dominoes") ==
   1`). CLAUDE.md/D-37's "60 tasks" framing is accurate as an evaluation
   count (the baseline really is Docker-evaluated 60 times) but not as a
   *unique*-task count — worth remembering when budgeting or reporting
   real baseline-evaluation numbers for Polyglot.
2. `S0Polyglot.solve()` run fully end-to-end for real against
   `javascript__queen-attack` (real vLLM model, real Docker container, real
   `process_entry` call): `usage=Usage(calls=1, prompt_tokens=1239,
   completion_tokens=722)`, `trace.op_counts() == {"single_call": 1}`,
   `eval_result="empty_patch"`, `passed=False` — the same no-tool-use,
   no-edit pattern observed in essentially every real run this session
   under `tool_choice="auto"`, not a bug in this new code; the pipeline
   correctly reports what actually happened rather than a spurious pass.

**Net result**: Phase 3's `S0` coverage now spans both of D-31's confirmed
native `S_evo` substrates against real infrastructure — SWE-bench Verified
(D-40) and Polyglot (this entry). `S_star`'s Polyglot analogue remains
explicitly unbuilt, needing the same container-splitting work D-40 did, not
a design question left unresolved.

---

## D-43 — stale Docker build-context staging silently reported as task
failure; found by the first at-scale batch run · **C**

**Symptom**: the first real 59-task `S0Polyglot` batch returned
`eval_result: "incomplete"` for its first task (`javascript__queen-attack`)
after 721s — a task that had completed normally (`empty_patch`, 146s) on
the *previous* instance days earlier.

**Root cause, from the real Docker log, not guessed**: HGM's `build_image`
stages a per-task Docker build context by `shutil.copytree`-ing the
exercise repo — *including its `.git/` directory* — into
`logs/build_images/instances/<image>/`. Git object files are created
**read-only** (mode `0444`). Because `/lambda/nfs/cbs-project` is a
*persistent* filesystem that survives instance termination, staging
directories written by an earlier session were still present, still
read-only, and the new run's `copytree` died with `[Errno 13] Permission
denied` on every one of them.

**Why it mattered far more than a slow first task**: `process_entry` wraps
its whole body in a broad `except`, and sets `eval_result = "incomplete"`
when the exception fires before grading. In the returned result dict, that
is **indistinguishable from a task the model genuinely failed**. This batch
exists specifically to establish the frozen model's baseline pass rate — an
infrastructure error silently counted as a model failure would have
depressed exactly the number being measured, in the direction that looks
like a real finding. Caught only because `"incomplete"` was an
`eval_result` value never seen before and got checked rather than assumed
to be a normal failure mode.

**Fixed in two places, deliberately:**
1. `scripts/polyglot_glue.py` now clears the per-task staging directory
   before each run (`_clear_stale_build_context`), with an `onerror` hook
   that restores write permission so read-only git objects can actually be
   unlinked. This removes the collision at its source rather than retrying
   around it. Only regenerable build context is touched, never source data.
2. The batch runner treats `"incomplete"`/`"error"` as infrastructure
   failures: retries once, and — importantly — **excludes them from the
   denominator** of the reported pass rate rather than counting them as
   zeros, reporting them as a separate line instead.

**Generalizes beyond Polyglot**: any harness that stages a git repo into a
build context on a persistent filesystem has this bug latent. Worth
checking `swebench_glue.py`'s path before its own at-scale run.

---

## D-44 — the frozen 7B baseline is at hard floor on Polyglot: 0 resolved,
**zero tool calls**, every result an empty patch · **P (batch in progress)**

**The measurement**: `S0Polyglot` across HGM's own 59-task Polyglot
baseline subset (D-42: 59 unique, not 60), real Docker, real vLLM-served
`Qwen/Qwen2.5-Coder-7B-Instruct`, the unmodified `measured_default_agent`
under its own `tool_choice="auto"` default. As of 27/59 completed, the
result is perfectly uniform across all six languages:

> **0 resolved · 27 `empty_patch` · 0 `tool_call` operations · 0 errors**

**Mechanism, confirmed from a real transcript rather than inferred**: the
model is not failing at *coding*. Given `cpp__diamond` it produces a
competent, well-structured natural-language analysis ("1. **Understand the
Pattern**: The diamond starts with 'A' and increases in size until it
reaches the given letter...") and then stops — `finish_reason='stop'`, no
`tool_calls`. It never invokes `edit`/`bash`, so it never modifies a file,
so `diff_versus_commit` yields an empty patch and the task auto-fails
before any test runs. It behaves as a chat model, not an agent.

This is the same behaviour D-38 flagged at n=9 and could not then
distinguish from small-sample noise. At n=27 across 6 languages with zero
variance, it is a property of this model-scaffold pair, not a fluke.

**Why this is a serious problem for the study, stated plainly**: `S0` is
what *defines* the frontier this instrument measures against. If `S0 = 0`,
the frontier sits at zero, and every one of the brief's four
frontier-crossing conditions degenerates — there is no task `S0` plausibly
fails-but-could-solve, the frontier estimator has nothing to estimate, and
any nonzero `S_evo` result would "cross" trivially. That is an
*uninformative* null (the instrument could not have detected an effect),
which is categorically weaker than a well-powered null (it could have, and
didn't). It also explains D-41's crash mechanically: with every archive
node at `mean_utility == 0`, `expand()`'s positive-utility filter is
*always* empty, so that degenerate branch is not an edge case here — it is
the expected path.

**The genuinely interesting finding inside the problem**: D-38 established
that forcing `tool_choice="required"` in an isolated agent copy produced a
real 64-round tool-using trajectory with a substantive patch that passed
most of the hidden suite. So a **one-line scaffold change** moves this
model-scaffold pair from *no action at all* to *near-solve*. That is an
enormous elicitation effect from a change that does nothing to the frozen
model — which is this project's own central thesis, appearing unbidden in a
validation run. It is not, however, a frontier-*crossing* result, and must
not be presented as one.

**Consequence for the experiment, not yet decided**: the frozen model
almost certainly has to change before any evolutionary run is worth paying
for. Constraint: the A10's 24GB. Quantized larger models fit at **no extra
GPU cost** (the instance bills per hour regardless) — `Qwen2.5-Coder-14B-
Instruct-AWQ` (~9GB, comfortable at 16k context) and
`Qwen2.5-Coder-32B-Instruct-AWQ` (~19GB, tight) are both downloaded and
staged. Quantization does not weaken the frozen-verifiability argument
(D-01/D-03): the weights are still fixed, self-hosted, and hashable; it
only needs stating.

**Confirmed while planning the swap**: no agent reconfiguration is needed.
`llm_withtools.py:62` resolves the served model dynamically
(`client.models.list().data[0].id`) whenever the model string contains
`vllm`, so the `"vllm-model:localhost"` string is model-agnostic and
swapping what vLLM serves is transparent to the agent.

**A hazard closed before it could bite**: results files did not record
*which* model produced each row. Since the batch runner resumes from a
checkpoint, stopping a run, swapping models, and resuming would have
silently merged two models' results into one file with no way to separate
them afterward — precisely the kind of quiet data corruption this project
tries to design out. The subset runner now records the served model per
result and warns loudly if a single file ever contains more than one.

**Methodological caution recorded now, before results exist**: switching
the agent to `tool_choice="required"` to make the numbers move would change
the scaffold mid-study and confound exactly the comparison this instrument
is built to make. Either a stronger model uses tools under the *unmodified*
default, or auto-vs-required becomes an explicit, pre-registered
experimental arm. `preregistration.md` §4 (models and task families) is
still legitimately open (`[TO FIX]`), so choosing and *documenting* the
model now is honest; changing it after seeing crossing results would not
be.

---

## D-45 — target a NeurIPS 2026 workshop first, not the main track; venue
chosen and paper scoped · **C**

**Why the venue changed.** The stated ambition has been a strong main-track
paper. Two facts make a workshop the correct *first* move rather than a
consolation: NeurIPS 2026's main-track deadline (2026-05-06) has already
passed, so the main track is a 2027 target regardless; and NeurIPS workshops
are **non-archival**, explicitly welcome work in progress, and permit
subsequent submission of the same work to an archival venue. Verified
directly against NeurIPS's own 2026 handbook and workshop guidance — dual
submission to non-archival workshops is permitted, and reviewers are
instructed not to reference a workshop version. So the workshop costs
nothing at the main track later and buys expert feedback before the
expensive experiment is committed to.

**Venue chosen: "Managing Agents that Manage Agents: Responsible Use of
Meta-Agents"** (NeurIPS 2026, deadline 2026-08-29 AoE, non-archival, 9-page
full / 4-page short). Selected over two same-deadline alternatives
("Evaluation of Interactive Agents"; "Who Verifies the Agents?") because
four of its six stated topics map directly onto work already built:
automated design/discovery of agent harnesses *and when they generalise
beyond tuning tasks* (the overfitting-gap machinery); self-improvement and
open-ended evolution (HGM/HyperAgents); **evaluation and benchmarks for
meta-agents — "tests determining whether improved agents genuinely
generalise"** (a near-verbatim restatement of this project's thesis); and
misalignment/safety via observable execution (the interception-based
tagging, which exists precisely because an evolved scaffold cannot be
trusted to self-report). The runner-up's emphasis on user simulators and
user-facing interactive systems is a materially worse fit.

**Scope deliberately cut to fit.** The workshop paper is the instrument plus
the D-44 scaffold-sensitivity result. It explicitly **excludes** the full
evolutionary run, multiple seeds, both forks, and SWE-bench Verified — those
are the main-track contribution, and attempting them now would miss the
deadline for no gain. Estimated compute for what *is* in scope is ~$50 on
the existing A10, versus ~$2,000 for the full main-track study.

**Draft lives in `paper/workshop_paper.tex`** (+ `paper/refs.bib`), written
against the NeurIPS style, with every not-yet-measured number wrapped in a
`\todoresult{}` macro so no placeholder can survive to submission
unnoticed. `scripts/analyze_scaffold_sensitivity.py` regenerates the main
table (including its LaTeX form) directly from the per-arm results files,
so the paper's numbers are reproducible rather than transcribed.

**One framing decision recorded here because it is load-bearing and could
otherwise look like spin.** If the `required` arm ends at the same resolve
rate as `auto` (both zero), the result is *not* "scaffold change produces
capability gain" — it plainly does not. The correct and more interesting
claim is that two scaffolds differing by two lines produce enormous
*behavioural* divergence (no actions at all vs. tens of tool calls and real
graded patches) while receiving *identical* utility under a pass-rate
objective. That makes the most consequential scaffold property invisible to
the very signal DGM/HGM/SICA all select on — and it is mechanically why
D-41's `argmax` crash exists, since the positive-utility filter is always
empty in this regime. This framing must not be swapped for the stronger
"capability gain" claim if the data does not support it.

---

## D-46 — a third agent failure mode ("acts on the wrong artifact"), and a
denominator correction made against our own interest · **C**

**Found by**: 9 of 59 tasks in the `14B × required` arm returned HGM's
`error` marker — a rate seen in no other arm (0/59 in all three others).

**Root cause, from the real container logs, not inferred.** The failing
sequence is always identical:

```
"No local changes to save"      <- git stash push <declared solution files>
"HEAD is now at 632858f"        <- git reset --hard test_commit
"Removing diamond_kata.py"      <- git clean -fd deletes the agent's work
"No stash entries found."       <- git stash pop fails, exit 1
```

The agent worked productively (tens to hundreds of tool calls) but created a
**new file of its own naming** instead of editing the exercise's declared
solution files — frequently in the wrong language entirely:
`diamond_kata.py` for the C++ task `cpp__diamond`; `robot.py` and
`spell_number.py` for JavaScript tasks; `src/main.rs` where the declared
solution is `src/lib.rs`; `school_roster.py` plus a self-authored test file.
Polyglot's harness stashes only the declared solution paths before revealing
hidden tests, so these attempts leave nothing gradeable behind.

**This is a genuine agent failure, not broken infrastructure**, and that
distinction changes the reported numbers. D-43 established the discipline of
excluding infrastructure failures from denominators. These rows superficially
match — same exception machinery, adjacent marker — but excluding them would
have reported `4/50 = 8.0%` where the correct figure is `4/59 = 6.8%`,
inflating the resolve rate by shrinking the denominator over precisely the
attempts that failed. **The correction was made in the direction that makes
the result weaker.** `scripts/analyze_scaffold_sensitivity.py` now separates
`incomplete` (exception *before* `eval_result` exists — true infra failure,
excluded) from `error` (exception *during* grading — counted as unresolved,
reported separately as `wrong-artifact`).

**Why it matters beyond bookkeeping.** This is a third distinct way the
scaffold–model *interface*, rather than coding ability, determines the
measured outcome — alongside (1) not acting at all under `tool_choice="auto"`
and (2) acting but failing the task. Like the other two it is invisible in a
pass-rate column, where it is indistinguishable from having tried and been
wrong. Note the failure modes *shift* rather than disappear with scale: the
7B `required` arm had 0 wrong-artifact failures but 6 `empty_patch`; the 14B
arm had 9 wrong-artifact and 0 `empty_patch`.

**Also fixed**: the inline summary in `run_s0_polyglot_subset.py` computed
`used_tools` over all rows while dividing by the infra-excluded denominator,
printing the impossible `59/50`. Cosmetic (the real analysis script filters
correctly first), but corrected so a glance at a run log is not misleading.

---

## Still open

| # | Decision | Status | Needed by |
|---|---|---|---|
| D-12 | ~~Primary `S_evo` fork~~ — **confirmed by user, 2026-08-04**: `metauto-ai/HGM` primary, `facebookresearch/HyperAgents` as a second independent variant (own CC-BY-NC-SA module), `jennyzzt/dgm` kept only as literature baseline. Not yet cloned/forked — real engineering work, next once a host exists | **C** | Phase 4 execution |
| D-13 | LiveCodeBench (investigated, real scope now known — needs sandbox stdin support + a new I/O-judge verification path, not just a loader; D-33) remains out of scope. SWE-bench Verified is now **load-bearing, confirmed** (D-12/D-31) rather than optional. HumanEval→HumanEval+ (D-34) and MBPP→MBPP+ (D-35) are **done** | **D** | before real capability claims |
| D-31 | ~~Option (a) vs (b) for task integration~~ — **confirmed by user, 2026-08-04**: option (b), native SWE-bench Verified/Polyglot, not (a)'s task-adaptation layer. Wrapper around HGM's own `swe_bench/harness.py` for `S0`/`S_star` still needs building | **C** | Phase 4 execution, alongside D-12 |
| D-14 | ~~Exact `N_max`, the `k`/`K` reliability threshold, and the interpretation matrix's three placement thresholds~~ — **signed off by user, 2026-08-04** (D-32); all ten rows in `preregistration.md` §3 now marked **locked**, not `[TO FIX]`. Must not be revisited after seeing results | **C** | before Phase 5 |
| D-15 | Whether to add a third model family | **D** | Phase 3 (done otherwise) |
| D-16 | Per-phase budget caps in dollars / GPU-hours | **D** | before first paid run |
| D-17 | ~~Which reasoning set is the transfer family~~ — resolved: `transfer_reasoning`, D-30 | **C** | — |
| D-23 | ~~Provision a Linux host with both GPU and Docker~~ — **resolved, D-37**: Lambda Labs A10 instance, Docker+GPU+vLLM confirmed working, Phase 0 DoD met for real | **C** | — |
| D-36 | Novelty check against current literature — citation-graph pass (DGM/HGM/SICA), full METR/Apollo read, and forward-citation check on the İşcan cluster all **done**; gap holds as of this check. **Only remaining piece: one more recency sweep close to the actual submission date** | **D** | before submission, ideally before large infra spend |
| D-37 | ~~Root-cause the `TS_sample` argmax-on-empty crash~~ — **root-caused and fixed, 2026-08-06 (D-41)**: the real trigger is any archive state where every node has `mean_utility == 0` (nothing has solved anything yet), not "too few tasks" — confirmed by reading `hgm.py`/`tree.py` directly. Patched `expand()` to fall back to all evaluated nodes when the positive-utility filter is empty; confirmed inert in the normal case and crash-free in both the 1-task and 60-task-scale degenerate cases via an isolated reproduction (`scripts/verify_hgm_ts_sample_fix.py`), no real GPU/Docker run needed | **C** | — |
| D-38 | ~~Validate `tool_call` classification against a real tool-invoking episode~~ — **resolved, 2026-08-05**: forced `tool_choice="required"` in an isolated (non-canonical) agent copy, observed a real 64-round tool-using trajectory, correctly classified end-to-end (`used_expanding: true` produced by a real run for the first time). Separately, whether the baseline agent invokes tools *unprompted* under normal `tool_choice="auto"` remains open — 9/9 unforced real attempts still show none | **C** | — |
| D-39 | ~~HyperAgents local-endpoint patch + full task run~~ — **done, 2026-08-05**: litellm patch, `network_mode="host"`, hardcoded-`--model` fix, and a real ~600s-hang root cause (eval-index suffix corrupting the model string) all found and fixed; a full real polyglot task now completes end-to-end in ~4s | **C** | — |
| D-40 | ~~`S0`/`S_star` on SWE-bench Verified~~ — **done and fully execution-verified, 2026-08-05/06**: both `S0SweBench` and `SStarSweBench` run fully end-to-end for real (real vLLM model, real Docker/HGM harness, real instance), each producing a correct resolved/unresolved verdict; best-of-N, execution feedback, self-consistency, and budget summation across multiple real trajectories all confirmed working. Found and fixed 4 more real bugs along the way (see write-up). **The one remaining gap — the repair branch itself — is now also closed**: a deterministic test (synthetic bad-diff-then-gold-patch, real Docker/`git apply`/`pytest` throughout) exercised `adaptive_prompt_rewrite` for the first time, ending in a genuinely correct `passed: True` verdict. No remaining gaps in this scaffold's mechanisms | **C** | — |
| D-41 | ~~Root-cause + fix `expand()`'s empty-argmax crash (D-37)~~ — **done, 2026-08-06**: real trigger identified from source (every archive node at `mean_utility == 0`, not "too few tasks"); `hgm.py`'s `expand()` patched to fall back to all evaluated nodes rather than crash; fix validated crash-free at both 1-task and 60-task-scale analogues, and provably inert in the normal case, via `scripts/verify_hgm_ts_sample_fix.py` — no real GPU/Docker spend needed | **C** | — |
| D-42 | ~~`S0` on the Polyglot benchmark~~ — **done and real-execution-verified, 2026-08-06**: `cbs.tasks.polyglot`/`cbs.scaffolds.polyglot_scaffold.S0Polyglot`/`scripts/polyglot_glue.py` built and run fully end-to-end for real (real vLLM model, real Docker/HGM harness); completes Phase 3's coverage of D-31's other confirmed native substrate. `SStarPolyglot` deliberately not built yet — no execution-feedback hook exists without the same container-splitting surgery D-40 did for SWE-bench Verified | **C** | — |

| D-43 | ~~Stale Docker build-context staging reported as task failure~~ — **fixed, 2026-08-07**: read-only git objects left on the persistent filesystem broke `copytree`, and `process_entry` reported the resulting exception identically to a genuine model failure. Cleared at source in `polyglot_glue.py`; infra failures now excluded from the pass-rate denominator | **C** | — |
| D-44 | **The frozen 7B is at hard floor on Polyglot** — **complete at n=59**: 0 resolved, 0 tool calls, 59/59 `empty_patch`, every task exactly one generation then stop (mean 902 completion tokens). `S0 = 0` makes the frontier degenerate and the crossing test uninformative. 2×2 scaffold-sensitivity sub-study declared (preregistration §4.1) and running; `7B × required` shows the regime change clearly so far (tens of tool calls/task, real graded patches) | **P** | before any `S_evo` run |
| D-45 | ~~Publication venue and scope~~ — **decided, 2026-08-10**: NeurIPS 2026 "Managing Agents that Manage Agents" workshop (2026-08-29, non-archival), instrument + D-44 result, explicitly excluding the evolutionary run/seeds/second fork/SWE-bench. Main track is a 2027 target since NeurIPS 2026's own deadline has passed | **C** | — |

D-14 is now locked in `preregistration.md` and must not be revisited after
seeing results. D-23 is resolved (D-37); D-37's own remaining crash is now
also resolved (D-41); D-12/D-31 are confirmed but not yet built, though
Phase 3's `S0` now covers both of D-31's confirmed substrates for real
(D-40/D-42). **The practical blocker on Phase 4/5 has changed as of D-44**:
it is no longer primarily engineering or an open decision, but an empirical
one — the frozen model as currently chosen produces no agentic behaviour at
all, so there is nothing for an evolutionary run to improve *on*. Resolving
D-44 (a base model with real dynamic range under the unmodified scaffold)
now gates everything downstream.
