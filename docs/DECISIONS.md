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

## Still open

| # | Decision | Status | Needed by |
|---|---|---|---|
| D-12 | Primary `S_evo` fork: `jennyzzt/dgm` vs `metauto-ai/HGM` | **D** | Phase 4 execution |
| D-13 | Remaining benchmark families: MBPP+, LiveCodeBench, SWE-bench Verified, and upgrading HumanEval→HumanEval+ (D-27) | **D** | before real capability claims |
| D-14 | Exact `N_max` and the `1/N_max` crossing cutoff, pre-registered | **D** | before Phase 5 |
| D-15 | Whether to add a third model family | **D** | Phase 3 (done otherwise) |
| D-16 | Per-phase budget caps in dollars / GPU-hours | **D** | before first paid run |
| D-17 | Which reasoning set is the transfer family | **D** | Phase 1 (real families) |
| D-23 | Provision a Linux host with both GPU and Docker | **D** | Phase 4 execution (blocking) |

D-14 in particular must be fixed in `preregistration.md` **before** the full run,
not after seeing results. D-23 is the practical blocker on actually *running*
Phase 4/5 — its measurement layer is built and tested (D-24 through D-26), but
has nothing safe to run against without it.
