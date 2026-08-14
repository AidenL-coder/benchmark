# Pre-registration — capability-boundary study of self-improving coding agents

**Status: DRAFT / INCOMPLETE. Not yet binding.**

Brief §9.4 requires this document be completed and committed **before** the full
run. Sections marked **[TO FIX]** hold values that must be chosen while blind to
results. Committing this file with those sections resolved is the gate for
Phase 5.

The point of pre-registering is narrow and specific: the interpretation matrix
(§7 below) maps results to conclusions, and every cell is publishable. That is a
strength only if the mapping is fixed in advance. Chosen afterwards, "which
threshold counts as a crossing" becomes a free parameter and the design's main
claim — that it cannot return an uninformative result — quietly becomes the
weaker claim that it cannot return an *unpublishable* one.

---

## 1. Hypotheses

**H1 (bounded search / null).** Evolved-agent gains do not exceed a strong fixed
human-designed generic scaffold (`S_star`) at matched inference compute, and any
apparent frontier-crossings are attributable to support-expanding primitives
already available to the base agent, plus eval-specific overfitting.

**H2 (genuine expansion).** Evolution discovers support-expanding scaffold
structures that reliably cross the frozen frontier on held-out **and** transfer
tasks, beyond what `S_star` achieves at matched compute.

Both are stated to be falsifiable. The study is powered to distinguish them only
in the regime described in §6.

---

## 2. Fixed definitions (already implemented)

| Quantity | Definition | Implementation |
|---|---|---|
| `p_hat(x)` | solve rate of `M` under minimal scaffold `S0`, over `N_max` samples on a fixed temperature schedule | `cbs.frontier.sampler` |
| Practical frontier at `N` | tasks with `p_hat(x) > 0` within `N` samples | `FrontierRecord.beyond_frontier` |
| Beyond-frontier | zero successes in `N_max`; `p` bounded by Clopper-Pearson upper limit and rule of three | `cbs.frontier.records` |
| Support partition | every scaffold op tagged preserving/expanding, with a written rationale | `cbs.scaffolds.tagging` |
| Matched compute | identical per-task allowance, **and** realised spend within tolerance | `cbs.budget.MatchedComputeHarness` |

`p_hat(x)` is the solve probability under the **temperature mixture**, not at a
single temperature. Estimates drawn under different schedules are not pooled;
the schedule is part of the shard's resume key.

---

## 3. Thresholds **(signed off 2026-08-04 — locked, see below)**

The combination logic for every row below is implemented and tested
(`cbs.crossing.evaluate_crossing`, `cbs.interpretation.place_in_interpretation_matrix`,
`cbs.stats.benjamini_hochberg` — see `docs/DECISIONS.md` D-28). Every value
below was chosen and locked while blind to any real Phase 4/5 results —
no run against a real evolved scaffold has happened yet (see
`docs/DECISIONS.md` D-32 for the full sign-off record and reasoning behind
each). **These values must not be revisited after seeing results.** Any
future change to a row below invalidates preregistration for runs that used
the old value; treat this table as frozen, not as a living default.

| Parameter | Symbol / arg | Locked value | Status |
|---|---|---|---|
| Frontier budget | `N_max` | 1000 | **locked** |
| Crossing cutoff | `p_hat(x) < 1/N_max` | 0.001 (follows from `N_max`) | **locked** |
| Reliability of a crossing | `evaluate_crossing(k, K, ...)` | k=6, K=10 | **locked** |
| Compute-match tolerance | `MatchedComputeHarness(tolerance=...)` | 0.05 | implemented default, not disputed |
| Confidence level | — | 0.95, Clopper-Pearson two-sided | implemented default, not disputed |
| Multiple-comparison correction | `cbs.stats.benjamini_hochberg(alpha=...)` | 0.05, across tasks within a family | **locked** |
| Near-zero crossing rate | `place_in_interpretation_matrix(crossing_rate_epsilon=...)` | 0.0 (exact) | **locked** |
| "High" transfer retention | `place_in_interpretation_matrix(transfer_retention_high=...)` | 0.5 | **locked** |
| "Large" overfitting gap | `place_in_interpretation_matrix(overfitting_gap_high=...)` | 0.2 | **locked** |
| Number of evolution seeds | — | ≥3 | **locked** |

A task counts as **crossed** only if all four conditions of brief §3.3 hold
simultaneously: reliable solve at `k/K`, `p_hat(x)` below cutoff, compute matched
to `N_max`, and ablating support-expanding ops removes the solve.

### Reasoning behind each locked value

**`N_max = 1000`.** Rule-of-three bounds a zero-success task at `p < 0.003` —
comfortably inside the brief's stated 10³–10⁴ range and a defensible
"beyond the practical frontier at this compute" claim. At ~100 tasks per
model this is ≈100k generations, ≈3 GPU-hours on a T4 for a 1.5B model
(D-03) — affordable for a first real run. Trade-off: a larger `N_max` (e.g.
5000) tightens the bound to `p < 0.0006` at 5× the cost; the recommendation
is to start here and scale up selectively only for the specific
model/task/split combination where a crossing claim's strength actually turns
on the tighter bound, not uniformly.

**`k=6, K=10` reliability (60%, unchanged proportion from the original k=3/K=5
proposal, more resolution).** The crossing/reliability check only runs on the
small subset of tasks already identified as beyond-frontier — a filtered,
much smaller population than the full held-out set `N_max` sampling covers.
That asymmetry means `K` here is comparatively cheap to raise: doubling it
from 5 to 10 barely moves total compute, but gives a materially tighter
confidence interval around the reliability estimate for the highest-stakes
claim in the whole study (a confirmed frontier-crossing). Recommend spending
that cheap margin rather than leaving it on the table.

**BH `alpha=0.05`.** Standard FDR-control convention; no basis for deviating
without a specific reason tied to how many held-out tasks end up flagged as
crossing candidates (a genuinely unknown quantity before real data exists).

**Near-zero crossing epsilon = exactly `0.0`.** The BH step already absorbs
"how many false-positive crossings are tolerable" upstream of this. Adding a
second, fuzzy tolerance here (`epsilon > 0`) would double-count that same
uncertainty rather than add information. Any single BH-*corrected*,
significant crossing should be enough to leave the "bounded search" cell —
that is the point of correcting first.

**Transfer retention "high" = `0.5`.** A defensible, round bar: at least half
of whatever gain `S_evo` shows on held-out tasks survives on a family it never
optimised on. Common enough a bar in transfer-learning framings generally
that it needs no exotic justification; tightening it (e.g. 0.7) would make
"genuine expansion" harder to claim, loosening it (0.3) easier — the
recommendation is the middle value precisely because this cell is the
*constructive* claim the whole design is built to be capable of making
truthfully, and it shouldn't be made either too easy or too hard to reach by
an arbitrary choice here.

**Overfitting gap "large" = `0.2`, tightened from the original 0.3 proposal.**
A 30-point train-held_out gap (e.g. 90% vs. 60%) is a fairly permissive bar —
it would let a meaningfully overfit result still land in "genuine expansion"
rather than "illusory expansion." Given that cell is the highest-stakes,
hardest-to-walk-back claim the study can make, biasing this specific threshold
conservative (more readily triggering "illusory expansion" on a smaller gap)
is the safer asymmetry: a real discovery survives a stricter overfitting
check; a false one is exactly what a stricter check should catch.

**Evolution seeds `≥3`.** Each seed is a full, independent evolutionary run —
a materially larger cost driver than frontier sampling or the crossing test,
since it means running the self-improvement loop itself, not just sampling a
frozen model. Three is a sensible floor to see seed-to-seed variance at all
without prohibitive cost; scale up opportunistically if budget allows, same
posture as `N_max`.

---

## 4. Models and task families **[TO FIX — implementation, not a judgment call]**

- Frozen models: ≥2 families × ≥2 sizes. Proposed Qwen2.5-Coder-1.5B / 7B, plus
  one of Llama-3.x-8B or DeepSeek-Coder.
- `S0`/`S_star` frontier estimation and calibration (coding): both original and
  evalplus-upgraded HumanEval and MBPP vendored and integrated
  (`cbs.tasks.families.humaneval`/`.humanevalplus`/`.mbpp`/`.mbppplus`;
  164/163 + 427/374 problems, every canonical/reference solution verified
  against the real sandbox — docs/DECISIONS.md D-27/D-29/D-34/D-35). Use the
  "+" variants (`humanevalplus`, `mbppplus`) over the originals for anything
  beyond instrument validation — both upgrades are done. Validating each
  surfaced real upstream bugs (a task whose own generated test fails its
  reference solution, a floating-point-tolerance gap, three tasks whose
  generated test asserts nothing at all) — all found empirically, excluded,
  and documented rather than silently patched into the vendored data; see
  D-34/D-35 for each.
- `S_evo` (D-12/D-31, confirmed 2026-08-04): evolves natively against HGM's own
  SWE-bench Verified and Polyglot harnesses, not the `humaneval`/`mbpp`
  families above — D-31's option (b). `S0`/`S_star` still get measured on
  SWE-bench Verified too (via a `cbs` task-family wrapper around HGM's own
  `swe_bench/harness.py`, not a from-scratch port) so all three scaffolds sit
  on the same substrate for the primary comparison. **Still genuinely
  `[TO FIX]`** — this is unbuilt engineering work, not an undecided threshold:
  the SWE-bench Verified wrapper and Polyglot integration don't exist yet.
  LiveCodeBench remains out of scope for now (D-33 — a materially
  bigger, cross-cutting change, not simply "vendor another file").
- Transfer (reasoning): resolved (D-17) — `transfer_reasoning`, 10
  hand-authored math/logic/combinatorics tasks (`cbs.tasks.families.transfer_reasoning`;
  D-30), never optimised on, frozen entirely into the `transfer` split.
- Splits frozen and hashed **before** sampling (`cbs splits freeze`); the suite
  hash is quoted in results. Frozen so far: `data/splits/humaneval.json`
  (85 train / 79 held-out), `data/splits/humanevalplus.json` (84 train / 79
  held-out), `data/splits/mbpp.json` (205 train / 222 held-out),
  `data/splits/mbppplus.json` (173 train / 201 held-out),
  `data/splits/transfer_reasoning.json` (10 transfer).

### 4.1 Scaffold-sensitivity sub-study — arms declared 2026-08-10, **before** running them

Recorded here *before* the corresponding runs execute, so the choice cannot
be rationalised after seeing which configuration produces more favourable
numbers. This is the discipline §3's locked thresholds already follow,
applied to a design decision §4 legitimately still leaves open.

**Why this sub-study exists (the prompting observation, already collected):**
`S0Polyglot` on HGM's own 59-task Polyglot baseline subset, with frozen
`Qwen2.5-Coder-7B-Instruct` under the unmodified agent's own
`tool_choice="auto"` default, returned **0/59 resolved, 0 tool calls, 59/59
`empty_patch`**, uniform across all six languages (D-44). The model writes a
competent analysis and stops without ever editing a file. Since `S0` is what
*defines* the frontier, `S0 = 0` makes the frontier degenerate and the
crossing test uninformative. That result is complete and is **not** re-run
or revised by anything below.

**Declared arms** — a 2×2 over the same frozen 59-task set, single
manipulated variable per axis:

| Axis | Levels |
|---|---|
| Base model | `Qwen2.5-Coder-7B-Instruct` · `Qwen2.5-Coder-14B-Instruct-AWQ` |
| Agent tool policy | `tool_choice="auto"` (unmodified) · `tool_choice="required"` |

**Manipulation is verified minimal, not asserted:** `toolcheck_agent_src/`
differs from the canonical `measured_default_agent/src/` in exactly two
lines, both the `tool_choice` flag (`diff -rq` confirms one differing file;
`diff` confirms two differing lines). The canonical frozen baseline is never
edited.

**Declared predictions, before the fact:**
1. `7B × required` will show substantially more tool use than
   `7B × auto` (a single real trajectory at n=1, D-38, showed 64 rounds and
   a near-solve; this arm tests whether that generalises across 59 tasks).
2. `14B × auto` will show non-zero tool use, i.e. the floor is a capability
   threshold rather than a property of the scaffold alone. **If `14B × auto`
   is also 0/59 with zero tool calls, that prediction is wrong and will be
   reported as wrong**, and the interpretation shifts to the scaffold's
   default policy being unusable below frontier-model scale.
3. No prediction is registered about absolute resolve rates in any arm.

**Model-choice rationale, recorded now:** the move from 7B toward a larger
model is motivated *solely* by the pre-existing floor result above, decided
before any 14B measurement was taken. Quantisation (AWQ) is used because it
fits the existing 24GB A10 at no additional hardware cost; it does not
weaken the frozen-model claim (D-01/D-03) — weights remain fixed,
self-hosted, and hashable — but it is stated rather than hidden.

**Scope limit:** this sub-study is a scaffold-sensitivity measurement on
`S0` only. It is **not** a frontier-crossing result, does not involve
`S_evo`, and must not be presented as either.

### 4.2 Outcomes against the declared predictions (recorded 2026-08-11)

Written after the runs, scoring §4.1's predictions as stated. **All four arms
complete at n = 59 each.**

| Arm | Resolved | Used tools | Tool calls | Generations |
|---|---|---|---|---|
| 7B × `auto` | 0/59 | 0/59 | 0 | 59 |
| 7B × `required` | 2/59 | 59/59 | 3663 | 3687 |
| 14B × `auto` | 0/59 | 0/59 | 0 | 59 |
| 14B × `required` | 4/59 | 59/59 | 5592 | 5617 |

Tool-use effect vs `auto`: Fisher exact p = 3.4 × 10⁻¹³ for **both** `required`
arms. Model-scale effect under `auto`: p = 1.0 (identical in every cell).
Resolve-rate differences: p = 0.50 (7B) and p = 0.12 (14B) — **neither
significant**, and neither claimed.

**Prediction 1 — CONFIRMED, decisively.** `7B × required` shows
substantially more tool use than `7B × auto`: 59/59 vs 0/59 tasks with at
least one tool call (Fisher exact p = 3.4 × 10⁻¹³), Clopper-Pearson
intervals [0.939, 1.000] vs [0.000, 0.061] — non-overlapping, with no
exception in any of the six languages. The n = 1 observation from D-38
generalises completely.

**Prediction 2 — FALSIFIED.** The prediction was that `14B × auto` would
show non-zero tool use, i.e. that the floor was a capability threshold the
7B model sat below. It is not. At twice the parameter count the 14B arm
reproduced the 7B arm in *every* cell: 0/59 resolved, 0/59 tasks with any
tool call, 59/59 `empty_patch`, the same per-task generation distribution
({1}) and patch-length distribution ({0}); mean completion tokens moved
only 902 → 976. Recorded as wrong, per §4.1's own commitment to do so.

**The interpretation §4.1 named in advance as the consequence of this
falsification therefore applies**: the scaffold's default tool policy — not
model capability — determines whether the agent acts at all, at least
across the scales tested. This is the more useful finding, and would have
been unavailable had the prediction not been registered first: the
comfortable post-hoc story ("a 7B is simply too weak to be an agent") was
both available and false.

---

#### Amendment, 2026-08-14: the measure used to score Prediction 2 was defective

**The scoring above stands as recorded and is not revised.** What follows is
added, not substituted — the prediction was falsified against the measure
declared in §4.1, and that remains the registered outcome. But the measure
itself has since been shown inadequate, and leaving that unstated would
misrepresent what was learned.

§4.1 operationalised "tool use" as **structured `tool_calls`** — the field
HGM's agent loop dispatches on. Reading the container transcripts (not done
at scoring time, because the `auto` arms produced no errors to investigate)
shows the 14B model *attempting* tool calls on **47/59 tasks**, of which
**30/59** name a real tool with every required argument and would have been
dispatched by a lenient parser. The 7B model attempts 5/59, runnable 1/59.
Structured `tool_calls` is 0/59 in both arms and in all six languages.

Consequences for the record, stated plainly:

1. **The prediction's substance was correct; its operationalisation was
   blind.** "The larger model will show non-zero tool use" is true by a wide
   margin (47 vs 5) under any measure that inspects behaviour rather than
   the API field.
2. **"Identical in every cell" is true only of recorded cells.** The two
   arms differ roughly tenfold in attempted actions. The claim as written in
   §4.2 is not wrong, but it is narrower than it reads.
3. **The interpretation above is now overstated and is corrected here.** The
   tool policy does not determine whether the agent *acts*; it determines
   whether the harness can *observe* an attempt. `tool_choice="required"`
   works by constraining decoding to the tool-call grammar, so it repairs
   protocol compliance rather than inducing volition. No claim that the
   policy changes the model's disposition to act is supported by these data.
4. **This is the same failure mode the study is about.** A pre-registered
   measure and a scaffold's selection signal failed identically, and for the
   same reason: both were defined over what the system records rather than
   what the agent does. We report it as such rather than as an incidental
   correction.

Verification supporting the amendment (all re-runnable):
`scripts/check_tool_template.py` confirms the chat template renders the tool
schemas and states the `<tool_call>` convention, so the model was told the
protocol; `scripts/verify_tool_zero.py` sends one identical request under
each policy, returning fenced JSON under `auto` and the same intent as a
structured call under `required`; `scripts/extract_prose_calls.py` produces
the counts above from the stored transcripts. Measurements are recorded in
`results/prose_toolcalls.json`.

**No threshold, arm, or §3 value is changed by this amendment.**

**Prediction 3 held as stated** (no prediction was registered about
absolute resolve rates, and none is claimed). For completeness: the
resolve-rate difference between `7B × auto` and `7B × required` is 0/59 vs
2/59, Fisher exact p = 0.50 — **not statistically significant**, and it is
not claimed as an improvement. This design cannot detect a resolve-rate
shift below roughly 6/59 at α = 0.05.

**One observation not predicted in advance, flagged as post-hoc.** The
0 → 2 movement in resolved tasks is statistically negligible but
algorithmically decisive: HGM's `expand()` filters the archive to nodes of
*strictly positive* utility before sampling, so at exactly zero the
candidate set is always empty and the search cannot take its first step
(the D-41 crash). This connection was noticed after seeing the results and
is reported as an observation, not a tested hypothesis.

**A second post-hoc finding, also flagged as unregistered (D-46).** In 9 of
59 tasks in the `14B × required` arm, the agent used tools productively but
wrote a *new file of its own naming* instead of editing the exercise's
declared solution files — often in the wrong language entirely
(`diamond_kata.py` for the C++ task `cpp__diamond`; `robot.py` and
`spell_number.py` for JavaScript tasks). Because the harness stashes only
the declared solution paths before revealing hidden tests, these attempts
leave nothing gradeable and surface as HGM's `error` marker.

**This required a correction to the analysis, made in the direction that
hurts the result rather than helps it.** D-43 established excluding
infrastructure failures from denominators. These `error` rows superficially
match that pattern but are *not* infrastructure failures — the agent genuinely
failed the task. Excluding them would have reported `4/50 = 8.0%` instead of
the correct `4/59 = 6.8%`, inflating the resolve rate by shrinking the
denominator over exactly the attempts that failed. The analysis now
distinguishes `incomplete` (exception *before* grading — true infra failure,
excluded) from `error` (exception *during* grading — counted as unresolved,
reported separately as `wrong-artifact`).

---

### 4.3 Minimal evolutionary run — declared 2026-08-12, **before** running it

Registered before any `sample_child` call has ever executed. Nothing in this
section has been observed; it is written to constrain what we may later
claim, and to make a null result reportable rather than a failure.

**Why this run is now possible at all, and why that matters for
interpretation.** §4.2 established that under the unmodified default policy
the frozen model scores 0/59 on Polyglot. HGM's `expand()` samples only
archive nodes of *strictly positive* utility, so at a uniformly-zero signal
the candidate set is always empty and the search cannot take its first step
(the D-41 crash). The evolutionary experiment was therefore not merely
unrun but **structurally impossible under the default configuration**. The
`required` arm scores 2/59 (7B) and 4/59 (14B), which is non-degenerate.
This run is only viable because of that, and the paper must say so rather
than presenting the seed choice as routine.

**Fixed configuration, declared now:**

| Parameter | Value |
|---|---|
| Frozen model | `Qwen2.5-Coder-14B-Instruct-AWQ` (best measured seed utility) |
| Seed agent | `toolcheck_agent_src` (the `required` variant), **not** the default |
| Task substrate | The same frozen 59-task Polyglot subset used throughout |
| Diagnosis model | The same local frozen model — **no hosted model**, so the loop remains fully self-hosted and no un-frozen component enters |
| Generations | 2–3, budget permitting; whatever completes is reported |
| Seeds | ≥1. Amended 2026-08-13, **still before any `sample_child` has executed**: additional independent seeds will be run if GPU budget permits, and *every* completed seed is reported, including ones that contradict the others. The amendment only raises intended sample size; no threshold, prediction, or outcome mapping is changed. If exactly one seed completes, that single-run limitation is stated, not glossed. |
| Instrumentation | `fork_bridge` network-layer interception throughout |

**Declared predictions:**

1. The loop will complete at least one `sample_child` → evaluate cycle
   without the D-41 crash. *(If it crashes for a new reason, that is
   reported as the result of this run.)*
2. Any child that improves measured resolve rate will do so **without**
   requiring a support-expanding operation the seed lacked — i.e. we predict
   **no frontier crossing** at this scale. This is the falsifiable one.
3. No prediction is registered about the magnitude of any improvement.

**Analysis plan, fixed in advance:**

- Primary quantity: per-generation resolve rate on the 59-task set, with
  Clopper–Pearson intervals, compared against the seed's own 4/59.
- Every child's operations are classified by interception (never
  self-report), and any resolve-rate gain is decomposed against the
  `S0` / best-of-N elicitation ladder from §4.1 and D-47 before being
  attributed to evolution.
- **A gain that does not exceed the best-of-N elicitation control at
  matched compute is reported as elicitation, not capability expansion.**
- Infrastructure failures follow the D-43/D-46 rules already fixed:
  `incomplete` excluded, `error` counted as unresolved.

**What we will conclude from each outcome, committed now:**

| Outcome | Reported conclusion |
|---|---|
| No child beats the seed | Null: no evidence of improvement at this scale/budget. Reported as such. |
| Child beats seed, but ≤ best-of-N control | Elicitation, not expansion — the headline interpretation. |
| Child beats seed *and* control, no expanding op required | Better search, not capability expansion. |
| Child beats both *and* ablation shows an expanding op was necessary | Candidate frontier crossing — reported as a single-seed, single-fork, one-task-family observation requiring replication, **not** as a general claim. |

**Scope limits stated in advance:** one seed, one fork, one model, one task
family, 2–3 generations. This cannot support a general claim about
self-improving agents, and no such claim will be made from it. It is a
demonstration that the instrument produces an interpretable decomposition on
a real evolutionary run.

---

## 5. Primary and secondary outcomes

**Primary.** Frontier-crossing rate on beyond-frontier held-out tasks, at matched
compute, for `S_evo` versus `S_star`.

**Secondary.**
1. Support-expansion dependency — fraction of crossings removed by ablating
   support-expanding ops.
2. Elicitation-adjusted gain — `S_evo − S_star` at matched compute on held-out.
3. Overfitting gap — `train − held_out` for `S_evo`.
4. Transfer retention — transfer gain ÷ held-out gain of the frozen evolved
   scaffold.

---

## 6. Analysis plan

- Clopper-Pearson CIs on every per-task rate; rule of three plus the exact
  one-sided bound for zero-success tasks.
- Bootstrap CIs on aggregate metrics (`cbs.stats.bootstrap_ci`, percentile
  method); report across evolution seeds as a distribution, never a single run.
- Multiple-comparison correction across tasks: Benjamini-Hochberg
  (`cbs.stats.benjamini_hochberg`), implemented and unit-tested against a
  hand-computed reference case; alpha **locked at 0.05** (§3).
- Any comparison not at matched compute is flagged as such in the output, not
  silently reported (`ComparisonRecord.realised_spend_matched`).
- Truncated runs (budget-exhausted) are excluded from capability claims rather
  than scored as failures; `FrontierRecord.n_budget_exhausted` records them.
- Per-task crossing verdicts computed by `cbs.crossing.evaluate_crossing`;
  aggregate placement into the interpretation matrix below by
  `cbs.interpretation.place_in_interpretation_matrix`, both mechanical once
  §3's thresholds are fixed.

---

## 7. Interpretation matrix (fixed in advance)

| Crossing rate | `S_evo` vs `S_star` @matched | Transfer | Conclusion |
|---|---|---|---|
| ≈ 0 | ≈ | — | **Bounded search** — rediscovers known scaffolding (deflationary) |
| ≈ 0 | `S_evo` > | high | Superhuman elicitation, still within the frontier |
| > 0 | `S_evo` > | high; crossings need support-expanding ops | **Genuine expansion via [mechanism]** (constructive) |
| > 0 | ≈ | vanish on transfer / large overfit gap | **Illusory expansion** — eval overfitting |

---

## 8. Stopping rules

- Frontier sampling stops at `N_max` per task or when a phase budget cap is hit,
  whichever first; truncation is recorded per task.
- No peeking at held-out or transfer results before thresholds in §3 are fixed.
- `S_evo` may evaluate on `train` only. Any held-out evaluation during evolution
  invalidates the overfitting analysis and voids the run.

---

## 9. Declared limitations

- The frontier estimator is a **lower bound**. Sampling can never establish that
  a solution is unreachable, only that it was not reached within `N_max`. Every
  claim is stated relative to the achieved budget. Enforced structurally: a
  record cannot express `p = 0`, and `beyond_frontier` always carries its
  qualifier.
- Findings on ≤8B models may not transfer to frontier scale.
- Where verification runs without a Docker sandbox, the affected records carry
  `verifier_is_security_boundary: false`.

---

*§3's thresholds are locked (signed off 2026-08-04). §4's remaining
**[TO FIX]** is unbuilt engineering (the SWE-bench Verified/Polyglot
integration for `S0`/`S_star`), not an open judgment call — complete it and
commit this file before the first Phase 5 run.*
