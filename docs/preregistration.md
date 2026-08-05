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
