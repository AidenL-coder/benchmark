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

## 3. Thresholds **[TO FIX before Phase 5]**

The combination logic for every row below is implemented and tested
(`cbs.crossing.evaluate_crossing`, `cbs.interpretation.place_in_interpretation_matrix`,
`cbs.stats.benjamini_hochberg` — see `docs/DECISIONS.md` D-28). What remains is
choosing the **values**, not writing the code that consumes them — each is a
required parameter with no default, specifically so it cannot be used
un-chosen.

| Parameter | Symbol / arg | Value | Notes |
|---|---|---|---|
| Frontier budget | `N_max` | **[TO FIX]** | proposed 1000; see docs/DECISIONS.md D-03 |
| Crossing cutoff | `p_hat(x) < 1/N_max` | **[TO FIX]** | follows from `N_max`; `evaluate_crossing` derives it from `FrontierRecord.beyond_frontier` when `S0` is sampled at exactly `N_max` |
| Reliability of a crossing | `evaluate_crossing(k, K, ...)` | **[TO FIX]** | proposed k=3, K=5 |
| Compute-match tolerance | `MatchedComputeHarness(tolerance=...)` | 0.05 | implemented default |
| Confidence level | — | 0.95 | Clopper-Pearson, two-sided |
| Multiple-comparison correction | `cbs.stats.benjamini_hochberg(alpha=...)` | **[TO FIX]** | implemented; proposed alpha=0.05, across tasks within a family |
| Near-zero crossing rate | `place_in_interpretation_matrix(crossing_rate_epsilon=...)` | **[TO FIX]** | proposed 0.0 (exact), applied to BH-*corrected* crossing claims, not raw ones |
| "High" transfer retention | `place_in_interpretation_matrix(transfer_retention_high=...)` | **[TO FIX]** | proposed 0.5 |
| "Large" overfitting gap | `place_in_interpretation_matrix(overfitting_gap_high=...)` | **[TO FIX]** | proposed 0.3 |
| Number of evolution seeds | — | **[TO FIX]** | proposed ≥3; loop is stochastic |

A task counts as **crossed** only if all four conditions of brief §3.3 hold
simultaneously: reliable solve at `k/K`, `p_hat(x)` below cutoff, compute matched
to `N_max`, and ablating support-expanding ops removes the solve.

---

## 4. Models and task families **[TO FIX]**

- Frozen models: ≥2 families × ≥2 sizes. Proposed Qwen2.5-Coder-1.5B / 7B, plus
  one of Llama-3.x-8B or DeepSeek-Coder.
- Primary (coding): original HumanEval vendored and integrated
  (`cbs.tasks.families.humaneval`; 164 problems, all canonical solutions
  verified against the real sandbox — docs/DECISIONS.md D-27). **[TO FIX]**
  before this backs a real capability claim: upgrade to HumanEval+ (the
  original's test suites under-specify correctness) and add MBPP+,
  LiveCodeBench slice, SWE-bench Verified subset.
- Transfer (reasoning): **[TO FIX]** — one set, checkable answers, never
  optimised on.
- Splits frozen and hashed **before** sampling (`cbs splits freeze`); the suite
  hash is quoted in results. HumanEval's is already frozen
  (`data/splits/humaneval.json`, 85 train / 79 held-out).

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
  hand-computed reference case; **[alpha TO FIX]**, proposed 0.05.
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

*Complete every **[TO FIX]** and commit before the first Phase 5 run.*
