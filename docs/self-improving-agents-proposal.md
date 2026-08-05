# Does Self-Improvement Expand Capability, or Only Search Within a Frozen Frontier?

### A Controlled Study of Capability Boundaries in Self-Referential Coding Agents

**Research proposal — draft v1**

---

## 1. One-paragraph summary

A new class of "self-improving" agents (the Darwin Gödel Machine, AlphaEvolve, Gödel Agent, and a 2026 wave of follow-ups) reports large gains by having a coding agent iteratively rewrite its own scaffold and validate each change on benchmarks. Every one of these systems wraps a **frozen** foundation model in an **evolving** scaffold — so it is not obvious whether the reported gains reflect *genuine capability expansion* or *bounded search over capability the frozen model already had*. This proposal designs a controlled study to answer that question. The central instrument is a method for estimating the frozen model's **reachable-solution frontier** and testing whether evolved agents ever cross it — and if they do, isolating *which* class of scaffold move is responsible. The result is informative either way: a deflationary finding ("self-improvement is bounded scaffold search") reframes a hyped subfield, and a constructive finding ("this specific mechanism crosses the frontier") identifies the real lever for open-ended improvement.

---

## 2. Motivation

### 2.1 The phenomenon

Self-referential improvement — an AI that rewrites its own code to get better at rewriting its own code — is one of the oldest goals in the field (Schmidhuber's Gödel Machine) and has become empirically active. The Darwin Gödel Machine (DGM) reports raising SWE-bench performance from 20% to 50% and Polyglot from ~14% to ~31% by evolving an archive of coding agents that modify their own Python scaffold and validate each change on coding benchmarks. Critically, the DGM authors keep the underlying foundation model frozen and explicitly defer the compute-intensive version (agents rewriting their own training and training new foundation models) to future work. AlphaEvolve, Gödel Agent, R&D-Agent, Group-Evolving Agents, and related 2026 systems share the same structure: a frozen model, an evolving wrapper, empirical validation on a target benchmark.

### 2.2 The unexamined assumption

Because the model is frozen and only the **scaffold** changes, the system's output distribution is always a *function of* the frozen model's distribution. This raises a question the incumbent papers do not rigorously answer, because their objective is to make the demo work, not to stress-test what the improvement *is*:

> When an evolved agent solves a problem the base agent could not, has the system acquired new capability, or has it merely found a better way to elicit capability that was already latent in the frozen model — possibly by overfitting its scaffold to the evaluation benchmark?

This is the direct analog of the reinforcement-learning-with-verifiable-rewards (RLVR) "expansion vs. resampling" debate, where the field split over whether RL expands a model's reasoning boundary or only sharpens selection within it, and where the disagreement turned out to hinge on measurement (which metric, how long, which tasks). That debate is now crowded. The self-improving-agent version of the same question is **not** — and it has a cleaner structural handle, because the scaffold's operations are inspectable and can be partitioned by whether they can, even in principle, exceed the frozen model's support.

### 2.3 Why it matters

- **Scientific:** It tells us whether current "self-improvement" is a genuine route to open-ended capability growth or a rebranding of scaffold engineering and benchmark adaptation.
- **Practical:** If gains are bounded by the frozen frontier and largely eval-specific, reported numbers overstate transferable progress; if a specific mechanism crosses the frontier, that mechanism is the thing worth scaling.
- **Safety-relevant:** Understanding the *limits* of self-improvement — where it plateaus, what it can and cannot do with a fixed model — is a prerequisite for reasoning about recursive self-improvement risk, and is the responsible, measurement-first version of this research area.

---

## 3. Research questions and hypotheses

**RQ1 (support).** Can an evolved agent, driven by a frozen model, reliably solve held-out problems that lie outside the frozen model's empirically estimated reachable-solution frontier?

**RQ2 (mechanism).** If it can, which class of scaffold operation is responsible — support-preserving moves (sampling, ranking, voting, self-consistency) or support-expanding moves (execution feedback, tool use, retrieval, problem decomposition)?

**RQ3 (elicitation vs. expansion vs. overfitting).** How much of the measured gain over a base agent is attributable to (a) genuine frontier-crossing, (b) better elicitation of latent capability at matched compute, and (c) scaffold overfitting to the target benchmark rather than transferable capability?

**RQ4 (transfer).** Do evolved scaffolds transfer their gains to a disjoint task family they never trained on, or are the gains benchmark-specific?

**Primary hypotheses (stated to be falsifiable):**

- **H1 (bounded-search / null).** Evolved-agent gains do not exceed a strong, fixed, human-designed generic scaffold at matched inference compute, and any apparent frontier-crossings are attributable to support-expanding *primitives already available to the base agent* plus eval-specific overfitting. Under H1, "self-improvement" reduces to automated rediscovery of known scaffolding tricks and adaptation to the eval set.
- **H2 (genuine expansion).** Evolution discovers support-expanding scaffold structures that reliably cross the frozen frontier on held-out and transfer tasks, beyond what the strong fixed baseline achieves at matched compute. Under H2, the specific discovered mechanism is the contribution.

The study is designed so that **either** outcome is a publishable result, and so that partial/mixed outcomes map onto specific interpretations (§7).

---

## 4. Key definitions and formal framing

Let `M` be a frozen foundation model and `S` a scaffold: a program that may query `M`, execute code, call tools, and select among candidate outputs, ultimately emitting a solution to a task `x`. A self-improving system evolves `S` while holding `M` fixed.

**Support-preserving vs. support-expanding operations.** Partition scaffold operations into two classes:

- **Support-preserving:** operations whose emitted solutions are always samples from `M`'s conditional distribution followed by selection/aggregation (best-of-N, majority vote, self-consistency, reranking). A support-preserving scaffold can raise success *rate* at fixed budget through better selection, but **cannot** emit a correct completion that has zero probability under `M`. (This mirrors support-based analyses of on-policy RL, which cannot amplify zero-probability completions.)
- **Support-expanding:** operations that route information *outside* `M`'s single-shot distribution — executing code and feeding back errors, calling external tools/solvers, retrieving documents, or decomposing `x` into subproblems each individually within `M`'s frontier and composing their solutions. These *can*, in principle, produce correct end-to-end solutions that `M` would never sample directly.

This partition is the theoretical spine: **if** genuine expansion occurs, it must run through support-expanding operations, and the study can localize it there.

**Reachable-solution frontier.** For task `x`, define `M`'s per-task base success probability `p(x)` under a *minimal fixed scaffold* (single call plus trivial formatting), estimated by `N`-sample best-of-N with a verifier. Define the **practical frontier** at budget `N` as the set of tasks with `p̂(x) > 0` at that budget. Tasks with zero successes across a large budget `N_max` are treated as **beyond the practical frontier**, with `p(x)` upper-bounded statistically (rule of three: with 0 successes in `N`, `p < 3/N` at ~95% confidence). We estimate the mass of *unseen* correct solutions with Good–Turing / Chao-style estimators to quantify how much frontier we may be under-counting.

**Frontier-crossing (operational).** An evolved agent "crosses the frontier" on task `x` if it solves `x` reliably (≥ k/K runs) while `M`'s estimated base success probability satisfies `p̂(x) < 1/N_max`, **and** the crossing is not explained by compute (the evolved agent is compared at inference compute matched to `N_max`), **and** ablating support-expanding operations from the evolved scaffold removes the crossing.

**Capability expansion vs. elicitation vs. overfitting.**
- *Expansion:* frontier-crossing that survives on held-out and transfer tasks at matched compute.
- *Elicitation:* gain over the base agent that the **strong fixed baseline** also achieves at matched compute (i.e., known generic scaffolding, not something evolution uniquely found).
- *Overfitting:* gain on the training/target benchmark that does **not** replicate on a held-out split of the same distribution or on transfer tasks.

---

## 5. Experimental design

### 5.1 Frozen models (the fixed core)

At least **two families × two sizes**, to separate model-specific from general effects, all open-weight and runnable at modest scale:

- Qwen2.5-Coder (e.g., 1.5B and 7B)
- Llama-3.x or DeepSeek-Coder (e.g., 8B-class)

The model is **frozen** throughout; only scaffolds evolve. Small models are a feature, not a limitation: they make the frontier estimable (see §5.4) and keep the study affordable.

### 5.2 Task families

- **Primary (coding):** a curated subset of an executable benchmark with reliable verification — e.g., MBPP+/HumanEval+, a LiveCodeBench slice, or a size-controlled SWE-bench-Verified subset. Executable ground truth is essential for automatic verification.
- **Transfer (reasoning):** a disjoint family the loop never optimizes on (e.g., a math/logic set with checkable answers), used only to test RQ4.
- **Splits:** each family is partitioned into `train` (the loop may evaluate on it), `held-out` (same distribution, never seen by the loop), and `transfer` (different distribution/family). Contamination is checked against the frozen models' likely pretraining exposure.

### 5.3 Systems compared

1. **Base agent** `S₀`: a minimal fixed scaffold (single call + formatting).
2. **Strong fixed baseline** `S★`: an expert-designed, generic agentic scaffold with best-of-N, self-consistency, execution feedback, and standard tool use — *held fixed*, no evolution. This is the critical control: it represents "known good scaffolding without self-improvement."
3. **Self-improving system** `S_evo`: a scaled-down DGM-style loop — an archive of scaffolds, open-ended selection, self-modification of scaffold code, empirical validation on `train`. Frozen `M`.

All three are evaluated at **matched total inference compute** (equal number of model-token calls per task), so that "improvement" is never confounded with "more sampling."

### 5.4 Frontier estimation protocol

For each model × each task in `held-out` and `transfer`:
- Run `N_max` best-of-N samples under `S₀` at a temperature schedule chosen to maximize solution diversity; verify each.
- Record `p̂(x)`, the empirical solve probability, and the *set* of distinct correct solutions found.
- Fit Good–Turing / Chao estimators to estimate unseen correct-solution mass and report the frontier as a function of budget, with confidence bounds.
- Flag `p̂(x) < 1/N_max` tasks as candidate "beyond-frontier" targets for the crossing test.

`N_max` is set as large as budget allows (e.g., 10³–10⁴ per task on small models), and all crossing claims are reported *relative to the achieved budget*, never as absolute unreachability (see §8).

### 5.5 Core tests and ablations

- **Crossing test (RQ1/RQ2):** On beyond-frontier tasks, measure whether `S_evo` solves them at matched compute. For any crossing, ablate support-expanding operations one at a time (disable execution feedback / tools / retrieval / decomposition) to identify the responsible mechanism.
- **Support-class decomposition (RQ2):** Instrument the evolved scaffold to tag every operation as support-preserving or support-expanding; measure what fraction of solved beyond-frontier tasks depend on at least one support-expanding operation. (Prediction: crossings, if any, require support-expanding ops — a falsifiable structural claim.)
- **Elicitation control (RQ3):** Compare `S_evo` vs. `S★` at matched compute on `held-out`. If `S_evo ≈ S★`, evolution merely rediscovered known scaffolding → not expansion.
- **Overfitting / archive ablation (RQ3):** Compare `S_evo` performance on `train` vs. `held-out`; inspect the archive for eval-specific hard-coding (e.g., pattern-matching to benchmark idioms). Quantify the `train − held-out` gap.
- **Transfer test (RQ4):** Apply the frozen evolved scaffold to `transfer` tasks with no further evolution; measure retained gain.
- **Compute-matching control (throughout):** Re-run all comparisons under equalized model-call budgets to rule out "improvement = more compute."

### 5.6 Metrics

- pass@1 and pass@k (and CoT-pass@k on reasoning tasks, scoring intermediate steps) for base, `S★`, and `S_evo`.
- **Frontier-crossing rate:** fraction of beyond-frontier tasks solved at matched compute.
- **Support-expansion dependency:** fraction of crossings eliminated by ablating support-expanding ops.
- **Elicitation-adjusted gain:** `S_evo − S★` at matched compute.
- **Overfitting gap:** `train − held-out` performance of `S_evo`.
- **Transfer-retention:** `transfer` gain of the frozen evolved scaffold ÷ its `held-out` gain.

---

## 6. Statistical plan

- Pre-register hypotheses, metric definitions, thresholds (`k/K`, `N_max`, the `1/N_max` crossing cutoff), and the interpretation matrix (§7) before running the full loop.
- Report per-task confidence intervals for `p̂(x)` (Clopper–Pearson) and rule-of-three bounds for zero-success tasks.
- Multiple seeds for the evolutionary loop (it is stochastic); report variance across seeds, not a single run.
- Correct for multiple comparisons across tasks when aggregating crossing claims.
- Bootstrap CIs for all headline aggregate metrics.

---

## 7. Interpretation matrix (result → conclusion)

| Crossing rate on beyond-frontier tasks | `S_evo` vs `S★` at matched compute | Transfer retention | Conclusion |
|---|---|---|---|
| ≈ 0 | `S_evo ≈ S★` | — | **Bounded search.** Self-improvement rediscovers known generic scaffolding; no expansion. (Strong deflationary result.) |
| ≈ 0 | `S_evo > S★` | high | Evolution finds *better elicitation* than experts, but stays within the frontier. (Nuanced: "self-improvement = superhuman scaffold engineering, not new capability.") |
| > 0 | `S_evo > S★` | high, crossings depend on support-expanding ops | **Genuine expansion via [mechanism].** Identify and characterize the lever. (Strong constructive result.) |
| > 0 | — | crossings vanish on transfer / high overfitting gap | **Illusory expansion.** Crossings are eval-specific overfitting, not capability. (Deflationary + a cautionary methodology contribution.) |

Every cell is a paper-worthy statement, which is the point: the design cannot return an uninformative result.

---

## 8. Threats to validity

- **The frontier estimator is a lower bound.** Sampling can never prove a solution is *unreachable*, only that it was not reached within budget `N_max`. Mitigation: report everything relative to budget, use Good–Turing unseen-mass estimates, and frame conclusions as "practical frontier at compute `N_max`," never as information-theoretic impossibility. This is the single most important limitation and must be stated plainly.
- **Verifier reliability.** Executable tasks with strong test suites minimize false positives; audit a sample of "solved" cases manually. On reasoning tasks, use CoT-pass@k to reduce guess-and-check false positives.
- **Benchmark contamination.** Frozen models may have seen benchmark solutions; run contamination probes and prefer newer or perturbed task variants.
- **Small-model generalization.** Findings on ≤8B models may not transfer to frontier scale. Frame claims accordingly and treat scale as an explicit axis for future work, not an implicit assumption.
- **Compute confounds.** Rigorously matched-compute comparisons throughout; any unmatched comparison is flagged.
- **Evolutionary stochasticity.** Multiple seeds; report distributions.

---

## 9. Feasibility, compute, and timeline

Designed for a resourced individual **without** a cluster:

- **Models:** ≤8B open weights, runnable on a single modern GPU (or modest cloud/Colab-Pro-class access); frontier estimation and evolution loops are batchable and can run on small task subsets.
- **Cost drivers:** the `N_max` frontier sampling and the evolutionary validation calls. Both are controllable by capping task-set size (e.g., 100–300 curated tasks) and `N_max`.
- **Rough phase plan:**
  1. *Weeks 1–3:* build/port a minimal DGM-style loop; assemble curated task splits + verifiers; implement operation-tagging (support class).
  2. *Weeks 4–6:* frontier estimation pipeline; validate estimators on a toy task with known ground truth.
  3. *Weeks 6–10:* run base / `S★` / `S_evo` at matched compute across models; collect crossing candidates.
  4. *Weeks 10–13:* ablations (support-class, elicitation, overfitting, transfer); statistics.
  5. *Weeks 13–16:* analysis, interpretation-matrix placement, writing.

Timeline is indicative; the task-set size and `N_max` are the main tunable knobs for fitting a real budget.

---

## 10. Related work and positioning

- **Self-improving agents:** Darwin Gödel Machine, AlphaEvolve, Gödel Agent, R&D-Agent, Group-Evolving Agents — this proposal studies *what their improvement is*, rather than building a stronger one.
- **RLVR capability-boundary debate:** the intellectual sibling (expansion vs. resampling), including support-based arguments that on-policy updates cannot amplify zero-probability completions; this proposal imports that lens into the scaffold-evolution setting, where the support partition is *inspectable*.
- **Open-endedness / quality-diversity:** MAP-Elites, novelty search (the machinery DGM-style archives use) — relevant to *why* evolution might or might not find support-expanding moves.
- **Memorization / contamination / benchmark validity:** grounds the overfitting analysis.

**Current literature check (2025-2026 wave, added after a targeted novelty
pass — see `docs/DECISIONS.md` D-36 for the full write-up and its caveats;
this was a search-plus-abstract/full-text pass on ~8 papers, not a systematic
citation-graph review, and should be redone properly before submission):**

- **Robeyns, Szummer & Aitchison, "A Self-Improving Coding Agent" (SICA,
  arXiv 2504.15228, NeurIPS 2025 preprint)** — the closest structural match:
  a frozen model wrapped by a self-editing scaffold, 17%→53% on a SWE-bench
  Verified subset. Read in full: it reports the performance delta as evidence
  of self-improvement without independently estimating the frozen model's own
  ceiling (no repeated-sampling/best-of-N/pass@k baseline), without testing
  an elicitation-vs-expansion distinction, and without preregistration or
  contamination discussion. This is the paper our contribution most needs to
  differentiate itself from, and on inspection the differentiation holds:
  SICA's methodology cannot answer RQ1-RQ3 as posed here.
- **Zhang, Zhao, Foerster, Clune, et al., "Hyperagents" (DGM-Hyperagents,
  arXiv 2603.19461, `facebookresearch/HyperAgents`)** — DGM's own original
  author extending DGM to make the meta-level modification procedure itself
  editable. Read in full: same gap as SICA (no elicitation-vs-expansion test,
  no independent frontier estimate), and the authors *themselves* list
  "evaluation protocols remain fixed" as a stated limitation of their own
  work — i.e., they've conceded the exact gap RQ1-RQ3 target. Also a strong
  candidate second `S_evo` implementation for this study (see
  `docs/DECISIONS.md` D-12), independent of its relevance here as prior art.
- **Starace, "Scaffold Effects on GAIA: A Controlled Comparison" (arXiv
  2606.08529)** — not a self-improvement paper, but the closest *methodological*
  relative: a preregistered, controlled comparison across fixed scaffolds and
  five frontier models, quantifying the elicitation gap directly (up to 28
  points from scaffold choice alone) and rejecting its own preregistered
  hypothesis that more-capable models are less scaffold-sensitive. Confirms
  the elicitation-gap framing and preregistration discipline this proposal
  already uses are the current bar in this space, not overkill. Its scaffolds
  are static, never evolving — the structural difference from RQ1/RQ2 here.
- **METR's time-horizon methodology** (arXiv 2503.14499) and **Apollo
  Research's forecasting work** (arXiv 2502.15850) — read in full, not just
  abstracts. Neither has a formal elicitation-vs-expansion framework: METR's
  "elicitation" is a pragmatic scaffold-selection step (pick a good scaffold,
  tune it, move on), not a measured, decomposed quantity, and its confidence
  intervals bound the *trend*'s slope, not any individual capability
  estimate's floor/ceiling. Apollo's "elicitation level" is a coarse
  low/high input variable to a forecasting model, not a mechanism decomposed
  into what specifically drives it. Both are strong citable grounding for
  *why* this space cares about elicitation at all and for the "measured
  capability is a lower bound" framing generally — but neither is a
  methodological competitor: neither studies an evolving scaffold, neither
  has a frontier-crossing concept, and neither attributes gains to specific
  operations.
- **A closely-related, genuinely important find from a proper citation-graph
  pass (not caught by the earlier keyword-search-only check) — a 3-paper
  cluster by İşcan et al., all frozen small code models, all preregistered:**
  "Try Again, Don't Look Back" (arXiv 2607.26117), "Falsification, Not
  Exposure" (arXiv 2606.31511), and "Form, Not Content?" (arXiv 2607.12962).
  These run exactly the comparison this proposal's own `example_agents.py`
  validates against synthetic data (`blind_best_of_n_agent` vs
  `feedback_repair_agent`) — blind resampling vs. execution-feedback-
  conditioned self-repair — at real scale (0.5B-7B), with real preregistration
  (Holm-corrected McNemar tests, a priori power analysis, hashed seed
  namespaces, placebo-controlled arms that ablate task-relevant content while
  keeping the scaffold's *form* identical). Their finding: blind resampling
  ties or beats execution feedback at these scales — feedback's measured
  value over a content-free placebo is statistically indistinguishable from
  zero. Read in full, not just abstracts, specifically to check whether this
  closes the gap this proposal targets. It does not, for two confirmed
  reasons: (1) **all three use a fixed, static retry loop** — the model's
  weights and the scaffold's structure are both frozen across attempts; one
  paper explicitly cites Darwin Gödel Machine and frames its own work as
  deliberately *contrasting* with "the broader agenda of agents that learn
  from their own histories," i.e. the authors themselves scope this as
  static-scaffold work, not evolving-scaffold work; (2) **none has a formal
  frontier-estimation or frontier-crossing concept** — their "dead" (unsolved)
  task-cell unit is defined operationally as zero passes in a small cached
  pool (N=8), explicitly caveated in one paper as making "no direct claim
  about latent sampling capacity, model distribution, or general
  program-synthesis ability" — exactly the gap this proposal's Clopper-Pearson/
  Good-Turing/Chao1 frontier estimator is built to close rigorously instead of
  leave caveated. Genuinely useful for this proposal in two ways beyond
  citation: their finding (feedback ≈ placebo in *fixed* scaffolds at these
  scales) is a strong prior that if this proposal's own `S_star` shows no
  gain from execution feedback either, that would corroborate rather than
  surprise — and if `S_evo`'s *evolved* scaffold does cross the frontier via
  a support-expanding operation a fixed retry loop never had access to (tool
  use, decomposition, retrieval), that is a materially more interesting
  contrast to draw explicitly in the eventual write-up than either result
  alone.
- **Forward-citation check on the İşcan cluster itself, done** (Semantic
  Scholar API, one query per arXiv ID). Two of the three papers have zero
  citing papers yet. The third (arXiv 2606.31511, "Falsification, Not
  Exposure") has exactly one: a survey, "Recursive Self-Improvement in AI:
  From Bounded Self-Refinement to Autonomous Research Loops" (arXiv
  2607.07663), taxonomizing 1,250 self-improvement papers (2024-2026) along
  what's-improved and degree-of-loop-closure axes — a survey, not new
  experiments. Read in full (not just the abstract) specifically to check
  whether it already joins this proposal's two building blocks: it names
  the Darwin Gödel Machine explicitly as an instance of open-ended RSI
  ("maintaining an open-ended archive of self-modifications validated
  against coding benchmarks") and separately discusses elicitation as
  amplifying latent capability rather than creating it — but never
  combines the two into a measured claim, and never uses the words
  "frontier crossing" or "capability frontier" in any form. Confirms both
  of this proposal's building blocks (DGM, the elicitation-gap framing) are
  independently on other researchers' radar as of mid-2026, but nobody has
  yet joined them the way this proposal does. Gap still holds.
- Checked and set aside as lower-overlap: **The Red Queen Gödel Machine**
  (arXiv 2606.26294 — co-evolves the *evaluator*, a different question),
  **Meta-Agent Challenge** (arXiv 2606.04455 — benchmarks meta-agents
  building artifacts from scratch against human baselines, not capability
  attribution), **DemoEvolve** (arXiv 2605.24539 — sparse-feedback harness
  evolution in long-horizon games), **SIA** (arXiv 2605.27276 — updates model
  weights *and* harness, outside this proposal's frozen-`M` premise but a
  useful citation for why keeping `M` frozen is necessary for clean
  attribution), and a **PACE / anytime-valid acceptance-testing line**
  (arXiv 2606.08106, 2607.00871) on the statistics of *committing* a change
  during self-evolution (avoiding the evolutionary loop "p-hacking itself")
  — a different question (accept/reject one step of evolution) from this
  proposal's (did the whole trajectory cross the frontier), but a relevant
  citation for the shared discipline of rigorous statistics in this space.

**The gap this fills:** no current work isolates whether frozen-model
self-improving agents cross a rigorously estimated capability frontier, nor
decomposes any gain into expansion vs. elicitation vs. overfitting, nor ties
crossings to an inspectable support-preserving/expanding partition. That gap
held up under a first novelty check against the current 2025-2026 wave (SICA
and Hyperagents came closest and both fall short of it on inspection), and
held up further under a proper citation-graph pass (via the Semantic Scholar
API against DGM/HGM/SICA's actual citing papers, not just keyword search) —
the closest work found, the İşcan preregistered self-repair cluster above,
studies the right *comparison* (feedback vs. resampling) with real rigor, but
on a fixed scaffold and without a frontier-crossing concept, which is exactly
where this proposal's contribution sits instead. A forward-citation check on
that cluster itself turned up one survey (arXiv 2607.07663) that independently
name-checks both DGM and the elicitation-gap framing this proposal relies on,
but never joins them into a measured claim — consistent with, not a
challenge to, the gap this proposal targets. Still not to be trusted as final
without one more pass closer to the actual submission date, since this field
moves fast enough that a paper posted the week before could still land in the
gap. The instrument itself — reusable across future self-improving
systems — is a contribution independent of which
hypothesis wins.

---

## 11. Expected contributions

1. A **reusable frontier-estimation-and-crossing methodology** for evaluating any frozen-model self-improving system.
2. The **support-preserving vs. support-expanding partition** as a theoretical handle for what self-improvement can and cannot do with a fixed model.
3. An **empirical answer**, on ≥2 model families, decomposing self-improvement gains into expansion / elicitation / overfitting.
4. Either a deflationary reframing of the self-improving-agent narrative or the identification of the specific mechanism that enables genuine crossing.

---

## 12. Honest bar for impact

- **Main-track-worthy:** a clean, controlled result on ≥2 model families and ≥2 task families, with the expansion/elicitation/overfitting decomposition done rigorously and the frontier-estimator limitations handled carefully.
- **Reach (oral/award-tier):** additionally, a crisp mechanistic account with a *falsifiable prediction the "it self-improves" story cannot make* — e.g., demonstrating the specific support-expanding lever that flips bounded to unbounded, or establishing a hard, characterized ceiling. This tier depends on execution and on the result being genuinely surprising, and it is not something the topic alone guarantees.

---

## 13. Responsible-research note

The study is deliberately small-scale, uses frozen models, runs all self-modifying code in a sandboxed environment, and does not aim to produce or release a maximally capable recursively self-improving artifact. Its orientation is toward *characterizing limits* — the measurement-first, safety-relevant version of self-improvement research.

---

*Draft v1 — intended as a working document to hand to a mentor and iterate on. `S_evo` fork choice (D-12: HGM primary, HyperAgents as a second variant, DGM as literature baseline only) and preregistration thresholds (D-14/D-32) are both confirmed/locked as of 2026-08-04 — only whether to add a third model family (D-15) remains an open question. A citation-graph novelty pass against the current literature, including a forward-citation check on the closest cluster found, is in §10/D-36 — one more recency sweep is still needed closer to actual submission.*
