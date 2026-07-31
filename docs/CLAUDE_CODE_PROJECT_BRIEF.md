# Project Brief — Capability-Boundary Study of Self-Improving Coding Agents

**Audience: a coding agent (Claude Code) that will implement this project.**
This document is the master spec. Read it fully before writing code. Work phase by
phase (§7). Each phase has a Definition of Done — do not advance until it is met.
Some steps spend real compute/money or execute untrusted model-generated code; obey
the guardrails in §10 and **ask the human before any large run or any live network/API
spend**.

---

## 1. How to use this document

- The **research goal** (§2–§3) tells you *why* — use it to resolve ambiguity. When an
  implementation choice affects what is being *measured*, favor scientific correctness
  over convenience, and surface the tradeoff to the human.
- **Do not rebuild the self-improvement loop from scratch.** Fork an existing one (§5)
  and add the measurement layer on top. Your novel work is the *measurement*, not the loop.
- Anything marked **[DECISION]** is for the human to confirm — list these and ask.
- Anything marked **[VERIFY]** may be stale in this brief; check the live repo/README/docs
  before relying on it.

---

## 2. Research question (the "why")

Modern "self-improving" agents (Darwin Gödel Machine / DGM, AlphaEvolve, Huxley-Gödel
Machine, etc.) wrap a **frozen** foundation model in an **evolving scaffold** and report
large benchmark gains. It is unestablished whether those gains are:

1. **Genuine capability expansion** — solving problems the frozen model truly could not, or
2. **Bounded search** — better *elicitation* of latent capability plus *overfitting* of the
   scaffold to the evaluation benchmark.

This project builds the measurement apparatus to decide this, and to localize any genuine
expansion to a specific mechanism.

**The result must be informative either way** (see interpretation matrix, §9.3).

---

## 3. Core scientific definitions the code MUST implement faithfully

These are not flavor text; they define what the code computes.

### 3.1 Support partition (the theoretical spine)
Every scaffold operation is tagged as one of:
- **support-preserving** — emits samples from the frozen model `M` followed by
  selection/aggregation (best-of-N, majority vote, self-consistency, reranking). *Cannot*
  produce a correct output that has ~zero probability under `M`.
- **support-expanding** — routes information outside `M`'s single-shot distribution:
  execution/error feedback loops, external tool/solver calls, retrieval, or decomposition
  of a task into subproblems each individually within `M`'s frontier.

Implement operation tagging in the scaffold so every solved task can be attributed to the
operation classes it used. **Prediction to test:** any genuine frontier-crossing depends on
at least one support-expanding operation.

### 3.2 Reachable-solution frontier
For a task `x` and frozen model `M`, `p(x)` = base success probability under a **minimal
fixed scaffold** (single call + trivial formatting), estimated by `N`-sample best-of-N with
a verifier.
- **Practical frontier at budget N** = tasks with `p̂(x) > 0` within `N` samples.
- **Beyond-frontier** = zero successes across a large budget `N_max`; bound `p` statistically
  (rule of three: 0 successes in `N` ⇒ `p < 3/N` at ~95%).
- Estimate *unseen* correct-solution mass with **Good–Turing / Chao** estimators; report the
  frontier as a function of budget with confidence bounds. **Never claim absolute
  unreachability — only "not reached at compute N_max."**

### 3.3 Frontier-crossing (operational)
An evolved agent crosses the frontier on `x` iff ALL hold:
- solves `x` reliably (≥ `k`/`K` runs),
- `M`'s base success `p̂(x) < 1/N_max`,
- comparison is at **inference compute matched** to `N_max` (crossing is not just "more samples"),
- ablating support-expanding ops from the evolved scaffold removes the crossing.

### 3.4 Decomposition of any measured gain
- **Expansion** = frontier-crossing that survives on held-out and transfer tasks at matched compute.
- **Elicitation** = gain over the base agent that a **strong fixed baseline** also achieves at matched compute.
- **Overfitting** = train-benchmark gain that does not replicate on a held-out split of the same distribution or on transfer tasks.

---

## 4. Systems to compare (build all three)

1. **`S0` base agent** — minimal fixed scaffold (single call + formatting).
2. **`S_star` strong fixed baseline** — expert-designed generic agentic scaffold: best-of-N,
   self-consistency, execution feedback, standard tool use — **held fixed, no evolution.**
   This is the critical control (represents "known good scaffolding without self-improvement").
3. **`S_evo` self-improving system** — the forked DGM-style loop with a **frozen** `M`.

All three evaluated at **matched total inference compute** (equal model-token/call budget per task).

---

## 5. What to reuse vs. build

### 5.1 Reuse (fork one; keep others as references) — [VERIFY all URLs/branches]
- **Primary fork (recommended): `github.com/jennyzzt/dgm`** — the original authors' code;
  matches the published method, so `S_evo` is a credible, literature-aligned baseline.
- **`github.com/metauto-ai/HGM`** (Huxley-Gödel Machine, ICLR 2026 oral, built on DGM) —
  strong current reference and a possible second `S_evo` variant.
- **`github.com/lemoz/darwin-godel-machine`** — good patterns for **sandboxed execution** and
  **cost instrumentation**; borrow these.
- **`github.com/mmtmn/Darwin-Godel-Machine`** — runs a **local LLM via Ollama**; useful if you
  prefer Ollama over vLLM for the frozen model.
- **`github.com/princeton-nlp/SWE-bench`** — SWE-bench harness + Verified subset.

[DECISION] Confirm primary fork with the human. Default: `jennyzzt/dgm`, backend swapped to a
local vLLM server (§6) so the frozen model is an open-weight model, not an API model.

### 5.2 Build (your novel contribution)
- **Frontier-estimation pipeline** (§3.2): high-throughput best-of-N sampling + verification +
  Good–Turing/Chao unseen-mass estimation.
- **Support-tagging instrumentation** (§3.1) on the scaffold.
- **`S_star`** strong fixed baseline (§4).
- **Matched-compute harness** — enforce equal call/token budgets across `S0`/`S_star`/`S_evo`
  and across the frontier estimator.
- **Ablation runners** (§9.2) and **analysis/statistics** (§9).

---

## 6. Environment & setup (Phase 0)

Do these in order. [VERIFY] exact commands against each repo's current README.

1. **OS/hardware:** Linux + NVIDIA GPU (≥24GB VRAM target for 7–8B models; smaller works for 1.5B).
   Confirm `nvidia-smi` works.
2. **Docker** (mandatory — all model-generated code runs sandboxed): install, then
   `docker run hello-world`. Add user to docker group if needed.
3. **Python env:** `python3 -m venv venv && source venv/bin/activate` (or conda per repo).
4. **Model serving:** install **vLLM**; serve the frozen open models with an OpenAI-compatible
   endpoint so the forked loop (which speaks the OpenAI API) can point at it by base-URL.
   (Alternative: Ollama, per the mmtmn fork.)
5. **Frozen models** — [VERIFY current best small open coding models]; defaults:
   - `Qwen2.5-Coder-1.5B` and `Qwen2.5-Coder-7B` (family 1),
   - a `Llama-3.x-8B` or `DeepSeek-Coder` model (family 2).
6. **Clone + install the primary fork**, then its benchmark deps (SWE-bench at the pinned
   commit the repo specifies; Polyglot via its prepare script).
7. **Datasets/benchmarks** (§8): SWE-bench Verified, Polyglot, HumanEval+/MBPP+, a
   LiveCodeBench slice, and a reasoning/transfer set with checkable answers.
8. **Tracking:** Weights & Biases (or equivalent) for runs; ample disk for the agent archive.
9. **Secrets:** provider API keys only if any comparison uses hosted models; keep them out of
   the repo (env vars).

**Phase 0 Definition of Done:** vLLM serves a frozen model; the forked loop runs one tiny
end-to-end iteration in Docker on a 5-task smoke set without errors; W&B logs it.

---

## 7. Implementation phases

Each phase: implement → test → meet DoD → surface [DECISION]s → proceed.

### Phase 1 — Task infrastructure & verification
- Assemble curated task splits per family: `train` (loop may evaluate on it), `held-out`
  (same distribution, never seen by loop), `transfer` (different family).
- Wrap each benchmark with a **reliable verifier** (executable tests; SWE-bench harness).
  Audit a sample of "solved" cases manually to bound false positives.
- Implement **contamination checks** against likely pretraining exposure; prefer newer or
  perturbed variants where possible.
- **DoD:** given (model, scaffold, task) you can produce a verified pass/fail deterministically
  (fixed seed), and splits are frozen and hashed.

### Phase 2 — Frontier-estimation pipeline
- Massive best-of-N sampling of `M` under the minimal scaffold `S0`, temperature schedule
  tuned for solution diversity; record `p̂(x)` and the *set of distinct* correct solutions.
- Implement rule-of-three bounds and **Good–Turing/Chao** unseen-mass estimation.
- Output per-task frontier records with CIs; flag `p̂(x) < 1/N_max` as beyond-frontier candidates.
- **DoD:** on a toy task with known ground-truth solve rate, estimator recovers it within CI;
  frontier report generated for `held-out` + `transfer` at the chosen `N_max`.

### Phase 3 — Baselines: `S0` and `S_star`
- `S0`: minimal scaffold. `S_star`: strong fixed generic agent (best-of-N, self-consistency,
  execution feedback, tools) — no evolution.
- Implement the **matched-compute harness**: a shared budget accountant (model calls + tokens)
  that all systems draw from equally.
- **DoD:** `S0` and `S_star` run across all models/splits at matched compute; budgets logged
  and equal within tolerance.

### Phase 4 — `S_evo` with support-tagging
- Adapt the forked loop to the frozen open model; keep the archive/selection/self-mod machinery.
- Instrument every scaffold operation with a **support-class tag** (§3.1) so each solved task
  carries the set of operation classes it used.
- Run with **multiple seeds** (loop is stochastic).
- **DoD:** `S_evo` completes seeded runs at matched compute; every solved task has a support-tag
  trace; archive persisted.

### Phase 5 — Core tests & ablations
- **Crossing test:** on beyond-frontier tasks, does `S_evo` solve them at matched compute?
- **Support-class ablation:** disable support-expanding ops one at a time; measure which
  crossings vanish.
- **Elicitation control:** `S_evo` vs `S_star` at matched compute on `held-out`.
- **Overfitting/archive ablation:** `train − held-out` gap; inspect archive for eval-specific
  hard-coding.
- **Transfer test:** freeze an evolved scaffold, apply to `transfer` with no further evolution.
- **DoD:** every metric in §9.1 computed with CIs and across seeds; ablation matrix (§9.2) filled.

### Phase 6 — Analysis & write-up support
- Place results in the **interpretation matrix** (§9.3); generate figures/tables; run the
  statistical plan (§9.4).
- Produce a results appendix (per-task frontier data, ablation tables, seed variance).
- **DoD:** a reproducible analysis notebook/script regenerates every headline number from raw logs.

---

## 8. Benchmarks/datasets

- **Coding (primary):** SWE-bench **Verified** (human-validated, solvable), Polyglot,
  HumanEval+/MBPP+, a LiveCodeBench slice for cheap iteration.
- **Reasoning (transfer):** a disjoint set with automatically checkable answers (math/logic);
  the loop never optimizes on it.
- **Splits:** `train` / `held-out` (same dist) / `transfer` (diff dist), frozen and hashed.
- **Cost control:** staged evaluation — agents pass a small sanity subset (e.g., 10 tasks)
  before running full sets. Cap curated task counts (e.g., 100–300) per [DECISION] budget.

---

## 9. Metrics, ablations, interpretation, statistics

### 9.1 Metrics
- pass@1, pass@k (and CoT-pass@k on reasoning tasks — score intermediate steps).
- **Frontier-crossing rate** (fraction of beyond-frontier tasks solved at matched compute).
- **Support-expansion dependency** (fraction of crossings removed by ablating support-expanding ops).
- **Elicitation-adjusted gain** = `S_evo − S_star` at matched compute.
- **Overfitting gap** = `train − held-out` for `S_evo`.
- **Transfer-retention** = transfer gain of frozen evolved scaffold ÷ its held-out gain.

### 9.2 Ablation matrix (fill for each model family)
| Ablation | Removes | Tests |
|---|---|---|
| Disable execution feedback | a support-expanding op | is it load-bearing for crossings? |
| Disable tool calls | a support-expanding op | " |
| Disable retrieval | a support-expanding op | " |
| Disable decomposition | a support-expanding op | " |
| Archive memory strip | eval-specific hard-coding | overfitting share of gain |
| Compute-unmatch → match | extra samples | is "gain" just more compute? |

### 9.3 Interpretation matrix (result → conclusion)
| Crossing rate | `S_evo` vs `S_star` @matched | Transfer | Conclusion |
|---|---|---|---|
| ≈ 0 | ≈ | — | **Bounded search** (deflationary; rediscovers known scaffolding) |
| ≈ 0 | `S_evo` > | high | Superhuman elicitation, still within frontier |
| > 0 | `S_evo` > | high; crossings need support-expanding ops | **Genuine expansion via [mechanism]** (constructive) |
| > 0 | — | vanish on transfer / big overfit gap | **Illusory expansion** (eval overfitting) |

### 9.4 Statistical plan
- **Pre-register** hypotheses, thresholds (`k/K`, `N_max`, `1/N_max` cutoff), and this matrix
  before the full run (commit a `preregistration.md`).
- Clopper–Pearson CIs for `p̂(x)`; rule-of-three for zero-success tasks.
- Multiple loop seeds; report distributions, not single runs.
- Bootstrap CIs for aggregate metrics; multiple-comparison correction across tasks.

---

## 10. Guardrails (safety, budget, reproducibility)

- **Sandbox everything.** All model-generated code executes inside Docker with no host mounts
  it doesn't need and constrained network. This is both safety and reproducibility.
- **Budget caps.** A hard accountant for model calls / tokens / dollars / GPU-hours. **Stop and
  ask** before exceeding a per-phase cap [DECISION: set caps].
- **Ask before live spend.** Any hosted-API call or large sampling run (frontier estimation is
  the big one) requires explicit human go-ahead; do a dry-run cost estimate first.
- **Determinism/repro.** Fixed seeds, pinned dependency versions, hashed data splits, config-
  driven runs (no magic constants in code — everything in a config file).
- **Checkpoint.** Persist the archive and all run state so long jobs resume.
- **No release** of a maximally capable self-improving artifact; this project characterizes
  *limits* at small scale.

---

## 11. Coding conventions

- Python; config-driven (YAML/JSON) experiments; one command reproduces one experiment.
- Keep the forked loop code isolated from your **measurement modules** (clean separation so the
  novel contribution is legible and reusable).
- Unit-test the verifier, the budget accountant, the frontier estimators (esp. Good–Turing/Chao),
  and the support-tagging.
- Log raw per-task records (not just aggregates) so analysis is re-derivable.
- Every figure/number in the paper must be regenerable from raw logs by a single script.

---

## 12. Definition of done (whole project)

- Clean, controlled results on **≥2 model families × ≥2 task families**, at matched compute.
- The expansion / elicitation / overfitting decomposition computed with CIs and seed variance.
- The frontier-estimator's lower-bound limitation handled explicitly in analysis and text.
- Every result placed in the interpretation matrix; reproducible end-to-end.
- (Reach) a crisp mechanistic account + a falsifiable prediction the "it self-improves" story
  cannot make (the specific support-expanding lever that flips bounded→unbounded, or a
  characterized ceiling).

---

## 13. Open decisions to surface to the human [DECISION]

1. Primary fork (`jennyzzt/dgm` default) and backend (vLLM local vs Ollama vs hosted).
2. Final benchmark selection and per-family task counts (cost driver #1).
3. `N_max` for frontier estimation (cost driver #2) and the `1/N_max` crossing cutoff.
4. Whether to add a third model family.
5. Per-phase budget caps (dollars / GPU-hours).
6. Which single reasoning set is the transfer family.

---

## 14. Key references (read before/while building) — [VERIFY links]

- **Darwin Gödel Machine** — Zhang, Hu, Lu, Lange, Clune (arXiv:2505.22954; ICLR 2026). Code: `jennyzzt/dgm`.
- **Huxley-Gödel Machine (HGM)** — ICLR 2026 oral. Code: `metauto-ai/HGM`.
- **RLVR capability-boundary debate** — the intellectual sibling (expansion vs. resampling);
  read the support-based analyses (on-policy RL cannot amplify zero-probability completions) —
  they justify the §3.1 partition.
- **AlphaEvolve** — coding agent for algorithmic discovery (arXiv:2506.13131).
- **Open-endedness / quality-diversity** — MAP-Elites, novelty search (explains archive dynamics).
- **SWE-bench / SWE-bench Verified** — `princeton-nlp/SWE-bench`.

---

*Build the measurement, not the hype. The loop is borrowed; the instrument is yours.*
