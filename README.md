# Capability-Boundary Study of Self-Improving Coding Agents

Measurement apparatus for deciding whether "self-improving" agents that wrap a
**frozen** foundation model in an **evolving scaffold** achieve genuine
capability expansion, or only better elicitation plus benchmark overfitting.

Specs: [`docs/self-improving-agents-proposal.md`](docs/self-improving-agents-proposal.md)
(the why) and [`docs/CLAUDE_CODE_PROJECT_BRIEF.md`](docs/CLAUDE_CODE_PROJECT_BRIEF.md)
(the master spec). Decisions made while building:
[`docs/DECISIONS.md`](docs/DECISIONS.md).

> *Build the measurement, not the hype. The loop is borrowed; the instrument is yours.*

---

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Environment, config, model/sandbox/budget layers | **done** |
| 1 | Task schema, verifier, frozen hashed splits: toy, HumanEval(+), MBPP(+), transfer_reasoning | **done** — 6 `Task`-shaped families, all "+" upgrades complete (D-34/D-35). LiveCodeBench scoped-not-started (D-33). SWE-bench Verified deliberately uses a separate, non-`Task` representation instead (`cbs.tasks.swebench`, D-40) — see Phase 4/5 |
| 2 | Frontier estimation + estimator validation | **done — DoD met** |
| 3 | `S0` / `S_star` baselines, matched-compute harness, elicitation control | **done** for the `Task`-shaped families; SWE-bench Verified variants (`S0SweBench`/`SStarSweBench`, D-40) also **done and real-execution-verified** — both run end-to-end against a real instance with a real vLLM model, producing correct resolved/unresolved verdicts |
| 4 | `S_evo` measurement layer (interception-based tagging, archive, ablation, network-layer proxy bridge) | measurement layer **done** (D-24/D-25, D-38's `cbs.scaffolds.fork_bridge`); a GPU+Docker host has now been proven working end-to-end twice, across two separate provisionings (D-37, then again this session) — the underlying interception/proxy/budget machinery is proven against real Docker+model infrastructure via HGM, HyperAgents (D-38/D-39), and SWE-bench Verified (D-40); *running an actual self-improving loop* (not just the frozen baseline agent) has still not happened |
| 5 | Crossing test and ablations | determination logic **done** (`cbs.crossing`); running it against a real evolved scaffold's results has still not happened |
| 6 | Analysis and write-up | statistical plan pieces (bootstrap CIs, BH correction) and interpretation-matrix placement **done** (`cbs.stats`, `cbs.interpretation`); write-up itself needs real results |

---

## Phase 4/5's remaining piece is a real evolving loop, not infrastructure

`S_evo`'s measurement layer (interception-based support tagging, archive
persistence, ablation) is built and tested — see DECISIONS.md D-24 through
D-26. Running a real forked self-improvement loop means executing
self-modifying orchestration code with live network access to a model
endpoint, which brief §10 requires Docker for; neither this machine (no
Docker) nor Colab (containers can't nest Docker) can provide that boundary,
so this all runs on a rented GPU+Docker host, provisioned and torn down
between sessions as needed (D-37).

**D-12/D-31 (which fork, how tasks meet its agent) are confirmed**:
`metauto-ai/HGM` as primary `S_evo`, `facebookresearch/HyperAgents` as a
second independently-implemented variant (its own CC-BY-NC-SA-licensed
module, kept separate from `cbs`), `jennyzzt/dgm` retained only as a
literature-baseline reference; `S_evo` evolves natively against HGM's own
SWE-bench Verified/Polyglot harnesses (D-31's option (b)), not adapted
`cbs` task families.

**Both forks' Docker/model integration is now execution-verified against a
real host, not just planned** (D-37/D-38/D-39): HGM's polyglot harness runs
real tasks against a real vLLM-served model end to end; HyperAgents needed
(and now has) its own `litellm`-based local-endpoint patch, a
`network_mode="host"` fix, and a real ~600-second-hang root cause fixed
(an eval-index suffix corrupting its model string) before its own harness
also ran real tasks end to end.

**`S0`/`S_star` on SWE-bench Verified are done too, and also
execution-verified** (D-40): `cbs.tasks.swebench` (a deliberately
non-`Task` representation — a repo-plus-diff problem doesn't fit
`Task`/`TaskSuite`'s prompt-plus-code-string shape), `cbs.scaffolds.
swebench_scaffold.S0SweBench`/`SStarSweBench` (unit-tested against
synthetic fakes), and `scripts/swebench_glue.py` (the real Docker/HGM glue)
have all been run for real against a real SWE-bench Verified instance,
producing correct resolved/unresolved verdicts for both scaffolds. See
`docs/DECISIONS.md` D-40 for the full validation trail, including four real
bugs found and fixed along the way and the honestly-flagged remaining gaps
(coarse pytest-exit-code pass/fail rather than per-test-node parsing; the
repair branch itself not yet observed in a live run).

**What's actually left for Phase 4/5**: forking HGM/HyperAgents into a real,
running self-improvement loop and observing whether it crosses the
frontier — the measurement instrument itself is now built and proven
against real infrastructure end to end, for both the frozen baseline agent
and the SWE-bench Verified substrate `S_evo` will run on.

---

## Quick start

```bash
py -m venv venv
./venv/Scripts/python.exe -m pip install -e ".[dev]"     # Windows
# python -m venv venv && source venv/bin/activate && pip install -e ".[dev]"

cbs env                                                  # what this host can do
cbs tasks verify                                         # every reference solution passes
cbs frontier validate                                    # Phase 2 DoD
cbs frontier estimate --config configs/frontier_toy_mock.yaml --dry-run
```

No GPU is needed for any of the above: the `mock` backend is a deterministic
frozen model with **known** `p(x)` and known solution richness, which is what
makes the estimators checkable against ground truth at all.

---

## What is actually novel here

The self-improvement loop is meant to be forked, not rebuilt (brief §5.1). The
contribution is the measurement layer:

**Frontier estimation** (`cbs.frontier`) — best-of-N sampling under the minimal
scaffold `S0` on a temperature schedule, with Clopper-Pearson intervals, the rule
of three for zero-success tasks, unbiased pass@k, Good-Turing unseen mass, Chao1
richness, and rarefaction curves. All implemented from scratch and unit-tested
against published reference values.

**Support partition** (`cbs.scaffolds.tagging`) — every scaffold operation is
classified support-preserving (selection among `M`'s own samples) or
support-expanding (routes information from outside `M`'s single-shot
distribution), each with a written rationale. Unknown operations raise, so an
evolved scaffold cannot emit untagged work. This is the handle for the study's
falsifiable claim: any genuine frontier-crossing must depend on at least one
support-expanding operation.

**Matched compute** (`cbs.budget`) — all model spend flows through a nesting
accountant. `MatchedComputeHarness` issues identical per-task allowances *and*
verifies realised spend afterwards, because equal allowances that one system
under-spends are not a matched comparison.

**Verification** (`cbs.tasks`) — executable, sandboxed, deterministic. Success
requires an explicit marker written to fd 1, so a candidate cannot forge a pass
via `sys.exit(0)` or by reassigning `print`.

**Task families** — `toy` (synthetic, known ground truth, validates the
instrument itself); `humaneval`/`humanevalplus` (164/163) and `mbpp`/`mbppplus`
(427/374), all real and vendored — brief §8's primary coding family, with both
evalplus "+" upgrades now done (D-34, D-35); and `transfer_reasoning` (10
hand-authored math/logic/combinatorics tasks — compound interest, Nim, the
Josephus problem, prime factorisation, and similar — D-17), frozen entirely
into the `transfer` split. Every canonical/reference solution across all real
families passes the actual sandbox and verifier (164/164, 163/163, 427/427,
374/374, 10/10), which is a materially stronger validation than the toy
family alone: real code has genuine syntax diversity, multi-line bodies, and
imports no hand-written example anticipates — running it surfaced **six real
upstream bugs across the four benchmarks**, all found by verification rather
than assumed away: two extraction bugs on `humaneval` (D-27); on
`humanevalplus`, a task whose own generated test fails its reference solution
(D-34) plus the need for an entirely new AST-based public-test derivation,
since HumanEval+'s tests are `inputs`/`results` list loops rather than flat
`assert` statements; and on `mbppplus`, a floating-point-tolerance gap in
evalplus's own helper plus three tasks whose generated test computes a result
and never asserts it — a non-functional test that accepts any candidate
(D-35). All excluded and documented, never silently patched into the vendored
data. `humaneval`/`mbpp` (original, not evalplus-extended) have under-specified
hidden tests and near-certain pretraining contamination — documented in each
family's `ATTRIBUTION.md` and DECISIONS.md D-27/D-29/D-30/D-34/D-35: fine for
instrument validation, not a real capability claim on their own; prefer the
"+" variants for that. LiveCodeBench (investigated and scoped as a materially
bigger change, D-33) remains out of scope. SWE-bench Verified is
**load-bearing rather than optional** (D-13, per D-12/D-31) — it's the
substrate `S_evo` will evolve against natively — and, via its own
deliberately-separate `cbs.tasks.swebench` representation (not this
family list), is now built and real-execution-verified for `S0`/`S_star`
(D-40, see "What is actually novel here" below).

**`S_star`** (`cbs.scaffolds.s_star`) — the strong fixed baseline (brief §4):
best-of-N, execution feedback, tool use (a static compile check), and
self-consistency (majority vote by canonical form), all support-tagged. The
hidden test oracle is queried exactly once, to score the final choice —
identical to `S0` — because execution feedback and selection read only a
separate, deliberately weaker `public_tests` subset (`Task.public_tests`, see
DECISIONS.md D-18/D-19). A real deployed agent does not get to try candidates
against the grading suite and keep the first that passes; letting `S_star` do
so would make any elicitation gain it shows an artefact of oracle access.

**Elicitation control** (`cbs.compare`) — `S0` vs `S_star` at matched
per-attempt compute `B`, the control that separates "genuine expansion" from
"known scaffolding tricks" (brief §9.1). `S0`'s side is read off `pass@B` on an
*oversampled* frontier record (never resampled at exactly `B`, which degenerates
— DECISIONS.md D-22); `S_star`'s side is measured empirically over independent
repetitions, each under a fresh per-rep budget. Run via `cbs compare`.

**`S_evo` measurement layer** (`cbs.scaffolds.evolved`, `cbs.archive`,
`cbs.ablation`) — the adapter any forked self-improvement loop's agent code
plugs into. Support-class tagging cannot be self-reported by evolved,
untrusted code the way `S0`/`S_star` self-tag (an optimisation process has no
incentive to honestly flag an operation that makes it look less like bounded
search), so `InterceptionSession` classifies every operation from the outside
by watching what the agent actually did — including resolving the one real
ambiguity (did a verifier result get used for *selection* or fed back to
*condition the next generation*?) from behavioural evidence: does a later
prompt contain a fragment of an earlier failure's error text? Ablation follows
the same logic: `disabled_ops` blocks the information channel itself, so an
"ablated" scaffold is denied a capability regardless of what its own code
tries to do, not just reconfigured and hoped to comply. `cbs.archive` computes
the overfitting gap and transfer retention (brief §9.1) and a crude
hard-coding triage heuristic. None of this requires running real
self-modifying code — see the Phase 4 note below for what's still needed
for that.

**`S0`/`S_star` on SWE-bench Verified** (`cbs.tasks.swebench`,
`cbs.scaffolds.swebench_scaffold`, `scripts/swebench_glue.py`, D-40) — a
repo-plus-diff problem doesn't fit `Task`/`TaskSuite`'s prompt-plus-code-
string shape, so this is a deliberately separate, non-`Task` measurement
path rather than a forced-fit loader. `S0SweBench` generalizes "one call to
`M`" to "one full agent trajectory" (the natural atomic unit once solving
requires real tool use, not a blind completion); `SStarSweBench` adds
best-of-N, execution feedback (against `PASS_TO_PASS`, never
`FAIL_TO_PASS` — the same public/hidden split `Task.public_tests`/`tests`
already enforce), and self-consistency across trajectories. Both run
end-to-end against real Docker infrastructure and a real vLLM-served
model, producing correct resolved/unresolved verdicts — not just
unit-tested against synthetic fakes, though that validation exists too.

---

## The estimator's central limitation

Sampling cannot prove a solution is unreachable — only that it was not reached
within `N_max`. This is the proposal's most important caveat (§8) and it is
enforced in the data model, not left to prose:

- a record cannot express `p = 0`; `p_upper_bound` is always positive,
- `beyond_frontier` always ships with the qualifier *"not reached at compute
  N=…; p(x) < … at 95% confidence. This is a budget-relative statement, not
  unreachability."*,
- Good-Turing and Chao1 return `None`, never `0.0`, on a zero-success task —
  because there is no sample to compute them from, and `0.0` would read as
  "fully covered" for exactly the tasks where nothing was observed.

---

## Validation

`cbs frontier validate` runs four independent checks:

| Check | What it catches | Latest result |
|---|---|---|
| CI coverage (simulated, 400 replicates × 6 true `p`) | miscalibrated intervals | ≥ nominal at every `p`, including `p=0` |
| Chao1 lower-bound behaviour | richness over-estimation | never below observed (200/200); ≤ truth 97.5% |
| Verifier vs known labels | **false positives** — the failure mode that would manufacture a crossing | 0 FP, 0 FN in 600 labelled samples |
| Full pipeline vs ground truth | plumbing faults end to end | 5/6 recovered; `P(≥1 miss \| calibrated) = 0.26` → pass |

The pipeline check passes on a binomial tail test rather than demanding 6/6:
a 95% interval is *supposed* to miss ~1 in 20, so an all-must-pass gate would
fail ~26% of the time on a healthy pipeline (see DECISIONS.md D-10). Calibration
itself is measured at high power separately — empirically 0.944 over 90
task-seeds.

```bash
./venv/Scripts/python.exe -m pytest       # ~350 tests (the four real benchmark families each verify every reference solution and public subset against the real sandbox, well over a thousand executions total; several minutes)
```

`cbs compare --config configs/compare_toy_mock.yaml` runs the same idea for
`S_star`: on the mock backend it should *not* reliably beat `S0` (the mock is
content-blind and cannot benefit from reading its own error messages — D-20),
and on two toy tasks it measurably loses to `S0` because a naive public-test
subset happens to admit a specific wrong answer that only the hidden oracle
catches (D-21). Both are documented, expected results, not bugs — the point of
running this on the mock is to validate the comparison harness itself before
trusting it on a real model.

---

## Running against a real model

Everything reaches the frozen model through one interface, so vLLM, Ollama and
hosted providers are a config change (DECISIONS.md D-01):

```yaml
model:
  backend: openai_compat
  base_url: http://localhost:8000
  model: Qwen/Qwen2.5-Coder-1.5B-Instruct
  local: true
```

See [`configs/frontier_vllm_colab.yaml`](configs/frontier_vllm_colab.yaml) and
[`notebooks/colab_bootstrap.ipynb`](notebooks/colab_bootstrap.ipynb).

**Sizing.** `N_max=1000` over ~100 tasks ≈ 200k generations ≈ 5–6 T4-hours per
model, and bounds a zero-success task at `p < 0.003`. Always `--dry-run` first;
it reports the sample count without spending anything.

**Resume.** Frontier runs are the dominant cost and Colab preempts, so samples
are appended to JSONL shards and `fsync`-ed. Re-running resumes where it stopped.
The temperature schedule is prefix-stable (D'Hondt), so widening `N_max` never
rewrites the temperatures of samples already drawn, and a truncated run still
spans the schedule instead of collecting only low-temperature draws. Changing any
condition that would move the frontier refuses to pool with the existing shard.

---

## Safety posture

All model-generated code runs sandboxed. Docker is the real boundary (network
`none`, read-only rootfs, `cap-drop ALL`, non-root, memory/PID limits); where
Docker cannot exist — Colab nests containers, the Windows dev box lacks it — the
subprocess backend is used and **is not a security boundary**. That fact is
recorded on every execution result and every frontier record rather than assumed
away.

Frontier sampling of a small frozen model on curated tasks is acceptable there.
**Running `S_evo`'s self-modifying scaffolds is not** — Phase 4 requires Docker
(DECISIONS.md D-02).

---

## Layout

```
src/cbs/
  budget.py        matched-compute accountant
  config.py        YAML configs with inheritance + fingerprinting
  cli.py           one command reproduces one experiment
  models/          frozen-model clients: mock (known ground truth), openai_compat
  sandbox/         docker (security boundary) | subprocess (not)
  tasks/           schema, verifier, canonicalisation, frozen hashed splits
  scaffolds/       support tagging, S0, S_star, and the S_evo interception adapter
  frontier/        estimators, sampler, records, validation
  compare.py       S0 vs S_star at matched compute (elicitation control)
  archive.py       overfitting gap / transfer retention / hard-coding triage
  ablation.py      scaffold-agnostic ablation runner
  crossing.py      four-part frontier-crossing determination (brief 3.3)
  interpretation.py  mechanical placement into the interpretation matrix
  stats.py         bootstrap CIs, Benjamini-Hochberg multiple-comparison correction
configs/           base + experiment profiles
docs/              specs, decision log, pre-registration
tests/             ~350 tests
```
