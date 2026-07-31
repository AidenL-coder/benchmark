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
| 1 | Task schema, verifier, frozen hashed splits, toy family | **done** (toy family; real benchmarks pending) |
| 2 | Frontier estimation + estimator validation | **done — DoD met** |
| 3 | `S0` / `S_star` baselines, matched-compute harness, elicitation control | **done** |
| 4 | `S_evo` (forked DGM loop) with support-tagging | not started |
| 5 | Crossing test and ablations | not started |
| 6 | Analysis and write-up | not started |

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
./venv/Scripts/python.exe -m pytest       # ~185 tests
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
  scaffolds/       support tagging + S0 minimal scaffold + S_star strong baseline
  frontier/        estimators, sampler, records, validation
  compare.py       S0 vs S_star at matched compute (elicitation control)
configs/           base + experiment profiles
docs/              specs, decision log, pre-registration
tests/             ~150 tests
```
