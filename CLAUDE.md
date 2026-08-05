# Orientation for the next Claude Code session

Read this first. It's the fast on-ramp — what this project is, what exists,
what's actually been verified vs. assumed, what's blocked, and where to look
for more depth. Last commit is `74be8f9` (message `f` — a checkpoint commit,
not a normal descriptive one; the fork_bridge/docs work from the D-38 session
landed there). There is further **uncommitted** work on top of it as of this
writing: `CLAUDE.md`, `docs/DECISIONS.md`, `docs/preregistration.md`, and
`docs/self-improving-agents-proposal.md` all modified, plus the second
review round's fixes to `src/cbs/scaffolds/fork_bridge.py` and
`tests/test_fork_bridge.py` (concurrency/drain fixes, status-aware trace
reconstruction — see D-38). Nothing from this stretch has been committed
yet — check `git status` before assuming otherwise. If code and this file
disagree, trust the code and `git log` — update this file when that happens.

**The bar just changed.** The user has explicitly stated the goal is not just
a working instrument but the strongest possible paper — targeting NeurIPS,
with a stated best-paper ambition. That reshapes several calls that were
previously "good enough": self-hosting an open-weight model instead of a
hosted API (D-01/D-03 — verifiable frozen-ness and contamination visibility
matter for reviewer trust), running two independently-implemented `S_evo`
forks instead of one (D-12), leaning toward the harder-but-more-faithful
option on D-31, and treating a real novelty/related-work check (D-36) as
something to redo properly before submission, not a one-off. Read D-36 in
`docs/DECISIONS.md` before assuming this space is unclaimed — it isn't
empty, but the specific gap this project targets looked real as of the last
check.

---

## 1. What this project is

A research-instrument build, not a product. The research question (full detail
in [`docs/self-improving-agents-proposal.md`](docs/self-improving-agents-proposal.md)):

> When a "self-improving" coding agent (Darwin Gödel Machine-style: a **frozen**
> foundation model wrapped in an **evolving** scaffold) solves a problem the
> base model couldn't, is that genuine capability expansion, or just better
> elicitation of latent capability plus overfitting to the eval?

The master spec is [`docs/CLAUDE_CODE_PROJECT_BRIEF.md`](docs/CLAUDE_CODE_PROJECT_BRIEF.md)
— read it before touching anything if you haven't. It defines the phases, the
support-preserving/support-expanding partition, the frontier-crossing
definition, and the guardrails. **This repo's job is the measurement
instrument, not a working self-improvement loop.** The brief is explicit:
"Build the measurement, not the hype. The loop is borrowed; the instrument is
yours." Don't rebuild a DGM-style loop from scratch — fork one (still an open
decision, see §4).

Every non-obvious choice made while building is logged in
[`docs/DECISIONS.md`](docs/DECISIONS.md), each with a **Why** and, where
relevant, what was measured to justify it. That file is the single most
important thing to skim before making a change that touches something already
built — a lot of things that look like arbitrary choices (why is `S_star`'s
selection blind to the hidden oracle? why does `pass_at_k` need `N_max` bigger
than the comparison budget?) are load-bearing decisions with a paper trail.
[`docs/preregistration.md`](docs/preregistration.md) holds the statistical
plan. Its §3 thresholds are now **locked** (signed off 2026-08-04, D-14/D-32)
and must not be revisited after seeing results, or the whole point of
pre-registering is lost. §4 (models and task families) still carries
`[TO FIX]`, but that's unbuilt engineering (the SWE-bench Verified/Polyglot
integration), not an open judgment call.

---

## 2. Where things stand

| Phase | Scope | Status |
|---|---|---|
| 0 | Config, model client, sandbox, budget accountant | **done** |
| 1 | Task schema, verifier, frozen hashed splits | **done** — 6 families: `toy`, `humaneval` (164), `humanevalplus` (163, D-34), `mbpp` (427), `mbppplus` (374, D-35), `transfer_reasoning` (10, hand-authored). Use the "+" variants over the originals for anything beyond instrument validation. SWE-bench Verified still open (D-13); LiveCodeBench investigated and scoped as a materially bigger, cross-cutting change, not started (D-33). |
| 2 | Frontier estimation (Clopper-Pearson, Good-Turing, Chao1, rarefaction) | **done, DoD met** — validated against known ground truth, verifier false-positive rate checked at 0/600 |
| 3 | `S0`, `S_star`, matched-compute harness, elicitation control | **done** |
| 4 | `S_evo` measurement layer: interception-based tagging, ablation, archive | measurement layer **done**; *running* a real evolved scaffold is blocked (§3) |
| 5 | Crossing test, ablation matrix | determination **logic** done (`cbs.crossing`); running it for real is blocked (§3) |
| 6 | Statistical plan, interpretation matrix, write-up | bootstrap CIs / BH correction / matrix placement **done**; the write-up itself needs real results |

380 tests, all passing (`./venv/Scripts/python.exe -m pytest`, a few minutes —
the HumanEval/MBPP tests each verify every reference solution *and* every
derived public-test subset against the real sandbox, so they alone are well
over a thousand subprocess executions).
[`README.md`](README.md) has the fuller version of this table plus the file
layout; it's written for a human reader rather than an agent, so it's a good
second stop after this file.

---

## 3. What's actually blocked, and why

**D-23 is resolved as of this session — read this before assuming infra is
still the blocker.** A real Docker+GPU host now exists and has been proven
working end-to-end (D-37 in `docs/DECISIONS.md` has the full write-up, and
it is worth reading in full — it includes a real self-correction, not just a
success story):
Lambda Labs on-demand instance (RunPod's shared multi-tenant Pods turned out
to disable privileged/nested-Docker mode entirely — confirmed, not assumed —
so Lambda was used instead; a real VM, not a shared container, so Docker just
works), 1x A10 (24GB), Docker confirmed with GPU passthrough, vLLM 0.26.0
serving `Qwen/Qwen2.5-Coder-7B-Instruct`, HGM cloned and run for a real
single-task end-to-end smoke test. **The first attempt's apparent success was
wrong** — it looked clean (no crash, a real evaluation report) but the
in-container log line it was judged by only proved the model client was
*configured*, not that the call *succeeded*; the actual agent output showed
five straight connection errors, because Docker's default bridge network
doesn't let a container reach the host's `localhost` at all. Caught only by
checking the real output file instead of trusting the summary, per this
project's own stated discipline. Two real fixes followed: `network_mode="host"`
added to both container-creation call sites (HGM's own and the vendored
SWE-bench harness's), and vLLM relaunched with `--enable-auto-tool-choice
--tool-call-parser hermes` (HGM's agent hard-requires structured
`tool_calls`, confirmed against the model's own chat template). The *second*
attempt is the real one: the containerized agent reached the model and got
a genuine 670-token substantive response — no error, `tool_calls=None`
because the model chose not to invoke a tool on that specific trial, which
is a question about agent behavior (exactly what this study measures), not
a broken pipeline. **The brief's Phase 0 DoD is met for real.** Three further
setup mistakes were made and fixed along the way (venv placed inside the
forked repo got swept into every archive-node snapshot; the editable
SWE-bench install had to be redone after moving the venv; the initial
baseline's task count turned out not to be controlled by `--max_task_evals`
at all) — full detail in D-37, worth reading before
repeating any of them. **One thing is not yet resolved**: a `TS_sample`
argmax-on-an-empty-sequence crash in the node-selection logic, hit only after
deliberately shrinking the task set to 1 for the smoke test — plausibly an
artifact of that shrinkage, not a real bug, but not reproduced or root-caused
against the real 60-task subset yet.

**The measurement-layer bridge itself is now built (D-38), correcting the
"just monkeypatch it" framing this file and D-12/D-37 used to have.** HGM's
agent runs as a separate OS process inside a Docker container, not something
`InterceptionSession` can wrap in-process — so `cbs.scaffolds.fork_bridge`
intercepts at the network layer instead (`ModelCallProxy`, a real reverse
proxy between the container and the model server). Unit-tested locally
(`tests/test_fork_bridge.py`, now 26 tests against a real backend server)
and validated against 7 real Polyglot tasks, catching three real bugs along
the way — a module-level global (`polyglot.harness.llm`) that has to be set
before calling `process_entry`, a missing `build_env_images` setup step, and
a genuine race condition in the proxy itself (caught by re-running tests
after an unrelated edit, not by design review). **A second, independent
4-dimension review round then found and fixed two more real critical bugs**
(reset()/stop() not draining in-flight handler threads — a real cross-task
contamination risk; unhandled backend exceptions silently dropping a call
from `.events` entirely) plus a moderate one (failed calls were being
counted as real generations) — see D-38 for the full, slightly humbling
story of both rounds, including what was deliberately left unfixed and why.
**Resolved, 2026-08-05: the `tool_call` classification path is now validated
against a real tool-call round trip, not just synthetic conversations.**
9/9 unforced real runs (`tool_choice="auto"`) still show the baseline agent
making one plain generation and stopping — plausibly real 7B-model behavior,
not a bug, and still an open question about baseline agent behavior in its
own right. But rather than keep waiting on it to happen unprompted, forced
the issue directly: an **isolated copy** of the frozen agent
(`hgm/toolcheck_agent_src/`, never the canonical `measured_default_agent/`)
had its two hardcoded `tool_choice="auto"` sites changed to `"required"` —
deliberate, clearly-scoped, validation-only, not real measurement data.
Result: a real 64-round tool-using trajectory, correctly classified
end-to-end — `used_expanding: true` produced by a real run for the first
time (previously only ever seen in synthetic tests), a genuine substantive
patch (passed most of the hidden suite, failed one edge case), not a
degenerate trace. Full detail in `docs/DECISIONS.md` D-38's closing update.

**D-12, D-31, and D-14 are now all confirmed by the user (2026-08-04)** —
read this before assuming any of them are still open judgment calls; only
the engineering work they imply remains.
- **D-12 — fork choice, confirmed.** `metauto-ai/HGM` is the primary `S_evo`
  (actively maintained, clade/subtree promise estimation, ICLR 2026 oral).
  `facebookresearch/HyperAgents` runs as a second, independently-implemented
  `S_evo` variant — its own authors admit "evaluation protocols remain
  fixed," i.e. they haven't solved the elicitation-vs-expansion question
  either, and showing a crossing result holds across two independently-built
  evolutionary mechanisms is much stronger than one implementation's
  idiosyncrasy. **HyperAgents is CC BY-NC-SA 4.0, not Apache-2.0** — keep any
  code touching it in its own clearly-separated module, never merged into
  the rest of `cbs`, since share-alike would otherwise pull that license onto
  whatever it's merged with. `jennyzzt/dgm` is kept available only as the
  brief's requested literature-baseline reference (§5.1), not run as a third
  `S_evo`. **Not yet built** — neither fork has actually been cloned into
  this repo yet.
- **D-31 — task integration, confirmed.** Option (b): `S_evo` evolves
  natively against HGM's own already-working `swe_bench/harness.py` and
  `polyglot/harness.py`, not `humaneval`/`mbpp`/`transfer_reasoning`, which
  remain useful for `S0`/`S_star` calibration only. `S0`/`S_star` will also
  get measured on SWE-bench Verified, via a thin `cbs` wrapper around HGM's
  own harness rather than a from-scratch port, so all three scaffolds sit on
  the same substrate for the primary comparison. **Not yet built.**

**D-39, new this session — both forks shallow-cloned locally (this machine
has no Docker but does have plain GitHub access) and checked against real
source rather than left as the D-12 comparison table's estimate:**
- **HGM needs no `llm.py` patch at all** (D-37's finding, re-confirmed
  against a fresh clone) — `create_client` already has a native
  `elif "vllm" in model.lower()` branch. Don't go looking for a patch to
  write here; there isn't one.
- **HyperAgents does need one, and it's a different shape than assumed.**
  Its `agent/llm.py:get_response_from_llm` is not shared code with HGM's
  `llm.py` — it's a separate implementation built on `litellm.completion()`
  with no `api_base`/vLLM awareness at all. Pointing it at a local vLLM
  server needs a real, small patch (~10-20 lines: pass `api_base=<url>` and
  a `hosted_vllm/<model>`-style prefix through `completion_kwargs` when a
  designated local-model string is used — litellm's own documented
  mechanism). Don't assume zero-cost parity with HGM here.
- **HyperAgents' Docker networking already defaults to `network_mode="host"`**
  (`utils/docker_utils.py`) — D-37's bridge-network-can't-reach-localhost fix
  needed for HGM does **not** need to be re-applied for HyperAgents. A real
  cost reduction, not just a wash against the litellm patch above.
- The model-call chokepoint is still singular in both forks (`task_agent.py`/
  `meta_agent.py` → `llm_withtools` → `agent/llm.py:get_response_from_llm`),
  so `fork_bridge.py`'s network-layer proxy should still work unmodified for
  HyperAgents in principle.
- Also found while reading `swe_bench/report.py`: actual pass/fail grading
  isn't in `harness()` at all (that only produces a candidate patch) — it's
  `make_report` → `run_evals`, which shells out to the vendored SWE-bench
  package's own `run_evaluation.py`. `cbs`'s SWE-bench Verified wrapper needs
  to call that, not just `harness()`, to get a real resolved/unresolved
  verdict. Full detail, plus a flagged-but-unresolved `SWE-bench` vs.
  `SWE-bench_Verified` dataset-loading discrepancy in HGM's own code, in
  `docs/DECISIONS.md` D-39.
- **None of this is execution-verified** — there is no Docker on this
  machine to run either fork's container path. Source-verified only, same
  epistemic status D-12 had before D-37 actually ran something for real.
  Don't let "read the source" get reported as "confirmed working."

Everything reachable *without a host* has been built: the full measurement
layer, validated end-to-end against a deterministic mock model and against
real vendored benchmark code, with the analysis logic (crossing determination,
interpretation-matrix placement, bootstrap CIs, multiple-comparison
correction) ready and waiting for real data to point at.

**Once a host is re-provisioned** (D-37's Lambda instance was terminated by
the user after the smoke test — nothing was lost except locally-cached
Docker images/model weights; the persistent filesystem with `hgm`/`cbs_pkg`
survives if the same filesystem is reattached), the natural next phase of
work is: writing and testing HyperAgents' local-endpoint patch (D-39),
forking HGM and HyperAgents into the repo for real, building D-31's
SWE-bench Verified/Polyglot wrapper against `make_report`/`run_evals`, and
bridging the agent function through `cbs.scaffolds.evolved.EvolvedScaffold`.

**D-14 — preregistration thresholds, signed off.** All ten rows in
`docs/preregistration.md` §3 are now marked **locked**, not `[TO FIX]` — see
D-32. Per this project's own preregistration discipline, **these values must
not be revisited after seeing any real Phase 4/5 results**. `[TO FIX]`
remains only on §4 (models and task families), which is unbuilt engineering
(the SWE-bench Verified/Polyglot integration D-31 just confirmed), not an
open judgment call. Both HumanEval+ (D-34) and MBPP+ (D-35) upgrades are
done.

**Both "+" upgrades each surfaced real upstream bugs, found by validating
against the real sandbox rather than trusting vendored "official" data**:
`humanevalplus` excludes `HumanEval/32` (its generated test fails its own
reference solution); `mbppplus` excludes four tasks (`Mbpp/590`: a
floating-point-tolerance gap in evalplus's own `is_floats` helper;
`Mbpp/737`/`787`/`794`: a generated `assertion()` function that computes a
result and never asserts it — a non-functional test that accepts any
candidate) and applies a per-task timeout override for one legitimately slow
(not wrong) task. See D-34/D-35 for the full investigation of each — every
exclusion is confirmed empirically (e.g. a `return None` stub actually made
to pass), not inferred from reading source alone. If a third "+"-style family
is ever added, budget time for this same validation pass; it has found real
bugs in *every* real family so far (D-27 too), not just these two.

**LiveCodeBench is investigated but deliberately not started** (D-33) — it
looked like "one more family to vendor" the way HumanEval/MBPP were, but
turned out to be a materially bigger, cross-cutting change: the data mixes at
least three test-execution conventions (stdin/stdout full-program judging,
LeetCode-style class-method "functional" tests with encoded arguments,
base64+zlib-compressed private tests), none of which fit `cbs`'s existing
assert-based `Task`/`Verifier` model. Doing it properly needs
`cbs.sandbox.ExecRequest` to grow stdin support in both backends, a parallel
I/O-judge verification path outside `Verifier`'s marker-based logic, and
touches `S_star`'s repair loop and `cbs.tasks.canonicalize`'s AST-based
renaming (both currently assume "one function to call"). Read D-33's full
write-up before starting this — it has the concrete schema, byte offsets that
found real example rows of each convention, and exactly what would need to
change where.

---

## 4. The instrument, in one screen

```
src/cbs/
  budget.py          matched-compute accountant (every model call charged here)
  config.py          YAML configs w/ `extends:` inheritance + fingerprinting
  cli.py             `cbs <command>` — one command reproduces one experiment
  models/            mock (known ground truth) | openai_compat (vLLM/Ollama/hosted)
  sandbox/           docker (real security boundary) | subprocess (NOT one)
  tasks/             schema, verifier, AST canonicalisation, frozen hashed splits
    families/        toy, humaneval, mbpp, transfer_reasoning
  scaffolds/
    tagging.py       the support-preserving/expanding partition + rationale registry
    s0.py            minimal scaffold (defines the frontier itself)
    s_star.py        strong fixed baseline (best-of-N, feedback, tool use, self-consistency)
    evolved.py        S_evo adapter: interception-based tagging (D-24), causal ablation (D-25)
    example_agents.py  synthetic agent functions used to validate evolved.py
  frontier/          estimators (Clopper-Pearson/Good-Turing/Chao1), sampler, records, validation
  compare.py         S0 vs S_star at matched compute (elicitation control)
  archive.py         overfitting gap / transfer retention / hard-coding triage
  ablation.py        scaffold-agnostic ablation runner
  crossing.py        the 4-part frontier-crossing verdict (brief §3.3)
  interpretation.py  mechanical placement into the interpretation matrix
  stats.py           bootstrap CIs, Benjamini-Hochberg correction
```

Quick commands:

```bash
cbs env                                                  # host capability report
cbs tasks verify --family humaneval                      # run real reference solutions
cbs frontier validate                                    # Phase 2 DoD, no GPU needed
cbs frontier estimate --config configs/frontier_toy_mock.yaml --dry-run
cbs compare --config configs/compare_toy_mock.yaml --dry-run
```

---

## 5. How this project works — conventions to keep following

- **Config-driven, one command = one experiment.** No magic constants in code.
  See `configs/*.yaml` for the pattern (`extends:` for inheritance).
- **Validate before building on top of something.** Every new task family got
  its reference solutions run through the *real* sandbox before being trusted
  — this found real bugs twice (HumanEval's multi-line/loop-scoped assertion
  extraction, D-27; a hand-compute  d test value in the transfer family, D-30).
  Assumption is cheap; a subprocess run is cheaper than a wrong result later.
- **Every non-obvious decision gets a `docs/DECISIONS.md` entry**, not just a
  code comment — rationale, what was measured, and what it rules in/out. If
  you make a call a future reader could reasonably challenge, write it down
  the same way.
- **Never claim absolute unreachability.** The frontier estimator's central
  discipline: a record can't express `p = 0`, `beyond_frontier` always carries
  its budget-relative qualifier. If you add a new "this failed" concept
  anywhere, ask whether it needs the same treatment.
- **Tagging is enforced, not trusted, for anything untrusted.** `S0`/`S_star`
  self-tag because we author them. `S_evo` cannot — see D-24 before writing
  anything that assumes an evolved scaffold will honestly report its own
  operations.
- **Full test suite before and after any nontrivial change**
  (`./venv/Scripts/python.exe -m pytest`). It's fast except for the two real
  benchmark families (marked `slow`, still included in a full run).
- **Git discipline**: commits only when explicitly asked, one phase of work
  per commit, descriptive bodies explaining *why* (see `git log` for the
  established style). Repo-local `user.name`/`user.email` are already
  configured — don't touch global git config.
- **Windows dev environment**: `venv/Scripts/python.exe`, not `venv/bin/`.
  `PYTHONIOENCODING=utf-8` needed for CLI output with non-ASCII in some
  shells. No Docker, no CUDA GPU — see D-02/D-23 for what that does and
  doesn't block.

---

*Keep this file current. When a future session finishes meaningful work,
update the phase table, the blocked-items list, and the commit hash at the
top — the next session (yours or someone else's) starts here.*
