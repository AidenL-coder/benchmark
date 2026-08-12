# Orientation for the next Claude Code session

Read this first. It's the fast on-ramp — what this project is, what exists,
what's actually been verified vs. assumed, what's blocked, and where to look
for more depth. Last commit is `64d436f` (message `edits` — a checkpoint
commit, not a normal descriptive one; it carries the full D-39/D-40 build:
`cbs.tasks.swebench`, `cbs.scaffolds.swebench_scaffold`,
`scripts/swebench_glue.py`, the `fork_bridge.usage_from_events` addition,
and the accompanying docs). There is further **uncommitted** work on top of
it as of this writing: `CLAUDE.md`, `README.md`, and `docs/DECISIONS.md`
all modified (documenting D-40's repair-branch closure, D-41's `TS_sample`
root-cause/fix, and D-42's Polyglot `S0` build), plus a small real fix to
`scripts/hgm_run_task_with_interception.py` (it captured a task's trace but
never charged its usage — now wired through `fork_bridge.usage_from_events`,
D-40's own function, re-validated against a real Polyglot task) and a
stale-docstring fix to `src/cbs/scaffolds/evolved.py` (it used to claim no
Docker-capable host existed anywhere in this project's infrastructure — no
longer true since D-23/D-37). New, not-yet-committed files: `src/cbs/
tasks/polyglot.py`, `src/cbs/scaffolds/polyglot_scaffold.py`,
`scripts/polyglot_glue.py`, `scripts/verify_hgm_ts_sample_fix.py`,
`tests/test_polyglot.py`, `tests/test_polyglot_scaffold.py` (D-42/D-41).
Nothing from this stretch has been committed yet — check `git status`
before assuming otherwise. If code and this file disagree, trust the code
and `git log` — update this file when that happens.

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
| 1 | Task schema, verifier, frozen hashed splits | **done** — 6 `Task`-shaped families: `toy`, `humaneval` (164), `humanevalplus` (163, D-34), `mbpp` (427), `mbppplus` (374, D-35), `transfer_reasoning` (10, hand-authored). Use the "+" variants over the originals for anything beyond instrument validation. SWE-bench Verified is deliberately **not** one of these — D-40 found `Task`/`TaskSuite` doesn't fit it (repo+diff, not prompt+code-string) and built `cbs.tasks.swebench.SweBenchInstance`/`SweBenchSuite` instead, now real-data-execution-verified end to end (see Phase 4/5 note and D-40). LiveCodeBench investigated and scoped as a materially bigger, cross-cutting change, not started (D-33). |
| 2 | Frontier estimation (Clopper-Pearson, Good-Turing, Chao1, rarefaction) | **done, DoD met** — validated against known ground truth, verifier false-positive rate checked at 0/600 |
| 3 | `S0`, `S_star`, matched-compute harness, elicitation control | **done** for the `Task`-shaped families; SWE-bench Verified variants (`S0SweBench`/`SStarSweBench`, D-40) also **done and real-execution-verified** — both run end-to-end against a real instance with the real vLLM model, producing correct resolved/unresolved verdicts. `S0Polyglot` (D-42) likewise done and real-execution-verified for D-31's other confirmed native substrate; `SStarPolyglot` not yet built (needs the same container-splitting surgery D-40 did) |
| 4 | `S_evo` measurement layer: interception-based tagging, ablation, archive | measurement layer **done**, and the underlying interception/proxy/budget machinery is now proven against real Docker+real-model infrastructure twice over (D-38's HGM/HyperAgents work, D-40's SWE-bench Verified work); *running an actual self-improving loop* (not just the frozen baseline agent) has still not happened |
| 5 | Crossing test, ablation matrix | determination **logic** done (`cbs.crossing`); running it against a real evolved scaffold's results has still not happened |
| 6 | Statistical plan, interpretation matrix, write-up | bootstrap CIs / BH correction / matrix placement **done**; the write-up itself needs real results |

437 tests, all passing (`./venv/Scripts/python.exe -m pytest`, a few minutes —
the HumanEval/MBPP tests each verify every reference solution *and* every
derived public-test subset against the real sandbox, so they alone are well
over a thousand subprocess executions).
[`README.md`](README.md) has the fuller version of this table plus the file
layout; it's written for a human reader rather than an agent, so it's a good
second stop after this file.

---

## 3. What's actually blocked, and why

> **READ THIS FIRST — the blocker changed on 2026-08-07 and is no longer
> engineering.** Everything below about infrastructure, forks, and
> measurement plumbing remains true and remains solved. But the first
> at-scale real measurement (D-44) found that the chosen frozen model,
> `Qwen2.5-Coder-7B-Instruct`, produces **zero agentic behaviour** under the
> unmodified agent's own `tool_choice="auto"` default: across the full
> Polyglot baseline subset it resolved **0 tasks, made 0 tool calls, and
> returned an empty patch every single time**, uniformly across all six
> languages. It writes a competent analysis of the problem and then stops.
> Since `S0` is what *defines* the frontier, `S0 = 0` makes the frontier
> degenerate and the whole crossing test uninformative — and it is also the
> mechanical reason D-41's `expand()` crash exists (every archive node sits
> at `mean_utility == 0`, so the positive-utility filter is *always* empty).
> **Do not spend GPU-hours on an evolutionary run until a base model with
> real dynamic range is confirmed.** 14B and 32B AWQ are staged and fit the
> existing A10 at no extra GPU cost; the task-matched comparison is the next
> step. Full detail in `docs/DECISIONS.md` D-44, including the one-line
> `tool_choice="required"` contrast (D-38) that turns the same model from
> no-action-at-all into a 64-round near-solve — a real elicitation finding,
> but explicitly *not* a crossing result.
>
> **Update 2026-08-10/11 — the project's near-term goal changed (D-45) and
> the sub-study is most of the way through.** The target is now a **NeurIPS
> 2026 workshop paper** ("Managing Agents that Manage Agents", deadline
> **2026-08-29**, non-archival so the main track is unaffected; NeurIPS 2026's
> own main-track deadline has already passed, so that is a 2027 target).
> Scope is deliberately cut to *instrument + D-44 result* — no evolutionary
> run, no seeds, no second fork, no SWE-bench. Draft lives in
> `paper/workshop_paper.tex`; `scripts/analyze_scaffold_sensitivity.py`
> regenerates its tables (and their LaTeX) from the results files.
> **The 2×2 is COMPLETE — all four arms at n=59 each** (`{7B,14B}` ×
> `{auto,required}`), declared in `preregistration.md` §4.1 **before** being
> run with explicit falsifiable predictions, outcomes scored in §4.2.
> Per-arm results are committed under `results/*.json` (raw logs are
> gitignored under `results/logs/`):
>
> | Arm | Resolved | Used tools | Tool calls | Generations |
> |---|---|---|---|---|
> | 7B × `auto` | 0/59 | 0/59 | 0 | 59 |
> | 7B × `required` | 2/59 | 59/59 | 3663 | 3687 |
> | 14B × `auto` | 0/59 | 0/59 | 0 | 59 |
> | 14B × `required` | 4/59 | 59/59 | 5592 | 5617 |
>
> **Tool policy → behaviour: Fisher p = 3.4e-13 at *both* scales. Model scale
> → behaviour: p = 1.0 (14B `auto` is identical to 7B `auto` in every cell,
> falsifying registered prediction 2). Resolve rate: never significant
> (p = 0.50, 0.12) and must not be claimed.** The sharp honest point: 0→2
> resolved is statistically negligible but *algorithmically* decisive, since
> it is exactly the boundary at which `expand()`'s positive-utility filter
> stops being empty (D-41).
>
> Three real data-quality traps found and handled, all documented:
> **D-43** (infra failures reported identically to model failures);
> `patch_len` dominated by build artifacts once the agent actually builds
> (one Rust "patch" was 2.97M chars, 3 real files of 1,935); and **D-46** —
> 9 rows where the agent used tools but wrote a *new file of its own naming*,
> often in the wrong language (`diamond_kata.py` for `cpp__diamond`), leaving
> nothing gradeable. That last one required a denominator correction made
> **against our own interest**: counting them as unresolved gives 4/59 = 6.8%
> rather than the flattering 4/50 = 8.0%.

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
repeating any of them. **The one thing D-37 left unresolved is now resolved
too (D-41)**: the `TS_sample` argmax-on-an-empty-sequence crash was
root-caused by reading `hgm.py`/`tree.py` directly, not by re-running the
smoke test — `expand()`'s `mean_utility > 0` filter goes empty whenever
*every* archive node has solved exactly zero tasks so far, which is a real
risk at 60-task scale too (not merely a 1-task artifact), given this
session's own repeated finding that the frozen 7B baseline rarely produces
a working fix under `tool_choice="auto"`. Patched `expand()` to fall back
to all evaluated nodes when that filter is empty, rather than crash;
confirmed crash-free at both 1-task and 60-task-scale analogues, and
provably inert in the normal case, via `scripts/verify_hgm_ts_sample_fix.py`
— a cheap, isolated reproduction, not a real GPU/Docker run.

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
- **HyperAgents needed a real patch, different in shape from HGM's — now
  written and empirically verified against a real vLLM server (2026-08-05),
  not just planned.** Its `agent/llm.py:get_response_from_llm` is a separate
  implementation from HGM's, built on `litellm.completion()` with no
  `api_base`/vLLM awareness. Cloned onto the persistent filesystem
  (`/lambda/nfs/cbs-project/HyperAgents/`), patched with a
  `"vllm-model:<host>"` branch that resolves the real served model name
  dynamically via `/v1/models` (a hardcoded model string 404s against vLLM
  0.26.0 — confirmed directly, this is *why* HGM's own generic-fallback code
  does the same dynamic lookup rather than something to skip). Confirmed
  HyperAgents needs **no separate tool-calling integration** — unlike HGM,
  `chat_with_agent` always does prompt-based tool parsing regardless of
  model, so the one `get_response_from_llm` patch is the complete surface.
  Also found and fixed a real `max_tokens` bug only by actually calling it:
  `chat_with_agent` never overrides the library's `MAX_TOKENS=16384`
  default, which left zero room for input against this deployment's
  `--max-model-len 16384` — capped to 4096 inside the vLLM branch,
  matching HGM's own `MAX_OUTPUT_TOKENS`. A dedicated minimal venv
  (`hyperagents_venv/`) was used rather than installing into the working
  `hgm_venv`, since HyperAgents' `requirements.txt` pulls in heavy,
  irrelevant robotics/RL packages for its other domains. Verified working
  end to end outside Docker: both a raw `get_response_from_llm` call and
  the actual `chat_with_agent` HyperAgents' own agent code calls returned
  real, correct responses. The patched file lives only on the remote clone
  and in scratch space, never in this repo, per the license-separation note
  above.
- **Correction, same session — the earlier "Docker networking already
  solved in HyperAgents" claim was wrong, caught by actually exercising it,
  not by re-reading more carefully.** It checked `utils/docker_utils.py`'s
  `network_mode="host"`, but that's a *different* container path (and one
  of its two hits is for the image-*build* step, not runtime networking at
  all). The container that actually matters —
  `domains/polyglot/harness.py:process_entry` copies `task_agent.py` into
  and runs it there, exactly like HGM's own agent container — is built by
  `domains/polyglot/docker_build.py:build_container`'s
  `client.containers.create(...)`, which had no `network_mode` set at all.
  Patched it the same way as D-37 (`.orig` backup kept). **HyperAgents does
  need this fix after all** — net cost is the litellm patch plus this one,
  not less than HGM's own D-37 lift as previously claimed.
- **Update, second instance (2026-08-05) — a full real HyperAgents task
  run now completes end-to-end.** Found the harness/Docker-orchestration
  layer (`domains/polyglot/harness.py`, directly analogous to HGM's own
  `polyglot/harness.py`), copied HGM's already-prepared
  `polyglot-benchmark/` data across, and pushed all the way through to a
  real, complete run. Two more real bugs found and fixed along the way:
  a missing `GitPython` dependency in the trimmed container
  `requirements.txt` (kept trimmed permanently — the full one only serves
  HyperAgents' other, unrelated robotics/RL domains and would waste real
  time installing Genesis etc. into every task container), and — the
  significant one — a genuine, fully root-caused ~600s hang: `harness()`'s
  own `process_evaluation` appends `"_<eval_idx>"` to the model string for
  output-directory labeling (e.g. `"vllm-model:localhost_0"`), and that
  suffixed string was reaching `--model`, turning the parsed-out hostname
  into the unresolvable `localhost_0` — `backoff` then silently retried
  the resulting connection error for the full configured timeout every
  time (confirmed to track the timeout value exactly across three
  reproductions, then confirmed directly and cheaply outside Docker).
  Fixed by stripping the eval-index suffix specifically at the `--model`
  call site. Result: `"Running the agent"` → done in ~4 seconds, a real
  task fully processed. Full root-cause narrative (four candidate
  hypotheses tested and eliminated in order, cheaply, before finding the
  real one) in `docs/DECISIONS.md` D-39's closing update — worth reading
  in full if this ever needs revisiting, since the debugging path itself
  is the reusable part.
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
- **Updated**: both forks' Docker/container paths are now execution-verified
  end to end against this project's real vLLM deployment — HGM since D-37,
  HyperAgents as of the second-instance update above. What remains
  unbuilt is `cbs`'s own SWE-bench Verified wrapper (for `S0`/`S_star`),
  which doesn't exist yet — that's the one concrete remaining piece, not
  "figure out if either fork's pipeline works at all."

**D-40, new this session — that remaining piece is now concretely scoped
(not guessed at), and two of its four pieces are actually built.**
`S0`/`S_star` are built entirely
around `cbs.tasks.schema.Task` (prompt + assert-based tests + one candidate
code string) — a SWE-bench Verified instance (repo + commit + problem
statement, solved by producing a diff) doesn't fit that shape at all, for
the same reason D-33 didn't force-fit LiveCodeBench. Worked out the actual
mapping by reading real code on both sides, not guessing:
- **The atomic unit generalizes cleanly**: one call to `chat_with_agent`
  (HGM's and HyperAgents' own agent-loop primitive, already exercised
  extensively this session) is the natural analogue of `S0`'s "one call to
  `M`" for a task that needs real tool use to solve at all — `S0`-for-
  SWE-bench is one such trajectory, no retries, submitted as-is.
- **`cbs.budget`'s accounting already generalizes** — confirmed by reading
  `budget.py`: nothing in `BudgetAccountant`/`Usage`/`MatchedComputeHarness`
  assumes a call maps to a `Task`. The one real gap was small and is now
  **built and tested**: `cbs.scaffolds.fork_bridge.usage_from_events` sums
  real `prompt_tokens`/`completion_tokens` out of already-captured
  `ProxiedCall`s (6 new tests, `tests/test_fork_bridge.py` now 32 total).
- **`S_star`'s execution-feedback mechanism has a real, verified analogue**,
  not a made-up one: queried the actual `princeton-nlp/SWE-bench_Verified`
  dataset directly and confirmed every instance carries `FAIL_TO_PASS` (the
  hidden regression test the fix must satisfy — this family's `tests`) and
  `PASS_TO_PASS` (the repo's own pre-existing tests, legitimately runnable
  without revealing whether the target bug is fixed — this family's
  `public_tests`). Same "hidden oracle queried exactly once" rule carries
  over unchanged.
- **Verification reuses the real harness** (`swe_bench/harness.py` +
  `report.py:make_report`/`run_evals`), per D-31/D-39 — not reimplemented.
- **The task representation itself is now built**: `cbs.tasks.swebench.
  SweBenchInstance`/`SweBenchSuite`/`load_swebench_verified`, loading
  `princeton-nlp/SWE-bench_Verified` directly (not HGM's two-dataset
  indirection). Validated against the real dataset (10 new tests,
  `tests/test_swebench.py`) — caught a real, non-obvious bug this same way:
  `FAIL_TO_PASS`/`PASS_TO_PASS` arrive as JSON-*encoded strings*, not native
  lists, despite printing like one. New `swebench` optional extra
  (`datasets>=2.14`) in `pyproject.toml`.
- **The pure scaffold logic is now built too**: `cbs.scaffolds.
  swebench_scaffold.S0SweBench`/`SStarSweBench`, using the exact
  injected-function pattern `cbs.scaffolds.evolved.EvolvedScaffold` already
  established (`SweBenchAgentFunction`/`SweBenchVerifyFunction` stand in for
  whatever Docker/HGM machinery actually runs a trajectory) — best-of-N, the
  PASS_TO_PASS-feedback repair loop, self-consistency, budget charging, and
  trace merging are all tested with synthetic fakes, no Docker needed
  (`tests/test_swebench_scaffold.py`, 18 tests, one real logic bug caught
  and fixed along the way). One real limitation surfaced and documented in
  the module itself: unlike `s_star.py`'s pre-check, a Docker trajectory's
  cost isn't known until it's already run, so budget charging here is
  unavoidably post-hoc.
- **Update — the user said to keep working autonomously, so the Docker-glue
  piece was written and validated for real too, not left at the scoping
  stage.** `scripts/swebench_glue.py` implements `real_agent_fn`/
  `real_verify_fn` by driving HGM's actual `swebench.harness.docker_build`/
  `test_spec` functions directly — the same ones `swe_bench/harness.py:
  process_entry` itself calls, not reimplemented. Validated incrementally:
  (1) `real_verify_fn` against the instance's own gold `patch` (mirroring
  this project's `reference_solution` discipline) — `PASS_TO_PASS` 13/13
  passed, `FAIL_TO_PASS` both passed, and an empty-diff negative control
  correctly failed; (2) `real_agent_fn` against the real vLLM model — one
  real trajectory, real usage recorded, no crash; (3) `S0SweBench.solve()`
  fully end-to-end — the model didn't actually fix the bug (no tool use,
  same pattern as every other real run this session), and the pipeline
  **correctly determined this**: a real, correct unresolved verdict, not
  just "didn't crash."
- **Four more real bugs found and fixed, each caught by running it, not by
  re-reading the design.** `make_test_spec` needs a `hints_text` key
  (confirmed "Unused" in swebench's own source, fixed with a placeholder,
  not added to `SweBenchInstance`). A missing `pip install -r
  requirements.txt` step inside the agent's container produced an
  immediate `ModuleNotFoundError`. Bash treats `test_separable[compound_
  model0-result0]`-style test IDs as glob patterns unless the shell sets
  `-f` first — silently mangled them otherwise. **The significant one**: a
  real correction to this entry's own earlier claim — "`PASS_TO_PASS` never
  needs `test_patch`" was wrong. Confirmed directly: collecting
  `test_separable.py` at `base_commit` found 11 tests without `test_patch`,
  15 with it — some `PASS_TO_PASS` entries are themselves new test_patch-
  introduced cases that happen to already pass pre-fix. The real rule:
  PASS_TO_PASS means "passes regardless of the fix, *given* `test_patch`
  is applied," not "doesn't need it." Oracle-safety holds a different way
  than assumed: `test_patch` is applied only inside the separate
  verify-only container, never the agent's own, and only a filtered pass/
  fail signal reaches the caller.
  Full write-up, including the still-open, explicitly-flagged
  simplifications (coarse exit-code pass/fail rather than per-test-node
  parsing; the agent's diff carries harmless environment-setup noise
  inherited from HGM's own harness), in `docs/DECISIONS.md` D-40.
- **`SStarSweBench` run for real too, right after `S0`'s success**: 2 real
  trajectories (`max_candidates=2`), both correctly tagged
  (`single_call`×2, `execution_feedback`×2, `self_consistency`×1), correct
  summed usage across both, both candidates passed `PASS_TO_PASS` and the
  final `FAIL_TO_PASS` check correctly confirmed the instance is still
  unresolved — agreeing with `S0`'s own verdict on the same instance.
  Best-of-N, execution feedback, self-consistency, and multi-trajectory
  budget accounting are all now confirmed against real infrastructure, not
  just synthetic tests.
- **The repair branch itself, closed for real, same session.** Rather than
  wait for the live 7B model to naturally produce a diff that fails
  `pass_to_pass` (never observed in ~10+ real runs this session), the gap
  was closed deterministically: a hand-built, deliberately-broken one-line
  diff for `astropy__astropy-12907` (`if transform.n_inputs == 1 and ...:` →
  `if True:`) as a synthetic first attempt, the instance's own gold `patch`
  as the synthetic repair — with the **real** `real_verify_fn` (real Docker,
  real `git apply`, real `pytest`) doing all the actual verification. The
  bad diff itself needed a fix first (a hand-escaped Python-string diff hit
  `corrupt patch at line 11` from `git apply`; rewritten as a plain heredoc
  file, anchored on a code line instead of the docstring, applied cleanly)
  and was confirmed genuinely broken by running it directly: 6 of 13
  `PASS_TO_PASS` tests failed. Wired into a real `SStarSweBench.solve()`
  call, the full repair mechanism fired for real: `trace.op_counts() ==
  {'execution_feedback': 2, 'adaptive_prompt_rewrite': 1, 'self_consistency':
  1}` — `adaptive_prompt_rewrite` (the actual repair step) had never fired
  in any prior real run. The second call's prompt was confirmed at runtime
  to carry the real failure text forward; its gold-patch diff passed
  `pass_to_pass` for real, and the final hidden-oracle `FAIL_TO_PASS` check
  came back `passed: True` — a genuinely resolved verdict, through three
  separate real Docker verify-container runs. Every scaffold/verification
  mechanism `SStarSweBench` has is now demonstrated against real
  infrastructure, ending in a correct positive result when the repair is
  actually correct — only the model call itself was synthetic. D-40's one
  stated gap is closed; see `docs/DECISIONS.md` D-40 for the full trace.

**D-42, new this session — `S0` built and real-execution-verified for
Polyglot, D-31's *other* confirmed native substrate**, completing what D-40
only did for SWE-bench Verified. Read HGM's own `polyglot_benchmark_
metadata.json` (225 real entries) and `polyglot/harness.py:process_entry`
directly before designing: unlike SWE-bench Verified, there's no `Task.
tests`/`public_tests`-style split at the data level (oracle safety instead
comes from the hidden test content only being revealed by `git reset --hard
test_commit`, inside the agent's own container, after it has already
stopped), and `process_entry` is atomic — one call runs the agent *and*
grades it, with no seam to split into separate agent/verify containers the
way SWE-bench Verified's harness had. Built `cbs.tasks.polyglot`
(`PolyglotInstance`/`PolyglotSuite`/`load_polyglot_benchmark`, keeping the
whole original entry dict rather than re-deriving fields, to avoid the
translation-mismatch bugs D-40 hit repeatedly), `cbs.scaffolds.
polyglot_scaffold.S0Polyglot` (one injected function, not an agent/verify
pair), and `scripts/polyglot_glue.py` (calls `process_entry` directly).
Validated against real infrastructure: the loader against the real
225-entry file (a real, previously undocumented finding — HGM's own
hardcoded "60-task" baseline subset is actually only 59 *unique* tasks,
since `python__dominoes` appears in both `medium.json` and `small.json`),
and `S0Polyglot.solve()` run fully end-to-end against a real Polyglot
instance with the real vLLM model and real Docker, producing a correct
`empty_patch`/`passed: False` verdict (same no-tool-use pattern as every
other real run this session). `SStarPolyglot` deliberately not built —
no execution-feedback hook exists without the same container-splitting
surgery D-40 did; a half-built repair loop with no real signal to feed it
would be worse than an honestly `S0`-only family for now.

Everything reachable *without a host* has been built: the full measurement
layer, validated end-to-end against a deterministic mock model and against
real vendored benchmark code, with the analysis logic (crossing determination,
interpretation-matrix placement, bootstrap CIs, multiple-comparison
correction) ready and waiting for real data to point at.

**Update — this paragraph described a host that had been terminated; that
is no longer the situation.** A Lambda instance has been live and in active
use for this entire D-38/D-39/D-40/D-41 stretch (re-provisioned once
already this project, per D-37's update; if it is ever terminated again,
the same re-provisioning steps apply — Docker group membership, PATH for
vLLM/ninja, the persistent filesystem reattaching `hgm`/`hgm_venv`/`cbs_pkg`
intact). Of the four things this paragraph used to list as the "next
phase": HyperAgents' local-endpoint patch (D-39) is done and validated
end-to-end; D-31's SWE-bench Verified/Polyglot wrapper is done and
execution-verified (D-40, including the repair branch); D-37's remaining
crash is root-caused and fixed (D-41). **Only "forking HGM and HyperAgents
into the repo for real" remains genuinely undone** — what exists today is
real, working clones on the remote instance's persistent filesystem
(`/lambda/nfs/cbs-project/hgm`, `.../HyperAgents`), patched and validated,
but not actual GitHub forks under the user's account with real commit
history. This specifically needs `gh` CLI or a personal access token
authenticated as the user — neither is available in this local dev
environment, so it's blocked on the user's own GitHub credentials, not on
any remaining design or engineering question.

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
