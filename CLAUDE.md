# Orientation for the next Claude Code session

Read this first. It's the fast on-ramp — what this project is, what exists,
what's actually been verified vs. assumed, what's blocked, and where to look
for more depth. Everything here was true as of commit `67dd4a7` (9 commits on
`main`, 354 tests passing). If code and this file disagree, trust the code and
`git log` — update this file when that happens.

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
[`docs/preregistration.md`](docs/preregistration.md) holds the statistical plan
and every threshold still marked `[TO FIX]` — those must be chosen **before**
Phase 5 runs for real, not after seeing results, or the whole point of
pre-registering is lost.

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

354 tests, all passing (`./venv/Scripts/python.exe -m pytest`, a few minutes —
the HumanEval/MBPP tests each verify every reference solution *and* every
derived public-test subset against the real sandbox, so they alone are well
over a thousand subprocess executions).
[`README.md`](README.md) has the fuller version of this table plus the file
layout; it's written for a human reader rather than an agent, so it's a good
second stop after this file.

---

## 3. What's actually blocked, and why

Two things, both logged as open decisions in `docs/DECISIONS.md`'s "Still
open" table — **check that table before assuming something needs to be
figured out from scratch; it may already be decided or explicitly deferred.**

- **D-23 — no Docker+GPU host.** This dev machine has no Docker, and **Colab's
  runtime cannot nest Docker either** (not a missing package — a platform
  constraint). Running `S_evo` means executing self-modifying orchestration
  code with live network access to a model endpoint, which the brief's own
  safety guardrail (§10) requires Docker for. There is currently nowhere in
  this project's reach that can do that safely. Real frontier sampling at
  scale also needs a GPU this machine doesn't have (a Radeon Pro 580 — no
  CUDA, ROCm-incompatible). **The fix converges both needs**: a rented Linux
  instance with GPU + Docker (RunPod/Lambda/vast.ai-class) satisfies D-23 and
  the GPU requirement in one box.
- **D-12 — which fork.** **Now researched three ways**, not two — source
  inspection + GitHub API metadata for all three, full write-up in
  `docs/DECISIONS.md`'s D-12 section. **Recommendation: `metauto-ai/HGM` as
  primary** over the brief's stated default (`jennyzzt/dgm`) — same
  integration cost, more actively maintained, more interesting selection
  mechanism (clade/subtree promise estimation, ICLR 2026 oral). **New as of
  this session: `facebookresearch/HyperAgents`** surfaced by the D-36 novelty
  check — it's DGM's own original author (`jennyzzt`) extending DGM herself
  (with Jeff Clune, Jakob Foerster), pushed as recently as yesterday relative
  to this check, and its own authors admit "evaluation protocols remain
  fixed" (i.e. they haven't solved the elicitation-vs-expansion question
  either). Given the paper's ambition, recommended as **a second `S_evo`
  variant to run alongside HGM, not instead of it** — showing a result holds
  across two independently-built evolutionary mechanisms is much stronger
  than one implementation's idiosyncrasy. One real catch: **HyperAgents is
  CC BY-NC-SA 4.0, not Apache-2.0** like DGM/HGM — fine for academic use, but
  keep any code touching it in its own clearly-separated module rather than
  merged into the rest of `cbs`, since share-alike would otherwise pull that
  license onto whatever it's merged with. **Still not yet confirmed by the
  user** — a recommendation in the log, not a unilateral decision.
- **D-31 — how `cbs` tasks meet the fork's agent — now concretely scoped**,
  not just framed as two abstract options. Shallow-cloned HGM and read the
  actual source: it already ships **complete, working harnesses for both
  SWE-bench Verified and Polyglot** (`swe_bench/harness.py`,
  `polyglot/harness.py`, a repo-root `Dockerfile`; `evaluate_agent.py --split
  Verified` is a real, already-supported invocation), and **both interception
  points D-24 needs already exist as single, well-defined choke points** —
  every model call funnels through `llm.py:get_response_from_llm`, every
  verification funnels through `hgm_utils.eval_agent` →
  `swe_bench.harness.harness(...)`. That means **option (b) — evolve natively
  against SWE-bench Verified/Polyglot, keep `humaneval`/`mbpp`/
  `transfer_reasoning` for `S0`/`S_star` only — turns out cheaper than option
  (a) (adapting `cbs` tasks into one-file git repos), not more expensive**,
  once actually scoped: (a) still requires inventing a translation layer
  between two task representations that were never designed to correspond;
  (b) reuses HGM's own harness code directly, and its one real cost (Docker-
  per-instance verification) is already required for D-23 regardless. **Still
  a recommendation pending confirmation**, but now "confirm (b)" rather than
  "figure out what (b) costs" — full write-up in `docs/DECISIONS.md`.

Everything reachable *without* those has been built: the full measurement
layer, validated end-to-end against a deterministic mock model and against
real vendored benchmark code, with the analysis logic (crossing determination,
interpretation-matrix placement, bootstrap CIs, multiple-comparison
correction) ready and waiting for real data to point at.

**If the user confirms the fork and provisions a host, that's the natural
next phase of work** — patching `llm.py`'s `create_client` for a local vLLM
endpoint (~20-30 lines; the DeepSeek/OpenRouter branches already show the
pattern), resolving D-31, and bridging the agent function through
`cbs.scaffolds.evolved.EvolvedScaffold`.

**If not**, the reachable work without infra is: getting sign-off on the
`preregistration.md` `[TO FIX]` thresholds (D-14/D-32 — each now has full
reasoning attached, not bare placeholders, so this is "review and confirm or
push back," not "figure out from scratch") — needs zero infra and can happen
with the user right now. Both HumanEval+ (D-34) and MBPP+ (D-35) upgrades are
done as of this session.

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
  extraction, D-27; a hand-computed test value in the transfer family, D-30).
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
