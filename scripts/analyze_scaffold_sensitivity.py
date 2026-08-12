"""Analysis for the scaffold-sensitivity sub-study (preregistration.md §4.1).

Consumes the per-arm results files written by `run_s0_polyglot_subset.py` /
`run_s0_polyglot_batch.py` and produces the paper's tables plus
Clopper-Pearson intervals, using `cbs`'s own estimator rather than a
separately-derived one.

Deliberately refuses to silently merge arms: every results file is checked
for a single (model, agent_src) pair, and any file mixing them is reported
as an error rather than averaged over -- the exact failure mode the
per-result provenance fields exist to catch (D-44).

Usage:
    analyze_scaffold_sensitivity.py <results.json> [<results.json> ...]
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

from cbs.frontier.estimators import clopper_pearson


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]].

    Implemented here rather than pulled from scipy because scipy is not a
    dependency of this project and adding one for a single hypergeometric
    sum is not worth it. Exact (not chi-squared) because the interesting
    cells here contain zero or one count, where the asymptotic test is
    badly wrong.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    if n == 0 or row1 == 0 or col1 == 0:
        return 1.0

    def p_table(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    observed = p_table(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(p_table(x) for x in range(lo, hi + 1) if p_table(x) <= observed + 1e-12))

#: `incomplete` is HGM's marker for an exception raised inside process_entry
#: *before* `eval_result` was ever set -- i.e. the harness broke before it
#: could grade anything. That is a genuine infrastructure failure, not a task
#: the model failed (D-43), and is excluded from denominators.
INFRA_MARKERS = {"incomplete", ""}

#: `error` is NOT treated as an infrastructure failure, despite also being an
#: exception marker (D-46). It means the exception fired *after* `eval_result`
#: existed -- in practice, during the grading sequence. Every occurrence
#: observed (9/59 in the 14B x required arm) had the same verified cause, and
#: it is a genuine agent failure rather than a broken measurement:
#:
#:   "No local changes to save"   <- git stash push <declared solution files>
#:   "Removing diamond_kata.py"   <- git clean -fd deletes the agent's work
#:   "No stash entries found."    <- git stash pop fails, exit 1
#:
#: The agent created a NEW file of its own naming -- frequently in the wrong
#: language entirely (`diamond_kata.py` for `cpp__diamond`; `robot.py` and
#: `spell_number.py` for JavaScript tasks) -- instead of editing the task's
#: declared solution files. Nothing gradeable changed, so the task was
#: certainly not solved. Excluding these would silently inflate the resolve
#: rate by shrinking the denominator on exactly the attempts that failed.
#: They are counted as unresolved.
AGENT_FAILURE_MARKERS = {"error"}

#: `solution_len` is NOT a usable measure of how much the agent actually
#: changed, and must not be reported as one. Once an agent really uses tools
#: it builds the project, and HGM's `diff_versus_commit` diffs the whole
#: working tree against the base commit -- so the recorded patch sweeps in
#: every build artifact. Observed directly: one Rust task produced a
#: 2,974,997-character "patch" spanning 1,935 files, of which exactly three
#: (`Cargo.toml`, `src/lib.rs`, `Cargo.lock`) were real; the remaining ~1,932
#: were `target/` build output. This does not affect grading -- Polyglot's
#: harness stashes only the declared solution files, resets to `test_commit`,
#: and runs `git clean -fd`, so artifacts never reach the tests -- but it
#: makes raw patch length meaningless as a behavioural metric. Report
#: non-empty-patch as a binary, and tool calls, instead.
BUILD_ARTIFACT_PREFIXES = (
    "target/", "node_modules/", "build/", ".gradle/", "__pycache__/",
    "vendor/", "dist/", ".pytest_cache/",
)

ARM_LABELS = {
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "measured_default_agent/src"): "7B x auto",
    ("Qwen/Qwen2.5-Coder-7B-Instruct", "toolcheck_agent_src"): "7B x required",
    ("Qwen/Qwen2.5-Coder-14B-Instruct-AWQ", "measured_default_agent/src"): "14B x auto",
    ("Qwen/Qwen2.5-Coder-14B-Instruct-AWQ", "toolcheck_agent_src"): "14B x required",
}


def load_arm(path: Path) -> dict:
    rows = json.loads(path.read_text())
    if not rows:
        raise ValueError(f"{path}: empty results file")

    models = {r.get("model", "UNRECORDED") for r in rows}
    agents = {r.get("agent_src", "UNRECORDED") for r in rows}
    if len(models) > 1 or len(agents) > 1:
        raise ValueError(
            f"{path}: MIXED ARMS in one file -- models={sorted(models)}, "
            f"agents={sorted(agents)}. Refusing to analyse; this file cannot "
            "be attributed to a single experimental condition."
        )

    model = next(iter(models))
    agent = next(iter(agents))

    valid = [r for r in rows if r.get("eval_result") not in INFRA_MARKERS]
    infra = len(rows) - len(valid)
    # Counted inside `valid` (as unresolved), reported separately -- see
    # AGENT_FAILURE_MARKERS for why these are not exclusions.
    wrong_artifact = sum(1 for r in valid if r.get("eval_result") in AGENT_FAILURE_MARKERS)

    resolved = sum(1 for r in valid if r.get("passed"))
    used_tools = sum(1 for r in valid if r.get("trace_op_counts", {}).get("tool_call", 0) > 0)
    total_tool_calls = sum(r.get("trace_op_counts", {}).get("tool_call", 0) for r in valid)
    total_gens = sum(r.get("trace_op_counts", {}).get("single_call", 0) for r in valid)
    nonempty = sum(1 for r in valid if r.get("eval_result") != "empty_patch")

    n = len(valid)
    ci_r = clopper_pearson(resolved, n) if n else None
    ci_t = clopper_pearson(used_tools, n) if n else None
    lo_r, hi_r = (ci_r.low, ci_r.high) if ci_r else (0.0, 0.0)
    lo_t, hi_t = (ci_t.low, ci_t.high) if ci_t else (0.0, 0.0)

    return {
        "path": path.name,
        "model": model,
        "agent": agent,
        "label": ARM_LABELS.get((model, agent), f"{model} x {agent}"),
        "n": n,
        "infra_excluded": infra,
        "wrong_artifact": wrong_artifact,
        "resolved": resolved,
        "resolved_ci": (lo_r, hi_r),
        "used_tools": used_tools,
        "used_tools_ci": (lo_t, hi_t),
        "total_tool_calls": total_tool_calls,
        "total_generations": total_gens,
        "nonempty_patch": nonempty,
        "mean_completion_tokens": (
            sum(r.get("usage", {}).get("completion_tokens", 0) for r in valid) / n if n else 0
        ),
        "mean_elapsed_s": sum(r.get("elapsed_s", 0) for r in valid) / n if n else 0,
        "rows": valid,
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(1)

    arms = []
    for p in paths:
        try:
            arms.append(load_arm(p))
        except (ValueError, FileNotFoundError) as exc:
            print(f"SKIPPED {p}: {exc}\n")

    if not arms:
        sys.exit("no analysable arms")

    print("=" * 96)
    print("SCAFFOLD-SENSITIVITY SUB-STUDY (preregistration.md 4.1)")
    print("Polyglot, HGM 59-task baseline subset, S0 (one trajectory, no retries)")
    print("=" * 96)
    hdr = (
        f"{'arm':<18}{'n':>4}{'resolved':>10}{'95% CI':>18}"
        f"{'used tools':>12}{'95% CI':>18}{'tool calls':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for a in arms:
        lo_r, hi_r = a["resolved_ci"]
        lo_t, hi_t = a["used_tools_ci"]
        print(
            f"{a['label']:<18}{a['n']:>4}"
            f"{a['resolved']:>6}/{a['n']:<3}"
            f"{f'[{lo_r:.3f}, {hi_r:.3f}]':>18}"
            f"{a['used_tools']:>8}/{a['n']:<3}"
            f"{f'[{lo_t:.3f}, {hi_t:.3f}]':>18}"
            f"{a['total_tool_calls']:>12}"
        )

    print()
    print("Secondary measures")
    print("-" * 96)
    for a in arms:
        produced = sum(1 for r in a["rows"] if r.get("solution_len", 0) > 0)
        print(
            f"  {a['label']:<18} produced-a-patch {produced:>3}/{a['n']:<3} | "
            f"graded non-empty {a['nonempty_patch']:>3}/{a['n']:<3} | "
            f"generations {a['total_generations']:>5} | "
            f"mean completion tok {a['mean_completion_tokens']:>7.0f} | "
            f"mean wall {a['mean_elapsed_s']:>6.0f}s | "
            f"wrong-artifact {a['wrong_artifact']:>2} | "
            f"infra-excl {a['infra_excluded']}"
        )
    print()
    print("  NB: raw patch length is not reported -- it is dominated by build")
    print("      artifacts once the agent actually builds the project. See")
    print("      BUILD_ARTIFACT_PREFIXES in this file for the measured example.")

    # Per-language breakdown, useful for showing the effect is not driven by
    # one language's harness quirk.
    print()
    print("Resolved by language")
    print("-" * 96)
    langs = sorted({r["language"] for a in arms for r in a["rows"]})
    print(f"{'arm':<18}" + "".join(f"{L:>12}" for L in langs))
    for a in arms:
        cells = []
        for L in langs:
            rs = [r for r in a["rows"] if r["language"] == L]
            cells.append(f"{sum(1 for r in rs if r['passed'])}/{len(rs)}")
        print(f"{a['label']:<18}" + "".join(f"{c:>12}" for c in cells))

    print()
    print("Tool use by language")
    print("-" * 96)
    print(f"{'arm':<18}" + "".join(f"{L:>12}" for L in langs))
    for a in arms:
        cells = []
        for L in langs:
            rs = [r for r in a["rows"] if r["language"] == L]
            n_t = sum(1 for r in rs if r.get("trace_op_counts", {}).get("tool_call", 0) > 0)
            cells.append(f"{n_t}/{len(rs)}")
        print(f"{a['label']:<18}" + "".join(f"{c:>12}" for c in cells))

    # Pairwise significance against the first arm (the unmodified default).
    if len(arms) > 1:
        print()
        print(f"Fisher exact vs baseline arm ({arms[0]['label']})")
        print("-" * 96)
        base = arms[0]
        for a in arms[1:]:
            p_res = fisher_exact_two_sided(
                base["resolved"], base["n"] - base["resolved"],
                a["resolved"], a["n"] - a["resolved"],
            )
            p_tool = fisher_exact_two_sided(
                base["used_tools"], base["n"] - base["used_tools"],
                a["used_tools"], a["n"] - a["used_tools"],
            )
            sig = lambda p: "significant" if p < 0.05 else "NOT significant"
            print(
                f"  {a['label']:<18} resolved  {base['resolved']}/{base['n']} vs "
                f"{a['resolved']}/{a['n']}   p = {p_res:<9.4f} {sig(p_res)}"
            )
            print(
                f"  {'':<18} tool use  {base['used_tools']}/{base['n']} vs "
                f"{a['used_tools']}/{a['n']}   p = {p_tool:<9.3e} {sig(p_tool)}"
            )

    # LaTeX for the paper's main table.
    print()
    print("LaTeX (main table)")
    print("-" * 96)
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Arm & $n$ & Resolved & Used tools & Tool calls \\")
    print(r"\midrule")
    for a in arms:
        lo_r, hi_r = a["resolved_ci"]
        print(
            f"{a['label']} & {a['n']} & "
            f"{a['resolved']} ({100*a['resolved']/a['n']:.1f}\\%) & "
            f"{a['used_tools']} ({100*a['used_tools']/a['n']:.1f}\\%) & "
            f"{a['total_tool_calls']} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
