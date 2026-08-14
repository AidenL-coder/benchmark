"""Elicitation-control arm: SStarPolyglotBestOfN over the 59-task subset (D-47).

Supplies the rung between S0 and an evolved scaffold that a reviewer of the
workshop paper will look for: how much is reachable by sampling the *same*
frozen model harder, with no evolution?

Reports two separate quantities, which must not be conflated:
  * resolved   -- the oracle-blind scaffold's own result (self-consistency
                  selection, never inspecting pass/fail). This is the
                  elicitation control proper.
  * pass_at_n  -- did ANY of the N trajectories pass. An upper bound on any
                  selection rule over the same samples, i.e. a
                  budget-relative estimate of the frozen model's reachable
                  set. NOT scaffold performance.

Usage: run_sstar_polyglot.py <n_candidates> <suffix>
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
from polyglot_glue import real_agent_fn
from cbs.tasks.polyglot import load_polyglot_benchmark
from cbs.scaffolds.polyglot_scaffold import SStarPolyglotBestOfN
from cbs.budget import BudgetAccountant

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else f"sstar_n{N}"

TASK_LIST = Path('/lambda/nfs/cbs-project/real_59_tasks.json')
OUT_FILE = Path(f'/lambda/nfs/cbs-project/s0_polyglot_{SUFFIX}_results.json')
LOG_DIR = Path(f'/lambda/nfs/cbs-project/polyglot_{SUFFIX}_out')

TASK_IDS = json.loads(TASK_LIST.read_text())
suite = load_polyglot_benchmark(
    'polyglot/polyglot_benchmark_metadata.json', instance_ids=TASK_IDS
)
by_id = suite.by_id()


def served_model() -> str:
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


MODEL_ID = served_model()
AGENT_SRC_ID = os.environ.get("CBS_AGENT_SRC", "measured_default_agent/src")
print(f"[{SUFFIX}] model={MODEL_ID} agent={AGENT_SRC_ID} N={N}", flush=True)

accountant = BudgetAccountant(f'polyglot_{SUFFIX}')
scaffold = SStarPolyglotBestOfN(n_candidates=N)

results = []
if OUT_FILE.exists():
    results = json.loads(OUT_FILE.read_text())
done = {r['instance_id'] for r in results}
print(f"[{SUFFIX}] resuming: {len(done)} done of {len(TASK_IDS)}", flush=True)

for i, task_id in enumerate(TASK_IDS):
    if task_id in done or task_id not in by_id:
        continue
    instance = by_id[task_id]
    t0 = time.time()
    try:
        res = scaffold.solve(
            instance, lambda inst, ps: real_agent_fn(inst, ps, LOG_DIR), accountant
        )
        rec = {
            "instance_id": task_id,
            "model": MODEL_ID,
            "agent_src": AGENT_SRC_ID,
            "n_candidates_cfg": N,
            "language": instance.language,
            "eval_result": res.eval_result,
            "passed": res.passed,                       # oracle-blind scaffold result
            "pass_at_n": res.metadata.get("pass_at_n"),  # upper bound, NOT performance
            "n_passing_candidates": res.metadata.get("n_passing_candidates"),
            "per_candidate_eval": res.metadata.get("per_candidate_eval"),
            "error": res.error,
            "usage": {
                "calls": res.usage.calls,
                "prompt_tokens": res.usage.prompt_tokens,
                "completion_tokens": res.usage.completion_tokens,
            },
            "trace_op_counts": res.trace.op_counts(),
            "solution_len": len(res.solution),
            "elapsed_s": round(time.time() - t0, 1),
        }
    except Exception as exc:
        rec = {
            "instance_id": task_id, "model": MODEL_ID, "agent_src": AGENT_SRC_ID,
            "n_candidates_cfg": N, "language": instance.language,
            "eval_result": "", "passed": False, "pass_at_n": False,
            "n_passing_candidates": 0, "per_candidate_eval": [],
            "error": f"{type(exc).__name__}: {exc}",
            "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "trace_op_counts": {}, "solution_len": 0,
            "elapsed_s": round(time.time() - t0, 1),
        }
    results.append(rec)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    n_sel = sum(1 for r in results if r["passed"])
    n_any = sum(1 for r in results if r.get("pass_at_n"))
    print(
        f"[{i+1}/{len(TASK_IDS)}] {task_id} ({instance.language}): "
        f"{rec['eval_result']!r} selected_passed={rec['passed']} "
        f"pass@{N}={rec['pass_at_n']} cands={rec['per_candidate_eval']} "
        f"{rec['elapsed_s']}s | totals selected {n_sel}/{len(results)}, "
        f"pass@{N} {n_any}/{len(results)}",
        flush=True,
    )

n_sel = sum(1 for r in results if r["passed"])
n_any = sum(1 for r in results if r.get("pass_at_n"))
n_infra = sum(1 for r in results if r["eval_result"] in ("incomplete", ""))
valid = len(results) - n_infra
print("\n" + "=" * 60, flush=True)
print(f"[{SUFFIX}] COMPLETE  (N={N})", flush=True)
print(f"  oracle-blind selected : {n_sel}/{valid}   <- elicitation control", flush=True)
print(f"  pass@{N} (upper bound) : {n_any}/{valid}   <- NOT scaffold performance", flush=True)
print(f"  headroom a perfect selector would win: {n_any - n_sel}", flush=True)
print(f"  infra-excluded : {n_infra}", flush=True)
print(f"  total usage    : {accountant.spent}", flush=True)
