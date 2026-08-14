"""Minimal evolutionary run driver (preregistration §4.3).

Runs HGM's own evolutionary loop (`hgm.py`) against the local frozen model,
with `cbs`'s network-layer interception proxy in the path so that every model
call the loop makes -- the seed evaluation, each `sample_child`
self-improvement attempt, each child's task evaluations, and the diagnosis
step -- is captured without trusting any component to self-report (D-24).

**Why a proxy rather than an in-process wrapper**: HGM runs its agents as
separate OS processes inside Docker containers, so there is nothing for an
`InterceptionSession` to wrap. `ModelCallProxy` sits between the containers
and the model server instead (D-38). The fork's own `llm.py` resolves
`"vllm-model:<host>"` to `http://<host>:8000/v1`, and the real vLLM server
listens on 8001 -- so port 8000 is exactly the seam the proxy occupies. This
is a deliberate convention shared with `scripts/polyglot_glue.py`, not a
coincidence.

**Memory note**: captured events are held in memory for the whole run and
deliberately never cleared mid-run. `ModelCallProxy.reset()` drains and
clears atomically but exposes no atomic "drain, return, and clear", so any
periodic flush built on it would have a window in which a call recorded
between the snapshot and the clear is silently dropped. A lossy trace is
worse than a large one, so the run holds everything and dumps at the end;
the periodic dump below is a non-clearing snapshot for monitoring only.

Usage:
    run_evolutionary_loop.py <max_task_evals> <max_workers> <run_label> [extra hgm.py args...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

from cbs.scaffolds.fork_bridge import (  # noqa: E402
    ModelCallProxy,
    reconstruct_trace_from_events,
    usage_from_events,
)

PROXY_PORT = 8000
VLLM_BACKEND = "http://127.0.0.1:8001"

MAX_TASK_EVALS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
MAX_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
RUN_LABEL = sys.argv[3] if len(sys.argv) > 3 else "evo1"
EXTRA_ARGS = sys.argv[4:]

NFS = Path("/lambda/nfs/cbs-project")
EVENTS_OUT = NFS / f"evo_{RUN_LABEL}_events.json"
SUMMARY_OUT = NFS / f"evo_{RUN_LABEL}_summary.json"

proxy = ModelCallProxy(PROXY_PORT, VLLM_BACKEND)
proxy.start()
print(f"[evo] interception proxy on :{PROXY_PORT} -> {VLLM_BACKEND}", flush=True)


def summarize(events) -> dict:
    """Cheap, allocation-light view of the run so far. Deliberately reports
    failed calls separately: a non-200 is not a generation (D-38)."""
    ok = [e for e in events if e.status == 200]
    usage = usage_from_events(ok)
    with_tools = sum(
        1
        for e in ok
        if (e.response_body.get("choices") or [{}])[0]
        .get("message", {})
        .get("tool_calls")
    )
    return {
        "run_label": RUN_LABEL,
        "calls_total": len(events),
        "calls_ok": len(ok),
        "calls_failed": len(events) - len(ok),
        "calls_with_tool_calls": with_tools,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


stop_monitor = threading.Event()


def monitor() -> None:
    while not stop_monitor.wait(120):
        try:
            s = summarize(proxy.events)
            SUMMARY_OUT.write_text(json.dumps(s, indent=2))
            print(f"[evo][monitor] {s}", flush=True)
        except Exception as exc:  # monitoring must never kill the run
            print(f"[evo][monitor] error: {type(exc).__name__}: {exc}", flush=True)


threading.Thread(target=monitor, daemon=True).start()

cmd = [
    sys.executable,
    "-u",
    "hgm.py",
    "--polyglot",
    "--max_task_evals",
    str(MAX_TASK_EVALS),
    "--max_workers",
    str(MAX_WORKERS),
    *EXTRA_ARGS,
]
print(f"[evo] launching: {' '.join(cmd)}", flush=True)

t0 = time.time()
rc = subprocess.call(cmd)
elapsed = time.time() - t0
print(f"[evo] hgm.py exited rc={rc} after {elapsed/3600:.2f}h", flush=True)

stop_monitor.set()
events = proxy.events
proxy.stop()

summary = summarize(events)
summary["hgm_exit_code"] = rc
summary["elapsed_hours"] = round(elapsed / 3600, 3)
try:
    summary["trace_op_counts"] = reconstruct_trace_from_events(
        [e for e in events if e.status == 200]
    ).op_counts()
except Exception as exc:
    summary["trace_op_counts_error"] = f"{type(exc).__name__}: {exc}"

SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
EVENTS_OUT.write_text(
    json.dumps(
        [
            {
                "status": e.status,
                "timestamp": e.timestamp,
                "model": e.request_body.get("model"),
                "n_messages": len(e.request_body.get("messages") or []),
                "tool_choice": e.request_body.get("tool_choice"),
                "usage": e.response_body.get("usage"),
                "finish_reason": (e.response_body.get("choices") or [{}])[0].get(
                    "finish_reason"
                ),
                "had_tool_calls": bool(
                    (e.response_body.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("tool_calls")
                ),
            }
            for e in events
        ],
        indent=2,
    )
)
print(f"[evo] COMPLETE {json.dumps(summary, indent=2)}", flush=True)
sys.exit(rc)
