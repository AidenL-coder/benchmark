"""Deploy-and-run glue for measuring one HGM agent variant against held-out
Polyglot tasks with cbs's interception active.

**Not part of the `cbs` package** -- this script imports HGM's own modules
(`polyglot.harness`) directly, so it only runs from inside a checked-out HGM
clone (i.e. copied to `.../hgm/` on the measurement host, run with that
directory as the working directory and HGM's own venv active), with `cbs`
importable in the same environment (`pip install -e` this repo, or add its
`src/` to `PYTHONPATH`).

**Validated against the real remote instance (docs/DECISIONS.md D-38)**: 7
real Polyglot tasks run end to end, cross-checked by hand against the raw
`.md` transcript for at least one (the trace's `single_call`/`had_tool_calls`
classification matched the actual model response byte-for-byte). Two real
bugs were found and fixed in the process, both now handled by this script
rather than left as a "remember to do X" note -- see the model-string global
below, and the `build_env_images` call further down. Still not validated on
real data: a task where the agent actually invokes a tool (every real run so
far made exactly one plain generation and stopped) -- the `tool_call`
classification path is only proven against synthetic conversations
(`tests/test_fork_bridge.py`), not yet against a real tool-call round trip.

Each record now also carries real `usage` (`calls`/`prompt_tokens`/
`completion_tokens`), via `fork_bridge.usage_from_events` (D-40) -- this
script previously captured the trace but never turned the same proxied
events into chargeable `Usage`, leaving a real gap between "this data is
already sitting in `.events`" and "actually charged to `cbs.budget`" for
this task family specifically.

Prerequisites this script does NOT set up for you (must already be true of
the environment it runs in):
  * The real vLLM server must be listening on a port OTHER than the one this
    script's proxy binds to (default 8000) -- e.g. relaunch vLLM on 8001 and
    pass --vllm-backend http://127.0.0.1:8001. If vLLM is still on 8000 when
    this script's proxy also tries to bind 8000, the proxy will fail to
    start, loudly, not silently misbehave.
  * `network_mode="host"` must already be patched into the container-
    creation call (docs/DECISIONS.md D-37) -- otherwise the containerized
    agent cannot reach the proxy at all, the same failure mode that produced
    the false-success incident this script's whole design is meant to avoid
    repeating blindly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cbs.scaffolds.fork_bridge import (
    ModelCallProxy,
    reconstruct_trace_from_events,
    usage_from_events,
)


def _load_polyglot_entry(task_id: str) -> dict:
    """Look up one task's full entry dict from HGM's own prepared metadata.

    `polyglot/polyglot_benchmark_metadata.json` is what `hgm_utils.py` itself
    reads (confirmed by reading `sample_child`'s `with open("polyglot/
    polyglot_benchmark_metadata.json")` call) -- reusing the exact same file
    HGM trusts, not a separately-derived copy that could drift from it.
    """
    with open("polyglot/polyglot_benchmark_metadata.json", encoding="utf-8") as fh:
        entries = json.load(fh)
    for entry in entries:
        if entry.get("instance_id") == task_id:
            return entry
    raise KeyError(f"task_id {task_id!r} not found in polyglot_benchmark_metadata.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-src", required=True, help="init_agent_path -- the frozen agent variant's src/ dir"
    )
    parser.add_argument(
        "--model-name", required=True, help="model_name_or_path -- HGM's commit_id/label for this variant"
    )
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vllm-backend", default="http://127.0.0.1:8001")
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument(
        "--llm-model-string",
        default="vllm-model:localhost",
        help="the actual model string llm.py's create_client() dispatches on -- "
        "NOT the same thing as --model-name, which is only a label. hgm.py's "
        "own main() sets this as a module-level global on polyglot.harness "
        "(`polyglot.harness.llm = llm_cfg.downstream_llm`) before calling "
        "process_entry; process_entry itself takes no model-string parameter "
        "at all. Forgetting this makes the agent fall through to "
        "create_client's OpenRouter branch silently (confirmed the hard way: "
        "see docs/DECISIONS.md D-37/D-38 for the run this bug produced).",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    import docker

    import polyglot.harness as polyglot_harness  # noqa: E402 -- must run from hgm/ checkout
    from polyglot.docker_build import build_env_images

    # Module-level global, not a process_entry parameter -- this is the one
    # thing hgm.py's main() sets that process_entry does not accept directly.
    polyglot_harness.llm = args.llm_model_string

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [_load_polyglot_entry(tid) for tid in args.task_ids]
    # process_entry does NOT build environment images itself -- that setup
    # step lives in harness()'s own batch orchestration, which calling
    # process_entry directly bypasses. Skipping this produced a real failure
    # the first time this script ran a task whose image wasn't already built
    # from an earlier session (docs/DECISIONS.md D-38): "Environment image
    # ... not found". Mirroring harness()'s own setup, not working around it.
    build_env_images(docker.from_env(), dataset=entries, max_workers=1, force_rebuild=False)

    proxy = ModelCallProxy(args.proxy_port, args.vllm_backend)
    proxy.start()
    records = []
    try:
        for task_id, entry in zip(args.task_ids, entries):
            proxy.reset()
            result = polyglot_harness.process_entry(
                entry,
                out_dname=out_dir,
                model_name_or_path=args.model_name,
                model_patch_paths=[],
                skip_existing=False,
                init_agent_path=args.agent_src,
            )
            events = proxy.events
            trace = reconstruct_trace_from_events(events)
            usage = usage_from_events(events)
            records.append(
                {
                    "task_id": task_id,
                    "eval_result": result.get("eval_result"),
                    "passed": result.get("eval_result") == "resolved",
                    "n_proxy_events": len(events),
                    "trace": trace.as_dict(),
                    "usage": {
                        "calls": usage.calls,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    },
                }
            )
    finally:
        proxy.stop()

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
