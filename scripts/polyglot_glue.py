"""Real Docker/HGM glue for cbs.scaffolds.polyglot_scaffold (D-42).

Must be run from inside the hgm/ checkout (needs polyglot/ modules
importable via cwd), exactly like scripts/hgm_run_task_with_interception.py
and scripts/swebench_glue.py. Implements PolyglotAgentFunction by calling
HGM's own polyglot.harness.process_entry directly -- unlike
swebench_glue.py, there is no lower-level container split needed here:
process_entry already is the single real function that builds the
container, runs the agent, reveals the hidden test_commit, and grades the
result, all in one atomic call (D-42's module docstring in
polyglot_scaffold.py has the full reasoning for why this collapses to one
injected function instead of two).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import docker

from cbs.budget import Usage
from cbs.scaffolds.fork_bridge import (
    ModelCallProxy,
    reconstruct_trace_from_events,
    usage_from_events,
)
from cbs.scaffolds.polyglot_scaffold import PolyglotRunResult
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.polyglot import PolyglotInstance

#: The frozen agent variant (D-38/D-39/D-40) -- not hgm's own top-level
#: coding_agent.py.
#:
#: Overridable via the CBS_AGENT_SRC environment variable so the
#: scaffold-sensitivity sub-study's two arms (preregistration.md §4.1) run
#: through *identical* code with only the agent directory swapped -- no
#: source edit between arms, which is what makes the comparison clean.
#: `toolcheck_agent_src/` differs from the default in exactly two lines,
#: both the `tool_choice` flag; verified by `diff`, not asserted.
AGENT_SRC = os.environ.get("CBS_AGENT_SRC", "measured_default_agent/src")
PROXY_PORT = 8000
VLLM_BACKEND = "http://127.0.0.1:8001"
LLM_MODEL_STRING = "vllm-model:localhost"

#: Where HGM's own `build_image` stages a per-task Docker build context.
BUILD_CONTEXT_ROOT = Path("logs/build_images/instances")


def _clear_stale_build_context(instance_id: str) -> None:
    """Remove any leftover Docker build-context staging for this task.

    **A real bug this fixes, found by running the 59-task batch (D-43)**, not
    a defensive nicety: HGM's `build_image` stages the build context by
    `shutil.copytree`-ing the exercise repo (including its `.git/`) into
    `logs/build_images/instances/<image_name>/`. Git object files are created
    **read-only** (mode 0444). On a *persistent* filesystem -- exactly this
    project's setup, where `/lambda/nfs/cbs-project` survives instance
    termination -- a staging directory left behind by an earlier session
    still holds those read-only objects, and the next run's `copytree` dies
    with `[Errno 13] Permission denied` trying to overwrite them.

    That failure is silently swallowed by `process_entry`'s own broad
    `except`, which sets `eval_result = "incomplete"` -- i.e. an
    infrastructure error is indistinguishable, in the result dict, from a
    task the model genuinely failed. Left unfixed it would have quietly
    depressed the measured baseline pass rate, which is precisely the number
    this batch exists to establish.

    Clearing the staging directory (regenerable build context, never source
    data) before each task removes the collision at its source rather than
    retrying around it. `onerror` restores write permission so the read-only
    git objects can actually be unlinked.
    """
    staging = BUILD_CONTEXT_ROOT / f"pb.eval.x86_64.{instance_id}__latest"
    if not staging.exists():
        return

    def _force_writable(func, path, exc_info):
        Path(path).chmod(stat.S_IWRITE | stat.S_IREAD)
        func(path)

    shutil.rmtree(staging, onerror=_force_writable)


def real_agent_fn(
    instance: PolyglotInstance, problem_statement: str, out_dir: Path
) -> PolyglotRunResult:
    """PolyglotAgentFunction: runs HGM's real `polyglot.harness.
    process_entry` against the local vLLM model (routed through
    fork_bridge's proxy for tagging and usage). `process_entry` both runs
    the agent trajectory and grades the result in one call -- there is no
    separate verify step to wire up here, unlike `swebench_glue.py`'s
    `real_agent_fn`/`real_verify_fn` split.

    `problem_statement` is accepted only for interface-consistency with
    `PolyglotAgentFunction` (`process_entry` always reads it off
    `instance.raw` itself, HGM's own harness has no separate parameter for
    it) -- there is no repair-loop caller yet that would ever pass a value
    different from `instance.raw["problem_statement"]` (see
    `polyglot_scaffold.py`'s module docstring on why `SStarPolyglot`
    doesn't exist yet).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    import polyglot.harness as polyglot_harness
    from polyglot.docker_build import build_env_images

    # Module-level global, not a process_entry parameter -- same gotcha
    # documented in hgm_run_task_with_interception.py (D-38).
    polyglot_harness.llm = LLM_MODEL_STRING

    # Must happen before build_env_images/process_entry touch the staging
    # directory -- see _clear_stale_build_context for the real failure this
    # prevents (D-43).
    _clear_stale_build_context(instance.instance_id)

    proxy = ModelCallProxy(PROXY_PORT, VLLM_BACKEND)
    proxy.start()
    try:
        # process_entry does not build environment images itself (D-38) --
        # mirror harness()'s own setup step rather than assume the image
        # already exists.
        build_env_images(
            docker.from_env(), dataset=[instance.raw], max_workers=1, force_rebuild=False
        )

        result = polyglot_harness.process_entry(
            instance.raw,
            out_dname=out_dir,
            model_name_or_path="cbs-measured-baseline",
            model_patch_paths=[],
            skip_existing=False,
            init_agent_path=AGENT_SRC,
        )

        events = proxy.events
        trace = reconstruct_trace_from_events(events)
        usage = usage_from_events(events)

        # process_entry's own return value is a small subset
        # ({"success", "instance_id", "eval_result"}) -- the actual
        # model_patch only lands in the JSON file it writes to out_dname,
        # confirmed by reading process_entry's source directly rather than
        # assumed from its return statement.
        solution_file = out_dir / f"{instance.instance_id}.json"
        model_patch = ""
        if solution_file.exists():
            model_patch = json.loads(solution_file.read_text()).get("model_patch", "")

        return PolyglotRunResult(
            solution=model_patch,
            eval_result=result.get("eval_result", ""),
            trace=trace,
            usage=usage,
        )
    except Exception as exc:  # noqa: BLE001 -- must always return a result, never crash the caller
        return PolyglotRunResult(
            solution="",
            eval_result="",
            trace=OperationTrace(),
            usage=Usage(),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        proxy.stop()
