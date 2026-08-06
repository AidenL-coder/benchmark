"""Real Docker/HGM glue for cbs.scaffolds.swebench_scaffold (D-40).

Must be run from inside the hgm/ checkout (needs swe_bench/prompts/utils
importable via cwd), exactly like scripts/hgm_run_task_with_interception.py.
Implements SweBenchAgentFunction/SweBenchVerifyFunction by driving HGM's
real swe_bench.harness Docker machinery directly (build_container/
make_test_spec, the same functions swe_bench/harness.py:process_entry
itself calls), not reimplementing it.

Correction found only by actually running this, not by re-reading the
design more carefully (D-40): the original assumption here was that
PASS_TO_PASS tests are pre-existing and never need test_patch applied.
That was wrong. Confirmed directly: collecting test_separable.py at
base_commit with no test_patch applied found 11 tests; with test_patch
applied it found 15. Some PASS_TO_PASS entries are themselves new
parametrized cases test_patch introduces that simply happen to already
pass before the fix -- not pre-existing tests at all. The real rule:
PASS_TO_PASS means "passes regardless of whether the code fix is applied,
*given* test_patch is applied" -- not "doesn't need test_patch". So
`real_verify_fn` always applies test_patch now, for both PASS_TO_PASS and
FAIL_TO_PASS checks. Oracle-safety is preserved a different way than
originally assumed: test_patch is never applied inside the *agent's* own
working container (only inside this function's separate verify-only
container), and only a pass/fail signal for the *requested* test IDs is
ever surfaced back to the caller.

First-pass simplification, stated explicitly rather than silently: pass/
fail is read off pytest's exit code (0 = every requested test passed), not
per-test-node parsing of `-rA` output. Good enough to validate the pipeline
end-to-end; refine to per-node parsing if a real run needs to distinguish
"some but not all PASS_TO_PASS tests broke" from "all of them did".
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import docker
from swebench.harness.docker_build import build_container, cleanup_container
from swebench.harness.test_spec import make_test_spec

from prompts.testrepo_prompt import get_test_description
from swe_bench.utils import (
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
)

from cbs.budget import Usage
from cbs.scaffolds.fork_bridge import (
    ModelCallProxy,
    reconstruct_trace_from_events,
    usage_from_events,
)
from cbs.scaffolds.swebench_scaffold import SweBenchAttempt, SweBenchVerifyResult
from cbs.scaffolds.tagging import OperationTrace
from cbs.tasks.swebench import SweBenchInstance

#: The frozen agent variant used throughout this session (D-38/D-39) --
#: keep using the same one here rather than the top-level hgm/coding_agent.py.
AGENT_SRC = "measured_default_agent/src"
PROXY_PORT = 8000
VLLM_BACKEND = "http://127.0.0.1:8001"
LLM_MODEL_STRING = "vllm-model:localhost"
AGENT_TIMEOUT_S = 1800


def _entry_dict(instance: SweBenchInstance) -> dict:
    """swebench's make_test_spec expects a raw HF-row-shaped dict."""
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "problem_statement": instance.problem_statement,
        "patch": instance.patch,
        "test_patch": instance.test_patch,
        "FAIL_TO_PASS": list(instance.fail_to_pass),
        "PASS_TO_PASS": list(instance.pass_to_pass),
        "environment_setup_commit": instance.environment_setup_commit or instance.base_commit,
        "version": instance.version,
        # Read by make_test_spec but explicitly marked "Unused" in swebench's
        # own source -- SweBenchInstance deliberately doesn't carry this
        # field since nothing in cbs reads it either; a placeholder here is
        # not a data-fidelity gap, just satisfying a required dict key.
        "hints_text": "",
    }


def real_agent_fn(
    instance: SweBenchInstance, problem_statement: str, log_dir: Path
) -> SweBenchAttempt:
    """SweBenchAgentFunction: builds a real container from the instance's
    real environment image, runs HGM's own coding_agent.py inside it
    against the local vLLM model (routed through fork_bridge's proxy for
    tagging and usage), and extracts the produced diff.

    Known, inherited characteristic, not something this glue introduces:
    `diff_versus_commit` (inside `coding_agent.py` itself, unmodified) diffs
    the whole working tree against `base_commit`, so the returned diff can
    include incidental changes from the environment setup step (`eval_
    script`'s own `pip install -e .[test]` touching `pyproject.toml`,
    observed directly on a real run) alongside anything the agent actually
    changed. HGM's own `swe_bench/harness.py` has this same characteristic
    (same `diff_versus_commit` call, same eval_script) -- not filtered here
    or there. Harmless for scoring (a real fix's own genuine hunks are still
    present and still what verification actually checks), just noisy to
    read directly.
    """
    entry = _entry_dict(instance)
    client = docker.from_env()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger = setup_logger(str(log_dir / f"{instance.instance_id}_{run_id}_docker.log"))
    test_spec = make_test_spec(entry)
    container_name = test_spec.get_instance_container_name(run_id)
    remove_existing_container(client, container_name)

    proxy = ModelCallProxy(PROXY_PORT, VLLM_BACKEND)
    proxy.start()
    container = None
    try:
        container = build_container(
            test_spec, client, run_id, logger, nocache=True, force_rebuild=False
        )
        container.start()

        for f in ["coding_agent.py", "requirements.txt", "pytest.ini", "LICENSE", "README.md"]:
            copy_to_container(container, os.path.join(AGENT_SRC, f), f"/hgm/{f}")
        for d in ["tools", "utils", "prompts"]:
            copy_to_container(container, os.path.join(AGENT_SRC, d) + "/", f"/hgm/{d}/")
        copy_to_container(container, os.path.join(AGENT_SRC, "llm.py"), "/hgm/llm.py")
        copy_to_container(
            container, os.path.join(AGENT_SRC, "llm_withtools.py"), "/hgm/llm_withtools.py"
        )

        eval_script = test_spec.eval_script
        eval_file = log_dir / f"{instance.instance_id}_{run_id}_eval.sh"
        eval_file.write_text(eval_script)
        copy_to_container(container, eval_file, "/eval.sh")
        setup_result = container.exec_run("/bin/bash /eval.sh", workdir="/")
        log_container_output(setup_result, raise_error=False)
        container.exec_run("rm /eval.sh", workdir="/")

        # The agent's own requirements (anthropic/openai/backoff/etc. --
        # llm.py imports these) -- omitting this step produced a real,
        # immediate ModuleNotFoundError the first time this ran for real.
        pip_result = container.exec_run(
            "python -m pip install -r /hgm/requirements.txt", workdir="/"
        )
        log_container_output(pip_result, raise_error=False)

        test_description = get_test_description(eval_script=eval_script, swerepo=True)

        chat_history_file_container = f"/hgm/{instance.instance_id}.md"
        cmd = [
            "timeout", str(AGENT_TIMEOUT_S),
            "python", "/hgm/coding_agent.py",
            "--problem_statement", problem_statement,
            "--git_dir", "/testbed/",
            "--chat_history_file", chat_history_file_container,
            "--base_commit", instance.base_commit,
            "--outdir", "/hgm/",
            "--test_description", test_description,
            "--instance_id", instance.instance_id,
            "--model", LLM_MODEL_STRING,
            "--timeout", str(AGENT_TIMEOUT_S),
        ]
        env_vars = {"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}
        exec_result = container.exec_run(cmd, environment=env_vars, workdir="/")
        agent_output = exec_result.output.decode(errors="replace")

        events = proxy.events
        trace = reconstruct_trace_from_events(events)
        usage = usage_from_events(events)

        diff_result = container.exec_run("cat /hgm/model_patch.diff")
        diff = (
            diff_result.output.decode(errors="replace")
            if diff_result.exit_code == 0
            else ""
        )

        error = (
            ""
            if exec_result.exit_code == 0
            else f"agent process exited {exec_result.exit_code}: {agent_output[-2000:]}"
        )
        return SweBenchAttempt(diff=diff, trace=trace, usage=usage, error=error)
    except Exception as exc:  # noqa: BLE001 -- must always return an Attempt, never crash the caller
        return SweBenchAttempt(
            diff="", trace=OperationTrace(), usage=Usage(), error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        proxy.stop()
        if container is not None:
            try:
                cleanup_container(client, container, logger)
            except Exception:
                pass


def real_verify_fn(
    instance: SweBenchInstance,
    diff: str,
    test_ids: tuple,
    log_dir: Path,
) -> SweBenchVerifyResult:
    """SweBenchVerifyFunction: applies `diff` to a fresh container built
    from the instance's real environment image and runs exactly `test_ids`.

    **Correction, found only by actually running this, not by re-reading
    the design more carefully**: `test_patch` is *always* applied now,
    regardless of which test set is being checked. The original assumption
    -- PASS_TO_PASS tests are pre-existing and never need `test_patch` --
    was wrong. Confirmed directly: collecting `test_separable.py` at
    `base_commit` with no `test_patch` applied found only 11 tests
    (`compound_model0`-`compound_model5`); with `test_patch` applied it
    found 15 (`compound_model0`-`compound_model9`). Some `PASS_TO_PASS`
    entries (`compound_model7`, `compound_model8` here) are themselves new
    parametrized cases `test_patch` introduces that simply happen to
    already pass before the fix -- not pre-existing tests at all. The real
    rule: PASS_TO_PASS means "passes regardless of whether the code fix is
    applied, *given* `test_patch` is applied" -- not "doesn't need
    `test_patch`". Oracle-safety is preserved a different way than
    originally assumed: `test_patch` is never applied inside the *agent's*
    own working container (only here, in this separate verify-only
    container), and only a pass/fail signal for the *requested* test IDs is
    ever surfaced back to the caller -- the agent never sees `test_patch`'s
    content or any FAIL_TO_PASS-specific result during the run, regardless
    of which test IDs get checked in this container.
    """
    if not diff.strip():
        return SweBenchVerifyResult(passed=False, raw_output="empty diff")

    entry = _entry_dict(instance)
    client = docker.from_env()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_verify"
    logger = setup_logger(str(log_dir / f"{instance.instance_id}_{run_id}_docker.log"))
    test_spec = make_test_spec(entry)
    container_name = test_spec.get_instance_container_name(run_id)
    remove_existing_container(client, container_name)

    container = None
    try:
        container = build_container(
            test_spec, client, run_id, logger, nocache=True, force_rebuild=False
        )
        container.start()
        # Defensive reset -- a fresh container from the instance image
        # should already be a clean checkout at base_commit, but this is
        # cheap insurance against any residual state.
        container.exec_run(
            "git config --global --add safe.directory /testbed", workdir="/"
        )
        container.exec_run(f"git checkout {instance.base_commit} -- .", workdir="/testbed")

        diff_file = log_dir / f"{instance.instance_id}_{run_id}.diff"
        diff_file.write_text(diff)
        copy_to_container(container, diff_file, "/tmp/model.diff")
        apply_result = container.exec_run("git apply -v /tmp/model.diff", workdir="/testbed")
        if apply_result.exit_code != 0:
            return SweBenchVerifyResult(
                passed=False,
                raw_output=(
                    f"patch did not apply: "
                    f"{apply_result.output.decode(errors='replace')[-1000:]}"
                ),
            )

        test_patch_file = log_dir / f"{instance.instance_id}_{run_id}_test.diff"
        test_patch_file.write_text(instance.test_patch)
        copy_to_container(container, test_patch_file, "/tmp/test.diff")
        test_patch_result = container.exec_run("git apply -v /tmp/test.diff", workdir="/testbed")
        if test_patch_result.exit_code != 0:
            return SweBenchVerifyResult(
                passed=False,
                raw_output=(
                    f"test_patch did not apply (harness bug, not a model-diff "
                    f"problem): {test_patch_result.output.decode(errors='replace')[-1000:]}"
                ),
            )

        # `set -f` (noglob) is essential: PASS_TO_PASS/FAIL_TO_PASS node IDs
        # routinely look like test_separable[compound_model0-result0] --
        # bash treats [...] as a glob pattern and silently mangles it
        # otherwise (confirmed empirically: without this, pytest reported
        # "collected 11 items" / "no tests ran", not a real 0/11 failure).
        test_cmd = (
            "set -f; source /opt/miniconda3/bin/activate && conda activate testbed "
            f"&& cd /testbed && pytest -rA {' '.join(test_ids)}"
        )
        result = container.exec_run(["/bin/bash", "-c", test_cmd], workdir="/testbed")
        output = result.output.decode(errors="replace")
        return SweBenchVerifyResult(passed=result.exit_code == 0, raw_output=output[-2000:])
    except Exception as exc:  # noqa: BLE001 -- must always return a result, never crash the caller
        return SweBenchVerifyResult(passed=False, raw_output=f"{type(exc).__name__}: {exc}")
    finally:
        if container is not None:
            try:
                cleanup_container(client, container, logger)
            except Exception:
                pass
