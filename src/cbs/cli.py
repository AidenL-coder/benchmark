"""`cbs` command line.

One command reproduces one experiment (brief section 11). Commands:

    cbs env                     host capability report (Phase 0 gate)
    cbs registry                dump the support-class partition
    cbs tasks verify            run every task's reference solution
    cbs splits freeze|verify    freeze / check hashed splits
    cbs frontier validate       Phase 2 DoD: estimators vs known ground truth
    cbs frontier estimate       run the frontier estimation from a config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cbs import __version__


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
def cmd_env(args) -> int:
    """Report what this host can actually do (brief section 6)."""
    import platform

    from cbs.sandbox import DockerSandbox, SubprocessSandbox, select_backend

    report: dict = {
        "cbs_version": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "sandbox": {
            "docker_available": DockerSandbox.available(),
            "subprocess_available": SubprocessSandbox.available(),
        },
    }

    chosen = select_backend("auto")
    report["sandbox"]["auto_selected"] = chosen.name
    report["sandbox"]["is_security_boundary"] = chosen.is_security_boundary

    try:
        import httpx  # noqa: F401

        report["serving_extra_installed"] = True
    except ImportError:
        report["serving_extra_installed"] = False

    gpu = _detect_gpu()
    report["gpu"] = gpu

    _print_json(report)

    problems = []
    if not chosen.is_security_boundary:
        problems.append(
            "No Docker: untrusted code will run WITHOUT a security boundary. "
            "Fine for the toy family and frontier sampling of small models; "
            "NOT acceptable for running S_evo self-modifying scaffolds."
        )
    if not gpu.get("cuda_available"):
        problems.append(
            "No CUDA GPU detected: local vLLM serving is unavailable here. Use "
            "the mock backend for instrument validation, and point "
            "model.base_url at a remote/Colab vLLM server for real runs."
        )
    if problems:
        print("\nCaveats:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    return 0


def _detect_gpu() -> dict:
    import shutil
    import subprocess

    out: dict = {"cuda_available": False, "detail": "nvidia-smi not found"}
    if shutil.which("nvidia-smi") is None:
        return out
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            out["cuda_available"] = True
            out["detail"] = proc.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        out["detail"] = f"nvidia-smi failed: {exc}"
    return out


# ---------------------------------------------------------------------------
def cmd_registry(args) -> int:
    from cbs.scaffolds.tagging import registry_snapshot

    snapshot = registry_snapshot()
    if args.json:
        _print_json(snapshot)
        return 0

    for cls in ("support-preserving", "support-expanding"):
        print(f"\n{cls.upper()}")
        print("=" * 78)
        for name, d in snapshot.items():
            if d["support_class"] != cls:
                continue
            flag = "  [CONTESTED]" if d["contested"] else ""
            print(f"\n  {name}{flag}")
            for line in _wrap(d["rationale"], 72):
                print(f"      {line}")
    print()
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


# ---------------------------------------------------------------------------
def _load_suite(family: str):
    if family == "toy":
        from cbs.tasks.families.toy import toy_suite

        return toy_suite()
    raise SystemExit(
        f"unknown task family {family!r}. Only 'toy' is vendored; real benchmark "
        "families need data fetched separately."
    )


def cmd_tasks_verify(args) -> int:
    """Run every task's reference solution (Phase 1 DoD)."""
    from cbs.sandbox import select_backend
    from cbs.tasks import Verifier

    suite = _load_suite(args.family)
    verifier = Verifier(select_backend(args.sandbox))
    print(f"suite={suite.name} n={len(suite)} sandbox={verifier.sandbox.name}")
    print(f"suite_hash={suite.suite_hash()}")
    print("-" * 78)

    failures = 0
    for task in suite.tasks:
        result = verifier.self_test(task)
        status = "OK  " if result.passed else "FAIL"
        if not result.passed:
            failures += 1
        print(f"{status} {task.task_id:28s} {result.reason}")
        if not result.passed and result.exec_result:
            print(f"       {result.exec_result.stderr[:300]}")
    print("-" * 78)
    print(f"{len(suite) - failures}/{len(suite)} reference solutions pass")
    return 1 if failures else 0


def cmd_splits_freeze(args) -> int:
    from cbs.tasks.splits import SplitRatios, assign_splits, write_manifest

    suite = _load_suite(args.family)
    ratios = SplitRatios(
        train=args.train, held_out=args.held_out, transfer=args.transfer
    )
    assigned = assign_splits(suite, ratios, salt=args.salt)
    manifest = write_manifest(assigned, ratios, Path(args.out), salt=args.salt)
    print(f"froze {len(assigned)} tasks -> {args.out}")
    print(f"suite_hash={manifest.suite_hash}")
    print(f"counts={manifest.counts}")
    return 0


def cmd_splits_verify(args) -> int:
    from cbs.tasks.splits import SplitRatios, assign_splits, verify_manifest

    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    suite = _load_suite(args.family)
    ratios = SplitRatios(**data["ratios"])
    assigned = assign_splits(suite, ratios, salt=data["salt"])
    result = verify_manifest(assigned, Path(args.manifest))
    _print_json(result)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
def cmd_frontier_validate(args) -> int:
    """Phase 2 DoD: estimators recover known ground truth."""
    from cbs.frontier.validate import (
        validate_chao1_lower_bound,
        validate_ci_coverage,
        validate_pipeline,
        validate_verifier_agreement,
    )

    ok = True

    print("=" * 78)
    print("1. CI COVERAGE (statistical procedure, simulated)")
    print("=" * 78)
    coverage = validate_ci_coverage(
        n_samples=args.coverage_samples, n_replicates=args.replicates
    )
    print(coverage.report())
    ok &= coverage.ok

    print()
    print("=" * 78)
    print("2. CHAO1 LOWER-BOUND BEHAVIOUR (simulated undersampling)")
    print("=" * 78)
    chao = validate_chao1_lower_bound()
    for key, value in chao.items():
        print(f"  {key}: {value}")
    ok &= bool(chao["ok"])

    print()
    print("=" * 78)
    print("3. VERIFIER vs KNOWN GROUND TRUTH (false positive / negative rate)")
    print("=" * 78)
    agreement = validate_verifier_agreement(n_per_task=args.agreement_samples)
    print(
        f"  checked {agreement['n_checked']} labelled samples over "
        f"{agreement['n_tasks']} tasks"
    )
    print(f"  false positives: {agreement['false_positives']}")
    print(f"  false negatives: {agreement['false_negatives']}")
    print(f"  agreement rate:  {agreement['agreement_rate']:.4f}")
    for d in agreement["disagreements"]:
        print(f"    {d}")
    ok &= bool(agreement["ok"])

    print()
    print("=" * 78)
    print("4. FULL PIPELINE vs KNOWN GROUND TRUTH")
    print("=" * 78)
    pipeline = validate_pipeline(
        n_samples=args.n_samples, output_dir=args.output_dir
    )
    print(pipeline.report())
    ok &= pipeline.ok

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "ok": ok,
                    "ci_coverage": coverage.as_dict(),
                    "chao1_lower_bound": chao,
                    "verifier_agreement": agreement,
                    "pipeline": pipeline.as_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")

    print()
    print("PHASE 2 DoD: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_frontier_estimate(args) -> int:
    """Run frontier estimation from a config file."""
    from cbs.budget import BudgetAccountant, BudgetCaps
    from cbs.config import load_config, parse_cli_overrides
    from cbs.frontier.sampler import FrontierSampler, TemperatureSchedule
    from cbs.models import build_model
    from cbs.sandbox import select_backend
    from cbs.tasks import Verifier
    from cbs.tasks.schema import Split
    from cbs.tasks.splits import SplitRatios, assign_splits

    config = load_config(args.config, parse_cli_overrides(args.set or []))
    print(f"config={args.config} fingerprint={config.fingerprint()}")

    model = build_model(config.section("model"))
    sandbox_cfg = config.section("sandbox")
    sandbox = select_backend(
        backend=sandbox_cfg.get("backend", "auto"),
        allow_insecure_fallback=sandbox_cfg.get("allow_insecure_fallback", False),
    )
    verifier = Verifier(sandbox)

    tasks_cfg = config.section("tasks")
    suite = _load_suite(tasks_cfg.get("family", "toy"))
    ratios = SplitRatios(**tasks_cfg.get("splits", {"train": 0.5, "held_out": 0.5}))
    suite = assign_splits(suite, ratios, salt=tasks_cfg.get("salt", "cbs-v1"))

    split_name = args.split or tasks_cfg.get("estimate_split")
    if split_name:
        suite = suite.filter_split(Split(split_name))

    frontier_cfg = config.section("frontier")
    stages = frontier_cfg.get("schedule")
    schedule = (
        TemperatureSchedule(tuple((float(t), float(f)) for t, f in stages))
        if stages
        else None
    )

    caps_cfg = config.section("budget").get("caps", {})
    accountant = BudgetAccountant("frontier", BudgetCaps(**caps_cfg))

    n_max = args.n_max or int(frontier_cfg.get("n_max", 100))
    output_dir = args.output_dir or config.get("output_dir", "runs/frontier")

    sampler_kwargs = dict(
        model=model,
        verifier=verifier,
        output_dir=output_dir,
        confidence=float(frontier_cfg.get("confidence", 0.95)),
        store_raw_completions=bool(frontier_cfg.get("store_raw_completions", False)),
    )
    if schedule:
        sampler_kwargs["schedule"] = schedule
    sampler = FrontierSampler(**sampler_kwargs)

    print(
        f"model={model.model_id} sandbox={sandbox.name} "
        f"tasks={len(suite)} split={split_name or 'all'} N_max={n_max}"
    )
    if args.dry_run:
        est_calls = len(suite) * n_max
        print(
            f"\nDRY RUN -- would draw {est_calls} samples "
            f"({len(suite)} tasks x {n_max}).\n"
            "Re-run without --dry-run to execute."
        )
        return 0

    records = sampler.estimate_suite(suite, n_max, accountant, resume=not args.no_resume)

    out_path = Path(output_dir) / "records.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json() + "\n")

    beyond = [r for r in records if r.beyond_frontier]
    print("\n" + "=" * 78)
    print(f"wrote {len(records)} records -> {out_path}")
    print(f"beyond-frontier at N={n_max}: {len(beyond)}/{len(records)}")
    for record in beyond:
        print(f"  {record.summary_line()}")
    print(f"spend: {accountant.spent.as_dict()}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cbs", description=__doc__)
    parser.add_argument("--version", action="version", version=f"cbs {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="host capability report").set_defaults(func=cmd_env)

    p_reg = sub.add_parser("registry", help="dump the support-class partition")
    p_reg.add_argument("--json", action="store_true")
    p_reg.set_defaults(func=cmd_registry)

    p_tasks = sub.add_parser("tasks", help="task operations")
    tasks_sub = p_tasks.add_subparsers(dest="tasks_command", required=True)
    p_tv = tasks_sub.add_parser("verify", help="run every reference solution")
    p_tv.add_argument("--family", default="toy")
    p_tv.add_argument("--sandbox", default="auto", choices=["auto", "docker", "subprocess"])
    p_tv.set_defaults(func=cmd_tasks_verify)

    p_splits = sub.add_parser("splits", help="frozen hashed splits")
    splits_sub = p_splits.add_subparsers(dest="splits_command", required=True)
    p_sf = splits_sub.add_parser("freeze")
    p_sf.add_argument("--family", default="toy")
    p_sf.add_argument("--out", default="data/splits/toy.json")
    p_sf.add_argument("--salt", default="cbs-v1")
    p_sf.add_argument("--train", type=float, default=0.5)
    p_sf.add_argument("--held-out", dest="held_out", type=float, default=0.5)
    p_sf.add_argument("--transfer", type=float, default=0.0)
    p_sf.set_defaults(func=cmd_splits_freeze)
    p_sv = splits_sub.add_parser("verify")
    p_sv.add_argument("--family", default="toy")
    p_sv.add_argument("--manifest", default="data/splits/toy.json")
    p_sv.set_defaults(func=cmd_splits_verify)

    p_front = sub.add_parser("frontier", help="frontier estimation")
    front_sub = p_front.add_subparsers(dest="frontier_command", required=True)

    p_fv = front_sub.add_parser("validate", help="Phase 2 DoD")
    p_fv.add_argument("--n-samples", type=int, default=300)
    p_fv.add_argument("--coverage-samples", type=int, default=300)
    p_fv.add_argument("--replicates", type=int, default=400)
    p_fv.add_argument("--agreement-samples", type=int, default=120)
    p_fv.add_argument("--output-dir", default="runs/validation")
    p_fv.add_argument("--json-out", default="")
    p_fv.set_defaults(func=cmd_frontier_validate)

    p_fe = front_sub.add_parser("estimate", help="run frontier estimation")
    p_fe.add_argument("--config", required=True)
    p_fe.add_argument("--set", action="append", metavar="KEY=VALUE")
    p_fe.add_argument("--n-max", type=int, default=0)
    p_fe.add_argument("--split", default="")
    p_fe.add_argument("--output-dir", default="")
    p_fe.add_argument("--no-resume", action="store_true")
    p_fe.add_argument(
        "--dry-run",
        action="store_true",
        help="report the sample count without spending compute (brief section 10)",
    )
    p_fe.set_defaults(func=cmd_frontier_estimate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
