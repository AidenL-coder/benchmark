"""Best-of-N frontier sampling with crash-safe resume.

Implements the protocol in brief section 3.2 / proposal section 5.4: for each
(model, task), draw `N_max` samples under the minimal scaffold `S0` on a
temperature schedule tuned for solution diversity, verify each, and record both
`p_hat(x)` and the *set of distinct* correct solutions.

Resume
------
This is the project's dominant cost (proposal section 9) and, on a preemptible
runtime like Colab, will be interrupted. Samples are therefore appended to a
JSONL shard and `fsync`-ed periodically; on restart the sampler counts what is
already on disk and continues from there. Resume is keyed on
(model_id, task_id, scaffold fingerprint, schedule), so changing any condition
that would move the frontier starts a fresh shard rather than silently mixing
samples drawn under different conditions into one estimate.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from cbs.budget import BudgetAccountant, BudgetExceeded, Usage
from cbs.frontier.records import DEFAULT_BUDGET_GRID, FrontierRecord, build_record
from cbs.models.base import ModelClient
from cbs.scaffolds.s0 import S0
from cbs.tasks.canonicalize import canonicalize_solution
from cbs.tasks.schema import Task, TaskSuite
from cbs.tasks.verifier import Verifier

__all__ = ["TemperatureSchedule", "DEFAULT_SCHEDULE", "FrontierSampler", "ShardPaths"]


@dataclass(frozen=True)
class TemperatureSchedule:
    """How the `N_max` budget is split across sampling temperatures.

    The frontier protocol asks for a schedule "chosen to maximize solution
    diversity" rather than a single temperature, because the quantity of interest
    is *reachability within budget*, not the solve rate at any one setting.

    That makes `p_hat(x)` the solve probability under **the mixture**, not under
    a fixed temperature. This is the right target for a frontier estimate, and it
    is why the schedule is recorded on every record and folded into the resume
    key: two estimates drawn under different schedules are not comparable and
    must not be pooled.
    """

    #: (temperature, fraction of budget). Fractions must sum to 1.
    stages: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("schedule needs at least one stage")
        total = sum(f for _, f in self.stages)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"schedule fractions must sum to 1.0, got {total}")
        if any(f <= 0 for _, f in self.stages):
            raise ValueError("schedule fractions must be positive")
        if any(t < 0 for t, _ in self.stages):
            raise ValueError("temperatures must be non-negative")

    def allocate(self, n: int) -> list[float]:
        """Per-sample temperatures, length exactly `n`.

        **Prefix-stable**: ``allocate(n)[:m] == allocate(m)`` for every ``m <= n``.
        Sample `i` therefore always draws the same temperature regardless of the
        budget it was requested under. Two properties depend on this, and both
        matter specifically because runs get interrupted:

        *   *Resume soundness.* Topping a shard up from `N_max=1000` to `5000`
            must not retroactively change the temperature of samples 0-999. A
            block allocation (all low-temperature draws first, then the next
            stage) fails this: widening the budget rewrites the earlier
            assignment, and the shard silently pools samples drawn under two
            different schedules.
        *   *Unbiased truncation.* A run killed by preemption or a budget cap
            keeps a prefix of the sequence. Under a block allocation that prefix
            would be entirely low-temperature, so the estimate would come from a
            schedule nobody chose. Interleaving keeps any prefix approximately
            proportional to the requested fractions.

        Uses D'Hondt apportionment: repeatedly award the next sample to the
        stage maximising ``fraction / (already_awarded + 1)``. Building the
        sequence incrementally is what makes it prefix-stable by construction.
        """
        if n <= 0:
            return []
        counts = [0] * len(self.stages)
        out: list[float] = []
        for _ in range(n):
            best, best_value = 0, -1.0
            for index, (_, fraction) in enumerate(self.stages):
                value = fraction / (counts[index] + 1)
                if value > best_value:
                    best, best_value = index, value
            counts[best] += 1
            out.append(self.stages[best][0])
        return out

    def temperature_at(self, index: int) -> float:
        """Temperature for a single sample index. O(index); prefer `allocate`."""
        return self.allocate(index + 1)[index]

    def as_list(self) -> list[dict]:
        return [{"temperature": t, "fraction": f} for t, f in self.stages]

    def fingerprint(self) -> str:
        return ";".join(f"{t:g}@{f:g}" for t, f in self.stages)


#: Default schedule. Weighted toward moderate-to-high temperature because the
#: goal is coverage of the solution space, with a low-temperature stage retained
#: so the modal solution is reliably observed.
DEFAULT_SCHEDULE = TemperatureSchedule(
    stages=((0.2, 0.10), (0.6, 0.25), (0.8, 0.35), (1.0, 0.20), (1.2, 0.10))
)


@dataclass
class ShardPaths:
    samples: Path
    solutions: Path
    meta: Path


@dataclass
class _ResumeState:
    n_done: int = 0
    species_counts: dict[str, int] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    n_verifier_errors: int = 0
    canon_methods: dict[str, int] = field(default_factory=dict)


class FrontierSampler:
    """Draws and verifies `N_max` samples per task, resumably."""

    def __init__(
        self,
        model: ModelClient,
        verifier: Verifier,
        scaffold: S0 | None = None,
        schedule: TemperatureSchedule = DEFAULT_SCHEDULE,
        output_dir: Path | str = "runs/frontier",
        flush_every: int = 25,
        confidence: float = 0.95,
        budget_grid: tuple[int, ...] = DEFAULT_BUDGET_GRID,
        store_raw_completions: bool = False,
    ):
        self.model = model
        self.verifier = verifier
        self.scaffold = scaffold or S0()
        self.schedule = schedule
        self.output_dir = Path(output_dir)
        self.flush_every = max(1, flush_every)
        self.confidence = confidence
        self.budget_grid = budget_grid
        self.store_raw_completions = store_raw_completions

    # -- shard identity ---------------------------------------------------
    def condition_key(self) -> str:
        """Identifies the sampling conditions. Any change starts a new shard."""
        fp = self.scaffold.config_fingerprint()
        return "|".join(
            [
                f"model={self.model.model_id}",
                f"scaffold={fp.get('name')}",
                f"temp_default={fp.get('temperature')}",
                f"max_tokens={fp.get('max_tokens')}",
                f"sys={hash_str(fp.get('system_prompt') or '')}",
                f"sched={self.schedule.fingerprint()}",
            ]
        )

    def shard_paths(self, task: Task) -> ShardPaths:
        safe_model = _slug(self.model.model_id)
        safe_task = _slug(task.task_id)
        base = self.output_dir / safe_model / safe_task
        return ShardPaths(
            samples=base / "samples.jsonl",
            solutions=base / "solutions.jsonl",
            meta=base / "shard.json",
        )

    # -- resume -----------------------------------------------------------
    def _load_resume(self, task: Task, paths: ShardPaths) -> _ResumeState:
        state = _ResumeState()
        if not paths.meta.exists() or not paths.samples.exists():
            return state

        meta = json.loads(paths.meta.read_text(encoding="utf-8"))
        if meta.get("condition_key") != self.condition_key():
            raise RuntimeError(
                f"{paths.meta} was written under different sampling conditions:\n"
                f"  on disk: {meta.get('condition_key')}\n"
                f"  current: {self.condition_key()}\n"
                "Samples drawn under different conditions must not be pooled into "
                "one frontier estimate. Delete the shard to re-sample, or restore "
                "the original configuration."
            )

        with paths.samples.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A partial final line means the process died mid-write.
                    # Everything before it is intact, so stop here and let the
                    # sampler top the shard back up to N_max.
                    break
                state.n_done += 1
                state.usage = state.usage + Usage(
                    calls=rec.get("calls", 0),
                    prompt_tokens=rec.get("prompt_tokens", 0),
                    completion_tokens=rec.get("completion_tokens", 0),
                    usd=rec.get("usd", 0.0),
                )
                if rec.get("passed"):
                    key = rec["species_key"]
                    state.species_counts[key] = state.species_counts.get(key, 0) + 1
                    method = rec.get("canon_method", "unknown")
                    state.canon_methods[method] = state.canon_methods.get(method, 0) + 1
                elif rec.get("reason") in ("verifier_error", "timeout"):
                    state.n_verifier_errors += 1
        return state

    # -- sampling ---------------------------------------------------------
    def estimate_task(
        self,
        task: Task,
        n_max: int,
        accountant: BudgetAccountant,
        resume: bool = True,
        progress_every: int = 0,
    ) -> FrontierRecord:
        """Sample `task` up to `n_max` times and build its frontier record."""
        paths = self.shard_paths(task)
        paths.samples.parent.mkdir(parents=True, exist_ok=True)

        state = self._load_resume(task, paths) if resume else _ResumeState()
        if not paths.meta.exists():
            paths.meta.write_text(
                json.dumps(
                    {
                        "condition_key": self.condition_key(),
                        "task_id": task.task_id,
                        "task_content_hash": task.content_hash(),
                        "model": self.model.describe(),
                        "scaffold": self.scaffold.config_fingerprint(),
                        "schedule": self.schedule.as_list(),
                        "n_max_requested": n_max,
                        "verifier_backend": self.verifier.sandbox.name,
                        "verifier_is_security_boundary": (
                            self.verifier.sandbox.is_security_boundary
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        temperatures = self.schedule.allocate(n_max)
        known_species = set(state.species_counts)
        n_budget_exhausted = 0
        since_flush = 0

        samples_fh = paths.samples.open("a", encoding="utf-8")
        solutions_fh = paths.solutions.open("a", encoding="utf-8")
        try:
            for index in range(state.n_done, n_max):
                temperature = temperatures[index]

                # Stop cleanly rather than raising: a truncated run is still a
                # valid estimate at the budget actually achieved, and the record
                # reports that budget. Raising would discard completed samples.
                if not accountant.can_afford(Usage(calls=1)):
                    n_budget_exhausted = n_max - index
                    break

                try:
                    result = self.scaffold.solve(
                        task,
                        self.model,
                        accountant,
                        verifier=self.verifier,
                        seed=index,
                        temperature=temperature,
                    )
                except BudgetExceeded:
                    n_budget_exhausted = n_max - index
                    break

                if result.budget_exhausted:
                    n_budget_exhausted = n_max - index
                    break

                state.usage = state.usage + result.usage
                passed = result.passed
                reason = (
                    result.verification.reason if result.verification else "unverified"
                )

                row = {
                    "i": index,
                    "t": temperature,
                    "passed": passed,
                    "reason": reason,
                    "calls": result.usage.calls,
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "usd": result.usage.usd,
                }

                if passed:
                    canon = canonicalize_solution(result.solution)
                    row["species_key"] = canon.key
                    row["canon_method"] = canon.method
                    state.species_counts[canon.key] = (
                        state.species_counts.get(canon.key, 0) + 1
                    )
                    state.canon_methods[canon.method] = (
                        state.canon_methods.get(canon.method, 0) + 1
                    )
                    if canon.key not in known_species:
                        known_species.add(canon.key)
                        solutions_fh.write(
                            json.dumps(
                                {
                                    "species_key": canon.key,
                                    "first_seen_index": index,
                                    "temperature": temperature,
                                    "canonical": canon.text,
                                    "method": canon.method,
                                    "example_solution": result.solution,
                                }
                            )
                            + "\n"
                        )
                elif reason in ("verifier_error", "timeout"):
                    state.n_verifier_errors += 1

                if self.store_raw_completions:
                    row["raw"] = result.raw_completion

                samples_fh.write(json.dumps(row) + "\n")
                since_flush += 1
                if since_flush >= self.flush_every:
                    _durable_flush(samples_fh)
                    _durable_flush(solutions_fh)
                    since_flush = 0

                if progress_every and (index + 1) % progress_every == 0:
                    done = index + 1
                    print(
                        f"  {task.task_id}: {done}/{n_max} "
                        f"({sum(state.species_counts.values())} correct, "
                        f"{len(state.species_counts)} distinct)",
                        flush=True,
                    )
        finally:
            _durable_flush(samples_fh)
            _durable_flush(solutions_fh)
            samples_fh.close()
            solutions_fh.close()

        n_done = _count_lines(paths.samples)
        return build_record(
            task_id=task.task_id,
            model_id=self.model.model_id,
            split=task.split.value if task.split else None,
            family=task.family,
            n_samples=n_done,
            species_counts=state.species_counts,
            temperature_schedule=self.schedule.as_list(),
            scaffold_fingerprint=self.scaffold.config_fingerprint(),
            usage=state.usage,
            verifier_backend=self.verifier.sandbox.name,
            verifier_is_security_boundary=self.verifier.sandbox.is_security_boundary,
            n_verifier_errors=state.n_verifier_errors,
            n_budget_exhausted=n_budget_exhausted,
            canonicalization_methods=state.canon_methods,
            confidence=self.confidence,
            budget_grid=self.budget_grid,
            metadata={
                "n_max_requested": n_max,
                "truncated": n_done < n_max,
                "condition_key": self.condition_key(),
                "task_content_hash": task.content_hash(),
            },
        )

    def estimate_suite(
        self,
        suite: TaskSuite,
        n_max: int,
        accountant: BudgetAccountant,
        resume: bool = True,
        progress: bool = True,
    ) -> list[FrontierRecord]:
        records: list[FrontierRecord] = []
        for i, task in enumerate(suite.tasks, start=1):
            if progress:
                print(
                    f"[{i}/{len(suite.tasks)}] {task.task_id} (N_max={n_max})",
                    flush=True,
                )
            started = time.monotonic()
            record = self.estimate_task(task, n_max, accountant, resume=resume)
            records.append(record)
            if progress:
                print(
                    f"    {record.summary_line()}  [{time.monotonic()-started:.1f}s]",
                    flush=True,
                )
        return records


# ---------------------------------------------------------------------------
def _durable_flush(fh) -> None:
    """Flush to disk hard enough to survive a preempted runtime."""
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except (OSError, ValueError):  # pragma: no cover - e.g. network drives
        pass


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def hash_str(text: str) -> str:
    import hashlib

    return hashlib.blake2b(text.encode("utf-8"), digest_size=6).hexdigest()
