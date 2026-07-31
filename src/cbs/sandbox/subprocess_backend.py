"""Subprocess sandbox -- resource-limited, but NOT a security boundary.

Use on hosts where Docker is unavailable (the Windows dev machine, Colab). It
enforces what a plain subprocess can enforce:

*   a wall-clock timeout, killing the whole process group;
*   a fresh temporary working directory, deleted afterwards;
*   a scrubbed environment;
*   on POSIX, address-space / CPU / file-size / process-count rlimits.

It does **not** confine filesystem or network access. Code executed here can read
the host filesystem and open sockets. That is acceptable for verifying solutions
to toy and benchmark coding tasks whose generations come from a small frozen
model, and unacceptable for running an evolved `S_evo` scaffold's self-modifying
code -- which is why `cbs.sandbox.select_backend` refuses to hand this backend to
callers that require a security boundary unless the config opts in explicitly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from cbs.sandbox.base import ExecRequest, ExecResult, SandboxBackend

__all__ = ["SubprocessSandbox"]

_IS_POSIX = os.name == "posix"


def _posix_limits(memory_mb: int, timeout_s: float):  # pragma: no cover - POSIX only
    """Return a preexec_fn applying rlimits and detaching the process group."""
    import resource

    def _apply() -> None:
        os.setsid()  # own process group, so a timeout kills children too
        mem_bytes = memory_mb * 1024 * 1024
        for res in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            try:
                resource.setrlimit(res, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
        cpu_s = int(timeout_s) + 1
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass
        try:  # cap written files at 64 MiB
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        except (ValueError, OSError):
            pass
        try:  # no fork bombs
            resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
        except (ValueError, OSError):
            pass

    return _apply


class SubprocessSandbox(SandboxBackend):
    name = "subprocess"
    is_security_boundary = False

    @classmethod
    def available(cls) -> bool:
        return True

    def run(self, request: ExecRequest) -> ExecResult:
        workdir = Path(tempfile.mkdtemp(prefix="cbs-sbx-"))
        started = time.monotonic()
        try:
            for rel, content in request.files.items():
                target = workdir / rel
                # Refuse path traversal out of the working directory.
                resolved = target.resolve()
                if not str(resolved).startswith(str(workdir.resolve())):
                    raise ValueError(f"file path escapes sandbox: {rel!r}")
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content, encoding="utf-8")

            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                # Deterministic hashing so canonicalisation is reproducible.
                "PYTHONHASHSEED": "0",
                **request.env,
            }
            if os.name == "nt":
                # Windows needs these for the interpreter to start at all.
                for key in ("SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
                    if key in os.environ:
                        env[key] = os.environ[key]

            popen_kwargs: dict = {}
            memory_enforced = False
            if _IS_POSIX:
                popen_kwargs["preexec_fn"] = _posix_limits(
                    request.memory_mb, request.timeout_s
                )
                memory_enforced = True
            else:
                popen_kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )

            proc = subprocess.Popen(
                request.entrypoint,
                cwd=str(workdir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )

            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=request.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    stdout, stderr = "", ""

            cap = request.max_output_bytes
            return ExecResult(
                exit_code=proc.returncode if proc.returncode is not None else -1,
                stdout=(stdout or "")[:cap],
                stderr=(stderr or "")[:cap],
                timed_out=timed_out,
                duration_s=time.monotonic() - started,
                backend=self.name,
                is_security_boundary=False,
                network_enforced=False,
                memory_enforced=memory_enforced,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if _IS_POSIX:  # pragma: no cover - POSIX only
                import signal

                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass


def default_python() -> list[str]:
    """The interpreter used to run generated code.

    `sys.executable` keeps the sandbox on the same Python as the harness, which
    matters because a task's tests may rely on version-specific behaviour.

    No isolation flags are passed, deliberately. `-S` would hide site-packages,
    which real benchmark tests need, and `-I` implies `-E`, which would discard
    the `PYTHONHASHSEED=0` that keeps solution canonicalisation reproducible.
    `run()` already builds the child environment from scratch (no inherited
    `PYTHONPATH`), which is what those flags were guarding against.
    """
    return [sys.executable]
