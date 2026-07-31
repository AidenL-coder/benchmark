"""Docker sandbox -- the real security boundary required by brief section 10.

Runs untrusted code in a throwaway container with no network, a read-only root
filesystem, dropped capabilities, and hard memory/CPU/PID limits. The working
directory is mounted from a host temp dir that is deleted afterwards.

This is the backend to use for anything involving `S_evo`'s self-modifying code.
On hosts without Docker (Windows dev box, Colab) `cbs.sandbox.select_backend`
falls back to the subprocess backend, and the resulting records say so.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from cbs.sandbox.base import ExecRequest, ExecResult, SandboxBackend, SandboxUnavailable

__all__ = ["DockerSandbox", "DEFAULT_IMAGE"]

DEFAULT_IMAGE = "python:3.11-slim"


class DockerSandbox(SandboxBackend):
    name = "docker"
    is_security_boundary = True

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        cpus: float = 1.0,
        pids_limit: int = 128,
        docker_bin: str = "docker",
    ):
        self.image = image
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.docker_bin = docker_bin

    @classmethod
    def available(cls, docker_bin: str = "docker") -> bool:
        if shutil.which(docker_bin) is None:
            return False
        try:
            proc = subprocess.run(
                [docker_bin, "info", "--format", "{{json .ServerVersion}}"],
                capture_output=True,
                timeout=20,
                text=True,
            )
            return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def ensure_image(self) -> None:
        """Pull the sandbox image if absent. Network use, so it is explicit."""
        proc = subprocess.run(
            [self.docker_bin, "image", "inspect", self.image],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode == 0:
            return
        pull = subprocess.run(
            [self.docker_bin, "pull", self.image], capture_output=True, timeout=900
        )
        if pull.returncode != 0:
            raise SandboxUnavailable(
                f"could not pull sandbox image {self.image}: "
                f"{pull.stderr.decode('utf-8', 'replace')[:400]}"
            )

    def run(self, request: ExecRequest) -> ExecResult:
        if not self.available(self.docker_bin):
            raise SandboxUnavailable("docker is not available on this host")

        workdir = Path(tempfile.mkdtemp(prefix="cbs-dsbx-"))
        started = time.monotonic()
        try:
            for rel, content in request.files.items():
                target = (workdir / rel).resolve()
                if not str(target).startswith(str(workdir.resolve())):
                    raise ValueError(f"file path escapes sandbox: {rel!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            cmd = [
                self.docker_bin,
                "run",
                "--rm",
                "--network", "none" if not request.network else "bridge",
                "--memory", f"{request.memory_mb}m",
                # Equal swap to memory means no swap headroom past the limit.
                "--memory-swap", f"{request.memory_mb}m",
                "--cpus", str(self.cpus),
                "--pids-limit", str(self.pids_limit),
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--read-only",
                # Writable scratch that vanishes with the container.
                "--tmpfs", "/tmp:rw,size=64m,noexec",
                "-v", f"{workdir}:/work:rw",
                "-w", "/work",
                "-u", "65534:65534",  # nobody
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "PYTHONHASHSEED=0",
                "-e", "PYTHONIOENCODING=utf-8",
            ]
            for key, value in request.env.items():
                cmd += ["-e", f"{key}={value}"]
            cmd += [self.image, *request.entrypoint]

            timed_out = False
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    # Grace period on top of the in-container limit, so Docker
                    # startup latency is not charged against the task's timeout.
                    timeout=request.timeout_s + 30,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = -1
                stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
                stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""

            cap = request.max_output_bytes
            return ExecResult(
                exit_code=exit_code,
                stdout=stdout[:cap],
                stderr=stderr[:cap],
                timed_out=timed_out,
                duration_s=time.monotonic() - started,
                backend=self.name,
                is_security_boundary=True,
                network_enforced=True,
                memory_enforced=True,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "available": self.available(self.docker_bin),
            "is_security_boundary": True,
            "image": self.image,
        }
