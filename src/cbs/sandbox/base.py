"""Sandboxed execution of untrusted, model-generated code.

The brief (section 10) makes Docker mandatory. Two deployment targets make a
Docker-only implementation unworkable in practice, so this is an interface with
swappable backends (docs/DECISIONS.md D-02):

*   the development machine (Windows, no Docker installed),
*   Google Colab (the runtime is itself a container; nested Docker is unavailable).

Every `ExecResult` records which backend ran it and whether that backend actually
enforced isolation, so a run can never quietly claim a safety property it did not
have. `SandboxBackend.is_security_boundary` is the field that matters: only the
Docker backend sets it True.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

__all__ = [
    "ExecRequest",
    "ExecResult",
    "SandboxBackend",
    "SandboxUnavailable",
]


class SandboxUnavailable(RuntimeError):
    """The requested sandbox backend cannot run on this host."""


@dataclass(frozen=True)
class ExecRequest:
    """A unit of untrusted execution.

    `files` is written into a fresh working directory; `entrypoint` is executed
    with that directory as cwd. Nothing outside it is exposed.
    """

    files: dict[str, str]
    entrypoint: list[str]
    timeout_s: float = 30.0
    memory_mb: int = 1024
    #: Untrusted code gets no network by default. Only the Docker backend can
    #: actually enforce this; others report the failure to enforce.
    network: bool = False
    env: dict[str, str] = field(default_factory=dict)
    #: Cap on captured output, to stop a runaway print loop filling the disk.
    max_output_bytes: int = 1_000_000


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    backend: str
    #: True only if the backend provides a real isolation boundary.
    is_security_boundary: bool
    #: True if the backend enforced the requested network policy.
    network_enforced: bool
    #: True if the backend enforced the requested memory limit.
    memory_enforced: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def as_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 4),
            "backend": self.backend,
            "is_security_boundary": self.is_security_boundary,
            "network_enforced": self.network_enforced,
            "memory_enforced": self.memory_enforced,
            "stdout_bytes": len(self.stdout),
            "stderr_bytes": len(self.stderr),
        }


class SandboxBackend(abc.ABC):
    name: str = "unset"
    #: Whether this backend is a genuine security boundary for untrusted code.
    is_security_boundary: bool = False

    @classmethod
    @abc.abstractmethod
    def available(cls) -> bool:
        """Whether this backend can run on the current host."""

    @abc.abstractmethod
    def run(self, request: ExecRequest) -> ExecResult:
        """Execute `request`, never raising for non-zero exit or timeout."""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "available": self.available(),
            "is_security_boundary": self.is_security_boundary,
        }
