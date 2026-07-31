"""Sandbox backends and selection policy."""

from __future__ import annotations

import warnings

from cbs.sandbox.base import (
    ExecRequest,
    ExecResult,
    SandboxBackend,
    SandboxUnavailable,
)
from cbs.sandbox.docker_backend import DEFAULT_IMAGE, DockerSandbox
from cbs.sandbox.subprocess_backend import SubprocessSandbox, default_python

__all__ = [
    "DEFAULT_IMAGE",
    "DockerSandbox",
    "ExecRequest",
    "ExecResult",
    "SandboxBackend",
    "SandboxUnavailable",
    "SubprocessSandbox",
    "default_python",
    "select_backend",
    "sandbox_report",
]


def select_backend(
    backend: str = "auto",
    require_security_boundary: bool = False,
    allow_insecure_fallback: bool = False,
    image: str = DEFAULT_IMAGE,
) -> SandboxBackend:
    """Pick a sandbox backend under an explicit safety policy.

    Parameters
    ----------
    backend:
        ``"auto"`` (Docker if present, else subprocess), ``"docker"``, or
        ``"subprocess"``.
    require_security_boundary:
        Set by callers running genuinely untrusted code -- notably `S_evo`'s
        self-modifying scaffolds. When True, a non-isolating backend is only
        returned if `allow_insecure_fallback` is also True, and even then it
        warns loudly.
    allow_insecure_fallback:
        Escape hatch for hosts where Docker cannot exist (Colab). Must be set
        deliberately in config; it is never the default.
    """
    if backend not in ("auto", "docker", "subprocess"):
        raise ValueError(f"unknown sandbox backend {backend!r}")

    if backend == "subprocess":
        chosen: SandboxBackend = SubprocessSandbox()
    elif backend == "docker":
        if not DockerSandbox.available():
            raise SandboxUnavailable(
                "sandbox.backend='docker' was requested but Docker is not "
                "available on this host"
            )
        chosen = DockerSandbox(image=image)
    else:  # auto
        chosen = DockerSandbox(image=image) if DockerSandbox.available() else SubprocessSandbox()

    if require_security_boundary and not chosen.is_security_boundary:
        if not allow_insecure_fallback:
            raise SandboxUnavailable(
                "this operation runs untrusted code and requires a real sandbox, "
                "but Docker is unavailable. Install Docker, or set "
                "sandbox.allow_insecure_fallback=true to proceed on the "
                "subprocess backend (NOT a security boundary -- see "
                "docs/DECISIONS.md D-02)."
            )
        warnings.warn(
            "Running untrusted code on the subprocess backend: this is NOT a "
            "security boundary. Generated code can read the host filesystem and "
            "open network connections. Results will be tagged "
            "is_security_boundary=false.",
            RuntimeWarning,
            stacklevel=2,
        )
    return chosen


def sandbox_report() -> dict:
    """Capability summary for `cbs env`."""
    return {
        "docker": DockerSandbox.describe(DockerSandbox()),
        "subprocess": SubprocessSandbox().describe(),
        "selected_auto": select_backend("auto").name,
    }
