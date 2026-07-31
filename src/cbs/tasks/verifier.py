"""Executable verification of candidate solutions.

`p_hat(x)` is only meaningful if "correct" is decided identically on every draw,
so verification is deterministic given (task, candidate, sandbox backend) and
never depends on model output beyond the extracted code.

False positives are the dangerous failure mode here: a verifier that passes wrong
code inflates `p_hat(x)`, shrinks the set of beyond-frontier tasks, and so makes
frontier-crossing *harder* to claim -- but a verifier that passes wrong code for
`S_evo` specifically would manufacture crossings. The brief (section 7, Phase 1)
therefore requires auditing a sample of passes by hand; `VerificationResult`
keeps the full execution record to make that audit possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cbs.sandbox import ExecRequest, ExecResult, SandboxBackend, default_python
from cbs.tasks.schema import Task

__all__ = [
    "VerificationResult",
    "Verifier",
    "extract_code",
]

_FENCE_RE = re.compile(
    r"```(?:python|py|python3)?\s*\n(.*?)(?:\n```|\Z)", re.DOTALL | re.IGNORECASE
)


def extract_code(text: str) -> str:
    """Pull Python source out of a model completion.

    Extraction is part of the *minimal* scaffold `S0` ("single call + trivial
    formatting", brief section 3.2). It must stay trivial: anything smarter --
    retrying, repairing, selecting among blocks by test outcome -- is a scaffold
    operation that would have to be support-tagged, and would silently raise the
    baseline the whole study measures against.

    Policy: if fenced blocks exist, concatenate them in order (models often split
    imports and body across blocks). Otherwise return the text unchanged.
    """
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return "\n\n".join(b.strip("\n") for b in blocks).strip()
    return text.strip()


@dataclass
class VerificationResult:
    passed: bool
    reason: str
    exec_result: ExecResult | None = None
    extracted_code: str = ""
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "exec": self.exec_result.as_dict() if self.exec_result else None,
            **({"metadata": self.metadata} if self.metadata else {}),
        }


class Verifier:
    """Runs a candidate against a task's tests inside a sandbox."""

    #: Marker printed by the harness on success. Requiring an explicit marker
    #: rather than trusting exit code 0 defends against a candidate that calls
    #: `sys.exit(0)` or swallows the assertion error.
    SUCCESS_MARKER = "__CBS_VERIFIED_OK__"

    def __init__(self, sandbox: SandboxBackend, python_cmd: list[str] | None = None):
        self.sandbox = sandbox
        self.python_cmd = python_cmd or default_python()

    def build_program(self, task: Task, code: str) -> str:
        """Assemble the file executed in the sandbox."""
        parts = []
        if task.setup:
            parts.append(task.setup)
        parts.append(code)
        parts.append(task.tests)
        # The marker is printed only if every test statement completed. It goes
        # out via an unbuffered os.write to fd 1, so a candidate that reassigns
        # `print` or replaces `sys.stdout` can neither forge nor suppress it.
        parts.append(
            "\n".join(
                [
                    "import os as _cbs_os",
                    f'_cbs_os.write(1, b"{self.SUCCESS_MARKER}\\n")',
                ]
            )
        )
        return "\n\n".join(parts) + "\n"

    def verify_code(self, task: Task, code: str) -> VerificationResult:
        """Verify already-extracted source."""
        if not code.strip():
            return VerificationResult(False, "empty_candidate", extracted_code=code)

        program = self.build_program(task, code)
        request = ExecRequest(
            files={"candidate.py": program},
            entrypoint=[*self.python_cmd, "candidate.py"],
            timeout_s=task.timeout_s,
            memory_mb=task.memory_mb,
            network=False,
        )
        result = self.sandbox.run(request)

        if result.timed_out:
            reason = "timeout"
        elif self.SUCCESS_MARKER in result.stdout:
            # Exit code is checked too: a test could pass yet the interpreter
            # still fail during shutdown (e.g. an atexit hook raising).
            reason = "ok" if result.exit_code == 0 else "marker_but_nonzero_exit"
        elif result.exit_code != 0:
            reason = "test_failed"
        else:
            reason = "no_success_marker"

        return VerificationResult(
            passed=(reason == "ok"),
            reason=reason,
            exec_result=result,
            extracted_code=code,
        )

    def verify_completion(self, task: Task, completion_text: str) -> VerificationResult:
        """Extract code from a raw model completion, then verify it."""
        return self.verify_code(task, extract_code(completion_text))

    def self_test(self, task: Task) -> VerificationResult:
        """Run the task's own reference solution.

        A task whose reference fails is broken; treating its zero successes as
        "beyond the frontier" would be a measurement artefact, not a result.
        `cbs tasks verify` runs this across a suite before any sampling starts.
        """
        if task.reference_solution is None:
            return VerificationResult(False, "no_reference_solution")
        return self.verify_code(task, task.reference_solution)
