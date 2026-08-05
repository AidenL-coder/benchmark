"""Bridge for external DGM-style forks whose agent code runs as a separate OS
process (inside Docker), not an in-process `AgentFunction` call.

Why this is a different mechanism from `cbs.scaffolds.evolved`
-----------------------------------------------------------------
`EvolvedScaffold`/`InterceptionSession` (D-24/D-25) assume the untrusted agent
is a Python callable invoked in-process, so every model/verifier call can be
intercepted by handing it wrapped objects. A forked loop like HGM does not
work that way: its agent code is copied into a Docker image and executed via
`container.exec_run(...)` as a genuinely separate process, with its own
Python interpreter and its own imported copy of the fork's `llm.py`. There is
no in-process object to wrap.

Confirmed empirically, not assumed (docs/DECISIONS.md D-37): the containerized
agent reaches the model over a real HTTP connection (`network_mode="host"`
was required precisely because the container and the host are different
network namespaces by default). That HTTP boundary is the only channel
available for interception, so `ModelCallProxy` below is a transparent
reverse proxy sitting between the container and the real model server,
recording every request/response pair it forwards. Nothing about the
candidate agent code itself is trusted or modified -- exactly the same
posture as `InterceptionSession`, just enforced at the network layer instead
of the Python-object layer because that is where this boundary actually is.

Verification (does the final patch solve the task) is different: it is
computed by ordinary host-side Python function calls
(`hgm_utils.eval_agent` -> `polyglot.harness.process_entry` /
`swe_bench.harness.process_entry`), not inside the container, so it does not
need this proxying trick -- see D-31/D-37 for why this is a legitimate
distinction, not an inconsistency.

Why this doesn't reuse `InterceptionSession._was_conditioned_on`
--------------------------------------------------------------------
That heuristic resolves a specific ambiguity: a verifier call might be pure
*selection* among several independently-generated candidates
(`test_guided_selection`, preserving) or *conditioning* the next generation on
a failure's error text (`execution_feedback`, expanding) -- both look
identical as "a sandboxed run happened", which is why `evolved.py` needs
behavioural evidence to tell them apart.

HGM's `coding_agent.py` does not have that ambiguity. It runs one
continuously-growing tool-use conversation per task (confirmed by reading
`llm_withtools.py`: each turn appends to a single accumulating message
history, not N independent generate-then-select branches). Every tool
invocation this agent makes is an external program whose output is folded
back into that same growing history and conditions everything the model
generates afterward. That is unambiguously `tool_call` (already registered
in `cbs.scaffolds.tagging`: "Injects the output of an external program into
the context") -- there is no separate "blind selection among candidates"
mode to confuse it with here, so no new ambiguity-resolution heuristic is
needed. Confirmed against the actual agent variant used in every validated
run (D-38): its tool surface (the top-level `hgm/tools/`, not the separate,
unused `best_agent/tools/` bundle an earlier draft of this module's history
mistakenly cited) is exactly `bash.py` and `edit.py` -- matching D-12's
original "entire tool surface" finding -- and both map onto the existing
`tool_call` entry without needing a new one.

The one-time, end-of-episode pass/fail check (does the submitted patch solve
the task) is the direct analogue of `EvolvedScaffold.solve()`'s single hidden-
oracle query, and for the same reason is *not* itself recorded as a scaffold
operation in the trace -- it is the thing being measured, not a move the
scaffold made.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cbs.scaffolds.tagging import OperationTrace

__all__ = [
    "ProxiedCall",
    "ModelCallProxy",
    "reconstruct_trace_from_events",
]

#: How long `reset()`/`stop()` will wait for in-flight handler threads to
#: finish recording before giving up (raising, for `reset()`; proceeding
#: anyway, for `stop()`). Generous relative to any real model-call latency
#: this project's tasks see, deliberately not "instant" -- a short wait here
#: is the entire fix for a real cross-task contamination bug (see below).
_DRAIN_TIMEOUT_S = 30.0
_DRAIN_POLL_INTERVAL_S = 0.02


@dataclass(frozen=True)
class ProxiedCall:
    """One `/v1/chat/completions`-shaped request/response pair the proxy
    observed, in the raw JSON shape the OpenAI-compatible API uses.

    `status` is stored (not just the bodies) specifically so a caller can
    tell a real generation from a failed/errored call attempt -- see
    `reconstruct_trace_from_events`, which uses it to avoid counting a
    backend error as a legitimate `single_call`.
    """

    request_body: dict
    response_body: dict
    status: int
    timestamp: float = field(default_factory=time.time)


class ModelCallProxy:
    """Transparent reverse HTTP proxy that records every chat-completion
    request/response pair it forwards.

    Runs on its own background thread. `reset()` between tasks so each task's
    `events` reflects only that task's conversation -- this is why measured
    runs must serialize task evaluation (one task's agent process talking to
    the proxy at a time); see D-37 on why that is an acceptable scoping
    choice, not a limitation snuck in silently.

    **Real bugs found by a second, independent review pass (D-38 update),
    fixed here rather than left as documented fragilities:**

    1. `reset()`/`stop()` used to not wait for in-flight handler threads.
       A response to task N's last call could still be mid-flight (recorded
       between `send_response` and this handler's own thread finishing) when
       the caller called `reset()` to start task N+1 -- a straggling record
       could land in the wrong task's event log, silently contaminating it.
       Fixed by tracking an in-flight counter (`_inflight`, guarded by the
       same lock as `_events`) that `reset()` now waits to hit zero before
       clearing, and raises (loudly, not silently) if that takes longer than
       `_DRAIN_TIMEOUT_S` -- a stuck handler should be a visible failure, not
       a quietly-corrupted trace.
    2. `urlopen`'s exception handling only covered `HTTPError`/`URLError`.
       A backend that is merely slow past the socket-level timeout, or that
       resets the connection mid-response, raises other exception types
       (e.g. plain `OSError`/`TimeoutError`) that propagated uncaught --
       silently dropping that call from `.events` entirely, with no trace it
       was ever attempted. Fixed by widening the catch to any `Exception`
       during the backend call, always producing a synthesized error
       response (never a silent gap).
    3. Failed/errored calls (whether a real backend error or the two cases
       above) used to be recorded and reconstructed exactly like a genuine
       generation, inflating `single_call` counts with calls that never
       actually sampled from `M`. Fixed: `ProxiedCall` now carries `status`,
       and `reconstruct_trace_from_events` only counts a `single_call` (and
       only advances its message-count bookkeeping) for a 2xx response.

    One fragility worth being explicit about rather than silent on:
    `reconstruct_trace_from_events` assumes each event's `messages` list is a
    strict, append-only extension of the previous event's -- true for this
    codebase today (confirmed: no truncation/history-trimming logic exists in
    `llm_withtools.py`), but would silently undercount `tool_call` events if a
    future agent ever trims context to fit a window.

    (An earlier version of this proxy recorded a call only *after* replying
    to the client, on the assumption that the client's own subsequent work
    would always be slow enough not to race a caller reading `.events`
    immediately after. That assumption was wrong -- it produced a real,
    intermittent test failure the first time it was checked for, not just a
    hypothetical one. Recording happens before responding now, which removes
    that race rather than relying on timing to avoid it.)
    """

    def __init__(self, listen_port: int, backend_base_url: str):
        self.listen_port = listen_port
        self.backend_base_url = backend_base_url.rstrip("/")
        self._events: list[ProxiedCall] = []
        self._lock = threading.Lock()
        self._inflight = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _drain(self, timeout: float) -> bool:
        """Block until no handler thread is mid-request, or `timeout`
        elapses. Returns whether it actually drained (`False` means the
        caller is proceeding despite still-active handlers)."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._inflight == 0:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_DRAIN_POLL_INTERVAL_S)

    def reset(self, timeout: float = _DRAIN_TIMEOUT_S) -> None:
        """Clear captured events for the next task.

        Waits for any in-flight request to finish recording first --
        clearing while a handler thread is between "backend responded" and
        "recorded" would let that straggling record land in the *next*
        task's event log instead of the one it actually belongs to. Raises
        rather than proceeding if a handler is stuck past `timeout`: a
        silently-corrupted trace is worse than a loud failure here.
        """
        if not self._drain(timeout):
            raise RuntimeError(
                f"ModelCallProxy.reset() timed out after {timeout}s waiting "
                "for an in-flight request to finish; refusing to clear "
                "events while a handler may still be about to record into "
                "this task's log"
            )
        with self._lock:
            self._events = []

    @property
    def events(self) -> list[ProxiedCall]:
        with self._lock:
            return list(self._events)

    def _record(self, request_body: dict, response_body: dict, status: int) -> None:
        with self._lock:
            self._events.append(ProxiedCall(request_body, response_body, status))

    def start(self) -> None:
        if self._server is not None:  # fast-path check; not itself atomic
            raise RuntimeError("ModelCallProxy already started")
        proxy = self

        class _Handler(BaseHTTPRequestHandler):
            #: Socket-level read/write timeout. Without this, a client that
            #: connects but never finishes sending a request (or never reads
            #: its response) blocks this handler thread -- and, before the
            #: in-flight tracking fix above, that thread's slot -- forever.
            timeout = 300

            def log_message(self, format, *args):  # noqa: A002 -- stdlib signature
                pass  # keep test/CI output quiet; failures still raise

            def _forward(self, method: str) -> None:
                with proxy._lock:
                    proxy._inflight += 1
                try:
                    length = int(self.headers.get("Content-Length", 0) or 0)
                    body = self.rfile.read(length) if length else b""
                    url = proxy.backend_base_url + self.path
                    req = urllib.request.Request(url, data=body or None, method=method)
                    if body:
                        req.add_header("Content-Type", "application/json")
                    try:
                        with urllib.request.urlopen(req, timeout=600) as resp:
                            resp_body = resp.read()
                            status = resp.getcode()
                    except urllib.error.HTTPError as exc:
                        resp_body = exc.read()
                        status = exc.code
                    except Exception as exc:  # noqa: BLE001 -- any backend
                        # failure (URLError, timeout, connection reset, ...)
                        # must still produce a response and a recorded
                        # event, never a silent gap in `.events`.
                        resp_body = json.dumps({"error": str(exc)}).encode()
                        status = 502

                    # Record BEFORE responding to the client, not after: the
                    # client proceeding is exactly the signal that the
                    # record must already exist by (a caller that reads
                    # `.events` immediately after its request returns must
                    # see this call in it -- confirmed as a real, not just
                    # theoretical, race: an earlier record-after-respond
                    # ordering here produced an intermittent test failure
                    # the very first time it was checked for).
                    if method == "POST" and self.path.startswith("/v1/chat/completions"):
                        try:
                            proxy._record(json.loads(body), json.loads(resp_body), status)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass  # a malformed pair is not worth crashing over

                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                finally:
                    with proxy._lock:
                        proxy._inflight -= 1

            def do_POST(self) -> None:  # noqa: N802 -- stdlib handler name
                self._forward("POST")

            def do_GET(self) -> None:  # noqa: N802 -- stdlib handler name
                self._forward("GET")

        # Built outside the lock (pure object construction, no shared state
        # touched yet); the check-and-set that actually guards against a
        # concurrent second start() happens atomically in one lock
        # acquisition right here, closing the redundant socket if it lost.
        server = ThreadingHTTPServer(("0.0.0.0", self.listen_port), _Handler)
        with self._lock:
            if self._server is not None:
                server.server_close()
                raise RuntimeError("ModelCallProxy already started")
            self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self, drain_timeout: float = _DRAIN_TIMEOUT_S) -> None:
        self._drain(drain_timeout)  # best-effort; proceed regardless
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


def reconstruct_trace_from_events(events: list[ProxiedCall]) -> OperationTrace:
    """Turn one task's captured conversation into an `OperationTrace`.

    `events` must be in chronological call order (the proxy records them in
    the order it forwards them, which is call order for a single-threaded
    agent conversation). For each event: any `"tool"`-role message newly
    present in its request (i.e. appended since the previous event) means a
    tool actually ran and its output is now conditioning this generation --
    recorded as `tool_call` *before* the generation it conditions, matching
    real chronology.

    A `single_call` is only recorded for a genuinely successful (2xx)
    response with a real `choices[0].message` -- a failed/errored call
    attempt (backend error, timeout, malformed response) never sampled from
    `M`, so it must not inflate the operation count, and the message-count
    bookkeeping (`prev_len`) is not advanced for it either: a client that
    gets an error does not append an assistant reply to its history and
    will typically resend the same (unchanged) history on retry, so holding
    `prev_len` steady is what keeps that retry's own tool-message scan
    correct. Tool-role messages already present in a failed call's request
    are still scanned and counted, since those genuinely happened regardless
    of whether the follow-up generation succeeded.
    """
    trace = OperationTrace()
    prev_len = 0
    for event in events:
        messages = event.request_body.get("messages", [])
        for message in messages[prev_len:]:
            if message.get("role") == "tool":
                trace.record_instant("tool_call", intercepted=True)
        prev_len = max(prev_len, len(messages))

        if not (200 <= event.status < 300):
            continue  # a failed call attempt is not a sample from M

        choices = event.response_body.get("choices", [])
        if not choices:
            continue  # 2xx with no choices is not a real generation either
        assistant_message = choices[0].get("message", {})
        trace.record_instant(
            "single_call",
            intercepted=True,
            had_tool_calls=bool(assistant_message.get("tool_calls")),
        )
        # The assistant's own reply becomes part of history for the next
        # turn's request -- but only on success; see docstring.
        prev_len += 1

    return trace
