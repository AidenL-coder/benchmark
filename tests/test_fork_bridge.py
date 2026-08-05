"""Tests for `cbs.scaffolds.fork_bridge` -- the network-layer interception
mechanism for external forks whose agent code runs as a separate OS process
(inside Docker), not an in-process `AgentFunction` (see the module docstring
for why `cbs.scaffolds.evolved`'s in-process interception cannot reach that
code, and docs/DECISIONS.md D-37 for the empirical finding that motivated
this).

Two independent things are under test here, deliberately without touching a
real fork or a real Docker container:

1. `ModelCallProxy` is a genuine network proxy -- it must be tested as one:
   a real backend server, a real client request through the proxy, checking
   the bytes that actually came back and what actually got recorded.
2. `reconstruct_trace_from_events` is a pure function over synthetic
   OpenAI-shaped conversation histories -- tested against hand-built message
   sequences whose correct classification is known by construction, the same
   style `tests/test_evolved.py` uses for `InterceptionSession`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from cbs.scaffolds.fork_bridge import (
    ModelCallProxy,
    ProxiedCall,
    reconstruct_trace_from_events,
)


class _FakeBackend:
    """A minimal fake OpenAI-compatible server the proxy forwards to.

    Returns a fixed, inspectable response for every POST so tests can assert
    on exactly what came back through the proxy versus what the backend sent.
    """

    def __init__(self, response_body: dict, status: int = 200):
        self.response_body = response_body
        self.status = status
        self.received_paths: list[str] = []
        self.received_bodies: list[dict] = []
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                backend.received_paths.append(self.path)
                backend.received_bodies.append(json.loads(body) if body else {})
                payload = json.dumps(backend.response_body).encode()
                self.send_response(backend.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                payload = b'{"status": "ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post_json(url: str, body: dict, timeout: float = 5.0) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def fake_backend():
    backend = _FakeBackend(
        response_body={
            "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}]
        }
    )
    yield backend
    backend.stop()


@pytest.fixture
def proxy(fake_backend):
    p = ModelCallProxy(listen_port=_free_port(), backend_base_url=f"http://127.0.0.1:{fake_backend.port}")
    p.start()
    yield p
    p.stop()


class TestModelCallProxyForwarding:
    def test_forwards_post_body_and_returns_backend_response(self, proxy, fake_backend):
        status, response = _post_json(
            f"http://127.0.0.1:{proxy.listen_port}/v1/chat/completions",
            {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert status == 200
        assert response == fake_backend.response_body
        assert fake_backend.received_paths == ["/v1/chat/completions"]
        assert fake_backend.received_bodies[0]["messages"][0]["content"] == "hello"

    def test_forwards_backend_error_status_unchanged(self, fake_backend):
        fake_backend.status = 400
        fake_backend.response_body = {"error": {"message": "bad request"}}
        p = ModelCallProxy(_free_port(), f"http://127.0.0.1:{fake_backend.port}")
        p.start()
        try:
            status, response = _post_json(
                f"http://127.0.0.1:{p.listen_port}/v1/chat/completions", {"messages": []}
            )
            assert status == 400
            assert response == {"error": {"message": "bad request"}}
        finally:
            p.stop()

    def test_get_requests_are_forwarded_but_not_recorded(self, proxy):
        with urllib.request.urlopen(f"http://127.0.0.1:{proxy.listen_port}/health", timeout=5) as resp:
            assert resp.getcode() == 200
        assert proxy.events == []


class TestModelCallProxyRecording:
    def test_chat_completion_calls_are_recorded_in_order(self, proxy):
        for i in range(3):
            _post_json(
                f"http://127.0.0.1:{proxy.listen_port}/v1/chat/completions",
                {"messages": [{"role": "user", "content": f"call {i}"}]},
            )
        events = proxy.events
        assert len(events) == 3
        assert [e.request_body["messages"][0]["content"] for e in events] == [
            "call 0",
            "call 1",
            "call 2",
        ]

    def test_non_chat_completion_post_paths_are_not_recorded(self, proxy):
        _post_json(f"http://127.0.0.1:{proxy.listen_port}/v1/completions", {"prompt": "x"})
        assert proxy.events == []

    def test_reset_clears_events(self, proxy):
        _post_json(
            f"http://127.0.0.1:{proxy.listen_port}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}]},
        )
        assert len(proxy.events) == 1
        proxy.reset()
        assert proxy.events == []

    def test_events_survive_independently_of_reset_snapshot(self, proxy):
        """`.events` returns a copy -- mutating it must not corrupt the
        proxy's own internal log (a bug here would silently corrupt every
        trace reconstructed downstream)."""
        _post_json(
            f"http://127.0.0.1:{proxy.listen_port}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}]},
        )
        snapshot = proxy.events
        snapshot.append("not a real event")
        assert len(proxy.events) == 1


def _event(messages: list[dict], tool_calls: list | None = None, status: int = 200) -> ProxiedCall:
    return ProxiedCall(
        request_body={"messages": messages},
        response_body={
            "choices": [
                {"message": {"role": "assistant", "content": "...", "tool_calls": tool_calls}}
            ]
        },
        status=status,
    )


class TestReconstructTraceFromEvents:
    def test_single_generation_with_no_tools_is_one_single_call(self):
        events = [_event([{"role": "user", "content": "solve this"}])]
        trace = reconstruct_trace_from_events(events)
        assert trace.op_counts() == {"single_call": 1}
        assert not trace.used_expanding

    def test_three_plain_generations_are_three_single_calls_no_tool_use(self):
        """Mirrors a multi-turn conversation where the agent just keeps
        talking -- growing history alone must not be mistaken for tool use."""
        base = [{"role": "user", "content": "solve this"}]
        events = [
            _event(base),
            _event(base + [{"role": "assistant", "content": "step 1"}]),
            _event(
                base
                + [
                    {"role": "assistant", "content": "step 1"},
                    {"role": "user", "content": "keep going"},
                ]
            ),
        ]
        trace = reconstruct_trace_from_events(events)
        assert trace.op_counts() == {"single_call": 3}
        assert not trace.used_expanding

    def test_a_real_tool_round_trip_is_classified_as_tool_call_and_expanding(self):
        """event 1: agent asks for a tool. event 2: the tool's output is now
        in the request history, conditioning the next generation -- this is
        the actual moment information from outside M enters the context."""
        user_msg = {"role": "user", "content": "run the tests"}
        assistant_asks_for_tool = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": "{}"}}],
        }
        tool_result = {"role": "tool", "tool_call_id": "1", "content": "3 tests failed"}

        event_1 = _event([user_msg], tool_calls=assistant_asks_for_tool["tool_calls"])
        event_2 = _event([user_msg, assistant_asks_for_tool, tool_result])

        trace = reconstruct_trace_from_events([event_1, event_2])

        assert [r.name for r in trace.records] == ["single_call", "tool_call", "single_call"]
        assert trace.used_expanding
        assert "tool_call" in trace.expanding_ops

    def test_tool_call_is_recorded_before_the_generation_it_conditions(self):
        """Chronology matters for any future analysis keyed on operation
        order -- the tool result must precede the generation it fed."""
        user_msg = {"role": "user", "content": "run the tests"}
        assistant_asks = {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}
        tool_result = {"role": "tool", "tool_call_id": "1", "content": "ok"}
        events = [
            _event([user_msg], tool_calls=[{"id": "1"}]),
            _event([user_msg, assistant_asks, tool_result]),
        ]
        trace = reconstruct_trace_from_events(events)
        names_in_order = [r.name for r in trace.records]
        # the tool_call record comes from the SECOND event but must be
        # ordered before that event's own single_call, matching the real
        # chronology (tool result arrives, then the model generates from it)
        assert names_in_order == ["single_call", "tool_call", "single_call"]

    def test_multiple_tool_calls_in_one_round_are_each_recorded(self):
        """A single turn can invoke more than one tool (e.g. bash and
        file_editor back to back) -- every tool-role message counts."""
        user_msg = {"role": "user", "content": "fix it"}
        assistant_asks = {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}, {"id": "2"}]}
        tool_result_1 = {"role": "tool", "tool_call_id": "1", "content": "file read"}
        tool_result_2 = {"role": "tool", "tool_call_id": "2", "content": "tests ran"}
        events = [
            _event([user_msg], tool_calls=[{"id": "1"}, {"id": "2"}]),
            _event([user_msg, assistant_asks, tool_result_1, tool_result_2]),
        ]
        trace = reconstruct_trace_from_events(events)
        assert trace.op_counts() == {"single_call": 2, "tool_call": 2}

    def test_empty_event_list_produces_an_empty_trace(self):
        trace = reconstruct_trace_from_events([])
        assert trace.records == []
        assert not trace.used_expanding

    def test_requesting_a_tool_without_a_later_response_is_not_yet_a_tool_call(self):
        """If the episode ends right after the model asks for a tool (budget
        ran out, container timed out) but the tool never actually ran, there
        is no evidence anything left M's context -- must not be tagged
        expanding on the strength of a request alone."""
        events = [_event([{"role": "user", "content": "go"}], tool_calls=[{"id": "1"}])]
        trace = reconstruct_trace_from_events(events)
        assert trace.op_counts() == {"single_call": 1}
        assert not trace.used_expanding

    def test_a_failed_call_is_not_counted_as_a_single_call(self):
        """A backend error/timeout never sampled from M -- counting it would
        inflate op_counts with calls that didn't actually happen."""
        events = [_event([{"role": "user", "content": "go"}], status=502)]
        trace = reconstruct_trace_from_events(events)
        assert trace.records == []
        assert trace.op_counts() == {}

    def test_a_failed_call_does_not_desync_a_following_retry(self):
        """The client resends the SAME (unchanged) history after a failure;
        prev_len must not have advanced past what the retry's request
        actually contains, or the retry's own messages get mis-scanned."""
        user_msg = {"role": "user", "content": "go"}
        events = [
            _event([user_msg], status=502),  # failed attempt, no history growth
            _event([user_msg]),  # retry with the identical (unretried) history
        ]
        trace = reconstruct_trace_from_events(events)
        assert trace.op_counts() == {"single_call": 1}  # only the retry actually produced a generation

    def test_a_tool_message_already_present_in_a_failed_calls_request_still_counts(self):
        """The tool genuinely ran before this attempt was made -- whether the
        attempt itself then failed doesn't erase that it happened."""
        user_msg = {"role": "user", "content": "go"}
        assistant_asks = {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}
        tool_result = {"role": "tool", "tool_call_id": "1", "content": "ok"}
        events = [
            _event([user_msg], tool_calls=[{"id": "1"}]),
            _event([user_msg, assistant_asks, tool_result], status=502),
        ]
        trace = reconstruct_trace_from_events(events)
        assert trace.op_counts() == {"single_call": 1, "tool_call": 1}
        assert trace.used_expanding

    def test_2xx_response_with_no_choices_is_not_counted_either(self):
        """A malformed-but-technically-200 response has no real generation
        in it -- must not be treated as a sample from M just because the
        status code looked fine."""
        event = ProxiedCall(
            request_body={"messages": [{"role": "user", "content": "go"}]},
            response_body={},  # no "choices" key at all
            status=200,
        )
        trace = reconstruct_trace_from_events([event])
        assert trace.records == []


class TestModelCallProxyFailureHandling:
    def test_genuinely_unreachable_backend_yields_a_clean_502_not_a_hang_or_crash(self):
        """No fake/mocked backend here -- point the proxy at a real closed
        port and confirm the existing URLError path (connection refused)
        still produces a clean response and a recorded event, not a hang."""
        p = ModelCallProxy(_free_port(), "http://127.0.0.1:1")  # port 1: nothing listens there
        p.start()
        try:
            status, response = _post_json(
                f"http://127.0.0.1:{p.listen_port}/v1/chat/completions",
                {"messages": [{"role": "user", "content": "x"}]},
                timeout=10.0,
            )
            assert status == 502
            assert "error" in response
            events = p.events
            assert len(events) == 1
            assert events[0].status == 502
        finally:
            p.stop()

    def test_arbitrary_backend_exceptions_are_caught_not_propagated(self, proxy, fake_backend, monkeypatch):
        """Not just HTTPError/URLError -- ANY exception talking to the
        backend must produce a clean synthesized response and a recorded
        event, never an unhandled crash that silently drops the call."""
        real_urlopen = urllib.request.urlopen
        backend_marker = f":{fake_backend.port}/"

        def flaky_urlopen(req, *args, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if backend_marker in url:
                raise ValueError("simulated non-URLError backend failure")
            return real_urlopen(req, *args, **kwargs)

        monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)

        status, response = _post_json(
            f"http://127.0.0.1:{proxy.listen_port}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}]},
        )
        assert status == 502
        assert "simulated non-URLError backend failure" in response.get("error", "")
        events = proxy.events
        assert len(events) == 1
        assert events[0].status == 502

    def test_status_is_recorded_on_every_event(self, proxy):
        _post_json(
            f"http://127.0.0.1:{proxy.listen_port}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "x"}]},
        )
        assert proxy.events[0].status == 200

    def test_error_status_from_backend_is_recorded_not_silently_dropped(self, fake_backend):
        fake_backend.status = 400
        fake_backend.response_body = {"error": {"message": "bad request"}}
        p = ModelCallProxy(_free_port(), f"http://127.0.0.1:{fake_backend.port}")
        p.start()
        try:
            _post_json(f"http://127.0.0.1:{p.listen_port}/v1/chat/completions", {"messages": []})
            events = p.events
            assert len(events) == 1
            assert events[0].status == 400
        finally:
            p.stop()


class TestModelCallProxyLifecycle:
    def test_reset_raises_if_a_handler_is_stuck_in_flight(self, proxy):
        """A silently-corrupted trace (clearing while a straggler could still
        land) is worse than a loud failure -- prove reset() actually refuses
        rather than proceeding, using a short timeout so the test is fast."""
        with proxy._lock:
            proxy._inflight += 1  # simulate a handler thread mid-request
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                proxy.reset(timeout=0.2)
        finally:
            with proxy._lock:
                proxy._inflight -= 1  # don't leak state into other tests

    def test_reset_succeeds_once_the_simulated_handler_finishes(self, proxy):
        with proxy._lock:
            proxy._inflight += 1
        import threading

        def finish_shortly():
            import time

            time.sleep(0.05)
            with proxy._lock:
                proxy._inflight -= 1

        threading.Thread(target=finish_shortly).start()
        proxy.reset(timeout=2.0)  # must not raise -- the drain should succeed

    def test_double_start_raises_and_does_not_leak_the_redundant_socket(self, proxy):
        with pytest.raises(RuntimeError, match="already started"):
            proxy.start()
        # the proxy must still be fully usable after the rejected second start
        status, _ = _post_json(
            f"http://127.0.0.1:{proxy.listen_port}/v1/chat/completions",
            {"messages": [{"role": "user", "content": "still works"}]},
        )
        assert status == 200

    def test_stop_without_start_does_not_raise(self):
        p = ModelCallProxy(_free_port(), "http://127.0.0.1:1")
        p.stop()  # must be a no-op, not an AttributeError on a None server
