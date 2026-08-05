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


def _event(messages: list[dict], tool_calls: list | None = None) -> ProxiedCall:
    return ProxiedCall(
        request_body={"messages": messages},
        response_body={
            "choices": [
                {"message": {"role": "assistant", "content": "...", "tool_calls": tool_calls}}
            ]
        },
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
