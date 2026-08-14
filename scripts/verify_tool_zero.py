"""Verify the zero-tool-call result is real, not a misconfiguration.

Sends HGM's *own* tool definitions -- imported from the measured agent
source, not hand-written -- to the same served model, once with
tool_choice="auto" and once with "required", on an identical prompt.

The two calls differ in exactly one parameter. If the "auto" call returns
prose with no tool_calls while "required" returns a real tool call, then the
tools were present, well-formed, and reachable; the model simply chose not
to invoke them. That is a behavioural result, not a broken pipeline.
"""

import json
import sys
import urllib.request

sys.path.insert(0, "/lambda/nfs/cbs-project/hgm/measured_default_agent/src")

from llm_withtools import convert_tool_info  # noqa: E402
from tools import load_all_tools  # noqa: E402

BASE = "http://127.0.0.1:8001/v1/chat/completions"


def served_model():
    with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


MODEL = served_model()

all_tools = load_all_tools()
tools = [convert_tool_info(t["info"], model=MODEL) for t in all_tools]
print(f"model      : {MODEL}")
print(f"tools sent : {[t['function']['name'] for t in tools]}")
print(f"tool schema bytes: {len(json.dumps(tools))}")

PROMPT = (
    "You are working in a git repository at /testbed. The file "
    "queen_attack.js contains a stub implementation that must be completed "
    "so the hidden tests pass. Implement the solution."
)

messages = [
    {"role": "system", "content": "You are a coding agent operating on a repository."},
    {"role": "user", "content": PROMPT},
]


def call(tool_choice):
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": 700,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        BASE,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


for choice in ("auto", "required"):
    print("\n" + "=" * 70)
    print(f"tool_choice = {choice!r}")
    print("=" * 70)
    try:
        resp = call(choice)
    except Exception as exc:
        print(f"REQUEST FAILED: {type(exc).__name__}: {exc}")
        continue
    msg = resp["choices"][0]["message"]
    tc = msg.get("tool_calls")
    print(f"finish_reason : {resp['choices'][0].get('finish_reason')}")
    print(f"prompt_tokens : {resp['usage']['prompt_tokens']}")
    print(f"tool_calls    : {'NONE' if not tc else [c['function']['name'] for c in tc]}")
    if tc:
        print(f"first call args: {tc[0]['function']['arguments'][:200]}")
    content = (msg.get("content") or "").strip()
    print(f"content ({len(content)} chars): {content[:400]}")
