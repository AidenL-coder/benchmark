"""Did the model ever get told how to emit a tool call?

Qwen2.5-Coder's chat template defines a <tool_call>...</tool_call> convention
and renders the tool schemas into the system turn. vLLM's hermes parser
extracts calls from those tags. If the template renders tools correctly, the
model has been told the convention and emitting markdown JSON instead is a
model-behaviour result. If the template does NOT render them, the model was
never told, and "0 tool calls" is our deployment's fault, not a finding.

Prints the rendered prompt so the answer is visible rather than inferred.
"""

import json
import sys

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"
tok = AutoTokenizer.from_pretrained(MODEL)

sys.path.insert(0, "/lambda/nfs/cbs-project/hgm/measured_default_agent/src")
from llm_withtools import convert_tool_info  # noqa: E402
from tools import load_all_tools  # noqa: E402

tools = [convert_tool_info(t["info"], model=MODEL) for t in load_all_tools()]

messages = [
    {"role": "system", "content": "You are a coding agent operating on a repository."},
    {"role": "user", "content": "Implement the solution in /testbed."},
]

with_tools = tok.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True
)
without = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("=" * 72)
print("RENDERED WITH tools=  (len %d)" % len(with_tools))
print("=" * 72)
print(with_tools[:2600])
print("...")
print()
print("has <tool_call> convention :", "<tool_call>" in with_tools)
print("has tool schemas rendered  :", "editor" in with_tools or "bash" in with_tools)
print("len without tools          :", len(without))
