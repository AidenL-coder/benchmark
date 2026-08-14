"""How many `auto`-arm transcripts contain a tool call that a lenient parser
would have executed?

The measured result was 0 structured tool calls. This asks the different
question: did the model emit a *well-formed* call in the text channel that
only the wire format prevented from running?

Three increasingly strict levels are reported, because they support
different claims:

  emitted   -- a JSON object naming a known tool appears in the assistant text
  parses    -- that object is valid JSON
  runnable  -- it names a real tool AND supplies every required argument,
               i.e. a lenient parser could have dispatched it as-is

Only `runnable` supports "the harness could have executed this".

The transcripts are Python reprs of the response object, so the content is
escape-encoded; it is unescaped before matching.
"""

import glob
import json
import os
import re
import sys
from collections import Counter

# Required args per tool, taken from the schemas the template actually
# rendered (see check_template.py output), not guessed.
REQUIRED = {
    "editor": {"command", "path", "file_text"},
    "bash": {"command"},
}

FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
BARE = re.compile(r'(\{\s*"name"\s*:\s*"(?:editor|bash)".*?\})', re.DOTALL)


def unescape(t: str) -> str:
    return (
        t.replace("\\\\n", "\n")
        .replace("\\n", "\n")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )


def candidates(text: str):
    for m in FENCE.finditer(text):
        yield m.group(1)
    for m in BARE.finditer(text):
        yield m.group(1)


def scan(arm_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(arm_dir, "*.md")))
    stats = {
        "arm": os.path.basename(arm_dir),
        "files": len(files),
        "emitted": 0,
        "parses": 0,
        "runnable": 0,
        "tools_used": Counter(),
        "commands_used": Counter(),
        "runnable_examples": [],
    }
    for f in files:
        raw = unescape(open(f, encoding="utf-8", errors="ignore").read())
        got_emit = got_parse = got_run = False
        for cand in candidates(raw):
            got_emit = True
            try:
                obj = json.loads(cand)
            except Exception:
                continue
            got_parse = True
            name = obj.get("name")
            args = obj.get("arguments")
            if name in REQUIRED and isinstance(args, dict):
                if REQUIRED[name].issubset(args.keys()):
                    got_run = True
                    stats["tools_used"][name] += 1
                    if name == "editor":
                        stats["commands_used"][args.get("command")] += 1
        stats["emitted"] += got_emit
        stats["parses"] += got_parse
        if got_run:
            stats["runnable"] += 1
            if len(stats["runnable_examples"]) < 3:
                stats["runnable_examples"].append(os.path.basename(f))
    stats["tools_used"] = dict(stats["tools_used"])
    stats["commands_used"] = dict(stats["commands_used"])
    return stats


for arm in sys.argv[1:]:
    print(json.dumps(scan(arm), indent=2))
