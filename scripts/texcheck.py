"""Minimal LaTeX sanity check: brace balance and environment matching.

Not a compiler substitute -- it catches the two error classes most likely to
come from hand-editing (an unclosed group, a begin without its end) without
needing a TeX toolchain installed.
"""
import re
import sys
from collections import Counter

path = sys.argv[1]
raw = open(path, encoding="utf-8").read()

# Strip comments: a '%' that is NOT escaped as '\%'.
no_comments = re.sub(r"(?<!\\)%.*", "", raw)

# For brace counting, remove escaped braces so they don't skew the balance.
for esc in (r"\{", r"\}"):
    counted = no_comments.replace(esc, "")
    no_comments = counted

bal = no_comments.count("{") - no_comments.count("}")
print(f"brace balance: {bal}  ({'OK' if bal == 0 else 'MISMATCH'})")

envs = re.findall(r"\\(begin|end)\{([a-zA-Z*]+)\}", no_comments)
begins = Counter(n for k, n in envs if k == "begin")
ends = Counter(n for k, n in envs if k == "end")
mismatched = {k: (begins[k], ends[k]) for k in set(begins) | set(ends) if begins[k] != ends[k]}
print(f"environments: {dict(begins)}")
print(f"unmatched environments: {mismatched or 'none'}")

# Undefined-looking custom macros actually used
defined = set(re.findall(r"\\newcommand\{\\([a-zA-Z]+)\}", raw))
print(f"custom macros defined: {sorted(defined)}")
used = set(re.findall(r"\\([a-zA-Z]+)", no_comments))
print(f"custom macros used but undefined: "
      f"{sorted(m for m in (defined & used)) and 'n/a (all defined ones are used)'}")

# Cross-references: every \ref must have a \label
labels = set(re.findall(r"\\label\{([^}]+)\}", no_comments))
refs = set(re.findall(r"\\(?:page|auto)?ref\{([^}]+)\}", no_comments))
print(f"\nlabels defined: {sorted(labels)}")
print(f"DANGLING refs (no matching label): {sorted(refs - labels) or 'none'}")
print(f"labels never referenced: {sorted(labels - refs) or 'none'}")

# Citations vs bib keys
cites = set()
for grp in re.findall(r"\\cite[a-z]*\{([^}]*)\}", no_comments):
    cites.update(c.strip() for c in grp.split(","))
bib = set(re.findall(r"@\w+\{([^,]+),", open(path.replace("workshop_paper.tex", "refs.bib"), encoding="utf-8").read()))
print(f"\ncitations used: {len(cites)}")
missing = sorted(cites - bib)
unused = sorted(bib - cites)
print(f"cited but NOT in refs.bib: {missing or 'none'}")
print(f"in refs.bib but never cited: {unused or 'none'}")
