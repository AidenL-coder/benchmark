"""Figure 1: the behaviour/pass-rate asymmetry, drawn from the real results.

Regenerates `paper/fig1_asymmetry.pdf` from `results/polyglot_*.json`. Nothing
is hardcoded: every plotted number is counted from the same files
`scripts/analyze_scaffold_sensitivity.py` reports from, so the figure cannot
drift from the tables.

**The one design decision that carries the argument**: both panels share a
single 0--59 y-axis. Plotting "tasks where the agent used tools" and "tasks
resolved" on a common scale is what makes the asymmetry visible rather than
merely stated -- the left panel saturates, the right panel barely leaves the
floor, under the identical intervention. Rescaling the right panel to its own
range would make a 0->4 change look like a large effect, which is precisely
the misreading the paper argues against (§Implications).

Denominator rules follow D-43/D-46 and match the analysis script exactly:
`incomplete`/`""` are infrastructure failures and are excluded; `error` rows
are genuine agent failures and count as unresolved.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper" / "fig1_asymmetry.pdf"

INFRA_MARKERS = {"incomplete", ""}

ARMS = [
    ("7b", "auto", "polyglot_7b_auto.json"),
    ("7b", "required", "polyglot_7b_required.json"),
    ("14b", "auto", "polyglot_14b_auto.json"),
    ("14b", "required", "polyglot_14b_required.json"),
]


def load(fname: str) -> dict:
    rows = json.loads((RESULTS / fname).read_text())
    valid = [r for r in rows if r.get("eval_result") not in INFRA_MARKERS]
    used_tools = sum(1 for r in valid if r.get("trace_op_counts", {}).get("tool_call"))
    resolved = sum(1 for r in valid if r.get("passed"))
    return {"n": len(valid), "used_tools": used_tools, "resolved": resolved}


data = {(m, p): load(f) for m, p, f in ARMS}
n = max(d["n"] for d in data.values())

# Attempted-but-unparsed calls, measured from the transcripts rather than the
# API (see results/prose_toolcalls.json). Plotting only the structured counts
# would reproduce the misreading this paper corrects: it would show the 14B
# auto arm at zero tool use when in fact it attempted a call on 47/59 tasks.
PROSE = json.loads((RESULTS / "prose_toolcalls.json").read_text())["arms"]
for scale in ("7b", "14b"):
    data[(scale, "auto")]["attempted"] = PROSE[f"{scale}_auto"]["emitted"]
    data[(scale, "required")]["attempted"] = data[(scale, "required")]["used_tools"]

# Muted blue / warm orange, distinguishable in greyscale by marker and dash
# pattern as well as hue, since printed reviewer copies are often monochrome.
#
# `dx` is load-bearing, not cosmetic. The two series coincide exactly in the
# left panel (0->59 for both scales) -- which IS the finding -- so drawn at
# the same x they occlude completely and the figure appears to show a single
# model. A small horizontal offset plus a dashed 14B line keeps both visible
# while preserving the fact that they are on top of each other.
STYLE = {
    "7b": {"color": "#3B6EA5", "marker": "o", "label": "7B", "ls": "-", "dx": -0.012},
    "14b": {"color": "#C1662F", "marker": "s", "label": "14B", "ls": "--", "dx": 0.012},
}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
panels = [
    ("used_tools", "Tasks where the agent used tools"),
    ("resolved", "Tasks resolved"),
]

for ax, (key, title) in zip(axes, panels):
    # On the tool-use panel only, overlay what the model *attempted*, counted
    # from transcripts. The gap between the hollow and filled markers at
    # x=auto is the paper's central measurement artifact.
    if key == "used_tools":
        for scale in ("7b", "14b"):
            st = STYLE[scale]
            ax.plot(
                [0 + st["dx"], 1 + st["dx"]],
                [data[(scale, "auto")]["attempted"], data[(scale, "required")]["attempted"]],
                marker=st["marker"],
                color=st["color"],
                linestyle=":",
                linewidth=1.2,
                markersize=6,
                markerfacecolor="white",
                markeredgecolor=st["color"],
                clip_on=False,
                zorder=1,
            )
        ax.annotate(
            f"attempted\n(unparsed): {data[('14b','auto')]['attempted']}",
            (0 + STYLE["14b"]["dx"], data[("14b", "auto")]["attempted"]),
            textcoords="offset points",
            xytext=(14, -4),
            fontsize=8,
            color=STYLE["14b"]["color"],
            annotation_clip=False,
        )

    for scale in ("7b", "14b"):
        ys = [data[(scale, "auto")][key], data[(scale, "required")][key]]
        st = STYLE[scale]
        xs = [0 + st["dx"], 1 + st["dx"]]
        ax.plot(
            xs,
            ys,
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
            linestyle=st["ls"],
            linewidth=1.7,
            markersize=6,
            clip_on=False,
            zorder=3 if scale == "7b" else 2,
        )
        # Slopegraph convention: values sit outboard of their endpoints
        # (left of the left column, right of the right column) rather than
        # above them. Placing them vertically collides with the axis at
        # y=0 and with each other at y=59, since both series share both
        # endpoints. The small per-series vertical stagger separates the
        # two labels where the lines coincide exactly.
        # Push labels inward near the axis limits: a point at y=0 would put
        # its label under the x-axis and a point at y=n would clip against
        # the panel top. `near` also keeps the two series' labels apart
        # where their values coincide.
        for i, (x, y) in enumerate(zip(xs, ys)):
            if y >= n * 0.9:  # at the ceiling: both labels go downward
                dy = -8 if scale == "7b" else -19
            elif y <= n * 0.1:  # at the floor: both labels go upward
                dy = 9 if scale == "7b" else 20
            else:
                dy = 7 if scale == "7b" else -9
            outboard = -8 if i == 0 else 8
            ax.annotate(
                f"{y}",
                (x, y),
                textcoords="offset points",
                xytext=(outboard, dy),
                ha="right" if i == 0 else "left",
                va="center",
                fontsize=9,
                color=st["color"],
                annotation_clip=False,
            )
    # Wide enough that the outboard value labels have room and are not
    # clipped at the panel edges.
    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(0, n)
    ax.set_xticks([0, 1])
    # Rendered by matplotlib, not LaTeX -- underscores are literal here and
    # must NOT be backslash-escaped or the escape prints verbatim.
    ax.set_xticklabels(['tool_choice="auto"', 'tool_choice="required"'], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

axes[0].set_ylabel(f"Tasks (of {n})", fontsize=9)
# Lower right: the panel's lines run bottom-left to top-right and the
# attempted-call annotation occupies the upper left, so this is the only
# quadrant left free.
axes[0].legend(frameon=False, fontsize=9, loc="lower right")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
print(json.dumps({f"{m}_{p}": data[(m, p)] for m, p, _ in ARMS}, indent=2))
