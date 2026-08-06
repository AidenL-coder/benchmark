"""Isolated reproduction of HGM's expand()/TS_sample crash (D-37) and
confirmation of the D-41 fix, without touching sample_child/Docker at all.

Not part of the `cbs` package -- like the other scripts here, this is
glue/validation for a vendored fork, not `cbs`'s own code. Replicates
exactly the two code paths (before/after the fix) using a fake node
registry, so this runs anywhere numpy is installed -- no HGM checkout, no
Docker, no GPU, no real model call needed. See docs/DECISIONS.md D-37/D-41
for the full root-cause writeup.
"""
import numpy as np


class FakeNode:
    def __init__(self, node_id, utility_measures):
        self.id = node_id
        self.utility_measures = utility_measures

    @property
    def num_evals(self):
        return len(self.utility_measures)

    @property
    def mean_utility(self):
        if self.num_evals == 0:
            return np.inf
        return np.sum(self.utility_measures) / self.num_evals

    def get_decendant_evals(self, num_pseudo=10):
        return self.utility_measures if self.num_evals < num_pseudo else [self.mean_utility] * num_pseudo


def TS_sample(evals):
    alphas = [1 + np.sum(de) for de in evals]
    betas = [1 + len(de) - np.sum(de) for de in evals]
    thetas = np.random.beta(alphas, betas)
    return np.argmax(thetas)


def expand_original(nodes_dict):
    """The exact pre-D-41 logic (hgm.py's expand(), before the patch) --
    expected to crash in the all-zero case."""
    nodes = [
        node
        for node in nodes_dict.values()
        if np.isfinite(node.mean_utility) and node.mean_utility > 0
    ]
    decendant_evals = [node.get_decendant_evals(num_pseudo=10) for node in nodes]
    selected_node = nodes[TS_sample(decendant_evals)]
    return selected_node


def expand_fixed(nodes_dict):
    """The D-41 patched logic, as applied to the real hgm.py on the remote
    instance (/lambda/nfs/cbs-project/hgm/hgm.py, hgm.py.orig kept as
    backup)."""
    nodes = [
        node
        for node in nodes_dict.values()
        if np.isfinite(node.mean_utility) and node.mean_utility > 0
    ]
    if not nodes:
        nodes = [node for node in nodes_dict.values() if np.isfinite(node.mean_utility)]
    decendant_evals = [node.get_decendant_evals(num_pseudo=10) for node in nodes]
    selected_node = nodes[TS_sample(decendant_evals)]
    return selected_node


def main() -> None:
    # Case 1: the exact real smoke-test scenario -- one node, evaluated on
    # one task, which failed (utility_measures=[0], mean_utility=0.0).
    degenerate = {0: FakeNode(0, [0])}

    crashed = False
    try:
        expand_original(degenerate)
    except ValueError as e:
        crashed = True
        print(f"expand_original on degenerate archive: CRASHED as expected -- {e}")
    assert crashed, "expected the original logic to crash on the degenerate case"

    result = expand_fixed(degenerate)
    print(f"expand_fixed on degenerate archive: did NOT crash, selected node id={result.id}")
    assert result.id == 0

    # Case 2: a slightly larger degenerate archive -- 60 real tasks' worth
    # of nodes, ALL with mean_utility == 0 (the real-scale analogue of the
    # crash: not "too few tasks", but "nothing has succeeded yet").
    degenerate_60 = {i: FakeNode(i, [0] * 60) for i in range(3)}
    crashed = False
    try:
        expand_original(degenerate_60)
    except ValueError as e:
        crashed = True
        print(f"expand_original on 60-task-scale all-zero archive: CRASHED as expected -- {e}")
    assert crashed, "expected the original logic to also crash at 60-task scale when all nodes are 0"

    result = expand_fixed(degenerate_60)
    print(f"expand_fixed on 60-task-scale all-zero archive: did NOT crash, selected node id={result.id}")

    # Case 3: normal case -- one node has succeeded on at least one task.
    # Fixed logic must behave identically to the original (fallback branch
    # never entered), i.e. only ever select among nodes with mean_utility > 0.
    normal = {
        0: FakeNode(0, [0, 0, 1]),  # mean_utility = 1/3 > 0
        1: FakeNode(1, [0, 0, 0]),  # mean_utility = 0, excluded either way
    }
    np.random.seed(0)
    orig_selected_ids = {expand_original(normal).id for _ in range(200)}
    np.random.seed(0)
    fixed_selected_ids = {expand_fixed(normal).id for _ in range(200)}
    print(f"normal case -- original ever selects: {orig_selected_ids}, fixed ever selects: {fixed_selected_ids}")
    assert orig_selected_ids == fixed_selected_ids == {0}, (
        "fixed logic must never select the mean_utility==0 node when a positive-utility node exists"
    )

    print(
        "\nALL CHECKS PASSED -- D-41 fix confirmed to eliminate the crash in both the exact "
        "1-task smoke-test scenario and its 60-task-scale analogue, without changing behavior "
        "in the normal (>=1 positive-utility node) case."
    )


if __name__ == "__main__":
    main()
