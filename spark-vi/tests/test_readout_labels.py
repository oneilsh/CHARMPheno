"""readout.labels.NodeLabeler (spec 2026-10-30 R6): a document's DAG labels -> the
per-node closure label and observation mask the heads fit against. Pure Python; the
expected vectors are written out by hand from the DAG."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.dag.nodeset import fit_node_set
from spark_vi.readout.labels import NodeLabeler

# Two top-level nodes, two leaves under each. Engine order = (depth, id), so the
# output index of node n is n here (index 0 = root).
PARENT = {1: None, 2: None, 3: 1, 4: 1, 5: 2, 6: 2}
DOCS = [[3], [4], [5], [6], [3, 1], []]


def _labeler(docs=DOCS, dag=PARENT, min_docs=1, **kw):
    return NodeLabeler.from_node_set(fit_node_set(dag, docs, min_docs_per_node=min_docs),
                                     **kw)


def test_closure_labels_and_background():
    lab = _labeler()
    assert lab.C == 7
    assert lab.active([3]) == (0, 1, 3)
    assert lab.active([3, 1, 3]) == (0, 1, 3)          # ancestors / duplicates
    assert lab.active([3, 5]) == (0, 1, 2, 3, 5)
    assert lab.active([2]) == (0, 2)
    assert lab.active([]) == () and lab.active(None) == ()


def test_full_mask_observes_everything():
    lab = _labeler()
    assert lab([4]) == ((0, 1, 4), None)
    y, m = lab.dense([[4], None])
    np.testing.assert_array_equal(y, [[1, 1, 0, 0, 1, 0, 0], [0] * 7])
    np.testing.assert_array_equal(m, np.ones((2, 7)))


def test_closure_mask_observes_closure_and_siblings():
    lab = _labeler(mask_mode="closure")
    # [3]: closure {0,1,3}; siblings of 1 = {1,2} (root's children), of 3 = {3,4}.
    assert lab([3]) == ((0, 1, 3), (0, 1, 2, 3, 4))
    # [2]: closure {0,2}; siblings of 2 = {1,2}; leaves under 2 stay unobserved.
    assert lab([2]) == ((0, 2), (0, 1, 2))
    y, m = lab.dense([[6], []])
    np.testing.assert_array_equal(y, [[1, 0, 1, 0, 0, 0, 1], [0] * 7])
    np.testing.assert_array_equal(m, [[1, 1, 1, 0, 0, 1, 1], [0] * 7])  # bg: nothing


def test_multi_parent_siblings():
    dag = {"a": None, "b": None, "x": ["a", "b"], "y": "a", "z": "b"}
    docs = [["x"], ["y"], ["z"]]
    ns = fit_node_set(dag, docs)
    lab = NodeLabeler.from_node_set(ns, mask_mode="closure")
    idx = {n: ns.engine_id[n] for n in ns.nodes}
    act, obs = lab(["x"])
    assert set(act) == {0, idx["a"], idx["b"], idx["x"]}
    # siblings of x through both parents: y (under a) and z (under b)
    assert set(obs) == {0, idx["a"], idx["b"], idx["x"], idx["y"], idx["z"]}


def test_pruned_label_lands_on_kept_ancestor():
    docs = [[3], [3], [4], [4], [5], [5], [6]]           # 6 has support 1
    ns = fit_node_set(PARENT, docs, min_docs_per_node=2)
    assert 6 in ns.pruned
    lab = NodeLabeler.from_node_set(ns)
    two = ns.engine_id[2]
    assert lab.active([6]) == (0, two)
    assert lab.C == len(ns.nodes) + 1


def test_spliced_chain_label_lands():
    dag = {"top": None, "mid": "top", "leaf": "mid", "other": None}
    ns = fit_node_set(dag, [["leaf"], ["other"]])
    assert set(ns.spliced) == {"top", "mid"}               # single-child chain
    lab = NodeLabeler.from_node_set(ns)
    assert lab.active(["mid"]) == ()                      # lands nowhere: background
    assert lab.active(["leaf"]) == (0, ns.engine_id["leaf"])


def test_unknown_labels_error_or_ignore():
    with pytest.raises(ValueError, match="'nope'"):
        _labeler().active([3, "nope"])
    lab = _labeler(unknown_labels="ignore")
    assert lab.active([3, "nope"]) == (0, 1, 3)
    assert lab.active(["nope"]) == ()


def test_ids_are_not_normalized():
    dag = {1: None, "1": None}
    lab = NodeLabeler.from_node_set(fit_node_set(dag, [[1], ["1"]]))
    assert lab.active([1]) != lab.active(["1"])


def test_bad_policies_refused():
    ns = fit_node_set(PARENT, DOCS)
    with pytest.raises(ValueError, match="labelMaskMode"):
        NodeLabeler.from_node_set(ns, mask_mode="partial")
    with pytest.raises(ValueError, match="unknownLabels"):
        NodeLabeler.from_node_set(ns, unknown_labels="drop")


def test_local_from_labels_matches_from_arrays():
    from spark_vi.readout import LocalReadoutData
    lab = _labeler(mask_mode="closure")
    theta = np.random.default_rng(0).dirichlet(np.ones(4), size=len(DOCS))
    a = LocalReadoutData.from_labels(theta, DOCS, lab)
    y, m = lab.dense(DOCS)
    b = LocalReadoutData.from_arrays(theta, y, m)
    assert a.C == b.C == 7
    for ra, rb in zip(a.rows, b.rows):
        np.testing.assert_array_equal(ra[2], rb[2])
        assert (ra[3] is None) == (rb[3] is None)
