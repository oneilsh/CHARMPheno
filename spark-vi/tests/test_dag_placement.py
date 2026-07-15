import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout, label_from_coded, strip_dag_node_codes

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}   # root 0 -> families 1,2 -> subtypes

def test_daglayout_structure_and_masks():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    assert lay.nodes == [1, 2, 3, 4, 5, 6]
    assert lay.K == 2 + 6                      # bg + one topic per node
    assert lay.closure(3) == [0, 1, 3]         # root..v
    assert lay.subtree(1) == {1, 3, 4}
    assert lay.depth(3) == 2 and lay.depth(1) == 1
    # allowed(v) = bg ∪ blocks along closure(v), excluding root
    assert list(lay.allowed(3)) == [0, 1] + lay.block[1] + lay.block[3]
    assert list(lay.allowed(1)) == [0, 1] + lay.block[1]

def test_daglayout_tpn_two():
    lay = DagLayout(PARENT, n_bg=1, tpn=2)
    assert lay.K == 1 + 6 * 2
    assert len(lay.block[3]) == 2

def test_label_same_path_is_deepest():
    lay = DagLayout(PARENT)
    # {1,3} lie on one path root->1->3 : most-specific = deepest = 3
    assert label_from_coded([1, 3], lay) == 3
    assert label_from_coded([3], lay) == 3

def test_label_siblings_is_lca():
    lay = DagLayout(PARENT)
    # {3,4} are siblings under 1 : LCA = 1
    assert label_from_coded([3, 4], lay) == 1
    # {3,5} cross-branch under root : LCA = 0
    assert label_from_coded([3, 5], lay) == 0

def test_strip_dag_node_codes():
    doc = np.array([10, 3, 11, 1, 12])          # 3 and 1 are DAG-node codes
    out = strip_dag_node_codes(doc, {1, 3})
    assert list(out) == [10, 11, 12]

def test_dag_placement_corpus_shapes():
    from tests._stm_synth import dag_placement_corpus

    docs, labels, node_codes = dag_placement_corpus(
        parent=PARENT, node_prev={1: .18, 2: .18, 3: .16, 4: .16, 5: .16, 6: .16},
        V=120, doc_len=40, seed=0)
    assert len(docs) == len(labels)
    assert set(labels.tolist()) <= set(PARENT.keys())
    assert set(node_codes.keys()) == set(PARENT.keys())
    # a node's exact code appears in items labeled at/below that node
    below3 = [d for d, y in zip(docs, labels) if y in {3}]
    assert any(node_codes[3] in d for d in below3)
