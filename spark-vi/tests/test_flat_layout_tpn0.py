"""A FLAT layout through the gated engine: `tpn=0`, `n_bg=K`.

Exp 0132 (spec 2026-10-06 Part B) fits HSLDA's flat shared topics with the
SAME engine, corpus and saved-fit format as every gated run: every node's
block is empty, K == n_bg, and every document's allowed set is the whole
topic range — i.e. plain online LDA, multi-domain λ included, with nothing
on the topic side knowing the DAG. These pin the three facts the run relies
on: the layout's K and blocks, the gate's allowed set, and the alpha init.
"""
import numpy as np

from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import (GatedBOWDocument, GatedOnlineLDA,
                                             equalized_alpha)

PARENT = {1: [0], 2: [0], 3: [1], 4: [1, 2]}


def test_tpn0_layout_is_flat():
    lay = DagLayout(PARENT, n_bg=7, tpn=0)
    assert lay.K == 7
    assert all(lay.block[u] == [] for u in lay.nodes)
    for fr in (frozenset(), frozenset({3}), frozenset({4, 2})):
        assert sorted(lay.allowed_set(fr).tolist()) == list(range(7))
    # The localized-head supports are the full range too (dense head).
    assert all(len(lay.allowed_with_path_cousins_kids(c)) == 7 for c in range(5))


def test_tpn0_gated_e_step_touches_every_topic_for_every_doc():
    lay = DagLayout(PARENT, n_bg=6, tpn=0)
    m = GatedOnlineLDA(lay, 12, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    assert gp["lambda"].shape == (6, 12)
    docs = [GatedBOWDocument(indices=np.array([5, 6], dtype=np.int32),
                             counts=np.array([2.0, 1.0]), length=3,
                             frontier=frozenset({3})),
            GatedBOWDocument(indices=np.array([1, 2, 7], dtype=np.int32),
                             counts=np.array([1.0, 3.0, 2.0]), length=6,
                             frontier=frozenset())]
    out = m.local_update(docs, gp)
    assert out["lambda_stats"].shape == (6, 12)
    assert (out["lambda_stats"].sum(axis=1) > 0).all()     # no topic is gated out


def test_tpn0_equalized_alpha_is_uniform_not_a_zero_division():
    lay = DagLayout(PARENT, n_bg=5, tpn=0)
    a = equalized_alpha(lay, {frozenset(): 10, frozenset({3}): 4}, 0.5)
    assert np.allclose(a, 0.5) and a.shape == (5,)
