"""`spectralBgAnchors` (exp 0134): anchor only the first N background topics;
the rest take the engine's random Gamma init.

Pins: (1) capping the anchored background to N reproduces, row for row, the
seed of a layout whose n_bg IS N — same background anchors (the greedy is
prefix-consistent), same deflation seeds, same node rows — and leaves the
unanchored background rows at zero in the seed; (2) the engine fills exactly
those rows from its random draw, single-array and per-domain dict alike;
(3) the estimator Param and the helper that names the rows.
"""
import numpy as np
import pytest

from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_init import (resolve_bg_anchors,
                                              scalable_block_aligned_lambda)
from spark_vi.models.topic.gated_lda import GatedOnlineLDA
from spark_vi.models.topic.types import GatedBOWDocument

V = 10


def _docs():
    def doc(idx, frontier):
        idx = sorted(idx)
        return GatedBOWDocument(indices=np.asarray(idx, dtype=np.int32),
                                counts=np.ones(len(idx)), length=len(idx),
                                frontier=frozenset(frontier))
    rows = []
    for _ in range(30):
        rows.append(doc([8, 9, 0, 1], []))
        rows.append(doc([0, 1, 8], [1]))
        rows.append(doc([2, 3, 0, 8], [2]))
        rows.append(doc([4, 5, 9], []))
    return rows


def test_resolve_bg_anchors():
    lay = DagLayout({1: 0, 2: 1}, n_bg=6, tpn=1)
    assert resolve_bg_anchors(lay, None) == 6
    assert resolve_bg_anchors(lay, 0) == 6
    assert resolve_bg_anchors(lay, 2) == 2
    assert resolve_bg_anchors(lay, 50) == 6


@pytest.mark.slow
def test_capped_background_reproduces_the_narrow_layouts_seed(spark):
    rdd = spark.sparkContext.parallelize(_docs(), 3)
    wide = DagLayout({1: 0, 2: 1}, n_bg=4, tpn=1)      # K = 6
    narrow = DagLayout({1: 0, 2: 1}, n_bg=2, tpn=1)    # K = 4
    lam_w = scalable_block_aligned_lambda(rdd, wide, V, seed=0, min_doc_freq=1,
                                          n_bg_anchors=2)
    lam_n = scalable_block_aligned_lambda(rdd, narrow, V, seed=0, min_doc_freq=1)
    assert lam_w.shape == (6, V) and lam_n.shape == (4, V)
    assert np.allclose(lam_w[:2], lam_n[:2])                       # anchored bg rows
    # unanchored rows carry only the seed's uniform floor (no anchor, no
    # recovery) — the engine replaces them with its random draw
    assert np.all(lam_w[2:4] < 1e-5) and np.ptp(lam_w[2:4]) == 0.0
    for u in (1, 2):                                               # node rows identical
        assert np.allclose(lam_w[wide.block[u][0]], lam_n[narrow.block[u][0]])


def test_engine_fills_random_rows_single_array():
    lay = DagLayout({1: 0, 2: 1}, n_bg=4, tpn=1)
    seed = np.zeros((lay.K, V)); seed[0, :3] = 50.0; seed[1, 3:6] = 50.0
    seed[4, 6:8] = 50.0; seed[5, 8:] = 50.0
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0, init="spectral")
    gp = m.initialize_global({"spectral_lambda": seed, "random_rows": [2, 3]})
    lam = gp["lambda"]
    assert np.allclose(lam[[0, 1, 4, 5]], seed[[0, 1, 4, 5]])     # seed rows untouched
    assert (lam[2] > 0).all() and (lam[3] > 0).all()               # filled
    assert not np.allclose(lam[2], lam[3])                         # distinct draws
    # the fill IS the random path's draw for those rows
    rnd = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0).initialize_global(None)
    assert np.allclose(lam[[2, 3]], rnd["lambda"][[2, 3]])


def test_engine_fills_random_rows_per_domain():
    lay = DagLayout({1: 0, 2: 1}, n_bg=4, tpn=1)
    dom = [6, 4]
    seed = {0: np.full((lay.K, 6), 7.0), 1: np.full((lay.K, 4), 3.0)}
    seed[0][2:4] = 0.0; seed[1][2:4] = 0.0
    m = GatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0, init="spectral",
                       domains=dom)
    gp = m.initialize_global({"spectral_lambda": seed, "random_rows": [2, 3]})
    lam = gp["lambda"]
    assert set(lam) == {0, 1}
    assert np.allclose(lam[0][[0, 1, 4, 5]], 7.0) and np.allclose(lam[1][[0, 1, 4, 5]], 3.0)
    assert (lam[0][2:4] > 0).all() and (lam[1][2:4] > 0).all()
    assert not np.allclose(lam[0][2], lam[0][3])


def test_estimator_param_and_random_rows_helper():
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator, _random_bg_rows
    est = OnlinePCLDAEstimator()
    assert est.getOrDefault("spectralBgAnchors") == 0
    assert OnlinePCLDAEstimator(spectralBgAnchors=8).getOrDefault("spectralBgAnchors") == 8
    lay = DagLayout({1: 0, 2: 1}, n_bg=5, tpn=1)
    assert _random_bg_rows(lay, 0) == {}
    assert _random_bg_rows(lay, 5) == {}
    assert _random_bg_rows(lay, 2) == {"random_rows": [2, 3, 4]}
