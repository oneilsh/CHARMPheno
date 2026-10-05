"""HPO-guided spectral anchors (spec 2026-10-05): the preferred-set two-stage
anchor search and its threading through both gated spectral paths.

The contract under test, in order of what would hurt most if broken:
  1. IDENTITY: `preferred=None` / empty reproduces the unguided search exactly —
     every existing spectral run stays byte-identical.
  2. STAGE ORDER: with enough eligible preferred rows, every anchor comes from
     the preferred set, chosen by the same farthest-point rule WITHIN it.
  3. FALLBACK: a short preferred set is used up first, then the open pool fills
     the rest, on one shared Gram–Schmidt basis.
  4. The floor still applies to preferred rows (a preferred word below the
     node's df floor cannot anchor), and seeds/degenerate rows are skipped.
  5. gated_init threads per-node sets and reports per-node counts; the dense
     and scalable paths agree on a planted corpus.
"""
import numpy as np
import pytest

from spark_vi.models.topic.spectral_init import find_anchors, word_cooccurrence
from spark_vi.models.topic.spectral_init_scalable import find_anchors_projected


def _Q(V=24, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((V, V))
    return A @ A.T + np.eye(V)


def test_dense_identity_when_unguided():
    Q = _Q()
    base = find_anchors(Q, 5)
    assert find_anchors(Q, 5, preferred=None) == base
    assert find_anchors(Q, 5, preferred=[]) == base
    assert find_anchors(Q, 5, preferred=np.array([], dtype=int)) == base


def test_dense_stage1_takes_only_preferred_in_farthest_point_order():
    Q = _Q()
    pref = [3, 7, 11, 15, 19, 21]
    # floor off: every preferred row is eligible (the floor's own effect on a
    # preferred row is tested separately below)
    got = find_anchors(Q, 4, preferred=pref, min_marginal_frac=0.0)
    assert len(got) == 4 and set(got) <= set(pref)
    # same rule WITHIN the preferred pool: an open search over a Q whose only
    # non-degenerate rows are the preferred ones must pick the same anchors in
    # the same order (row normalization makes the scaling invisible to the
    # geometry; the near-zero rows fall under the `norms <= EPS` guard).
    Qr = Q.copy()
    keep = np.zeros(Q.shape[0], dtype=bool); keep[pref] = True
    Qr[~keep] = 0.0
    assert find_anchors(Qr, 4, preferred=None, min_marginal_frac=0.0) == got


def test_dense_floor_applies_to_preferred_rows_too():
    Q = _Q()
    marginal = Q.sum(axis=1)
    low = int(np.argmin(marginal))              # surely below the mean-marginal floor
    got = find_anchors(Q, 3, preferred=[low])
    assert low not in got and len(got) == 3     # fell back to the open pool


def test_dense_fallback_fills_from_open_pool_on_shared_basis():
    Q = _Q()
    pref = [5, 9]
    got = find_anchors(Q, 4, preferred=pref, min_marginal_frac=0.0)
    assert set(got[:2]) == {5, 9}
    assert got[2] not in pref and got[3] not in pref
    # the two fallback picks are exactly what the open search chooses when
    # seeded with the two preferred rows (one shared basis, same criterion)
    tail = find_anchors(Q, 2, seed_rows=got[:2], min_marginal_frac=0.0)
    assert got[2:] == tail


def test_dense_preferred_rows_that_are_seeds_or_out_of_range_are_skipped():
    Q = _Q()
    got = find_anchors(Q, 3, seed_rows=[5], preferred=[5, 999, -1, 9],
                       min_marginal_frac=0.0)
    assert 5 not in got and 999 not in got
    assert got[0] == 9                      # the one usable preferred row


def _sketch(V=24, d=8, seed=1):
    rng = np.random.default_rng(seed)
    QR = rng.random((V, d))
    p_w = rng.random(V) + 0.1
    df_w = np.full(V, 10)
    return QR, p_w, df_w


def test_projected_identity_and_stage_order():
    QR, p_w, df_w = _sketch()
    base = find_anchors_projected(QR, p_w, df_w, 4)
    assert find_anchors_projected(QR, p_w, df_w, 4, preferred=None) == base
    assert find_anchors_projected(QR, p_w, df_w, 4, preferred=[]) == base
    pref = [2, 6, 10, 14, 18]
    got = find_anchors_projected(QR, p_w, df_w, 4, preferred=pref)
    assert len(got) == 4 and set(got) <= set(pref)


def test_projected_floor_applies_to_preferred_rows_then_falls_back():
    QR, p_w, df_w = _sketch()
    df_w = df_w.copy(); df_w[6] = 1            # below the floor of 5
    got = find_anchors_projected(QR, p_w, df_w, 3, preferred=[2, 6], min_doc_freq=5)
    assert got[0] == 2 and 6 not in got and len(got) == 3


# --------------------------------------------------------------------------- #
# gated_init threading                                                         #
# --------------------------------------------------------------------------- #
def _planted():
    """Parent 1 (tokens 0,1), child 2 (tokens 2,3 + inherits 0), background 8,9;
    tokens 4..7 are a 'confound' that co-occurs with node 2's docs more purely
    than its own phenotype tokens, so the OPEN search anchors node 2 on it and
    the GUIDED search (preferred = node 2's phenotype tokens) does not."""
    docs, labels = [], []
    for i in range(40):
        docs.append(np.array([8, 9, 0, 1])); labels.append(frozenset())
        docs.append(np.array([0, 1, 8])); labels.append(frozenset({1}))
        # node 2: phenotype tokens 2,3 in every doc; confound 4..7 in a sharp
        # sub-stratum that never carries anything else
        if i % 2 == 0:
            docs.append(np.array([2, 3, 0, 8])); labels.append(frozenset({2}))
        else:
            docs.append(np.array([4, 5, 6, 7])); labels.append(frozenset({2}))
    return docs, labels


def test_dense_gated_init_threads_candidates_and_reports_counts():
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.gated_init import spectral_block_aligned_lambda

    docs, labels = _planted()
    lay = DagLayout({1: 0, 2: 1}, n_bg=2, tpn=1)
    V = 10
    ds = {"train_docs": docs, "train_labels": labels}
    stats_open: dict = {}
    lam_open = spectral_block_aligned_lambda(ds, lay, V, anchor_stats=stats_open)
    stats_g: dict = {}
    lam_g = spectral_block_aligned_lambda(
        ds, lay, V, anchor_candidates={2: [2, 3]}, anchor_stats=stats_g)
    # unguided stats: every node seeded, none guided
    assert set(stats_open) == {1, 2} and all(v[0] == 0 for v in stats_open.values())
    # guided: node 2 drew its one anchor from its preferred set; node 1 untouched
    assert stats_g[2][0] == 2 and stats_g[2][2] == 1 and stats_g[2][3] == 1
    assert stats_g[1] == stats_open[1]
    b2_open, b2_g = lam_open[lay.block[2][0]], lam_g[lay.block[2][0]]
    assert np.allclose(lam_open[lay.block[1][0]], lam_g[lay.block[1][0]])
    # the guided block puts more of its mass on the phenotype tokens than the
    # open one does
    assert b2_g[[2, 3]].sum() / b2_g.sum() > b2_open[[2, 3]].sum() / b2_open.sum()


def test_summarize_anchor_stats_pools_to_counts_only():
    from spark_vi.models.topic.gated_init import summarize_anchor_stats
    st = {1: (0, None, 0, 1), 2: (3, 2, 1, 1), 3: (4, 0, 0, 1), 4: (2, 2, 1, 2)}
    s = summarize_anchor_stats(st, n_nodes=5, tpn=2)
    assert s["nodes_seeded"] == 4 and s["nodes_total"] == 5
    assert s["nodes_guided"] == 3 and s["anchors_from_profile"] == 2
    assert s["anchors_total"] == 5 and s["anchors_possible"] == 10
    assert (s["nodes_fully_guided"], s["nodes_partially_guided"],
            s["nodes_fallback_only"]) == (1, 1, 1)
    assert summarize_anchor_stats(None, n_nodes=3, tpn=1)["nodes_guided"] == 0


@pytest.mark.slow
def test_scalable_and_dense_agree_on_guided_anchors(spark):
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument
    from spark_vi.models.topic.gated_init import (
        scalable_block_aligned_lambda, spectral_block_aligned_lambda)

    docs, labels = _planted()
    lay = DagLayout({1: 0, 2: 1}, n_bg=2, tpn=1)
    V = 10
    rows = [GatedBOWDocument(indices=np.asarray(sorted(d), dtype=np.int32),
                             counts=np.ones(len(d)), length=len(d),
                             frontier=frozenset(l)) for d, l in zip(docs, labels)]
    rdd = spark.sparkContext.parallelize(rows, 3)
    st_s: dict = {}
    scalable_block_aligned_lambda(rdd, lay, V, seed=0, min_doc_freq=1,
                                  anchor_candidates={2: [2, 3]}, anchor_stats=st_s)
    st_d: dict = {}
    spectral_block_aligned_lambda({"train_docs": docs, "train_labels": labels},
                                  lay, V, anchor_candidates={2: [2, 3]},
                                  anchor_stats=st_d)
    # both paths guided node 2 from its set, and both left node 1 unguided
    assert st_s[2][2] == st_d[2][2] == 1 and st_s[1][0] == st_d[1][0] == 0
    assert st_s[2][1] == 2                       # scalable reports the df-floor count


def test_estimator_param_roundtrip_and_random_init_guard():
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    est = OnlinePCLDAEstimator()
    assert est.getSpectralAnchorCandidates() == {}
    est.setSpectralAnchorCandidates({3: [7, 2, 7], 5: []})
    assert est.getSpectralAnchorCandidates() == {3: [2, 7]}     # deduped, sorted, empty dropped
    est.setSpectralAnchorCandidates(None)
    assert est.getSpectralAnchorCandidates() == {}
    with pytest.raises(TypeError):
        est.setSpectralAnchorCandidates({3: [1.5]})              # float id must not truncate
