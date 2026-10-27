"""The preferred-set two-stage anchor search in the flat anchor-word init
(`find_anchors` / `find_anchors_projected`; spec 2026-10-05). Its threading
through the DAG spectral init was removed with that init (ADR 0050).

The contract under test, in order of what would hurt most if broken:
  1. IDENTITY: `preferred=None` / empty reproduces the unguided search exactly —
     every existing spectral run stays byte-identical.
  2. STAGE ORDER: with enough eligible preferred rows, every anchor comes from
     the preferred set, chosen by the same farthest-point rule WITHIN it.
  3. FALLBACK: a short preferred set is used up first, then the open pool fills
     the rest, on one shared Gram–Schmidt basis.
  4. The floor still applies to preferred rows (a preferred word below the
     node's df floor cannot anchor), and seeds/degenerate rows are skipped.
"""
import numpy as np
import pytest

from spark_vi.models.topic.spectral_init import find_anchors
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


def test_projected_marginal_floor_guided_and_all():
    QR, p_w, df_w = _sketch()
    p_w = p_w.copy(); p_w[[2, 6]] = 1e-4            # two rare preferred words
    base = find_anchors_projected(QR, p_w, df_w, 3, preferred=[2, 6, 10])
    assert base[0] in (2, 6, 10)
    g = find_anchors_projected(QR, p_w, df_w, 3, preferred=[2, 6, 10], marginal_floor="guided")
    assert g[0] == 10 and 2 not in g[:1] and 6 not in g[:1]     # rare preferred rows cannot lead
    a = find_anchors_projected(QR, p_w, df_w, 3, preferred=[2, 6, 10], marginal_floor="all")
    assert 2 not in a and 6 not in a                           # nor anchor at all
    assert find_anchors_projected(QR, p_w, df_w, 3, marginal_floor="none") == \
        find_anchors_projected(QR, p_w, df_w, 3)
    with pytest.raises(ValueError):
        find_anchors_projected(QR, p_w, df_w, 3, marginal_floor="bogus")
