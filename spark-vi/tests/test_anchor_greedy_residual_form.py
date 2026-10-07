"""The anchor greedy in residual-matrix form equals the basis-reprojection
form it replaced (exp 0134: 1,200 background anchors made the old
O(n^2 V d) Python-loop form unrunnable).

The reference below IS the old implementation, kept here as the oracle:
an orthonormal basis of chosen residuals, every candidate row re-projected
against all of it at every step. Both functions must pick the same anchors
in the same order on random data, with seeds, with a preferred stage, and
with degenerate (all-zero / duplicate) rows.
"""
import numpy as np
import pytest

from spark_vi.models.topic.spectral_init import find_anchors
from spark_vi.models.topic.spectral_init_scalable import (
    _row_normalize_projected, find_anchors_projected)

EPS = 1e-12


def _reference_greedy(Qbar, norms, candidate, n, *, seed_rows=None, preferred=None):
    V = Qbar.shape[0]
    basis = []

    def project_out(vec):
        r = vec.copy()
        for b in basis:
            r = r - (r @ b) * b
        return r

    def add_to_basis(row_id):
        r = project_out(Qbar[row_id])
        nrm = np.sqrt(r @ r)
        if nrm > EPS:
            basis.append(r / nrm)

    for s in (seed_rows or []):
        add_to_basis(int(s))
    anchors = []
    chosen = set(int(s) for s in (seed_rows or []))

    def greedy(pool, n_left):
        for _ in range(n_left):
            best_id, best_res = -1, -np.inf
            for i in range(V):
                if i in chosen or norms[i] <= EPS or not pool[i]:
                    continue
                r = project_out(Qbar[i])
                res = r @ r
                if res > best_res:
                    best_res, best_id = res, i
            if best_id < 0:
                break
            anchors.append(best_id)
            chosen.add(best_id)
            add_to_basis(best_id)

    if preferred is not None and len(preferred):
        pref = np.zeros(V, dtype=bool); pref[list(preferred)] = True
        greedy(candidate & pref, n)
    greedy(candidate, n - len(anchors))
    return anchors


def _sketch(seed, V=60, d=24, n_zero=2):
    rng = np.random.default_rng(seed)
    QR = rng.random((V, d)) * (rng.random((V, d)) < 0.4)
    QR[:n_zero] = 0.0                      # degenerate rows never anchor
    QR[V - 1] = QR[V - 2]                  # a duplicate row: zero residual
    p_w = QR.sum(axis=1) + 1e-9
    df = rng.integers(0, 40, size=V); df[:n_zero] = 0
    return QR, p_w, df


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_projected_greedy_matches_the_reprojection_reference(seed):
    QR, p_w, df = _sketch(seed)
    Qbar = _row_normalize_projected(QR, p_w)
    norms = (Qbar * Qbar).sum(axis=1)
    candidate = df >= 5
    got = find_anchors_projected(QR, p_w, df, 12, min_doc_freq=5)
    assert got == _reference_greedy(Qbar, norms, candidate, 12)


def test_projected_greedy_with_seeds_and_a_preferred_stage_matches():
    QR, p_w, df = _sketch(7)
    Qbar = _row_normalize_projected(QR, p_w)
    norms = (Qbar * Qbar).sum(axis=1)
    candidate = df >= 5
    seeds = [5, 9, 9, 0]                      # a repeat and a zero row among them
    pref = [3, 11, 17, 23, 29]
    got = find_anchors_projected(QR, p_w, df, 10, seed_rows=seeds,
                                 min_doc_freq=5, preferred=pref)
    assert got == _reference_greedy(Qbar, norms, candidate, 10,
                                    seed_rows=seeds, preferred=pref)
    assert not (set(got) & set(seeds))


@pytest.mark.parametrize("seed", [0, 1])
def test_dense_greedy_matches_the_reprojection_reference(seed):
    from spark_vi.models.topic.spectral_init import _row_normalize
    rng = np.random.default_rng(seed)
    V = 40
    Q = rng.random((V, V)) * (rng.random((V, V)) < 0.5)
    Q = Q + Q.T
    Q[0] = 0.0; Q[:, 0] = 0.0
    Qbar = _row_normalize(Q)
    norms = (Qbar * Qbar).sum(axis=1)
    marginal = Q.sum(axis=1)
    candidate = marginal >= 0.5 * marginal[marginal > 0].mean()
    got = find_anchors(Q, 8, min_marginal_frac=0.5, seed_rows=[3, 4])
    assert got == _reference_greedy(Qbar, norms, candidate, 8, seed_rows=[3, 4])


def test_projected_greedy_scales_to_a_wide_background():
    """1,200 anchors from an 11.6k x 2048 sketch is exp 0134's background. The
    old form needed days; this must take seconds (no timing assert — the point
    is that it finishes and returns the asked-for count from distinct rows)."""
    rng = np.random.default_rng(3)
    V, d, n = 3000, 512, 400
    QR = rng.random((V, d)) * (rng.random((V, d)) < 0.1)
    p_w = QR.sum(axis=1) + 1e-9
    df = np.full(V, 10)
    got = find_anchors_projected(QR, p_w, df, n, min_doc_freq=5)
    assert len(got) == n and len(set(got)) == n
