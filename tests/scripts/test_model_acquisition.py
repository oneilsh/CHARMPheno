"""WP-D part 3b: model-based acquisition (model_acquisition.py) — θ re-inferred from
the revealed codes by the deployment fold-in, scored by saved readout heads."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_acquisition as ma  # noqa: E402

# K = 4 topics: 0, 1 shared background; 2 emits condition code 0 (member A's marker);
# 3 emits condition code 1 (member B's). Two domains: condition (6), drug (4).
K, V0, V1 = 4, 6, 4


def _fit():
    lam0 = np.full((K, V0), 0.5)
    lam0[0, 2:] = 50.0
    lam0[1, 2:] = 50.0
    lam0[2, 0] = 200.0
    lam0[3, 1] = 200.0
    lam1 = np.full((K, V1), 5.0)
    alpha = np.full(K, 0.1)
    # readout nodes: 0 root, 1 group G, 2 member A, 3 member B
    Vh = np.zeros((4, K)); Vh[2, 2] = 20.0; Vh[3, 3] = 20.0
    b = np.array([8.0, 8.0, -3.0, -3.0])
    heads = (Vh, b, np.zeros(4, bool), np.zeros(4), 0)
    closure = {2: [0, 1, 2], 3: [0, 1, 3]}
    return ma.ModelEngine([lam0, lam1], alpha, heads, closure, [2, 3])


def _patients(rng, n=60):
    pats = []
    for i in range(n):
        t = int(rng.random() < 0.5)
        bag0 = {2: 4.0, 3: 2.0, 4: 1.0, (0 if t == 0 else 1): 3.0}
        pats.append(([t], {0: bag0, 1: {0: 2.0}}, {0: 4.0, 1: 1.0}))
    return pats


POOL = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (0, 0), (0, 1)]


def test_empty_record_folds_in_to_alpha_and_markers_move_the_posterior():
    eng = _fit()
    g = eng.fold_in([], [])
    assert np.allclose(g, eng.alpha)
    q0 = eng.posterior(g / g.sum())
    assert q0 == pytest.approx([0.5, 0.5])
    gA = eng.fold_in([eng.gidx(0, 0)], [3.0])
    gB = eng.fold_in([eng.gidx(0, 1)], [3.0])
    assert eng.posterior(gA / gA.sum())[0] > 0.9
    assert eng.posterior(gB / gB.sum())[1] > 0.9


def test_batched_candidate_refold_matches_the_full_fold_in():
    eng = _fit()
    eng.cand_iters = 200
    idx, cnt = [eng.gidx(0, 2)], [4.0]
    g = eng.fold_in(idx, cnt)
    cand = np.array([eng.gidx(0, 0), eng.gidx(1, 1)])
    th = eng.candidate_thetas(g, idx, cnt, cand)
    for j, c in enumerate(cand):
        gf = eng.fold_in(idx + [int(c)], cnt + [1.0])
        assert np.allclose(th[j], gf / gf.sum(), atol=1e-3)


def test_greedy_asks_the_markers_first_and_absences_change_nothing():
    eng = _fit()
    rng = np.random.default_rng(0)
    pats = _patients(rng)
    res = ma.simulate_set_model(eng, POOL, pats, steps=3, seed=1)
    g, r = res["strategies"]["greedy"]["by_step"], res["strategies"]["random"]["by_step"]
    assert g[0]["p_true"] == pytest.approx(0.5)
    assert g[2]["top1"] >= 0.95                 # a marker within two questions
    assert g[2]["p_true"] > r[2]["p_true"]
    assert res["full"]["top1"] >= 0.95 and res["record"]["top1"] >= 0.95
    # a patient with no marker: every answer that is NO leaves q at the empty record's
    pool_g = np.array([eng.gidx(d, v) for d, v in POOL])
    present = np.zeros(len(POOL), bool)
    traj, hits = ma.simulate_patient_model(
        eng, pool_g, np.array([d for d, _ in POOL]), present, np.zeros(len(POOL)),
        {0: 4.0, 1: 1.0}, strategy="greedy", steps=4, Eb_pool=eng.ebeta_cols(POOL))
    assert np.allclose(traj, traj[0]) and not hits.any()


def test_format_reports_both_ceilings_and_suppresses_small_sets():
    eng = _fit()
    rng = np.random.default_rng(3)
    small = ma.simulate_set_model(eng, POOL, _patients(rng, n=8), steps=2)
    assert "suppressed" in ma.format_model_acquisition("x", ["A", "B"], small)
    res = ma.simulate_set_model(eng, POOL, _patients(rng, n=30), steps=2)
    txt = ma.format_model_acquisition("x", ["A", "B"], res)
    assert "whole record (incl. diagnosis codes)" in txt and "| greedy | 2 |" in txt
    assert "empty record" in txt
