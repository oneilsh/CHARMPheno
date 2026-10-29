"""ModelEngine (model_engine.py): θ folded in by the deployment CAVI, scored by saved
readout heads, renormalized over a candidate set's members."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_engine import ModelEngine  # noqa: E402

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
    return ModelEngine([lam0, lam1], alpha, heads, closure, [2, 3])


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




def test_topm_and_degenerate_heads_and_batched_posterior():
    eng = _fit()
    gA = eng.fold_in([eng.gidx(0, 0)], [3.0])
    th = np.stack([gA / gA.sum(), eng.alpha / eng.alpha.sum()])
    q = eng.posterior(th)
    assert q.shape == (2, 2) and np.allclose(q.sum(axis=1), 1.0)
    assert np.allclose(q[0], eng.posterior(th[0]))
    eng.deg[:] = [False, False, True, False]     # member A's own head is constant
    eng.const[:] = [0.0, 0.0, 1e-6, 0.0]
    assert eng.posterior(th[0])[0] < 0.5
