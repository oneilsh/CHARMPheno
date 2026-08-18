"""Post-fit localized head-convergence diagnostic (_converge_localized_head):
converges the engine's localized ridge-Newton head on frozen theta to test whether
the co-fit head merely under-converged. Pure numpy — no Spark."""
import numpy as np

from gated_pc_cloud import _converge_localized_head, _localized_head_proba


def test_converged_localized_head_recovers_signal_and_stays_on_support():
    rng = np.random.default_rng(0)
    N, K, C = 800, 8, 3
    Pi = rng.normal(size=(N, K))
    # node c's TRUE signal is topic c (in its support); topics 5-7 are off-support noise.
    support_cols = [np.array([0, 1, 2, c], dtype=int) for c in range(C)]
    y = np.stack([(Pi[:, c] + 0.3 * rng.normal(size=N) > 0).astype(float)
                  for c in range(C)], axis=1)
    obs = np.ones((N, C))

    w = _converge_localized_head(Pi, y, obs, support_cols, C, head_l2=1e-2,
                                 head_newton_ridge=0.05, n_iters=25)

    # (1) off-support weights stay exactly 0 (localization invariant).
    for c in range(C):
        off = [k for k in range(K) if k not in set(support_cols[c].tolist())]
        assert np.allclose(w[c, off], 0.0), f"node {c} leaked off support"

    # (2) the converged head discriminates its signal (AUC well above chance) — the
    #     ceiling the under-converged co-fit head should reach.
    proba = _localized_head_proba(Pi, w)
    for c in range(C):
        order = np.argsort(proba[:, c])
        yc = y[order, c]
        # rank-sum AUC
        pos = yc.sum()
        neg = len(yc) - pos
        auc = (np.sum(np.where(yc == 1)[0]) - pos * (pos - 1) / 2) / (pos * neg)
        assert auc > 0.85, f"node {c} converged AUC {auc:.3f} too low"


def test_converge_handles_degenerate_and_empty():
    C, K = 2, 4
    # single-class node + empty-support node -> no crash, no update.
    Pi = np.random.default_rng(1).normal(size=(20, K))
    y = np.zeros((20, C)); y[:, 0] = 1.0            # node 0 all-ones (single class)
    obs = np.ones((20, C))
    support_cols = [np.array([0, 1], dtype=int), np.array([], dtype=int)]
    w = _converge_localized_head(Pi, y, obs, support_cols, C, head_l2=1e-2,
                                 head_newton_ridge=0.05, n_iters=10)
    assert w.shape == (C, K) and np.allclose(w, 0.0)   # both skip -> stays 0
    # empty theta -> zero-row proba, correct shape.
    assert _localized_head_proba(np.zeros((0, K)), w).shape == (0, C)
