import numpy as np
from spark_vi.eval.topic.concentration_recovery import (
    laplace_theta_samples, marginalized_predictive_loglik, _predictive_loglik,
)

def test_samples_shape_and_simplex_and_masking():
    K = 5
    allowed = np.array([0, 2, 4])          # topics 1,3 disallowed
    reference = 0                           # reference pinned at eta=0
    eta_hat = np.full(K, -np.inf); eta_hat[allowed] = [0.0, 0.3, -0.2]
    nu_d = np.zeros((K, K))
    free = np.array([2, 4])                 # allowed minus reference
    nu_d[np.ix_(free, free)] = [[0.4, 0.05], [0.05, 0.3]]
    S = 64
    th = laplace_theta_samples(eta_hat, nu_d, allowed, K,
                               reference=reference, n_samples=S,
                               rng=np.random.default_rng(0))
    assert th.shape == (S, K)
    assert np.allclose(th.sum(axis=1), 1.0)         # simplex
    assert np.allclose(th[:, [1, 3]], 0.0)          # disallowed -> exactly 0
    assert (th[:, reference] > 0).all()             # reference alive

def test_zero_covariance_samples_reduce_to_mode_theta():
    # nu_d = 0 -> every draw is the mode -> marginalized == plug-in at the mode.
    K = 4; allowed = np.array([0, 1, 2, 3]); reference = 0
    eta_hat = np.array([0.0, 0.5, -0.3, 0.1])
    nu_d = np.zeros((K, K))
    th = laplace_theta_samples(eta_hat, nu_d, allowed, K, reference=reference,
                               n_samples=8, rng=np.random.default_rng(1))
    mode = th[0]
    assert np.allclose(th, mode)                    # all draws identical
    beta = np.abs(np.random.default_rng(2).normal(size=(K, 6)))
    beta /= beta.sum(axis=1, keepdims=True)
    held_i = np.array([0, 3, 5]); held_c = np.array([2.0, 1.0, 3.0])
    marg = marginalized_predictive_loglik(th, beta, held_i, held_c)
    plug = _predictive_loglik(mode, beta, held_i, held_c)
    assert abs(marg - plug) < 1e-9

def test_log_of_average_not_average_of_log():
    # With real spread, log-of-average (marginalized) must EXCEED average-of-log
    # (Jensen) — this is the ordering that IS the fix.
    K = 3; allowed = np.array([0, 1, 2]); reference = 0
    eta_hat = np.array([0.0, 0.2, -0.1])
    nu_d = np.zeros((K, K)); nu_d[np.ix_([1, 2], [1, 2])] = [[1.5, 0.0], [0.0, 1.5]]
    th = laplace_theta_samples(eta_hat, nu_d, allowed, K, reference=reference,
                               n_samples=256, rng=np.random.default_rng(3))
    beta = np.abs(np.random.default_rng(4).normal(size=(K, 8)))
    beta /= beta.sum(axis=1, keepdims=True)
    held_i = np.array([1, 4, 7]); held_c = np.array([1.0, 2.0, 1.0])
    log_of_avg = marginalized_predictive_loglik(th, beta, held_i, held_c)
    # average-of-log baseline
    avg_of_log = np.mean([_predictive_loglik(t, beta, held_i, held_c) for t in th])
    assert log_of_avg > avg_of_log
