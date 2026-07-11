import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit
from spark_vi.models.topic.pg_stm import (
    gated_theta, gated_expected_log_theta, gated_counts,
    stick_to_simplex, stick_trials)


def test_gated_theta_sums_to_one_and_composes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        psi_bg = rng.normal(size=3)      # B=4 background topics
        psi_gate = float(rng.normal())
        psi_fg = rng.normal(size=2)      # m_g=3 foreground topics
        theta = gated_theta(psi_bg, psi_gate, psi_fg)
        assert theta.shape == (4 + 3,)
        assert np.all(theta > 0) and abs(theta.sum() - 1.0) < 1e-12
        # background mass = sigma(gate), foreground mass = 1-sigma(gate)
        assert abs(theta[:4].sum() - expit(psi_gate)) < 1e-12
        assert abs(theta[4:].sum() - (1 - expit(psi_gate))) < 1e-12
        # within-block proportions match the flat map
        assert np.allclose(theta[:4] / theta[:4].sum(), stick_to_simplex(psi_bg))
        assert np.allclose(theta[4:] / theta[4:].sum(), stick_to_simplex(psi_fg))


def test_gated_counts():
    n_bg = np.array([3.0, 0.0, 5.0])     # N_bg=8
    n_fg = np.array([2.0, 4.0])          # N_fg=6
    ga, gb, b_bg, b_fg = gated_counts(n_bg, n_fg)
    assert ga == 8.0 and gb == 14.0
    assert np.allclose(b_bg, stick_trials(n_bg))
    assert np.allclose(b_fg, stick_trials(n_fg))


def _quad_elog_gated(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg, nodes=64):
    # Exact E[log theta] via per-stick Gaussian quadrature (deterministic reference).
    x, w = hermegauss(nodes); w = w / np.sqrt(2 * np.pi)
    def e_log_sig(m, v, sign):  # E[log sigma(sign*psi)], psi~N(m,v)
        return float(w @ np.log(expit(sign * (m + np.sqrt(v) * x))))
    def e_log_theta_flat(m, v):
        K = len(m) + 1
        lp = np.array([e_log_sig(m[j], v[j], +1) for j in range(len(m))])
        lm = np.array([e_log_sig(m[j], v[j], -1) for j in range(len(m))])
        out = np.empty(K); cum = np.concatenate([[0.0], np.cumsum(lm)])
        out[:K-1] = lp + cum[:K-1]; out[K-1] = cum[K-1]; return out
    eg_bg = e_log_sig(m_gate, v_gate, +1)      # E[log sigma(gate)]
    eg_fg = e_log_sig(m_gate, v_gate, -1)      # E[log (1-sigma(gate))]
    return np.concatenate([eg_bg + e_log_theta_flat(m_bg, v_bg),
                           eg_fg + e_log_theta_flat(m_fg, v_fg)])


def test_gated_expected_log_theta_matches_quadrature():
    m_bg = np.array([0.3, -0.5]); v_bg = np.array([0.4, 0.2])
    m_gate, v_gate = 0.2, 0.3
    m_fg = np.array([0.1]); v_fg = np.array([0.5])
    approx = gated_expected_log_theta(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg)
    exact = _quad_elog_gated(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg)
    assert np.allclose(approx, exact, atol=2e-3)   # same delta-method accuracy as Task 4
