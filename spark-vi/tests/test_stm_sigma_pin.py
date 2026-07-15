import numpy as np
import pytest
from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic._linalg import topic_correlation


def _drive_mstep(m, gp, S, N, lr=1.0):
    """Call update_global with a planted scatter S and support N (the only Sigma
    inputs), returning the new Sigma. Mirrors test_stm_global_scale._drive_mstep."""
    K, V, P = m.K, m.V, m.P
    stats = {
        "residual_outer_stat": S,
        "n_pairs_stat": N,
        "lambda_stats": np.zeros((K, V)),
        "XtX": np.zeros((P, P)),
        "XtX_groups": [np.zeros((P, P)) for _ in m._effective_partition().groups],
        "XtMu": np.zeros((P, K)),
        "n_docs_per_topic": np.ones(K),
    }
    return m.update_global(gp, stats, learning_rate=lr)["Sigma"]


def _toy_stats(K=3):
    # All self-supported (N=50), diag_var=[4,4,4], off-diag correlation r_01=0.5.
    N = np.full((K, K), 50.0)
    S = np.array([
        [200.0, 100.0, 0.0],
        [100.0, 200.0, 0.0],
        [0.0, 0.0, 200.0],
    ])
    return S, N


def test_pin_1_is_byte_identical_to_unit():
    S, N = _toy_stats()

    m_default = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0)
    gp_default = m_default.initialize_global(None)
    Sig_default = _drive_mstep(m_default, gp_default, S.copy(), N.copy(), lr=1.0)

    m_pin1 = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0, sigma_diagonal_pin=1.0)
    gp_pin1 = m_pin1.initialize_global(None)
    Sig_pin1 = _drive_mstep(m_pin1, gp_pin1, S.copy(), N.copy(), lr=1.0)

    assert np.array_equal(Sig_default, Sig_pin1)


def test_pin_c_sets_diagonal_and_scales_offdiagonals():
    S, N = _toy_stats()
    c = 5.0

    m_unit = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0)
    gp_unit = m_unit.initialize_global(None)
    Sig_unit = _drive_mstep(m_unit, gp_unit, S.copy(), N.copy(), lr=1.0)

    m_pin = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0, sigma_diagonal_pin=c)
    gp_pin = m_pin.initialize_global(None)
    gp_pin["Sigma"] = np.eye(3) * c   # c-consistent starting Sigma (mirrors global_scale tests)
    Sig_pin = _drive_mstep(m_pin, gp_pin, S.copy(), N.copy(), lr=1.0)

    # Diagonal is exactly c (all topics free/self-supported here).
    np.testing.assert_allclose(np.diag(Sig_pin), c, atol=1e-9)

    # Supported off-diagonal (0,1): Sigma_ij == c * R_ij where R_ij is the
    # unit-pin correlation for the same stats.
    np.testing.assert_allclose(Sig_pin[0, 1] / c, Sig_unit[0, 1], atol=1e-9)


def test_correlation_is_scale_invariant():
    S, N = _toy_stats()
    c = 5.0

    m_unit = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0)
    gp_unit = m_unit.initialize_global(None)
    Sig_unit = _drive_mstep(m_unit, gp_unit, S.copy(), N.copy(), lr=1.0)

    m_pin = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0, sigma_diagonal_pin=c)
    gp_pin = m_pin.initialize_global(None)
    gp_pin["Sigma"] = np.eye(3) * c
    Sig_pin = _drive_mstep(m_pin, gp_pin, S.copy(), N.copy(), lr=1.0)

    R_unit = topic_correlation(Sig_unit)
    R_pin = topic_correlation(Sig_pin)
    np.testing.assert_allclose(R_unit, R_pin, atol=1e-9)


def test_pin_conflicts_with_estimators():
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0,
                  sigma_diagonal_pin=5.0, estimate_global_scale=True)
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0,
                  sigma_diagonal_pin=5.0, estimate_sigma_diagonal=True)
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0, sigma_diagonal_pin=0.0)
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0, sigma_diagonal_pin=-1.0)
