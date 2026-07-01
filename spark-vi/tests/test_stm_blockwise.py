import numpy as np
import pytest
from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.partition import TopicBlockPartition


def _drive_mstep(m, gp, S, N, lr=1.0):
    """Call update_global with a planted scatter S and support N (the only Sigma
    inputs), returning the new Sigma. Other stats are zero-shaped placeholders."""
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


def test_mstep_output_is_unit_diagonal():
    m = OnlineSTM(K=4, vocab_size=8, P=1, random_seed=0)
    gp = m.initialize_global(None)
    S = np.eye(4) * 3.0 + 0.5          # arbitrary PSD-ish scatter
    N = np.full((4, 4), 100.0)
    Sig = _drive_mstep(m, gp, S, N)
    np.testing.assert_allclose(np.diag(Sig), 1.0, atol=1e-12)


def test_mstep_supported_offdiag_is_standardized_correlation():
    m = OnlineSTM(K=3, vocab_size=8, P=1, random_seed=0)
    gp = m.initialize_global(None)
    S = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
    N = np.full((3, 3), 50.0)
    Sig = _drive_mstep(m, gp, S, N)          # lr=1 -> full move to target
    # r_01 = (S01/N)/sqrt((S00/N)(S11/N)) = 1/sqrt(4*9) = 1/6
    assert abs(Sig[0, 1] - (1.0 / 6.0)) < 1e-9
    assert abs(Sig[1, 0] - (1.0 / 6.0)) < 1e-9


def test_mstep_unsupported_pair_is_lazy_kept():
    m = OnlineSTM(K=3, vocab_size=8, P=1, min_pair_support=10, random_seed=0)
    gp = m.initialize_global(None)
    gp["Sigma"] = np.array([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]])
    S = np.eye(3) * 2.0
    N = np.full((3, 3), 50.0)
    N[0, 2] = N[2, 0] = 0.0                   # pair (0,2) unsupported
    Sig = _drive_mstep(m, gp, S, N)
    assert abs(Sig[0, 2] - 0.3) < 1e-9        # kept at previous value, not zeroed


def test_mstep_pins_runaway_variance_to_one():
    # A scatter whose free-variance MLE would inflate topic 1's diagonal to 500;
    # block-wise pins it to 1 regardless.
    m = OnlineSTM(K=2, vocab_size=8, P=1, random_seed=0)
    gp = m.initialize_global(None)
    S = np.array([[1.0, 0.0], [0.0, 500.0]])
    N = np.full((2, 2), 20.0)
    Sig = _drive_mstep(m, gp, S, N)
    assert abs(Sig[1, 1] - 1.0) < 1e-12       # pinned, not 500/20
