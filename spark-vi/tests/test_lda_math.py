"""Pure-numpy tests for VanillaLDA's CAVI inner loop and ELBO machinery.

No Spark — these test the math at module level. Single document, hand-checked
shapes and values where possible.
"""
import numpy as np
import pytest


def _peaked_expElogbeta(K: int, V: int, sharpness: float = 5.0) -> np.ndarray:
    """Build a stylized expElogbeta where each topic peaks on a single word.

    Returns (K, V) array. Topic k peaks on word k (mod V). Used to make CAVI
    tests deterministic and visually inspectable.
    """
    eb = np.full((K, V), -sharpness, dtype=np.float64)
    for k in range(K):
        eb[k, k % V] = 0.0
    return np.exp(eb)


def test_cavi_doc_inference_converges_to_dominant_topic():
    """A document whose tokens are all word-0 should drive gamma toward topic 0."""
    from spark_vi.models.lda import _cavi_doc_inference

    K, V = 3, 5
    expElogbeta = _peaked_expElogbeta(K, V)  # topic 0 peaks on word 0
    indices = np.array([0], dtype=np.int32)
    counts = np.array([20.0], dtype=np.float64)
    alpha = 0.1
    gamma_init = np.full(K, 1.0)  # symmetric init

    gamma, expElogthetad, phi_norm, n_iter = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha, gamma_init,
        max_iter=100, tol=1e-4,
    )
    # Topic 0 should dominate the posterior.
    assert np.argmax(gamma) == 0
    assert gamma[0] > gamma[1] + 5.0
    assert gamma[0] > gamma[2] + 5.0
    # Phi normalizer is per-unique-token; one unique token here.
    assert phi_norm.shape == (1,)
    # expElogthetad is K-vector, all positive.
    assert expElogthetad.shape == (K,)
    assert (expElogthetad > 0).all()
    # Should converge well under the iteration cap.
    assert n_iter < 100


def test_cavi_doc_inference_respects_max_iter():
    """If tol is impossibly tight, max_iter is the hard ceiling."""
    from spark_vi.models.lda import _cavi_doc_inference

    K, V = 2, 3
    expElogbeta = _peaked_expElogbeta(K, V)
    indices = np.array([0, 1], dtype=np.int32)
    counts = np.array([1.0, 1.0])
    gamma_init = np.full(K, 1.0)

    _, _, _, n_iter = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha=0.1, gamma_init=gamma_init,
        max_iter=3, tol=1e-100,
    )
    assert n_iter == 3


def test_cavi_doc_inference_matches_explicit_phi_implementation():
    """Lee/Seung trick (implicit phi) must agree numerically with the
    explicit-phi formulation on a small fixture. The production path uses
    the implicit form for memory efficiency; this test pins them to the
    same answer.
    """
    from spark_vi.models.lda import _cavi_doc_inference

    K, V = 3, 6
    rng = np.random.default_rng(0)
    expElogbeta = np.exp(rng.normal(size=(K, V)) * 0.3)
    indices = np.array([0, 2, 5], dtype=np.int32)
    counts = np.array([3.0, 1.0, 2.0])
    alpha = 0.5
    gamma_init = np.ones(K) * 1.0
    max_iter = 50
    tol = 1e-8

    # Implicit (production):
    gamma_impl, _, _, _ = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha, gamma_init.copy(),
        max_iter=max_iter, tol=tol,
    )

    # Explicit phi reference, same recurrence:
    eb_d = expElogbeta[:, indices]                 # (K, n_unique)
    gamma_exp = gamma_init.copy()
    from scipy.special import digamma
    for _ in range(max_iter):
        prev = gamma_exp.copy()
        eEt = np.exp(digamma(gamma_exp) - digamma(gamma_exp.sum()))
        # phi_dnk ∝ eEt[k] * eb_d[k, n]; normalize over k per token n.
        unnorm = eb_d * eEt[:, None]               # (K, n_unique)
        phi_explicit = unnorm / unnorm.sum(axis=0, keepdims=True)
        gamma_exp = alpha + (phi_explicit * counts[None, :]).sum(axis=1)
        if np.mean(np.abs(gamma_exp - prev)) < tol:
            break

    np.testing.assert_allclose(gamma_impl, gamma_exp, atol=1e-6)
