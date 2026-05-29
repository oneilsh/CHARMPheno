"""Integration tests for OnlineSTM: synthetic data with known parameters."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.types import STMDocument


def _synthetic_corpus(
    *,
    K: int, V: int, P: int, D: int, doc_len: int,
    Gamma_true: np.ndarray, Sigma_true: np.ndarray,
    seed: int = 0,
) -> tuple[list[STMDocument], np.ndarray]:
    """Generate D docs from the STM generative process.

    Returns the docs plus the true beta matrix (K, V).
    """
    rng = np.random.default_rng(seed)
    # β_k ~ Dirichlet(0.1 / V).
    beta = rng.dirichlet(np.full(V, 0.1), size=K)   # (K, V)
    # x_d ~ N(0, I); η_d ~ N(Γᵀ x_d, diag(Σ)).
    docs = []
    for d in range(D):
        x = rng.normal(size=P)
        mu = Gamma_true.T @ x
        eta = rng.normal(loc=mu, scale=np.sqrt(Sigma_true))
        theta = np.exp(eta - eta.max())
        theta = theta / theta.sum()
        # Multinomial draw of doc_len tokens.
        z = rng.choice(K, size=doc_len, p=theta)
        w = np.array([rng.choice(V, p=beta[zi]) for zi in z])
        unique, counts = np.unique(w, return_counts=True)
        docs.append(STMDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
            x=x,
        ))
    return docs, beta


@pytest.mark.slow
def test_synthetic_recovery_full_batch():
    """Train OnlineSTM full-batch on synthetic data with known Γ; recover Γ̂
    within reasonable tolerance.

    Marked slow because full-batch fit over 200 docs at K=4, V=30, P=2
    takes ~30 seconds. The point of this test is qualitative recovery —
    Γ̂'s sign pattern and rough magnitudes — not exact identity.
    """
    K, V, P, D, doc_len = 4, 30, 2, 200, 60
    Gamma_true = np.array([
        [+1.5, -0.5, 0.0, +0.2],
        [-0.2, +1.0, -1.0, +0.1],
    ])
    Sigma_true = np.full(K, 0.5)

    docs, beta_true = _synthetic_corpus(
        K=K, V=V, P=P, D=D, doc_len=doc_len,
        Gamma_true=Gamma_true, Sigma_true=Sigma_true, seed=42,
    )

    model = OnlineSTM(
        K=K, vocab_size=V, P=P,
        sigma_init=1.0, lbfgs_max_iter=80, lbfgs_tol=1e-5,
        random_seed=42,
    )
    gp = model.initialize_global(None)

    # Run full-batch (ρ=1) for N outer iters.
    for _ in range(30):
        stats = model.local_update(docs, gp)
        gp = model.update_global(gp, stats, learning_rate=1.0)

    Gamma_hat = gp["Gamma"]

    # Topic labels are unidentifiable up to permutation. We check that
    # *some* column-permutation of Γ̂ recovers Γ_true's sign pattern.
    from itertools import permutations
    best = max(
        permutations(range(K)),
        key=lambda perm: float(np.sum(np.sign(Gamma_hat[:, list(perm)]) == np.sign(Gamma_true))),
    )
    Gamma_hat_aligned = Gamma_hat[:, list(best)]
    sign_match = float(np.mean(np.sign(Gamma_hat_aligned) == np.sign(Gamma_true)))
    # 75% (6/8 entries) sign-match is the floor for a "qualitative recovery"
    # check at this corpus size; tighten if the implementation hits higher
    # consistently. Perfect recovery would be 100%.
    assert sign_match >= 0.75, f"Γ̂ sign pattern off: {sign_match=}"
