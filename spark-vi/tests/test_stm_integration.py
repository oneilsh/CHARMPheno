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


@pytest.mark.slow
def test_minibatch_converges_to_neighborhood_of_full_batch():
    """Mini-batch fit on the same synthetic corpus produces Γ̂ that:
    (1) Sign-pattern matches full-batch Γ̂ on most entries.
    (2) ELBO at convergence is within ~5% of full-batch.

    Failure of either gate is a signal that ρ-blended stochastic-EM on
    Γ, Σ has correctness issues — investigate before shipping STM
    with mini-batch enabled by default (ADR 0023).
    """
    K, V, P, D, doc_len = 4, 30, 2, 200, 60
    Gamma_true = np.array([
        [+1.5, -0.5, 0.0, +0.2],
        [-0.2, +1.0, -1.0, +0.1],
    ])
    Sigma_true = np.full(K, 0.5)

    docs, _ = _synthetic_corpus(
        K=K, V=V, P=P, D=D, doc_len=doc_len,
        Gamma_true=Gamma_true, Sigma_true=Sigma_true, seed=42,
    )

    # Full-batch reference.
    model_fb = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=42)
    gp_fb = model_fb.initialize_global(None)
    for _ in range(30):
        stats = model_fb.local_update(docs, gp_fb)
        gp_fb = model_fb.update_global(gp_fb, stats, learning_rate=1.0)

    # Mini-batch run with the same seed; ρ_t = (t + 64)^{-0.7}.
    rng = np.random.default_rng(42)
    model_mb = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=42)
    gp_mb = model_mb.initialize_global(None)
    batch_size = 20
    n_outer = 200
    for t in range(n_outer):
        idx = rng.choice(D, size=batch_size, replace=False)
        batch = [docs[i] for i in idx]
        stats = model_mb.local_update(batch, gp_mb)
        # Corpus-scale stats by (D / batch_size) so they represent the
        # full-corpus target, then ρ-blend.
        scale = D / batch_size
        # Scale every stat to the full-corpus target. Scaling all keys
        # generically (rather than an explicit allowlist) keeps the gating
        # stats added by ADR 0027 — XtX_groups, n_docs_per_topic — present;
        # an earlier hand-picked dict dropped them and update_global raised
        # KeyError('XtX_groups').
        scaled_stats = {
            k: (v * scale if isinstance(v, (np.ndarray, int, float)) else v)
            for k, v in stats.items()
        }
        rho_t = (t + 64) ** -0.7
        gp_mb = model_mb.update_global(gp_mb, scaled_stats, learning_rate=rho_t)

    # Align Γ̂_mb to Γ̂_fb up to column permutation.
    from itertools import permutations
    best = max(
        permutations(range(K)),
        key=lambda perm: float(np.sum(
            np.sign(gp_mb["Gamma"][:, list(perm)]) == np.sign(gp_fb["Gamma"])
        )),
    )
    Gamma_mb_aligned = gp_mb["Gamma"][:, list(best)]
    sign_match = float(np.mean(np.sign(Gamma_mb_aligned) == np.sign(gp_fb["Gamma"])))
    assert sign_match >= 0.75, (
        f"Mini-batch Γ̂ sign pattern diverges from full-batch: {sign_match=}. "
        f"Stochastic-EM blending of Γ may have a correctness bug; see ADR 0023."
    )

    # ELBO comparison.
    stats_fb_final = model_fb.local_update(docs, gp_fb)
    stats_mb_final = model_mb.local_update(docs, gp_mb)
    elbo_fb = model_fb.compute_elbo(gp_fb, stats_fb_final)
    elbo_mb = model_mb.compute_elbo(gp_mb, stats_mb_final)
    rel_diff = abs(elbo_mb - elbo_fb) / abs(elbo_fb)
    assert rel_diff < 0.05, (
        f"Mini-batch ELBO too far from full-batch: rel_diff={rel_diff:.3f}. "
        f"See ADR 0023 mini-batch convergence-to-neighborhood discussion."
    )


def test_gated_stm_recovers_planted_minority_phenotype():
    """A planted vocabulary cluster expressed ONLY by the rare group must be
    recovered by a foreground topic, and majority docs must not express it."""
    import numpy as np
    from spark_vi.models.topic.stm import OnlineSTM, _softmax
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.types import STMDocument

    rng = np.random.default_rng(11)
    V = 12
    # Background vocab = tokens 0..7; rare-only phenotype = tokens 8..11.
    bg_tokens = np.arange(0, 8)
    rare_tokens = np.arange(8, 12)
    part = TopicBlockPartition("g", background_k=3, foreground=(("rare", 2),))
    K = part.K  # 5

    def make_doc(is_rare):
        toks = rng.choice(bg_tokens, size=3, replace=False)
        if is_rare:
            toks = np.concatenate([toks, rng.choice(rare_tokens, size=2, replace=False)])
        toks = np.unique(toks)
        return STMDocument(indices=toks.astype(np.int32),
                           counts=np.ones(len(toks)), length=len(toks),
                           x=np.array([1.0]),
                           groups=frozenset({"rare"}) if is_rare else frozenset())

    docs = [make_doc(i % 5 == 0) for i in range(400)]  # ~20% rare
    model = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=2, topic_blocks=part)
    gp = model.initialize_global(None)
    for _ in range(30):
        stats = model.local_update(docs, gp)
        gp = model.update_global(gp, stats, learning_rate=0.5)

    lam = gp["lambda"]
    beta = lam / lam.sum(axis=1, keepdims=True)
    fg = part.block_indices("rare")
    # Some foreground topic concentrates on the rare-only tokens.
    fg_mass_on_rare = beta[fg][:, rare_tokens].sum(axis=1).max()
    bg_mass_on_rare = beta[part.background_indices()][:, rare_tokens].sum(axis=1).max()
    assert fg_mass_on_rare > 0.5, fg_mass_on_rare
    # Background barely touches rare tokens (majority never expresses them).
    assert bg_mass_on_rare < 0.1, bg_mass_on_rare


def test_gated_diagnostics_include_block_labels():
    import numpy as np
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))
    model = OnlineSTM(K=3, vocab_size=4, P=1, random_seed=1, topic_blocks=part)
    gp = model.initialize_global(None)
    diag = model.iteration_diagnostics(gp)
    assert list(diag["topic_block_labels"]) == ["background", "background", "rare"]
    assert "blocks[bg=" in model.iteration_summary(gp)


def test_reference_fit_pins_reference_end_to_end():
    """A fitted reference model, gated and non-gated, keeps the reference topic
    pinned (eta=0) through the export path and yields a valid theta. This is the
    deterministic invariant; the *recovery* payoff is measured by the ablation
    script, not asserted at a guessed threshold."""
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from _stm_synth import synthetic_ehr_corpus, synthetic_gated_corpus, fit_stm
    from spark_vi.models.topic.stm import OnlineSTM

    # Non-gated.
    docs, _ = synthetic_ehr_corpus(K_rare=4, V=80, D=200, doc_len=25,
                                   bg_frac=0.6, seed=5)
    m = OnlineSTM(K=5, vocab_size=80, P=1, sigma_init=1.0,
                  random_seed=42, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(30):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    out = m.infer_local(docs[0], gp)
    assert out["eta"][0] == 0.0
    assert abs(float(out["theta"].sum()) - 1.0) < 1e-12
    assert np.allclose(gp["Gamma"][:, 0], 0.0)

    # Gated.
    docs_g, _, part = synthetic_gated_corpus(groups=("maj", "rare"),
                                             fg_per_group=2, bg_k=2, V=120,
                                             D=200, doc_len=30, bg_frac=0.5,
                                             seed=8)
    mg = OnlineSTM(K=part.K, vocab_size=120, P=1, sigma_init=1.0,
                   random_seed=42, topic_blocks=part, reference_topic=True)
    gpg = mg.initialize_global(None)
    for _ in range(30):
        gpg = mg.update_global(gpg, mg.local_update(docs_g, gpg), learning_rate=0.5)
    og = mg.infer_local(docs_g[0], gpg)
    assert og["eta"][0] == 0.0
    assert np.allclose(gpg["Gamma"][:, 0], 0.0)
