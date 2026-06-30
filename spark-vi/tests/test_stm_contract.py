"""Tests for OnlineSTM's VIModel contract conformance."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import OnlineSTM
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument


def _docs(rng, n, V, P, groups_fn):
    out = []
    for i in range(n):
        nz = rng.choice(V, size=2, replace=False)
        out.append(STMDocument(
            indices=np.sort(nz).astype(np.int32),
            counts=np.array([2.0, 1.0]),
            length=3, x=rng.random(P), groups=groups_fn(i)))
    return out


def test_canonical_collapse_update_global_matches_original_formulas():
    # With partition=None, update_global must use EXACTLY the pre-gating closed
    # forms: a single XtX solve for Gamma and a scalar n_docs divisor for Sigma.
    # (Together with Task 3's masked==unmasked inference test, this pins the full
    # None path as byte-identical to the original engine.)
    rng = np.random.default_rng(3)
    V, P, K = 6, 2, 4
    docs = _docs(rng, 8, V, P, lambda i: frozenset())
    # reference_topic=False: pins the non-reference closed forms (reference path covered in test_stm_reference.py)
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=None, reference_topic=False)
    gp0 = model.initialize_global(None)
    stats = model.local_update(list(docs), gp0)
    out = model.update_global(gp0, stats, 0.5)
    # Original closed forms recomputed independently from the stats.
    from spark_vi.models.topic._linalg import pd_complete
    ridge = model.sigma_ridge * np.eye(P)
    Gamma_target = np.linalg.solve(stats["XtX"] + ridge, stats["XtMu"])
    exp_Gamma = 0.5 * gp0["Gamma"] + 0.5 * Gamma_target
    # Full-cov Σ M-step: per-pair MLE = scatter / support, ρ-blend, pd_complete.
    # No-gating path: every pair is supported (observed), so pd_complete has no
    # free entries and returns the ρ-blended sample covariance unchanged.
    Sigma_target = stats["residual_outer_stat"] / stats["n_pairs_stat"]
    blended = 0.5 * gp0["Sigma"] + 0.5 * Sigma_target
    exp_Sigma = pd_complete(blended, np.ones((K, K), dtype=bool))
    target_lam = float(gp0["eta"]) + stats["lambda_stats"]
    exp_lam = 0.5 * gp0["lambda"] + 0.5 * target_lam
    # The None-path support must be uniformly n_docs (every topic pair allowed).
    np.testing.assert_array_equal(stats["n_docs_per_topic"],
                                  np.full(K, float(stats["n_docs"])))
    np.testing.assert_array_equal(stats["n_pairs_stat"],
                                  np.full((K, K), float(stats["n_docs"])))
    np.testing.assert_allclose(out["Gamma"], exp_Gamma, atol=1e-12)
    np.testing.assert_allclose(out["Sigma"], exp_Sigma, atol=1e-12)
    np.testing.assert_allclose(out["lambda"], exp_lam, atol=1e-12)


def test_zero_foreground_contribution_from_majority():
    # Majority docs (no groups) must contribute ZERO to foreground lambda rows.
    rng = np.random.default_rng(5)
    V, P = 6, 2
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 2),))
    K = part.K  # 4; foreground topics = [2, 3]
    # All docs are majority (no groups) -> foreground never allowed.
    docs = _docs(rng, 10, V, P, lambda i: frozenset())
    # reference_topic=False: pins the non-reference closed forms (reference path covered in test_stm_reference.py)
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=part, reference_topic=False)
    gp = model.initialize_global(None)
    stats = model.local_update(docs, gp)
    assert np.all(stats["lambda_stats"][2:, :] == 0.0)
    assert np.all(stats["n_docs_per_topic"][2:] == 0.0)
    assert stats["n_docs_per_topic"][0] == 10.0  # background trained on all


def test_block_aware_sigma_divisor_uses_per_topic_counts():
    rng = np.random.default_rng(7)
    V, P = 6, 2
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))
    K = part.K  # 3; foreground topic = [2]
    # 6 majority + 4 'rare' docs.
    docs = (_docs(rng, 6, V, P, lambda i: frozenset())
            + _docs(rng, 4, V, P, lambda i: frozenset({"rare"})))
    # reference_topic=False: pins the non-reference closed forms (reference path covered in test_stm_reference.py)
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=part, reference_topic=False)
    gp = model.initialize_global(None)
    stats = model.local_update(docs, gp)
    np.testing.assert_array_equal(stats["n_docs_per_topic"], [10.0, 10.0, 4.0])


def test_absent_group_block_left_unchanged_lazy_update():
    # ADR 0027: a minibatch with NO documents for group 'rare' carries no
    # information about that block, so its Gamma columns and Sigma entry must be
    # left exactly as-is rather than decaying toward (Gamma=0, Sigma=floor) via
    # a rho-blend toward a target built from zero documents.
    rng = np.random.default_rng(11)
    V, P = 6, 2
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))
    K = part.K  # 3; foreground topic index 2
    docs = _docs(rng, 8, V, P, lambda i: frozenset())  # all majority; 'rare' absent
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=part)
    gp = model.initialize_global(None)
    # Seed distinctive, non-default values on the rare block so any decay shows.
    gp["Gamma"][:, 2] = np.array([0.7, -0.4])
    gp["Sigma"][2, 2] = 0.9
    stats = model.local_update(docs, gp)
    assert stats["n_docs_per_topic"][2] == 0.0  # precondition: 'rare' absent
    out = model.update_global(gp, stats, 0.5)
    # Rare block untouched (the fix): its variance keeps its current value
    # (no support -> lazy no-op). Under pd_complete (Task 2) the absent topic's
    # variance is observed-from-prior (carries the current Σ diagonal) so the
    # completion preserves it exactly; its cross-entries are completed.
    np.testing.assert_array_equal(out["Gamma"][:, 2], np.array([0.7, -0.4]))
    # The unsupported variance is BIT-IDENTICAL: an absent topic's diagonal carries
    # the current Σ diagonal (ρ-blend of 0.9 with itself = 0.9) and pd_complete
    # preserves observed entries exactly.
    assert out["Sigma"][2, 2] == 0.9
    # Background block variances DID update (sanity: lazy logic didn't freeze all).
    assert not np.allclose(np.diag(out["Sigma"])[:2], np.diag(gp["Sigma"])[:2])
    assert not np.allclose(out["Gamma"][:, :2], gp["Gamma"][:, :2])


def test_present_group_block_updates_via_block_formula():
    # Regression guard: when the group IS present, its block must still update
    # via the per-block ridge solve + rho-blend (lazy logic must not skip it).
    rng = np.random.default_rng(13)
    V, P = 6, 2
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 1),))
    K = part.K  # 3; foreground topic index 2
    docs = (_docs(rng, 6, V, P, lambda i: frozenset())
            + _docs(rng, 4, V, P, lambda i: frozenset({"rare"})))
    model = OnlineSTM(K=K, vocab_size=V, P=P, random_seed=1, topic_blocks=part)
    gp = model.initialize_global(None)
    stats = model.local_update(docs, gp)
    assert stats["n_docs_per_topic"][2] == 4.0  # precondition: 'rare' present
    out = model.update_global(gp, stats, 0.5)
    from spark_vi.models.topic._linalg import pd_complete
    ridge = model.sigma_ridge * np.eye(P)
    cols = part.block_indices("rare")  # [2]
    Gt = np.linalg.solve(stats["XtX_groups"][0] + ridge, stats["XtMu"][:, cols])
    exp_Gamma_col = 0.5 * gp["Gamma"][:, cols] + 0.5 * Gt
    np.testing.assert_allclose(out["Gamma"][:, cols], exp_Gamma_col, atol=1e-12)
    # Full-cov Σ M-step (Task 2): per-pair sample cov on SUPPORTED pairs
    # (N >= min_pair_support), ρ-blend, then pd_complete fills the free
    # (unsupported) cross-pairs with the zero-precision max-determinant
    # completion. An absent topic's diagonal (here the reference topic 0, which
    # carries no residual scatter -> N[0,0]=0) is observed-from-prior: it carries
    # the current Σ diagonal so the lazy-keep is exact.
    S, N = stats["residual_outer_stat"], stats["n_pairs_stat"]
    supported = N >= model.min_pair_support
    mle = np.where(supported, S / np.where(N > 0, N, 1.0), 0.0)
    observed = supported.copy()
    np.fill_diagonal(observed, True)
    Sigma_target = np.where(supported, mle, gp["Sigma"])
    diag_supported = np.diag(N) >= model.min_pair_support
    kept_diag = np.where(diag_supported, np.diag(mle), np.diag(gp["Sigma"]))
    np.fill_diagonal(Sigma_target, kept_diag)
    blended = 0.5 * gp["Sigma"] + 0.5 * Sigma_target
    exp_Sigma = pd_complete(blended, observed)
    np.testing.assert_allclose(out["Sigma"], exp_Sigma, atol=1e-12)
    assert stats["n_pairs_stat"][2, 2] == 4.0  # 'rare' pair support is the group docs


def test_topic_blocks_k_mismatch_raises():
    part = TopicBlockPartition("g", background_k=2, foreground=(("rare", 2),))
    import pytest
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=6, P=2, topic_blocks=part)  # part.K == 4


class TestConstructor:
    def test_constructs_with_minimal_args(self):
        m = OnlineSTM(K=5, vocab_size=100, P=3)
        assert m.K == 5
        assert m.V == 100
        assert m.P == 3

    def test_rejects_invalid_K(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            OnlineSTM(K=0, vocab_size=100, P=3)

    def test_rejects_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            OnlineSTM(K=5, vocab_size=0, P=3)

    def test_rejects_invalid_P(self):
        with pytest.raises(ValueError, match="P must be >= 1"):
            OnlineSTM(K=5, vocab_size=100, P=0)

    def test_rejects_invalid_sigma_ridge(self):
        with pytest.raises(ValueError, match="sigma_ridge must be >= 0"):
            OnlineSTM(K=5, vocab_size=100, P=3, sigma_ridge=-1.0)


class TestInitializeGlobal:
    def test_returns_lambda_eta_gamma_sigma_shapes(self):
        m = OnlineSTM(K=4, vocab_size=20, P=2, random_seed=42)
        gp = m.initialize_global(data_summary=None)
        assert gp["lambda"].shape == (4, 20)
        assert gp["eta"].shape == ()
        assert gp["Gamma"].shape == (2, 4)
        assert gp["Sigma"].shape == (4, 4)
        # Sigma starts at sigma_init * I (full (K,K) covariance).
        np.testing.assert_allclose(gp["Sigma"], np.eye(4) * 1.0)
        # Gamma starts at zeros (covariates have no effect at init).
        np.testing.assert_allclose(gp["Gamma"], np.zeros((2, 4)))

    def test_seeded_init_is_deterministic(self):
        gp1 = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=7).initialize_global(None)
        gp2 = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=7).initialize_global(None)
        np.testing.assert_array_equal(gp1["lambda"], gp2["lambda"])


class TestGetMetadata:
    def test_returns_K_V_P(self):
        m = OnlineSTM(K=5, vocab_size=100, P=3)
        md = m.get_metadata()
        assert md == {"K": 5, "V": 100, "P": 3}


class TestLocalUpdate:
    def test_returns_expected_keys_and_shapes(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        # Inject a non-degenerate Σ so Laplace doesn't collapse.
        gp["Sigma"] = np.eye(3)
        from spark_vi.models.topic.types import STMDocument
        docs = [
            STMDocument(
                indices=np.array([0, 3, 7], dtype=np.int32),
                counts=np.array([2.0, 1.0, 1.0]),
                length=4,
                x=np.array([1.0, 0.5]),
            ),
            STMDocument(
                indices=np.array([1, 4, 8], dtype=np.int32),
                counts=np.array([1.0, 3.0, 1.0]),
                length=5,
                x=np.array([-0.5, 1.0]),
            ),
        ]
        ss = m.local_update(docs, gp)
        assert ss["lambda_stats"].shape == (3, 10)
        assert ss["XtX"].shape == (2, 2)
        assert ss["XtMu"].shape == (2, 3)
        assert ss["residual_outer_stat"].shape == (3, 3)
        assert ss["n_pairs_stat"].shape == (3, 3)
        assert ss["n_docs"].shape == ()
        assert float(ss["n_docs"]) == 2.0
        # ELBO suff stats.
        assert ss["doc_loglik_sum"].shape == ()
        assert ss["doc_eta_kl_sum"].shape == ()

    def test_lambda_stats_only_touches_seen_columns(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        from spark_vi.models.topic.types import STMDocument
        doc = STMDocument(
            indices=np.array([2, 5], dtype=np.int32),
            counts=np.array([1.0, 1.0]),
            length=2,
            x=np.array([0.0, 0.0]),
        )
        ss = m.local_update([doc], gp)
        touched = set(np.flatnonzero(ss["lambda_stats"].sum(axis=0)).tolist())
        assert touched == {2, 5}

    def test_empty_partition_returns_zero_stats(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        ss = m.local_update([], gp)
        assert float(ss["n_docs"]) == 0.0
        np.testing.assert_array_equal(ss["lambda_stats"], np.zeros((3, 10)))
        np.testing.assert_array_equal(ss["XtX"], np.zeros((2, 2)))
        np.testing.assert_array_equal(ss["XtMu"], np.zeros((2, 3)))


class TestUpdateGlobal:
    def _make_state_with_stats(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        target_stats = {
            "lambda_stats": np.ones((3, 10)) * 0.5,
            "XtX": np.eye(2) * 100.0,
            "XtX_groups": np.zeros((0, 2, 2)),          # no foreground groups
            "XtMu": np.array([[1.0, -1.0, 0.5], [0.5, 0.0, -0.5]]),
            # Full-cov residual scatter (K,K, SPD-ish) + per-pair support.
            "residual_outer_stat": np.array([[5.0, 1.0, 0.5],
                                             [1.0, 3.0, 0.2],
                                             [0.5, 0.2, 2.0]]),
            "n_pairs_stat": np.full((3, 3), 50.0),
            "n_docs_per_topic": np.full(3, 50.0),        # all docs see all topics
            "doc_loglik_sum": np.array(-100.0),
            "doc_eta_kl_sum": np.array(5.0),
            "n_docs": np.array(50.0),
        }
        return m, gp, target_stats

    def test_lambda_natural_gradient_step(self):
        m, gp, target = self._make_state_with_stats()
        lam_before = gp["lambda"].copy()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        # At ρ=1.0 the update fully replaces λ with η + lambda_stats.
        expected = float(gp["eta"]) + target["lambda_stats"]
        np.testing.assert_allclose(gp_new["lambda"], expected)

    def test_lambda_partial_step(self):
        m, gp, target = self._make_state_with_stats()
        lam_before = gp["lambda"].copy()
        gp_new = m.update_global(gp, target, learning_rate=0.3)
        # (1-ρ)·old + ρ·target.
        expected = 0.7 * lam_before + 0.3 * (float(gp["eta"]) + target["lambda_stats"])
        np.testing.assert_allclose(gp_new["lambda"], expected)

    def test_gamma_solves_ridge_regression(self):
        m, gp, target = self._make_state_with_stats()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        # Γ̂ = (XᵀX + ridge·I)⁻¹ Xᵀμ.
        ridge = m.sigma_ridge
        expected_Gamma = np.linalg.solve(
            target["XtX"] + ridge * np.eye(2), target["XtMu"]
        )
        np.testing.assert_allclose(gp_new["Gamma"], expected_Gamma)

    def test_sigma_sample_covariance(self):
        m, gp, target = self._make_state_with_stats()
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        # Full-cov per-pair MLE = scatter / support; every pair supported (N=50,
        # min_pair_support=1) so pd_complete has no free entries and returns the
        # MLE unchanged (the fully-observed special case of the completion).
        from spark_vi.models.topic._linalg import pd_complete
        mle = target["residual_outer_stat"] / target["n_pairs_stat"]
        observed = np.ones((m.K, m.K), dtype=bool)
        expected_Sigma = pd_complete(mle, observed)
        np.testing.assert_allclose(gp_new["Sigma"], expected_Sigma)

    def test_sigma_minimum_floor(self):
        """Σ must stay SPD (eigenvalues floored) to keep Laplace well-defined."""
        m, gp, target = self._make_state_with_stats()
        target["residual_outer_stat"] = np.zeros((3, 3))
        gp_new = m.update_global(gp, target, learning_rate=1.0)
        assert np.all(np.linalg.eigvalsh(gp_new["Sigma"]) > 0)


class TestComputeELBO:
    def test_returns_finite_float(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        aggregated = {
            "doc_loglik_sum": np.array(-50.0),
            "doc_eta_kl_sum": np.array(3.0),
            "n_docs": np.array(10.0),
        }
        elbo = m.compute_elbo(gp, aggregated)
        assert np.isfinite(elbo)

    def test_includes_negative_global_beta_kl(self):
        """ELBO = doc_loglik - doc_eta_kl - global_beta_kl. Increasing the
        beta KL should decrease the ELBO."""
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp_low_kl = m.initialize_global(None)
        # Concentrate λ on one column → high KL vs uniform prior.
        gp_high_kl = {**gp_low_kl, "lambda": gp_low_kl["lambda"].copy()}
        gp_high_kl["lambda"][:, 0] *= 100.0
        agg = {
            "doc_loglik_sum": np.array(-50.0),
            "doc_eta_kl_sum": np.array(3.0),
            "n_docs": np.array(10.0),
        }
        elbo_low_kl = m.compute_elbo(gp_low_kl, agg)
        elbo_high_kl = m.compute_elbo(gp_high_kl, agg)
        assert elbo_high_kl < elbo_low_kl


class TestInferLocal:
    def test_returns_eta_theta_for_single_doc(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        from spark_vi.models.topic.types import STMDocument
        doc = STMDocument(
            indices=np.array([0, 3], dtype=np.int32),
            counts=np.array([2.0, 1.0]),
            length=3,
            x=np.array([0.5, -0.3]),
        )
        out = m.infer_local(doc, gp)
        assert out["eta"].shape == (3,)
        assert out["theta"].shape == (3,)
        np.testing.assert_allclose(out["theta"].sum(), 1.0, atol=1e-10)
        assert np.all(out["theta"] > 0)


class TestIterationSummary:
    def test_returns_compact_string(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        s = m.iteration_summary(gp)
        assert isinstance(s, str)
        assert "Γ" in s or "Gamma" in s
        assert "Σ" in s or "Sigma" in s


class TestIterationDiagnostics:
    def test_returns_traceable_arrays(self):
        m = OnlineSTM(K=3, vocab_size=10, P=2, random_seed=0)
        gp = m.initialize_global(None)
        d = m.iteration_diagnostics(gp)
        assert "Gamma" in d
        assert "Sigma" in d
        assert d["Gamma"].shape == (2, 3)
        assert d["Sigma"].shape == (3, 3)

    def test_full_sigma_diagnostic_keys_present_and_finite(self):
        """All four full-Σ health-signal keys must be present and finite."""
        m = OnlineSTM(K=4, vocab_size=20, P=2, random_seed=1)
        gp = m.initialize_global(None)
        d = m.iteration_diagnostics(gp)
        for key in ("sigma_eig_min", "sigma_eig_max", "sigma_cond", "max_abs_offdiag_corr"):
            assert key in d, f"missing diagnostic key: {key!r}"
            assert np.isfinite(d[key]), f"diagnostic key {key!r} is not finite: {d[key]}"
