"""Tests for the COLD (reference) single-document predictive-gain oracle.

``doc_predictive_gain`` computes the leave-one-topic-out held-out predictive
gain Delta_k = LL(allowed) - LL(allowed \\ {k}) for a single STM document,
by brute-force re-inference under each topic's ablation (see
spark_vi/mllib/topic/predictive_gain.py for the design). This file plants a
tiny hand-checkable, non-gated, disjoint-vocabulary two-topic model so the
expected sign and rough magnitude of every Delta_k can be reasoned about
directly: a document built entirely from topic 0's signature words should
show a large Delta for topic 0 (removing the generating topic hurts held-out
prediction) and a near-zero Delta for topic 1 (removing an irrelevant topic
costs ~nothing -- the "auto-floor" property) -- and symmetrically for a
topic-1-only document.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import digamma

from spark_vi.eval.topic.concentration_recovery import _predictive_loglik, heldout_split
from spark_vi.mllib.topic.stm import _gated_mode_theta
from spark_vi.models.topic._linalg import safe_inverse
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.stm import _stm_doc_inference
from spark_vi.models.topic.types import STMDocument


K, V = 2, 8


def _sharp_global_params():
    """2-topic, disjoint-vocab (V=8) global_params: topic 0 owns words
    {0,1,2,3}, topic 1 owns words {4,5,6,7}. lambda carries a large
    pseudo-count on-signature and a tiny one off-signature so both
    expElogbeta (inference) and beta_prob = E[beta] (scoring) are sharply
    peaked on the owning topic's words. Gamma=0 (neutral prior mean),
    Sigma=I (unit generative scale, non-gated: R == Sigma already)."""
    lam = np.full((K, V), 0.01)
    lam[0, 0:4] = 1000.0
    lam[1, 4:8] = 1000.0
    Gamma = np.zeros((1, K))
    Sigma = np.eye(K)
    return {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}


# A large scale c (weak Gaussian prior precision 1/c) lets eta_hat follow the
# strong multinomial evidence to its (near-)extreme MAP value instead of
# being pulled back toward the neutral Gamma^T x = 0 prior mean; this is what
# makes the auto-floor property ("removing an irrelevant topic costs ~0")
# hold to a tight numeric tolerance in a tiny hand-built fixture. At c=1 the
# posterior mode is a genuine ~91/9 mix (the Gaussian prior meaningfully
# regularizes with only 4 signature words of evidence) and delta[1] is a
# real, non-negligible -0.5 nats, not a numerical near-zero -- so the
# auto-floor assertion needs the weaker prior to isolate the property being
# tested from ordinary shrinkage.
_C_WEAK_PRIOR = 1000.0


def _sharp_global_params_wide_vocab(V_wide=500):
    """Same disjoint-vocab sharp fixture as ``_sharp_global_params``, but
    padded out to a much larger vocabulary (default V=500, vs. K=2/V=8
    above) with uniform low-lambda "filler" words everywhere past index 8.

    Task 2's null tests need this widening: ``null_delta`` shuffles a
    topic's beta row across the FULL vocabulary, and at V=8 that shuffle is
    only an 8-way permutation of a 4-signature/4-background row, so there is
    a non-negligible (hypergeometric) chance the shuffled row still lands
    a couple of its big values back on the document's own 4 held-out words
    by pure coincidence -- which does NOT collapse to a near-zero Delta (it
    still explains some of the document), defeating the point of the null
    ("a topic that explains nothing"). Diluting the same four signature
    values across hundreds of filler words instead makes that coincidence
    rare, so the permuted topic reliably lands on genuinely irrelevant words
    and its Delta reliably collapses toward the null floor -- exactly the
    behavior the null band is supposed to characterize."""
    lam = np.full((K, V_wide), 0.01)
    lam[0, 0:4] = 1000.0
    lam[1, 4:8] = 1000.0
    Gamma = np.zeros((1, K))
    Sigma = np.eye(K)
    return {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}


def _non_gated_partition():
    # background_k == K, foreground empty: allowed_indices(frozenset()) == [0, 1]
    # for every doc regardless of doc.groups (non-gated model).
    return TopicBlockPartition(group_var="g", background_k=K, foreground=())


def _doc(indices, groups=frozenset()):
    """A document whose tokens are all drawn from `indices`, count 5 each
    (20 tokens total -- enough for heldout_split(holdout_frac=0.3) to yield
    a non-empty visible AND held half)."""
    indices = np.asarray(indices, dtype=np.int32)
    counts = np.full(indices.shape, 5.0, dtype=np.float64)
    return STMDocument(
        indices=indices, counts=counts, length=int(counts.sum()),
        x=np.array([1.0]), groups=groups,
    )


class TestColdAutoFloor:
    def test_cold_topic0_doc_delta_signs(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])  # all topic-0 signature words

        dg = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg is not None
        assert np.array_equal(dg.allowed, np.array([0, 1]))
        assert len(dg.delta) == len(dg.allowed) == 2
        assert np.all(np.isfinite(dg.delta))

        # Removing topic 0 (the generating topic) should hurt held-out
        # prediction a lot; removing topic 1 (irrelevant) should cost ~0.
        assert dg.delta[0] > 1.0
        assert dg.delta[1] == pytest.approx(0.0, abs=1e-2)

    def test_cold_topic1_doc_delta_signs_mirror(self):
        """Mirror case: a document built entirely from topic 1's signature
        words should show the opposite pattern -- delta[1] large, delta[0]
        ~0."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([4, 5, 6, 7])  # all topic-1 signature words

        dg = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg is not None
        assert dg.delta[1] > 1.0
        assert dg.delta[0] == pytest.approx(0.0, abs=1e-2)


class TestColdFiniteness:
    def test_cold_finite_and_theta_full_is_a_gated_simplex(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        dg = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg is not None
        assert np.isfinite(dg.ll_full)
        assert np.all(np.isfinite(dg.delta))
        assert np.all(np.isfinite(dg.dedup_delta))
        assert dg.n_held > 0

        # theta_full: a full-K vector, mass only on `allowed`, summing to 1.
        assert dg.theta_full.shape == (K,)
        assert dg.theta_full.sum() == pytest.approx(1.0, abs=1e-8)
        off_allowed = np.setdiff1d(np.arange(K), dg.allowed)
        assert np.all(dg.theta_full[off_allowed] == 0.0)


class TestColdDegenerateSplit:
    def test_cold_too_short_doc_returns_none(self):
        """A 1-token document can't be split into a non-empty visible AND
        held half (heldout_split's guard); doc_predictive_gain must mirror
        that skip by returning None rather than raising."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = STMDocument(
            indices=np.array([0], dtype=np.int32),
            counts=np.array([1.0]),
            length=1, x=np.array([1.0]), groups=frozenset(),
        )

        dg = doc_predictive_gain(doc, gp, part, c=1.0, seed=0)
        assert dg is None


class TestColdRNormalizationInvariance:
    """The module normalizes Sigma to a correlation R = Sigma / sqrt(outer(d,
    d)) (d = diag(Sigma)) before inverting and scaling by 1/c (see module
    docstring). R is invariant to rescaling Sigma by any positive diagonal
    D: corr(D Sigma D) == corr(Sigma). So doc_predictive_gain must return
    identical results for Sigma's that share a correlation, even though the
    raw Sigma entries differ wildly. A buggy implementation that inverted
    raw Sigma (skipping the R normalization) would NOT be invariant here."""

    def test_diagonal_sigma_rescaling_gives_identical_delta(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        lam = np.full((K, V), 0.01)
        lam[0, 0:4] = 1000.0
        lam[1, 4:8] = 1000.0
        Gamma = np.zeros((1, K))
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])  # all topic-0 signature words

        gp_identity = {"lambda": lam, "Gamma": Gamma, "Sigma": np.eye(K)}
        # Any positive-diagonal Sigma has correlation R == I (off-diagonal
        # terms are all 0, so dividing by sqrt(outer(d, d)) always yields
        # exactly 1 on the diagonal and 0 off it) -- so this must give the
        # SAME delta as gp_identity above. Under a raw-Sigma-invert bug
        # (skipping the R = Sigma/sqrt(outer(d,d)) normalization), Sigma_inv
        # would be (1/c)*diag(1/4, 1/9) instead of (1/c)*I, and the two
        # results would differ.
        gp_diag = {"lambda": lam, "Gamma": Gamma, "Sigma": np.diag([4.0, 9.0])}

        dg_identity = doc_predictive_gain(doc, gp_identity, part, c=_C_WEAK_PRIOR, seed=0)
        dg_diag = doc_predictive_gain(doc, gp_diag, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg_identity is not None and dg_diag is not None
        np.testing.assert_allclose(dg_diag.delta, dg_identity.delta, atol=1e-9)
        np.testing.assert_allclose(dg_diag.dedup_delta, dg_identity.dedup_delta, atol=1e-9)
        assert dg_diag.ll_full == pytest.approx(dg_identity.ll_full, abs=1e-9)

    def test_off_diagonal_correlation_diagonal_rescaling_invariance(self):
        """Stronger variant: a genuinely correlated Sigma (off-diagonal !=
        0) rescaled by a positive diagonal D must give the identical delta,
        because corr(D Sigma D) == corr(Sigma) exactly (D cancels in
        Sigma_ij / sqrt(Sigma_ii * Sigma_jj))."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        lam = np.full((K, V), 0.01)
        lam[0, 0:4] = 1000.0
        lam[1, 4:8] = 1000.0
        Gamma = np.zeros((1, K))
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        Sigma_base = np.array([[1.0, 0.3], [0.3, 1.0]])
        D = np.diag([2.0, 3.0])
        Sigma_scaled = D @ Sigma_base @ D  # [[4.0, 1.8], [1.8, 9.0]]

        gp_base = {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma_base}
        gp_scaled = {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma_scaled}

        dg_base = doc_predictive_gain(doc, gp_base, part, c=_C_WEAK_PRIOR, seed=0)
        dg_scaled = doc_predictive_gain(doc, gp_scaled, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg_base is not None and dg_scaled is not None
        np.testing.assert_allclose(dg_scaled.delta, dg_base.delta, atol=1e-9)
        np.testing.assert_allclose(dg_scaled.dedup_delta, dg_base.dedup_delta, atol=1e-9)
        assert dg_scaled.ll_full == pytest.approx(dg_base.ll_full, abs=1e-9)


class TestColdReferenceAndSingleAllowedTopic:
    """Two branches doc_predictive_gain short-circuits without any extra
    L-BFGS inference (see the `if allowed_k.size == 0 or (reference is not
    None and k == reference)` guard): ablating the pinned reference topic
    (undefined -- `_stm_doc_inference` raises ValueError if asked to
    optimize with the reference removed from `allowed`) and ablating the
    sole topic of a single-allowed-topic document (no contrast to make)."""

    def test_reference_topic_ablation_is_skipped_not_raised(self):
        """3-topic all-background model; topic 0 is pinned as reference,
        topics 1 and 2 are sharp on disjoint vocab. A document built from
        topic 1's words: ablating topic 0 (the reference) must be skipped
        (delta 0, no raise) while ablating topic 2 (irrelevant) floors near
        0 and ablating topic 1 (the generator, non-reference) costs real
        held-out likelihood -- proving non-reference ablations still keep
        the reference inside allowed_k rather than dropping it too."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        K3, V3 = 3, 6
        lam = np.full((K3, V3), 0.01)
        lam[0, 0:2] = 1000.0  # reference topic's signature (irrelevant to the doc)
        lam[1, 2:4] = 1000.0  # generating topic
        lam[2, 4:6] = 1000.0  # irrelevant, non-reference topic
        Gamma = np.zeros((1, K3))
        Sigma = np.eye(K3)
        gp = {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}

        part = TopicBlockPartition(group_var="g", background_k=K3, foreground=())
        doc = _doc([2, 3])  # all topic-1 signature words

        dg = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, reference=0)

        assert dg is not None
        assert np.array_equal(dg.allowed, np.array([0, 1, 2]))
        assert len(dg.delta) == 3

        # Reference is topic 0, at position 0 in `allowed` -- ablating it is
        # skipped by the guard and set to exactly 0.0 (not computed via
        # inference; a missing-guard bug would instead call
        # _stm_doc_inference with reference=0 no longer in allowed_k, which
        # raises ValueError -- so simply reaching this assertion without an
        # exception is itself part of what this test checks).
        assert dg.delta[0] == 0.0
        assert dg.dedup_delta[0] == 0.0

        # The non-reference topic that actually generated the held-out
        # tokens (topic 1, position 1) shows a clearly positive delta.
        assert dg.delta[1] > 1.0

        # The other irrelevant, non-reference topic (topic 2, position 2)
        # floors near 0 -- and its inference call kept the reference (topic
        # 0) inside allowed_k rather than mistakenly stripping it too.
        assert dg.delta[2] == pytest.approx(0.0, abs=1e-2)

    def test_single_allowed_topic_has_no_contrast(self):
        """A model with exactly one (background) topic: allowed = [0] for
        every doc, so the ablation loop's `allowed_k.size == 0` branch fires
        immediately -- delta and dedup_delta must be exactly [0.0], with no
        contrast possible, and the result must still be finite."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        K1, V1 = 1, 4
        lam = np.full((K1, V1), 1.0)
        Gamma = np.zeros((1, K1))
        Sigma = np.eye(K1)
        gp = {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}

        part = TopicBlockPartition(group_var="g", background_k=K1, foreground=())
        doc = _doc([0, 1])

        dg = doc_predictive_gain(doc, gp, part, c=1.0, seed=0)

        assert dg is not None
        assert np.array_equal(dg.allowed, np.array([0]))
        assert len(dg.delta) == 1
        assert np.array_equal(dg.delta, np.array([0.0]))
        assert np.array_equal(dg.dedup_delta, np.array([0.0]))
        assert np.isfinite(dg.ll_full)
        assert np.all(np.isfinite(dg.delta))


class TestColdInferenceScoringConvention:
    """Pins the inference-vs-scoring convention at MODERATE lambda (where
    expElogbeta and beta_prob differ materially -- unlike a very sharp/very
    flat lambda, where digamma-exp shrinkage and the raw ratio nearly
    coincide and a swap bug would be numerically invisible). Independently
    reconstructs ll_full using the primitives directly, then shows the
    fully-swapped reconstruction (infer with beta_prob, score with
    expElogbeta) differs materially -- proving a production swap would be
    caught."""

    def test_ll_full_matches_correct_convention_and_swap_would_be_caught(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        c = 5.0
        seed = 42
        holdout_frac = 0.3
        max_iter, tol = 50, 1e-4

        lam = np.full((K, V), 0.05)
        lam[0, 0:4] = 2.0  # on-signature, MODERATE (not 1000/0.01)
        lam[1, 4:8] = 2.0
        Gamma = np.zeros((1, K))
        Sigma = np.eye(K)
        gp = {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}

        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        # Independently reconstruct both beta conventions from lambda.
        lam_rowsum = lam.sum(axis=1, keepdims=True)
        expElogbeta = np.exp(digamma(lam) - digamma(lam_rowsum))
        beta_prob = lam / lam_rowsum
        # Confirm the two conventions actually differ materially at this
        # lambda scale (otherwise the test below would have no power).
        assert np.max(np.abs(expElogbeta - beta_prob)) > 1e-2

        d = np.diag(Sigma)
        R = Sigma / np.sqrt(np.outer(d, d))
        allowed = part.allowed_indices(doc.groups)

        split = heldout_split(doc, holdout_frac=holdout_frac, seed=seed)
        assert split is not None
        visible, held_idx, held_cnt = split
        assert held_cnt.size > 0

        Sigma_inv_allowed = (1.0 / c) * safe_inverse(R[np.ix_(allowed, allowed)])

        # CORRECT convention: infer with expElogbeta, score with beta_prob.
        eta, _, _ = _stm_doc_inference(
            indices=visible.indices, counts=visible.counts,
            expElogbeta=expElogbeta, Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed,
            x=doc.x, max_iter=max_iter, tol=tol, allowed=allowed, reference=None,
        )
        theta = _gated_mode_theta(eta, allowed, K)
        ll_correct = _predictive_loglik(theta, beta_prob, held_idx, held_cnt)

        dg = doc_predictive_gain(
            doc, gp, part, c=c, seed=seed, holdout_frac=holdout_frac,
            lbfgs_max_iter=max_iter, lbfgs_tol=tol,
        )
        assert dg is not None
        assert dg.ll_full == pytest.approx(ll_correct, abs=1e-9)

        # SWAPPED convention: infer with beta_prob, score with expElogbeta.
        # This is what a swap bug in production would actually compute --
        # it must differ materially from ll_correct, or this test would be
        # vacuous (unable to distinguish correct from swapped).
        eta_swapped, _, _ = _stm_doc_inference(
            indices=visible.indices, counts=visible.counts,
            expElogbeta=beta_prob, Gamma=Gamma, Sigma_inv_allowed=Sigma_inv_allowed,
            x=doc.x, max_iter=max_iter, tol=tol, allowed=allowed, reference=None,
        )
        theta_swapped = _gated_mode_theta(eta_swapped, allowed, K)
        ll_swapped = _predictive_loglik(theta_swapped, expElogbeta, held_idx, held_cnt)

        assert abs(ll_swapped - ll_correct) > 1e-3


class TestNullDeltaBand:
    """``null_delta`` builds the permuted-topic NULL BAND: for a topic whose
    beta row has been shuffled across the vocabulary (so it carries no real
    word-signature, i.e. "explains nothing"), what Delta does it produce
    against the SAME held-out split used by ``doc_predictive_gain``? Because
    the Gaussian prior regularizes the MAP even when a topic is pure noise,
    this null Delta is small but NOT exactly 0 -- it is the model-generated
    presence threshold a real topic's Delta must clear, replacing any hard
    zero cutoff."""

    def test_null_mean_near_zero_and_small_relative_to_real_signature_delta(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain, null_delta

        gp = _sharp_global_params_wide_vocab()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])  # all topic-0 signature words

        real = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)
        assert real is not None
        assert real.delta[0] > 1.0  # the real signature Delta (see auto-floor test above)

        nulls = null_delta(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, n_perm=8, rng_seed=0)

        assert nulls is not None
        assert nulls.shape == (8,)
        assert np.all(np.isfinite(nulls) | np.isnan(nulls))

        mean_null = np.nanmean(nulls)
        # Much smaller than the real signature Delta...
        assert mean_null < 0.25 * real.delta[0]
        # ...and small in absolute terms too, but NOT exactly 0 -- the
        # Gaussian prior regularizes the MAP even for a noise topic, so the
        # null band has a small-but-nonzero mean by construction (this is
        # exactly why a hard zero threshold is wrong and the null band is
        # needed as the presence bar instead). Observed magnitude at this
        # fixture/seed is ~1e-3 (three orders of magnitude below the real
        # signature Delta of ~69), so 0.5 is a generous tolerance, not a
        # tight numerical pin.
        assert abs(mean_null) < 0.5

    def test_null_delta_is_deterministic_given_rng_seed(self):
        from spark_vi.mllib.topic.predictive_gain import null_delta

        gp = _sharp_global_params_wide_vocab()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        nulls_a = null_delta(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, n_perm=8, rng_seed=0)
        nulls_b = null_delta(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, n_perm=8, rng_seed=0)

        assert nulls_a is not None and nulls_b is not None
        np.testing.assert_array_equal(nulls_a, nulls_b)

    def test_null_delta_varies_with_different_rng_seed(self):
        """Non-vacuous determinism check: a DIFFERENT rng_seed must actually
        change which topic gets permuted / how, not just be accepted and
        ignored."""
        from spark_vi.mllib.topic.predictive_gain import null_delta

        gp = _sharp_global_params_wide_vocab()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        nulls_seed0 = null_delta(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, n_perm=8, rng_seed=0)
        nulls_seed1 = null_delta(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, n_perm=8, rng_seed=1)

        assert nulls_seed0 is not None and nulls_seed1 is not None
        assert not np.array_equal(nulls_seed0, nulls_seed1)

    def test_null_delta_too_short_doc_returns_none(self):
        """Mirrors doc_predictive_gain's degenerate-split guard exactly: a
        1-token document can't be split into a non-empty visible AND held
        half, so null_delta must return None rather than raising."""
        from spark_vi.mllib.topic.predictive_gain import null_delta

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = STMDocument(
            indices=np.array([0], dtype=np.int32),
            counts=np.array([1.0]),
            length=1, x=np.array([1.0]), groups=frozenset(),
        )

        nulls = null_delta(doc, gp, part, c=1.0, seed=0, n_perm=4, rng_seed=0)
        assert nulls is None


class TestCorpusPredictiveGainGated:
    """Tests for ``corpus_predictive_gain_gated`` (numpy) and its distributed
    twin ``corpus_predictive_gain_gated_rdd``: per-topic aggregation of the
    per-document COLD gain (``doc_predictive_gain``) and permuted null
    (``null_delta``) across a gated corpus. See
    spark_vi/mllib/topic/predictive_gain.py for the full design (depth as
    Sigma-num/Sigma-den, within-group denominators, the per-document paired-
    null presence decision)."""

    @staticmethod
    def _corpus(seed=0, D=120):
        from tests._stm_synth import synthetic_gated_corpus

        docs, planted, part = synthetic_gated_corpus(
            groups=("a", "b"), fg_per_group=1, bg_k=2, V=200, D=D,
            doc_len=60, bg_frac=0.5, seed=seed,
        )
        K = part.K
        gp = {
            "lambda": planted * 500.0,
            "Gamma": np.zeros((1, K)),
            "Sigma": np.eye(K),
        }
        return docs, part, gp, K

    def test_signature_topic_exceeds_diluted_background_and_shapes_and_within_group(self):
        """Signature check: group a's foreground topic (used by EVERY group-a
        doc) must show higher mean_gain than a background topic (used by only
        ~half of group a's docs, since synthetic_gated_corpus's docs mix ONE
        of the bg_k background topics with their group's single foreground
        topic -- the other background topic is allowed but never the actual
        generator, diluting its average with near-zero-Delta docs). Also
        checks shapes and the within-group count_k denominator."""
        from spark_vi.mllib.topic.predictive_gain import corpus_predictive_gain_gated

        docs, part, gp, K = self._corpus()
        result = corpus_predictive_gain_gated(docs, gp, part, c=1.0, reference=None, seed=0)

        fg_a = int(part.block_indices("a")[0])
        bg0 = int(part.background_indices()[0])

        assert result["mean_gain"].shape == (K,)
        assert result["depth"].shape == (K,)
        assert result["depth_num"].shape == (K,)
        assert result["depth_den"].shape == (K,)
        assert result["presence"].shape == (K,)
        assert result["prominence_hist"].shape == (K, 50)
        assert result["length_corr"].shape == (K,)
        assert result["dedup_mean_gain"].shape == (K,)
        assert result["count_k"].shape == (K,)

        assert result["mean_gain"][fg_a] > result["mean_gain"][bg0]

        depth_den_positive = result["depth_den"] > 0
        assert depth_den_positive.any()
        assert np.all(np.isfinite(result["depth"][depth_den_positive]))

        # Within-group denominator: fg_a's count_k must equal the number of
        # GROUP-A documents exactly (background-only / group-b docs never
        # touch it, since it is not in their allowed set).
        n_group_a = sum(1 for d in docs if d.groups == frozenset({"a"}))
        assert result["count_k"][fg_a] == n_group_a

        assert result["n_docs"] == len(docs)

    def test_depth_is_summed_ratio_not_per_doc_mean(self):
        """Non-vacuous pin on the depth formula: depth[k] must equal
        depth_num[k] / depth_den[k] computed from the RETURNED totals (proves
        the final division uses SUMMED numerator/denominator, never an
        average of per-doc ratios)."""
        from spark_vi.mllib.topic.predictive_gain import corpus_predictive_gain_gated

        docs, part, gp, K = self._corpus(D=60)
        result = corpus_predictive_gain_gated(docs, gp, part, c=1.0, seed=0)

        mask = result["depth_den"] > 0
        assert mask.any()
        expected = result["depth_num"][mask] / result["depth_den"][mask]
        np.testing.assert_allclose(result["depth"][mask], expected, rtol=1e-10)

    def test_depth_is_nan_iff_depth_den_not_positive(self):
        """Pins the depth_den > 0 guard (not != 0): depth is only defined as
        a SHARE of total predictive structure when that total is positive --
        Delta can be negative, so a `!= 0` guard would let a negative
        depth_den through and produce a nonsensical negative/exploded
        "share" instead of nan. This asserts the invariant directly on the
        returned arrays against the existing corpus fixture (non-vacuous as
        long as at least one topic has depth_den > 0, which
        test_depth_is_summed_ratio_not_per_doc_mean already establishes for
        this same fixture)."""
        from spark_vi.mllib.topic.predictive_gain import corpus_predictive_gain_gated

        docs, part, gp, K = self._corpus(D=60)
        result = corpus_predictive_gain_gated(docs, gp, part, c=1.0, seed=0)

        depth_den = result["depth_den"]
        depth = result["depth"]
        assert (depth_den > 0).any()  # non-vacuous: some k actually exercises the finite branch

        for k in range(K):
            if depth_den[k] > 0:
                assert np.isfinite(depth[k])
                assert depth[k] == pytest.approx(
                    result["depth_num"][k] / depth_den[k], rel=1e-10
                )
            else:
                assert np.isnan(depth[k])

    def test_empty_corpus_raises(self):
        from spark_vi.mllib.topic.predictive_gain import corpus_predictive_gain_gated

        _, part, gp, _ = self._corpus(D=1)
        with pytest.raises(ValueError):
            corpus_predictive_gain_gated([], gp, part, c=1.0)


class TestCorpusPredictiveGainGatedRddParity:
    def test_numpy_rdd_parity(self, spark):
        from spark_vi.mllib.topic.predictive_gain import (
            corpus_predictive_gain_gated, corpus_predictive_gain_gated_rdd,
        )

        docs, part, gp, K = TestCorpusPredictiveGainGated._corpus(seed=1, D=16)

        expected = corpus_predictive_gain_gated(docs, gp, part, c=1.0, seed=0)

        rdd = spark.sparkContext.parallelize(docs, 4)
        result = corpus_predictive_gain_gated_rdd(
            rdd, gp, part, c=1.0, seed=0, sample_cap=None,
        )

        assert result["n_docs"] == expected["n_docs"]
        np.testing.assert_array_equal(result["count_k"], expected["count_k"])
        np.testing.assert_array_equal(result["prominence_hist"], expected["prominence_hist"])
        for field in ("mean_gain", "depth", "presence", "length_corr", "dedup_mean_gain"):
            np.testing.assert_allclose(
                result[field], expected[field], rtol=1e-8, equal_nan=True
            )
        np.testing.assert_allclose(
            result["prominence_bin_edges"], expected["prominence_bin_edges"], rtol=1e-12,
        )
        assert result["null_band"]["n"] == expected["null_band"]["n"]
        assert result["observed_delta_range"] == pytest.approx(
            expected["observed_delta_range"]
        )

    def test_empty_rdd_raises(self, spark):
        from spark_vi.mllib.topic.predictive_gain import corpus_predictive_gain_gated_rdd

        _, part, gp, _ = TestCorpusPredictiveGainGated._corpus(D=1)
        empty = spark.sparkContext.parallelize([], numSlices=1)
        with pytest.raises(ValueError):
            corpus_predictive_gain_gated_rdd(empty, gp, part, c=1.0)


# ---------------------------------------------------------------------------
# Task 4: the warm-start one-Newton-step DOWNDATE (fast=True) + the real-data
# cold-solve discrepancy audit.
#
# The fast path re-scores each topic ablation by polishing the full mode with a
# few warm-started Newton steps instead of a cold L-BFGS re-solve, with a
# fallback to the exact cold solve when the warm-start Newton stalls (a dropped
# high-mass topic pushing the objective into its non-convex region). It is an
# APPROXIMATION validated against the COLD oracle.
#
# The agreement fixtures below deliberately use GENUINE-MIXTURE documents (held
# tokens explained by several remaining topics), the regime where the cold
# L-BFGS oracle actually reaches the mode and is therefore a trustworthy
# reference. (On the single-planted-topic ``synthetic_gated_corpus`` docs,
# removing the sole generating topic leaves a pathologically steep objective on
# which cold L-BFGS-B itself terminates abnormally at a non-stationary point --
# there the downdate can be MORE accurate than the oracle, and cold-vs-fast
# "disagreement" is really cold being wrong. Surfacing exactly that on real
# data is the job of ``predictive_gain_downdate_audit``.)
# ---------------------------------------------------------------------------


def _gated_mixture_model():
    """A 4-topic gated model (2 background + 1 foreground per group a/b) on a
    disjoint 12-word vocab at MODERATE concentration (lambda 50 on-signature).
    Returns (gp, part, ia, ib) where ia/ib are the foreground topic indices for
    groups a/b. Moderate lambda + genuine-mixture documents keep the ablated
    per-doc objective well-conditioned, so the cold L-BFGS oracle converges to
    the true mode and is a valid reference for the downdate."""
    part = TopicBlockPartition(
        group_var="g", background_k=2, foreground=(("a", 1), ("b", 1)))
    K = part.K
    V = 12
    lam = np.full((K, V), 0.05)
    lam[0, 0:3] = 50.0   # background topic 0 -> words 0,1,2
    lam[1, 3:6] = 50.0   # background topic 1 -> words 3,4,5
    ia = int(part.block_indices("a")[0])
    ib = int(part.block_indices("b")[0])
    lam[ia, 6:9] = 50.0  # group-a foreground -> words 6,7,8
    lam[ib, 9:12] = 50.0  # group-b foreground -> words 9,10,11
    gp = {"lambda": lam, "Gamma": np.zeros((1, K)), "Sigma": np.eye(K)}
    return gp, part, ia, ib


def _mix_doc(indices, counts, groups):
    idx = np.asarray(indices, dtype=np.int32)
    cnts = np.asarray(counts, dtype=np.float64)
    return STMDocument(
        indices=idx, counts=cnts, length=int(cnts.sum()),
        x=np.array([1.0]), groups=groups)


def _three_topic_dominant_doc():
    """A non-gated 3-topic disjoint-sharp model and a document DOMINATED by
    topic 0 (theta_full[0] ~ 0.71) but with real minority mass on topics 1,2.
    Removing the dominant topic forces a large, locally non-convex softmax
    renormalization that a SINGLE Newton step under-captures -- the fixture the
    high-mass multi-step trigger is exercised on."""
    K3, V3 = 3, 6
    lam = np.full((K3, V3), 0.01)
    lam[0, 0:2] = 1000.0
    lam[1, 2:4] = 1000.0
    lam[2, 4:6] = 1000.0
    gp = {"lambda": lam, "Gamma": np.zeros((1, K3)), "Sigma": np.eye(K3)}
    part = TopicBlockPartition(group_var="g", background_k=K3, foreground=())
    doc = _mix_doc([0, 1, 2, 3, 4, 5], [20, 20, 3, 3, 2, 2], frozenset())
    return gp, part, doc


class TestFastDowndateAgreesWithCold:
    """The GATE: fast=True (warm-start Newton downdate) must reproduce the COLD
    oracle's per-topic Delta within tolerance -- tightly for high-mass topics
    (where the multi-step trigger does its work)."""

    def test_fast_matches_cold_sharp_two_topic_fixture(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        cold = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, fast=False)
        fast = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0, fast=True)

        assert cold is not None and fast is not None
        np.testing.assert_allclose(fast.delta, cold.delta, atol=1e-2)
        for p, k in enumerate(cold.allowed):
            if cold.theta_full[k] > 0.2:
                assert abs(fast.delta[p] - cold.delta[p]) < 3e-3

    def test_fast_matches_cold_gated_mixture_all_high_mass(self):
        """A gated group-a document that is a genuine 3-way mixture of the two
        background topics and group a's foreground topic (each theta ~1/3, all
        above the high-mass bound). Every ablation forces a real
        renormalization, so this exercises the multi-step trigger on a GATED
        doc; the downdate must still track cold to the tight high-mass
        tolerance."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp, part, ia, ib = _gated_mixture_model()
        # words 0,1 (bg0), 3,4 (bg1), 6,7 (fg a) -> a genuine 3-topic mixture.
        doc = _mix_doc([0, 1, 3, 4, 6, 7], [8, 8, 6, 6, 7, 7], frozenset({"a"}))

        for c in (0.5, 1.0, 3.0):
            cold = doc_predictive_gain(doc, gp, part, c=c, seed=0, fast=False)
            fast = doc_predictive_gain(doc, gp, part, c=c, seed=0, fast=True)
            assert cold is not None and fast is not None
            # gated: only background + group-a foreground are allowed.
            assert set(cold.allowed.tolist()) == {0, 1, ia}
            np.testing.assert_allclose(fast.delta, cold.delta, atol=1e-2)
            # all three allowed topics carry > 0.2 mass here -> tight tolerance.
            for p, k in enumerate(cold.allowed):
                assert cold.theta_full[k] > 0.2
                assert abs(fast.delta[p] - cold.delta[p]) < 3e-3


class TestFastFalseIsUnchanged:
    """fast=False must be byte-identical to the default (Task-1) behavior."""

    def test_fast_false_byte_identical_to_default(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        default = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)
        explicit = doc_predictive_gain(
            doc, gp, part, c=_C_WEAK_PRIOR, seed=0, fast=False)

        np.testing.assert_array_equal(default.delta, explicit.delta)
        np.testing.assert_array_equal(default.dedup_delta, explicit.dedup_delta)
        assert default.ll_full == explicit.ll_full
        np.testing.assert_array_equal(default.theta_full, explicit.theta_full)


class TestFastHighMassMultiStepIsLoadBearing:
    """The high-mass multi-step trigger must do real work: on a document
    dominated by one topic, ablating that topic with a SINGLE Newton step
    leaves a materially larger cold-vs-fast discrepancy than with the default
    3-step cap. (Vacuous only if one step already matched -- the fixture is
    engineered so it does not.)"""

    def test_three_vs_one_step_on_dominant_topic(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp, part, doc = _three_topic_dominant_doc()

        cold = doc_predictive_gain(doc, gp, part, c=1.0, seed=0, fast=False)
        fast1 = doc_predictive_gain(
            doc, gp, part, c=1.0, seed=0, fast=True, max_fast_steps=1)
        fast3 = doc_predictive_gain(
            doc, gp, part, c=1.0, seed=0, fast=True, max_fast_steps=3)

        p0 = int(np.nonzero(cold.allowed == 0)[0][0])
        assert cold.theta_full[0] > 0.2   # topic 0 is the dominant, high-mass topic

        disc1 = abs(fast1.delta[p0] - cold.delta[p0])
        disc3 = abs(fast3.delta[p0] - cold.delta[p0])

        assert disc1 > 1e-5             # non-vacuous: one step is materially off
        assert disc3 < 0.1 * disc1      # three steps close most of that gap


class TestDowndateAudit:
    """``predictive_gain_downdate_audit`` returns the per-topic aggregate
    cold-vs-fast discrepancy on a small in-memory corpus."""

    @staticmethod
    def _mixture_corpus(n=12, seed=0):
        gp, part, ia, ib = _gated_mixture_model()
        rng = np.random.default_rng(seed)
        docs = []
        for _ in range(n):
            g = "a" if rng.random() < 0.5 else "b"
            fgw = [6, 7] if g == "a" else [9, 10]
            idx = [0, 1, 3, 4] + fgw
            cnts = rng.integers(4, 10, size=len(idx)).astype(float)
            docs.append(_mix_doc(idx, cnts, frozenset({g})))
        return docs, gp, part

    def test_audit_returns_small_per_topic_discrepancy(self):
        from spark_vi.mllib.topic.predictive_gain import predictive_gain_downdate_audit

        docs, gp, part = self._mixture_corpus()
        K = part.K

        audit = predictive_gain_downdate_audit(docs, gp, part, c=1.0, seed=0)

        assert audit["max_abs_discrepancy"].shape == (K,)
        assert audit["mean_abs_discrepancy"].shape == (K,)
        assert isinstance(audit["max_abs_overall"], float)
        assert audit["n_docs_audited"] > 0
        assert audit["n_docs_audited"] == len(docs)
        # On this well-conditioned (cold-reliable) corpus the downdate tracks
        # the oracle very tightly.
        assert audit["max_abs_overall"] < 5e-2

    def test_audit_skips_degenerate_docs(self):
        """A 1-token document (degenerate split -> both paths return None) must
        not be audited or crash the aggregation."""
        from spark_vi.mllib.topic.predictive_gain import predictive_gain_downdate_audit

        docs, gp, part = self._mixture_corpus(n=4)
        tiny = STMDocument(
            indices=np.array([0], dtype=np.int32), counts=np.array([1.0]),
            length=1, x=np.array([1.0]), groups=frozenset({"a"}))
        audit = predictive_gain_downdate_audit(
            docs + [tiny], gp, part, c=1.0, seed=0)
        assert audit["n_docs_audited"] == len(docs)  # the tiny doc is skipped
