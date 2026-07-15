"""Tests for the iterated pooled generative-variance scale export pass.

corpus_eta_scale_gated{,_rdd} estimate a SINGLE data-driven scalar `c` for the
STM generative covariance Sigma_gen = c*R (R the fit's unit-diagonal block-wise
correlation), at export time with beta and R FROZEN. Each round runs the same
per-doc Laplace E-step as infer_local under prior Sigma = c*R, accumulates a
per-topic between-doc variance of eta_hat (Welford) PLUS the mean per-doc Laplace
posterior variance (diag of nu_d) -- the two terms of the law of total variance --
pools those per-topic totals over the free, observed topics into one scalar, and
iterates c toward the self-consistent value. Because c is a single pooled number
and beta is frozen, no topic can run away (insight 0033).

Tests: (a) bounded + climbs above the ~c=1 single-pass estimate on a known-scale
synthetic; (b) converges within max_iter (last step <= tol*c); (c) the reference
topic (eta==0) is excluded from the pool; (d) the _rdd path matches the numpy
oracle on a real local SparkContext.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument


def _planted_beta(K, V):
    """Peaked topic-word beta: topic k owns a disjoint signature block, so a
    document sampled from softmax(eta)@beta carries real token evidence that
    pulls its recovered eta_hat toward the planted eta (rather than collapsing
    to the prior mean under a flat likelihood)."""
    beta = np.full((K, V), 1e-3)
    blk = V // K
    for k in range(K):
        beta[k, k * blk:(k + 1) * blk] += 2.0
    beta /= beta.sum(axis=1, keepdims=True)
    return beta


def _make_global_params(K, V, Gamma, Sigma, *, beta=None, seed=0):
    """global_params with an INFORMATIVE lambda: lambda = beta * concentration,
    so expElogbeta is peaked at the planted topics and token evidence genuinely
    pulls each doc's eta_hat toward its generative eta. A near-uniform lambda
    (flat likelihood) would leave every eta_hat pinned at the prior mean and the
    recovered between-doc variance -- and hence the pooled scale -- would be ~0
    regardless of the true planted spread, defeating the test."""
    if beta is None:
        beta = _planted_beta(K, V)
    lam = beta * (500.0 * V) + 0.01
    return {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}


def _planted_scale_corpus(*, K_scale, seed=0, D_group=60):
    """Two background topics + one foreground group 'A' (K=3, reference topic 0
    pinned). eta over each doc's ALLOWED topics is drawn ~ N(0, K_scale * I) then
    softmaxed and used to sample codes from a planted (near-uniform) beta, so the
    TRUE generative eta spread on the free topics is ~K_scale > 1. The unit-prior
    single-pass estimate under-recovers this (posterior modes shrink toward 0),
    and the iterated scale climbs back up toward it."""
    part = TopicBlockPartition(
        group_var="g", background_k=2, foreground=(("A", 1),))
    K = part.K  # 3: [0,1] background, [2] foreground A
    V = 60
    rng = np.random.default_rng(seed)
    Gamma = np.zeros((1, K))                       # P=1 (intercept only): prior mean 0
    # Planted beta: disjoint signature block per topic so a doc's theta maps to
    # recoverable token evidence (not a flat likelihood that ignores eta).
    beta = _planted_beta(K, V)

    docs = []
    # Background-only docs (allowed = {0,1}) and group-A docs (allowed = {0,1,2}).
    for grp in (frozenset(), frozenset({"A"})):
        allowed = np.sort(part.allowed_indices(grp))
        for _ in range(D_group):
            eta = np.zeros(K)
            draw = rng.normal(scale=np.sqrt(K_scale), size=allowed.shape[0])
            eta[allowed] = draw
            eta[0] = 0.0                            # reference pinned at 0
            theta = np.zeros(K)
            theta[allowed] = np.exp(eta[allowed] - eta[allowed].max())
            theta /= theta.sum()
            toks = rng.choice(V, size=40, p=theta @ beta)
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0]), groups=grp))
    Sigma = np.eye(K)                              # R: unit-diagonal correlation
    gp = _make_global_params(K, V, Gamma, Sigma, beta=beta, seed=seed + 1)
    return docs, part, gp, K


# ---------------------------------------------------------------------------
# (a) bounded + climbs above the single-pass unit estimate; plausible band
# ---------------------------------------------------------------------------

class TestScaleBoundedAndClimbs:
    def test_scale_bounded_and_above_unit_pass(self):
        from spark_vi.mllib.topic.stm import (
            corpus_eta_scale_gated, corpus_eta_variance_gated,
        )

        K_scale = 4.0
        docs, part, gp, K = _planted_scale_corpus(K_scale=K_scale, seed=7)

        # ~c=1 baseline: one E-step pass under the unit prior, pooled mean over
        # the free, observed topics (mirrors what the iterated estimator does at
        # its first round, before c climbs).
        unit_var = corpus_eta_variance_gated(docs, gp, part, reference=0)
        free_obs = [1, 2]  # topic 0 is the reference; 1,2 are free & observed
        unit_pooled = float(np.mean([unit_var[k] for k in free_obs]))

        c = corpus_eta_scale_gated(docs, gp, part, reference=0)

        # (i) bounded -- no runaway.
        assert np.isfinite(c)
        assert c < 50.0
        # (ii) climbed above the single-pass unit-prior estimate.
        assert c > unit_pooled
        # (iii) plausible band around the planted scale (Laplace under-correction
        # means c lands below K_scale but well above 1).
        assert 1.5 < c < K_scale * 1.3


# ---------------------------------------------------------------------------
# (b) converges within max_iter (last step <= tol*c, not truncated)
# ---------------------------------------------------------------------------

class TestScaleConverges:
    def test_converges_within_max_iter(self):
        from spark_vi.mllib.topic.stm import corpus_eta_scale_gated

        docs, part, gp, K = _planted_scale_corpus(K_scale=4.0, seed=13)
        max_iter, tol = 15, 0.02

        # Reproduce the loop here to confirm it STOPPED (converged) rather than
        # ran out of iterations: track the trace and assert the final step is
        # within tol*c and the round count is < max_iter.
        trace = corpus_eta_scale_gated(
            docs, gp, part, reference=0, max_iter=max_iter, tol=tol,
            _return_trace=True)
        assert len(trace) >= 2
        assert len(trace) <= max_iter
        # Converged: the last increment is within tol of the last value.
        assert abs(trace[-1] - trace[-2]) <= tol * trace[-1]
        # Did not consume the full budget (a genuine early stop).
        assert len(trace) < max_iter


# ---------------------------------------------------------------------------
# (c) reference topic excluded from the pool
# ---------------------------------------------------------------------------

class TestReferenceExcluded:
    def test_reference_topic_not_in_pool(self):
        from spark_vi.mllib.topic.stm import corpus_eta_scale_gated

        # The reference topic has eta pinned at 0 -> ~0 between-doc variance and
        # 0 posterior variance. If it leaked into the pool it would drag the
        # mean DOWN. Run with reference=0; a hypothetical run that (wrongly)
        # included topic 0's ~0 total would give a strictly smaller c.
        docs, part, gp, K = _planted_scale_corpus(K_scale=4.0, seed=21)
        c_excl = corpus_eta_scale_gated(docs, gp, part, reference=0)

        # Emulate the polluted pool: average the same per-topic totals but
        # INCLUDE topic 0 (its ~0 total). We reconstruct per-topic totals from
        # the estimator's own final round via the trace-returning debug hook.
        totals = corpus_eta_scale_gated(
            docs, gp, part, reference=0, _return_totals=True)
        free_obs_totals = [totals[k] for k in (1, 2)]
        polluted = float(np.mean([totals[0]] + free_obs_totals))
        clean = float(np.mean(free_obs_totals))

        # The returned c equals the clean (reference-excluded) pool, and the
        # polluted pool would have been strictly smaller.
        np.testing.assert_allclose(c_excl, clean, rtol=1e-9)
        assert polluted < clean
        # topic 0's total is essentially zero (pinned).
        assert totals[0] < 1e-6


# ---------------------------------------------------------------------------
# (d) _rdd path matches the numpy oracle on a real local SparkContext
# ---------------------------------------------------------------------------

class TestScaleRDDParity:
    def test_rdd_matches_numpy_oracle(self, spark):
        from spark_vi.mllib.topic.stm import (
            corpus_eta_scale_gated, corpus_eta_scale_gated_rdd,
        )

        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1), ("B", 1)))
        K = part.K  # 4
        V = 40
        rng = np.random.default_rng(31)
        Gamma = np.zeros((1, K))
        beta = _planted_beta(K, V)
        Sigma = np.eye(K)
        gp = _make_global_params(K, V, Gamma, Sigma, beta=beta, seed=2)

        docs = []
        groups_cycle = [frozenset(), frozenset({"A"}), frozenset({"B"})]
        for i in range(48):
            g = groups_cycle[i % 3]
            allowed = np.sort(part.allowed_indices(g))
            eta = np.zeros(K)
            eta[allowed] = rng.normal(scale=np.sqrt(3.0), size=allowed.shape[0])
            eta[0] = 0.0
            theta = np.zeros(K)
            theta[allowed] = np.exp(eta[allowed] - eta[allowed].max())
            theta /= theta.sum()
            toks = rng.choice(V, size=30, p=theta @ beta)
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0]), groups=g))

        expected = corpus_eta_scale_gated(docs, gp, part, reference=0)

        rdd = spark.sparkContext.parallelize(docs, numSlices=4)
        result = corpus_eta_scale_gated_rdd(rdd, gp, part, reference=0)

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)

    def test_rdd_raises_on_empty(self, spark):
        from spark_vi.mllib.topic.stm import corpus_eta_scale_gated_rdd

        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1),))
        K = part.K
        gp = _make_global_params(K, 10, np.zeros((1, K)), np.eye(K), seed=0)
        empty = spark.sparkContext.parallelize([], numSlices=1)
        with pytest.raises(ValueError, match="empty"):
            corpus_eta_scale_gated_rdd(empty, gp, part)
