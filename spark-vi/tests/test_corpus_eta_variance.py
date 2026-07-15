"""Tests for the distributed empirical per-topic eta-variance export pass.

corpus_eta_variance_gated{,_rdd} runs the same per-doc Laplace E-step as
infer_local to get eta_hat over each doc's allowed (gated) topic set, then
accumulates a per-topic streaming mean + M2 (Welford) over documents. The core
accumulate/combine logic is unit-tested directly (no Spark needed); the numpy
in-memory driver is tested against a synthetic gated corpus with a known
planted spread; and (Spark being available in this environment) the `_rdd`
path is tested end-to-end against the numpy oracle.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument


# ---------------------------------------------------------------------------
# Welford accumulate + combine (the core distributed-variance logic)
# ---------------------------------------------------------------------------

class TestWelfordAccumulateAndCombine:
    def test_single_partition_matches_np_var_ddof1(self):
        from spark_vi.mllib.topic.stm import _welford_update, _welford_combine

        rng = np.random.default_rng(0)
        K = 4
        etas = rng.normal(loc=2.0, scale=3.0, size=(20, K))

        n = np.zeros(K, dtype=np.float64)
        mean = np.zeros(K, dtype=np.float64)
        M2 = np.zeros(K, dtype=np.float64)
        for row in etas:
            allowed = np.arange(K, dtype=np.int64)
            n, mean, M2 = _welford_update(n, mean, M2, row, allowed)

        var = np.where(n > 1, M2 / np.maximum(n - 1, 1), 0.0)
        np.testing.assert_allclose(var, etas.var(axis=0, ddof=1), rtol=1e-10)

    def test_two_partition_combine_matches_np_var_ddof1(self):
        from spark_vi.mllib.topic.stm import _welford_update, _welford_combine

        rng = np.random.default_rng(1)
        K = 3
        etas = rng.normal(loc=-1.0, scale=2.0, size=(30, K))
        allowed = np.arange(K, dtype=np.int64)

        def accumulate(rows):
            n = np.zeros(K, dtype=np.float64)
            mean = np.zeros(K, dtype=np.float64)
            M2 = np.zeros(K, dtype=np.float64)
            for row in rows:
                n, mean, M2 = _welford_update(n, mean, M2, row, allowed)
            return n, mean, M2

        part_a = accumulate(etas[:11])
        part_b = accumulate(etas[11:])
        n, mean, M2 = _welford_combine(part_a, part_b)

        var = np.where(n > 1, M2 / np.maximum(n - 1, 1), 0.0)
        np.testing.assert_allclose(var, etas.var(axis=0, ddof=1), rtol=1e-10)
        np.testing.assert_allclose(mean, etas.mean(axis=0), rtol=1e-10)

    def test_many_partition_treereduce_style_combine_matches_np_var(self):
        # Simulates a tree-reduce over many small partitions (as treeReduce
        # would do), verifying the combine is associative/order-independent.
        from spark_vi.mllib.topic.stm import _welford_update, _welford_combine

        rng = np.random.default_rng(2)
        K = 5
        etas = rng.normal(loc=0.5, scale=1.5, size=(37, K))
        allowed = np.arange(K, dtype=np.int64)

        chunks = np.array_split(etas, 7)
        accs = []
        for chunk in chunks:
            n = np.zeros(K, dtype=np.float64)
            mean = np.zeros(K, dtype=np.float64)
            M2 = np.zeros(K, dtype=np.float64)
            for row in chunk:
                n, mean, M2 = _welford_update(n, mean, M2, row, allowed)
            accs.append((n, mean, M2))

        # Combine in a non-trivial tree order (not left-to-right fold).
        combined = _welford_combine(
            _welford_combine(accs[0], accs[1]),
            _welford_combine(
                _welford_combine(accs[2], accs[3]),
                _welford_combine(accs[4], _welford_combine(accs[5], accs[6])),
            ),
        )
        n, mean, M2 = combined
        var = np.where(n > 1, M2 / np.maximum(n - 1, 1), 0.0)
        np.testing.assert_allclose(var, etas.var(axis=0, ddof=1), rtol=1e-9)

    def test_topic_skipped_when_not_allowed_does_not_pollute_variance(self):
        # A doc that does not allow topic k must not update topic k's
        # accumulator at all (not even with a zero).
        from spark_vi.mllib.topic.stm import _welford_update

        K = 2
        n = np.zeros(K, dtype=np.float64)
        mean = np.zeros(K, dtype=np.float64)
        M2 = np.zeros(K, dtype=np.float64)
        eta = np.array([10.0, -np.inf])   # topic 1 disallowed
        allowed = np.array([0], dtype=np.int64)
        n, mean, M2 = _welford_update(n, mean, M2, eta, allowed)
        assert n[0] == 1
        assert n[1] == 0
        assert mean[0] == 10.0


# ---------------------------------------------------------------------------
# Numpy in-memory driver: corpus_eta_variance_gated
# ---------------------------------------------------------------------------

def _make_global_params(K, V, Gamma, Sigma, seed=0):
    """A lambda whose expElogbeta is (numerically) close to uniform, so
    document token evidence contributes only a mild pull on eta_hat away
    from its prior mean Gamma^T x -- keeping the planted covariate-driven
    eta spread visible in the recovered eta_hat."""
    rng = np.random.default_rng(seed)
    lam = np.full((K, V), 50.0) + rng.normal(scale=0.01, size=(K, V))
    lam = np.abs(lam) + 1.0
    return {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}


class TestCorpusEtaVarianceGatedNumpy:
    def _build_corpus(self, *, seed=0):
        """Two background topics + one foreground group 'A' with one topic
        (K=3, reference topic 0 pinned). Background-only docs never see the
        foreground topic; group-A docs see background + their own foreground
        topic. x is a scalar covariate drawn per-doc so Gamma^T x varies and
        real between-document eta spread exists to recover."""
        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1),))
        K = part.K  # 3: [0, 1] background, [2] foreground A
        V = 40
        rng = np.random.default_rng(seed)
        Gamma = np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 4.0]])  # P=2 (intercept + x)

        docs = []
        n_bg_only = 30
        n_group_a = 30
        for _ in range(n_bg_only):
            x_val = rng.normal()
            toks = rng.integers(0, V, size=25)
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0, x_val]),
                groups=frozenset(),
            ))
        for _ in range(n_group_a):
            x_val = rng.normal()
            toks = rng.integers(0, V, size=25)
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0, x_val]),
                groups=frozenset({"A"}),
            ))
        Sigma = np.eye(K) * 1.0
        gp = _make_global_params(K, V, Gamma, Sigma, seed=seed)
        return docs, part, gp, K

    def test_variance_in_right_ballpark_for_foreground_topic(self):
        from spark_vi.mllib.topic.stm import corpus_eta_variance_gated

        docs, part, gp, K = self._build_corpus(seed=3)
        var = corpus_eta_variance_gated(docs, gp, part, reference=0)

        assert var.shape == (K,)
        assert np.all(var >= 0.0)
        # Reference topic (0) is pinned eta=0 for every doc -> exactly 0 variance.
        assert var[0] == 0.0
        # Foreground topic 2's eta is driven by Gamma[:,2]^T x = 4*x_val, x_val ~
        # N(0,1) plus small Sigma=1 prior noise damped by data evidence, so its
        # variance should be substantial (order Gamma_row_norm^2 = 16, generously
        # bounded) and clearly nonzero.
        assert var[2] > 1.0

    def test_background_only_docs_do_not_contribute_to_foreground_variance(self):
        # Excluding the group-A docs must change the foreground topic's
        # variance answer (proves gating actually restricts which docs feed
        # which topic's accumulator, not just a post-hoc mask).
        from spark_vi.mllib.topic.stm import corpus_eta_variance_gated

        docs, part, gp, K = self._build_corpus(seed=5)
        bg_only_docs = [d for d in docs if not d.groups]
        group_a_docs = [d for d in docs if d.groups]
        assert bg_only_docs and group_a_docs

        var_all = corpus_eta_variance_gated(docs, gp, part, reference=0)
        # A background-only doc's allowed set excludes topic 2 entirely, so
        # feeding ONLY background-only docs must yield a topic-2 variance of
        # exactly 0 (zero documents ever touch that accumulator).
        var_bg_only = corpus_eta_variance_gated(bg_only_docs, gp, part, reference=0)
        assert var_bg_only[2] == 0.0
        assert var_all[2] > 0.0
        assert var_all[2] != var_bg_only[2]

    def test_reference_topic_forced_zero_even_if_not_exactly_zero_numerically(self):
        from spark_vi.mllib.topic.stm import corpus_eta_variance_gated

        docs, part, gp, K = self._build_corpus(seed=9)
        var = corpus_eta_variance_gated(docs, gp, part, reference=0)
        assert var[0] == 0.0

    def test_topic_allowed_by_zero_documents_gets_zero_variance(self):
        # A foreground topic for a group with no docs in the corpus at all.
        from spark_vi.mllib.topic.stm import corpus_eta_variance_gated

        part = TopicBlockPartition(
            group_var="g", background_k=1, foreground=(("A", 1), ("B", 1)))
        K = part.K  # 3
        V = 20
        rng = np.random.default_rng(11)
        Gamma = np.zeros((1, K))
        Sigma = np.eye(K)
        gp = _make_global_params(K, V, Gamma, Sigma, seed=1)
        docs = []
        for _ in range(10):
            toks = rng.integers(0, V, size=15)
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0]), groups=frozenset({"A"}),
            ))
        # No doc ever has group "B" -> topic for B (index 2) allowed by 0 docs.
        var = corpus_eta_variance_gated(docs, gp, part, reference=0)
        assert var[2] == 0.0


# ---------------------------------------------------------------------------
# Distributed _rdd path: end-to-end against the numpy oracle
# ---------------------------------------------------------------------------

class TestCorpusEtaVarianceGatedRDD:
    def test_rdd_matches_numpy_oracle(self, spark):
        from spark_vi.mllib.topic.stm import (
            corpus_eta_variance_gated, corpus_eta_variance_gated_rdd,
        )

        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1), ("B", 1)))
        K = part.K  # 4
        V = 30
        rng = np.random.default_rng(21)
        Gamma = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 3.0, -2.0]])
        Sigma = np.eye(K)
        gp = _make_global_params(K, V, Gamma, Sigma, seed=2)

        docs = []
        groups_cycle = [frozenset(), frozenset({"A"}), frozenset({"B"})]
        for i in range(45):
            g = groups_cycle[i % 3]
            x_val = rng.normal()
            toks = rng.integers(0, V, size=20)
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0, x_val]), groups=g,
            ))

        expected = corpus_eta_variance_gated(docs, gp, part, reference=0)

        rdd = spark.sparkContext.parallelize(docs, numSlices=4)
        result = corpus_eta_variance_gated_rdd(rdd, gp, part, reference=0)

        np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-10)
        assert result[0] == 0.0

    def test_rdd_raises_on_empty(self, spark):
        from spark_vi.mllib.topic.stm import corpus_eta_variance_gated_rdd

        part = TopicBlockPartition(group_var="g", background_k=2, foreground=(("A", 1),))
        K = part.K
        gp = _make_global_params(K, 10, np.zeros((1, K)), np.eye(K), seed=0)
        empty = spark.sparkContext.parallelize([], numSlices=1)
        with pytest.raises(ValueError, match="empty"):
            corpus_eta_variance_gated_rdd(empty, gp, part)
