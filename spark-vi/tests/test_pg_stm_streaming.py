"""Task 2/3 — distributed PG-STM: StreamingPGSTM (SVI) + the distributed Gibbs-Sigma pass.

The equivalence gate (test 1) is the load-bearing one: full-batch StreamingPGSTM must
reproduce the validated single-machine PGSTMVI.fit, since both call the SAME pg_mstep on
the SAME reduced sufficient statistics. Any divergence beyond float reduction-order drift
is a bug in the distribution, not the model.
"""
import numpy as np
import pytest

from spark_vi.mllib.topic.pg_stm import StreamingPGSTM
from spark_vi.models.topic.pg_stm import PGSTMVI, pg_stm_gibbs, stick_layout
from tests._stm_synth import gated_ln_corpus, gated_ln_corpus_stick, planted_recovery


def _corr(S):
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def test_streaming_fullbatch_matches_single_machine(spark):
    """Full-batch StreamingPGSTM == PGSTMVI.fit (same init, same M-step; only the
    partition reduction order differs -> float drift, bounded well under 1e-6)."""
    docs, part, _St, _b = gated_ln_corpus(
        group_weights={"A": 0.6, "B": 0.4}, fg_per_group=1, bg_k=3,
        V=40, D=120, doc_len=30, seed=0)
    P = docs[0].x.shape[0]
    ref = PGSTMVI(K=part.K, V=40, partition=part, P=P, n_iter=30, seed=0).fit(docs)
    rdd = spark.sparkContext.parallelize(docs, 4)
    got = StreamingPGSTM(K=part.K, V=40, partition=part, P=P, seed=0).fit(
        rdd, max_iter=30, batch="all")
    assert np.allclose(got["beta"], ref["beta"], atol=1e-6)
    assert np.allclose(got["Gamma"], ref["Gamma"], atol=1e-6)
    assert np.allclose(got["Sigma"], ref["Sigma"], atol=1e-6)


def test_streaming_minibatch_converges(spark):
    """Mini-batch SVI recovers the planted topics and returns a bounded, PD Sigma."""
    docs, part, _St, beta = gated_ln_corpus(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=1, bg_k=3,
        V=40, D=400, doc_len=30, seed=0)
    P = docs[0].x.shape[0]
    rdd = spark.sparkContext.parallelize(docs, 4)
    # 50% mini-batches with a faster-adapting schedule (tau0=16) recover the planted
    # topics in 200 iters (measured 5/5); the default tau0=64 needs more passes.
    out = StreamingPGSTM(K=part.K, V=40, partition=part, P=P, seed=0).fit(
        rdd, max_iter=200, batch=0.5, tau0=16.0)
    assert planted_recovery(out["beta"], beta) >= part.K - 2
    assert np.max(np.abs(out["Sigma"])) < 1e2
    np.linalg.cholesky(out["Sigma"])                       # PD
