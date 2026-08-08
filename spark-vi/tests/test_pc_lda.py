"""Increment-1 recovery-parity gate for OnlinePCLDA (weight_y == 0).

The faithful VI port of ``analysis/pc``, increment 1, is the UNSUPERVISED SVI
path: at ``weight_y == 0`` the Prediction-Constrained objective is unsupervised
LDA-MAP, so ``OnlinePCLDA`` must (a) recover planted topics on the ``analysis/pc``
synthetic known-signal generator, (b) do so comparably to the exact in-memory
oracle ``PCTopicModel(weight_y=0)``, and (c) be equivalent to plain
``OnlineLDA`` (same code path — the ``weight_y == 0`` faithfulness gate).

This is RECOVERY PARITY, not numeric identity vs the oracle: the reference's
label-free local step is a NEF-MAP point estimate (100 unrolled exp-grad
steps), ``OnlineLDA``'s is mean-field CAVI Dirichlet γ. Same role, different
estimator — they agree on strong planted signal, not bit-for-bit (design
§"estimator gap"). So the oracle comparison is on matched-topic cosine, and the
OnlineLDA comparison IS an equivalence (identical estimator).

The planted corpus is the ``analysis/pc`` generator's ``_make_corpus`` (K_DOM
dominant disjoint-block topics + one predictive block); we reuse it verbatim and
fit with K = K_true so the unsupervised model can isolate every disjoint block.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

# The analysis/pc synthetic known-signal generator (reused verbatim) + its regime
# constants. analysis.pc.tests is a package; importing the module for its helper
# does not trigger collection here.
from analysis.pc.tests.test_synthetic_signal import (
    _make_corpus, SEED, D, V, K_DOM, SIG_BLOCK,
)

K_TRUE = K_DOM + 1  # generator plants K_DOM dominant blocks + 1 predictive block


def _planted_topics() -> np.ndarray:
    """Reconstruct the generator's (K_true, V) topic-word matrix.

    Mirrors ``_make_corpus``'s fixed, seed-independent block layout: K_DOM
    dominant topics on disjoint word blocks of the first V-SIG_BLOCK columns,
    plus one predictive topic owning the last SIG_BLOCK columns.
    """
    topics = np.full((K_TRUE, V), 0.01)
    dom_region = V - SIG_BLOCK
    dom_bl = dom_region // K_DOM
    for k in range(K_DOM):
        topics[k, k * dom_bl:(k + 1) * dom_bl] += 1.0
    topics[K_DOM, V - SIG_BLOCK:] += 1.0
    topics /= topics.sum(axis=1, keepdims=True)
    return topics


def _matched_cosine(beta_fit: np.ndarray, beta_true: np.ndarray) -> np.ndarray:
    """Hungarian-matched per-topic cosine between two (K, V) row-stochastic
    topic-word matrices. Returns the K matched cosines."""
    nf = beta_fit / np.linalg.norm(beta_fit, axis=1, keepdims=True)
    nt = beta_true / np.linalg.norm(beta_true, axis=1, keepdims=True)
    cos = nf @ nt.T
    fi, ti = linear_sum_assignment(-cos)
    return cos[fi, ti]


def _x_to_pc_docs(X: np.ndarray):
    """(D, V) dense count matrix -> list[PCDocument] (label-free placeholders)."""
    from spark_vi.models.topic.types import PCDocument
    docs = []
    for row in X:
        idx = np.nonzero(row)[0].astype(np.int32)
        cnt = row[idx].astype(np.float64)
        docs.append(PCDocument(
            indices=idx, counts=cnt, length=int(cnt.sum()),
            y=np.zeros(1), label_mask=np.zeros(1),
        ))
    return docs


def _fit_beta(model_cls_kwargs, docs, spark, cfg_kwargs):
    """Fit a topic VIModel via VIRunner over a Spark RDD of docs; return beta."""
    from spark_vi.core import VIConfig, VIRunner
    rdd = spark.sparkContext.parallelize(docs, numSlices=4).persist()
    rdd.count()
    model = model_cls_kwargs()
    result = VIRunner(model, config=VIConfig(**cfg_kwargs)).fit(rdd)
    rdd.unpersist(blocking=False)
    lam = result.global_params["lambda"]
    return lam / lam.sum(axis=1, keepdims=True), result


@pytest.mark.slow
def test_pc_lda_recovers_planted_topics_and_parity_with_oracle(spark):
    """OnlinePCLDA(weight_y=0) recovers the planted topics, comparably to the
    exact PCTopicModel(weight_y=0) oracle."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    from analysis.pc.model import PCTopicModel

    X, y = _make_corpus(SEED)
    n_te = int(0.3 * D)
    Xtr = X[:D - n_te]
    beta_true = _planted_topics()
    docs = _x_to_pc_docs(Xtr)

    # Full-batch fit (no sampling randomness) so the OnlineLDA-equivalence test
    # below can assert a tight match; K = K_true, disjoint blocks are recoverable.
    cfg = dict(max_iterations=60, learning_rate_tau0=10.0,
               learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-9)
    beta_pc, _ = _fit_beta(
        lambda: OnlinePCLDA(K=K_TRUE, vocab_size=V, C=1, weight_y=0.0,
                            random_seed=0),
        docs, spark, cfg,
    )
    cos_pc = _matched_cosine(beta_pc, beta_true)

    # Oracle: exact in-memory unsupervised LDA-MAP on the same train counts.
    oracle = PCTopicModel(K=K_TRUE, C=1, weight_y=0.0, alpha=1.1,
                          pi_iters=100, max_iter=150, seed=0).fit(Xtr, y[:D - n_te])
    cos_oracle = _matched_cosine(oracle.topics_, beta_true)

    print(f"\n[increment-1 recovery] OnlinePCLDA matched cosine: "
          f"min={cos_pc.min():.3f} mean={cos_pc.mean():.3f}  |  "
          f"oracle PCTopicModel(wy=0): min={cos_oracle.min():.3f} "
          f"mean={cos_oracle.mean():.3f}")

    # (a) OnlinePCLDA recovers the planted topics.
    assert cos_pc.mean() > 0.85, f"weak topic recovery: matched cosines {cos_pc}"
    assert cos_pc.min() > 0.60, f"a planted topic was missed: {cos_pc}"

    # (b) Recovery parity with the oracle: comparable mean matched cosine (not
    # numeric identity — CAVI vs NEF-MAP estimator gap).
    assert cos_pc.mean() > cos_oracle.mean() - 0.10, (
        f"OnlinePCLDA recovery {cos_pc.mean():.3f} not comparable to oracle "
        f"{cos_oracle.mean():.3f}"
    )


@pytest.mark.slow
def test_pc_lda_weight_y_zero_equivalent_to_online_lda(spark):
    """At weight_y == 0, OnlinePCLDA is OnlineLDA on the numbers (same code path,
    same seeds => same fitted topics up to matched cosine ~ 1.0)."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    from spark_vi.models.topic.lda import OnlineLDA

    X, _ = _make_corpus(SEED)
    docs = _x_to_pc_docs(X[:300])
    cfg = dict(max_iterations=40, learning_rate_tau0=10.0,
               learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-9)

    # OnlinePCLDA consumes PCDocuments; OnlineLDA consumes the same objects
    # (it only reads .indices/.counts), so both see identical inputs.
    beta_pc, _ = _fit_beta(
        lambda: OnlinePCLDA(K=K_TRUE, vocab_size=V, C=1, weight_y=0.0,
                            random_seed=0),
        docs, spark, cfg,
    )
    beta_lda, _ = _fit_beta(
        lambda: OnlineLDA(K=K_TRUE, vocab_size=V, random_seed=0),
        docs, spark, cfg,
    )
    cos = _matched_cosine(beta_pc, beta_lda)
    print(f"\n[increment-1 equivalence] OnlinePCLDA vs OnlineLDA matched cosine: "
          f"min={cos.min():.4f} mean={cos.mean():.4f}")
    assert cos.min() > 0.999, (
        f"OnlinePCLDA(weight_y=0) diverged from OnlineLDA: matched cosines {cos}"
    )
