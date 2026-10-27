"""LocalRunner agrees with VIRunner on Spark (ADR 0051 §2): the one slow check.

Both engines drive the same loop (`spark_vi.core.loop`) and the same model
methods, so they can differ only in the E-step's minibatch draw and in floating-
point summation order. Two claims, on the DAG-gated LDA with the leaves-first gate
schedule on a small planted corpus:

1. Full batch: identical up to summation order (λ to ~1e-8 relative), so the
   gate schedule, step sizes and scaling are the same code path.
2. Mini batch: the same fit in distribution. Spark's and NumPy's samplers draw
   different batches, so the check is relative: the Spark fit is no further from a
   local fit than two local fits with different sample seeds are from each other.
   Measured on this corpus: 0.096 vs 0.071 (threshold 0.16); a wrong sampling rate
   (0.05 for 0.2) lands at 0.35 and an ungated fit at 0.53. Step-size changes are
   within sampling noise here, which is why claim 1 pins the schedule exactly.
"""
import numpy as np
import pytest

from _stm_synth import dag_placement_corpus


def _corpus():
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.models.topic.types import GatedBOWDocument

    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    docs, labels, _ = dag_placement_corpus(
        parent=parent, node_prev={u: 1 for u in range(1, 7)},
        V=90, doc_len=50, seed=1)
    bow = []
    for d, y in zip(docs[:600], labels[:600]):
        idx, cnt = np.unique(np.asarray(d), return_counts=True)
        bow.append(GatedBOWDocument(idx.astype(np.int32), cnt.astype(np.float64),
                                    int(cnt.sum()), frontier=frozenset({int(y)})))
    return DagLayout(parent, n_bg=2, tpn=1), bow


def _model(lay):
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA

    m = DagGatedOnlineLDA(lay, 90, alpha=0.1, eta=0.02, random_seed=0)
    m.set_gate_schedule(warm=2)
    return m


def _topics(res):
    lam = res.global_params["lambda"]
    return lam / lam.sum(axis=1, keepdims=True)


def _dist(a, b):
    """Mean over topics of the L1 distance between topic-word distributions. No
    topic matching is needed: both fits start from the same seeded λ and the gate
    pins every non-background topic to its node."""
    return float(np.abs(a - b).sum(axis=1).mean())


@pytest.mark.slow
def test_local_runner_agrees_with_spark(spark):
    from spark_vi.core import LocalRunner, VIConfig, VIRunner

    lay, bow = _corpus()
    rdd = spark.sparkContext.parallelize(bow, 4).persist()
    rdd.count()

    # 1. Full batch: the same computation.
    cfg = VIConfig(max_iterations=8, convergence_tol=1e-12, learning_rate_tau0=4.0,
                   learning_rate_kappa=0.51)
    s = VIRunner(_model(lay), cfg).fit(rdd)
    loc = LocalRunner(_model(lay), cfg).fit(bow)
    np.testing.assert_allclose(loc.global_params["lambda"], s.global_params["lambda"],
                               rtol=1e-8)
    np.testing.assert_allclose(loc.elbo_trace, s.elbo_trace, rtol=1e-8)
    np.testing.assert_array_equal(loc.global_params["gate_open_at"],
                                  s.global_params["gate_open_at"])

    # 2. Mini batch: the same in distribution.
    def mb(seed):
        return VIConfig(max_iterations=30, convergence_tol=1e-12, mini_batch_fraction=0.2,
                        random_seed=seed, learning_rate_tau0=4.0,
                        learning_rate_kappa=0.51)

    s = _topics(VIRunner(_model(lay), mb(0)).fit(rdd))
    l0 = _topics(LocalRunner(_model(lay), mb(0)).fit(bow))
    l1 = _topics(LocalRunner(_model(lay), mb(1)).fit(bow))
    within, across = _dist(l0, l1), _dist(s, l0)
    print(f"local-vs-local {within:.4f}  spark-vs-local {across:.4f}")
    assert across <= 2.0 * within + 0.02, (across, within)
