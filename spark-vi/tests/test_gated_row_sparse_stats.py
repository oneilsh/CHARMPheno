"""Row-sparse partition statistics of the gated E-step (spark_vi.core.row_sparse).

The gated local_update returns `lambda_stats` as the touched topic rows only; these
pin that the values are BIT-IDENTICAL to the dense (K, V) accumulation, that
combine_stats / scaling reproduce the dense arithmetic exactly, and that a fit does
not depend on how the corpus is split into partitions (chunks)."""
import pickle

import numpy as np
import pytest
from scipy.special import digamma

from spark_vi.core import LocalRunner, VIConfig, VIModel
from spark_vi.core.row_sparse import RowSparseStats, add_row_sparse, as_dense
from spark_vi.dag.layout import DagLayout
from spark_vi.models.topic.dag_gated_lda import (DagGatedOnlineLDA,
                                                 expElogbeta_from_lambda)
from spark_vi.models.topic.lda import (_cavi_doc_inference, _dirichlet_kl,
                                       _hashed_gamma_init)
from spark_vi.models.topic.types import GatedBOWDocument

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
V = 30


def _docs(n=24, seed=0):
    """Background docs (empty frontier), single- and multi-label docs."""
    rng = np.random.default_rng(seed)
    frontiers = [frozenset(), frozenset({3}), frozenset({4, 5}), frozenset({6}),
                 frozenset({2}), frozenset()]
    docs = []
    for i in range(n):
        idx, cnt = np.unique(rng.integers(0, V, size=12), return_counts=True)
        docs.append(GatedBOWDocument(idx.astype(np.int32), cnt.astype(np.float64),
                                     int(cnt.sum()), frontier=frontiers[i % 6]))
    return docs


def _dense_reference(m, rows, gp):
    """The dense (K, V) gated E-step: one zeros((K, V)) and a scatter-add per doc."""
    gate_level = (m._gate_level(int(gp["gate_iter"])) if "gate_iter" in gp else None)
    expElogbeta = (gp["expElogbeta"] if "expElogbeta" in gp
                   else expElogbeta_from_lambda(gp["lambda"]))
    alpha = gp["alpha"]
    stats = np.zeros((m.K, m.V))
    ll = kl = 0.0
    node = np.zeros(m._block_sizes.shape[0]) if m.optimize_alpha else None
    for doc in rows:
        allowed = m.lay.allowed_set(doc.frontier, max_height=gate_level)
        g0 = _hashed_gamma_init(doc.indices, doc.counts, len(allowed),
                                m.gamma_shape, m.random_seed)
        gamma, eth, phi_norm, _ = _cavi_doc_inference(
            indices=np.arange(len(doc.indices)), counts=doc.counts,
            expElogbeta=expElogbeta[np.ix_(allowed, doc.indices)],
            alpha=alpha[allowed], gamma_init=g0,
            max_iter=m.cavi_max_iter, tol=m.cavi_tol)
        stats[np.ix_(allowed, doc.indices)] += np.outer(eth, doc.counts / phi_norm)
        ll += float(np.sum(doc.counts * np.log(phi_norm)))
        kl += _dirichlet_kl(gamma, alpha[allowed])
        if node is not None:
            np.add.at(node, m._topic_to_tied[allowed],
                      digamma(gamma) - digamma(gamma.sum()))
    return stats, ll, kl, node


def _models():
    hist = {frozenset(): 8, frozenset({3}): 4, frozenset({4, 5}): 4,
            frozenset({6}): 4, frozenset({2}): 4}
    lay = DagLayout(PARENT, n_bg=3, tpn=2)
    plain = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    multi = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0,
                              domains=[18, 12], optimize_alpha=True,
                              frontier_histogram=hist)
    leaves = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    leaves.set_gate_schedule(warm=2)                    # gate_iter 0: leaves only
    return {"plain": plain, "multi_domain": multi, "leaves_first": leaves}


@pytest.mark.parametrize("name", ["plain", "multi_domain", "leaves_first"])
@pytest.mark.parametrize("shipped", [False, True])
def test_row_sparse_e_step_is_bit_identical_to_dense(name, shipped):
    m = _models()[name]
    gp = m.initialize_global(None)
    if name == "leaves_first":
        assert m._gate_level(int(gp["gate_iter"])) == 0
    params = m.broadcast_params(gp) if shipped else gp
    docs = _docs()
    out = m.local_update(iter(docs), params)
    ref, ll, kl, node = _dense_reference(m, docs, params)
    st = out["lambda_stats"]
    assert isinstance(st, RowSparseStats) and st.shape == (m.K, m.V)
    assert np.array_equal(st.toarray(), ref)
    # Stored rows are exactly the touched ones; everything else is zero.
    assert np.array_equal(st.rows, np.flatnonzero(ref.any(axis=1)))
    assert float(out["doc_loglik_sum"]) == ll
    assert float(out["doc_theta_kl_sum"]) == kl
    if node is not None:
        assert np.array_equal(out["e_log_theta_node_sum"], node)


def test_partition_touching_few_topics_stays_small():
    """Background-only docs touch the background block alone: the partial (and its
    pickle, the executor -> driver payload) is the bg rows, not K rows."""
    lay = DagLayout(PARENT, n_bg=3, tpn=4)
    m = DagGatedOnlineLDA(lay, V, alpha=0.1, eta=0.02, random_seed=0)
    gp = m.initialize_global(None)
    bg_docs = [d for d in _docs() if not d.frontier]
    st = m.local_update(iter(bg_docs), m.broadcast_params(gp))["lambda_stats"]
    assert st.rows.tolist() == [0, 1, 2]
    assert len(pickle.dumps(st)) < 0.25 * m.K * m.V * 8
    back = pickle.loads(pickle.dumps(st))
    assert np.array_equal(back.toarray(), st.toarray())


def test_empty_partition_gives_empty_row_sparse():
    m = _models()["plain"]
    st = m.local_update(iter([]), m.initialize_global(None))["lambda_stats"]
    assert st.rows.size == 0 and st.shape == (m.K, V)
    assert np.array_equal(st.toarray(), np.zeros((m.K, V)))


# --- combine / scaling ---------------------------------------------------------

def _rs(rows, n=10, v=4, seed=0):
    rng = np.random.default_rng(seed)
    return RowSparseStats(np.array(rows), rng.random((len(rows), v)), n)


def test_sparse_plus_sparse_below_threshold_stays_sparse_and_exact():
    a, b = _rs([0, 2, 5], seed=1), _rs([2, 3], seed=2)
    s = add_row_sparse(a, b)
    assert isinstance(s, RowSparseStats) and s.rows.tolist() == [0, 2, 3, 5]
    assert np.array_equal(s.toarray(), a.toarray() + b.toarray())
    assert np.array_equal(add_row_sparse(b, a).toarray(), s.toarray())   # commutes


def test_sparse_union_past_half_k_densifies():
    a, b = _rs([0, 1, 2], seed=1), _rs([3, 4, 5], seed=2)     # union 6 > 0.5 * 10
    s = add_row_sparse(a, b)
    assert type(s) is np.ndarray
    assert np.array_equal(s, a.toarray() + b.toarray())
    # exactly half stays sparse
    assert isinstance(add_row_sparse(_rs([0, 1]), _rs([2, 3, 4])), RowSparseStats)


def test_sparse_with_dense_is_dense_and_exact_either_order():
    a = _rs([1, 7], seed=3)
    d = np.random.default_rng(4).random((10, 4))
    d0 = d.copy()
    for s in (add_row_sparse(a, d), add_row_sparse(d, a)):
        assert type(s) is np.ndarray
        assert np.array_equal(s, a.toarray() + d)
    assert np.array_equal(d, d0)                                 # inputs untouched
    assert np.array_equal(add_row_sparse(d, d), d + d)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        add_row_sparse(_rs([0], n=10), _rs([0], n=11))
    with pytest.raises(ValueError):
        add_row_sparse(_rs([0], n=10), np.zeros((11, 4)))


@pytest.mark.parametrize("s", [3.5, np.float64(3.5), np.array(3.5)])
def test_scalar_scaling_both_sides_matches_dense(s):
    a = _rs([1, 4], seed=5)
    for scaled in (a * s, s * a):
        assert isinstance(scaled, RowSparseStats)
        assert np.array_equal(scaled.toarray(), a.toarray() * s)


def test_no_matrix_or_elementwise_product_with_arrays():
    """`*` is scalar scaling only: an array operand is refused, never densified or
    matrix-multiplied behind the caller's back."""
    a = _rs([1, 4])
    with pytest.raises(TypeError):
        a * np.ones((10, 4))
    with pytest.raises(TypeError):
        np.ones((10, 4)) * a
    assert np.array_equal(as_dense(a), np.asarray(a))
    assert np.array_equal(as_dense(np.eye(2)), np.eye(2))


def test_model_combine_stats_sums_every_key():
    m = _models()["plain"]
    gp = m.initialize_global(None)
    docs = _docs()
    a = m.local_update(iter(docs[:6]), gp)
    b = m.local_update(iter(docs[6:]), gp)
    c = m.combine_stats(a, b)
    whole = m.local_update(iter(docs), gp)
    assert set(c) == set(whole)
    assert np.array_equal(as_dense(c["lambda_stats"]),
                          as_dense(a["lambda_stats"]) + as_dense(b["lambda_stats"]))
    assert float(c["n_docs"]) == float(whole["n_docs"]) == len(docs)
    np.testing.assert_allclose(as_dense(c["lambda_stats"]),
                               as_dense(whole["lambda_stats"]), rtol=1e-12)


# --- whole fits ---------------------------------------------------------------

class _DenseGated(DagGatedOnlineLDA):
    """The dense path: densified partials summed by the default VIModel combine."""

    def local_update(self, rows, global_params):
        out = super().local_update(rows, global_params)
        out["lambda_stats"] = out["lambda_stats"].toarray()
        return out

    def combine_stats(self, a, b):
        return VIModel.combine_stats(self, a, b)


@pytest.mark.parametrize("minibatch", [None, 0.5])
def test_local_fit_is_identical_across_chunkings_and_to_the_dense_path(minibatch):
    lay = DagLayout(PARENT, n_bg=3, tpn=2)
    docs = _docs(40)
    cfg = VIConfig(max_iterations=4, mini_batch_fraction=minibatch, random_seed=1,
                   convergence_tol=1e-12)

    def fit(cls, chunk):
        m = cls(lay, V, alpha=0.1, eta=0.02, random_seed=0, domains=[18, 12])
        m.set_gate_schedule(warm=1)
        return LocalRunner(m, cfg, chunk_size=chunk).fit(docs).global_params["lambda"]

    whole = fit(DagGatedOnlineLDA, None)
    for got in (fit(DagGatedOnlineLDA, 1), fit(_DenseGated, 1)):
        for d in whole:
            assert np.array_equal(got[d], whole[d])
    # Any chunking equals the dense path under the same chunking.
    sparse7, dense7 = fit(DagGatedOnlineLDA, 7), fit(_DenseGated, 7)
    for d in whole:
        assert np.array_equal(sparse7[d], dense7[d])
