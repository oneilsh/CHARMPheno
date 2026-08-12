"""DagClosureHead — the label-side HIERARCHY flavor of the PC supervised-head seam.

Mondo is a DAG (not a tree), so HSLDA's parent-gating (Perotte 2011, ICD-9 tree)
generalizes to a closure PRODUCT that counts each ancestor once (diamond-safe):

    log P(node_l = 1) = Σ_{a ∈ closure(l)} log σ(w_a · π) .

These tests pin (i) the closure matrix incl. a diamond, (ii) the monotone is-a
consistency P(child) ≤ P(parent) that the product buys by construction, (iii) that
the base autograd accumulators differentiate the DAG loss correctly (finite-
difference, both the topic-correction and the head gradient), and (iv) that the
model accepts the injected head and — since the DAG head has no closed-form Fisher —
'newton' gracefully degrades (no head_hess_stat emitted). Pure numpy except (iv).
"""
import numpy as np
import pytest

from spark_vi.models.topic.pc import DagClosureHead, SupervisedHead


# 0=root; 1,2 under root; 3 under BOTH 1 and 2 (a diamond). closure(3)={3,1,2,0}.
DIAMOND_PARENTS = [(), (0,), (0,), (1, 2)]


def _dag_docs(seed, C, V, n=6):
    """A tiny fixed PCDocument batch with C labels and a non-trivial observed mask,
    guaranteeing one fully-observed doc so the batch is non-degenerate."""
    from spark_vi.models.topic.types import PCDocument
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n):
        nnz = int(rng.integers(3, 7))
        idx = np.sort(rng.choice(V, size=nnz, replace=False)).astype(np.int32)
        cnt = rng.integers(1, 6, size=nnz).astype(np.float64)
        y = rng.integers(0, 2, size=C).astype(np.float64)
        mask = rng.integers(0, 2, size=C).astype(np.float64)
        docs.append(PCDocument(indices=idx, counts=cnt, length=int(cnt.sum()),
                               y=y, label_mask=mask))
    docs[0] = PCDocument(indices=docs[0].indices, counts=docs[0].counts,
                         length=docs[0].length, y=np.ones(C), label_mask=np.ones(C))
    return docs


def test_closure_matrix_is_diamond_safe():
    """closure(l) = l ∪ all ancestors, each counted ONCE even under a diamond."""
    h = DagClosureHead(DIAMOND_PARENTS)
    assert isinstance(h, SupervisedHead)
    assert h.C == 4
    M = h._closure_matrix
    assert M[0].tolist() == [1, 0, 0, 0]            # root: just itself
    assert M[1].tolist() == [1, 1, 0, 0]            # {1, 0}
    assert M[2].tolist() == [1, 0, 1, 0]            # {2, 0}
    assert M[3].tolist() == [1, 1, 1, 1]            # {3,1,2,0} — shared ancestor 0 once
    assert M[3].sum() == 4                          # not 5 (0 not double-counted)


def test_monotone_is_a_consistency():
    """P(child) ≤ P(parent) for ANY head weights — the product of ≤1 sigmoids over a
    growing closure can only shrink. This is the hierarchy baked into the model."""
    h = DagClosureHead(DIAMOND_PARENTS)
    rng = np.random.default_rng(0)
    K = 3
    for _ in range(50):
        W = rng.standard_normal((4, K)) * 2.0
        theta = rng.dirichlet(np.ones(K))
        ls = np.log(1.0 / (1.0 + np.exp(-(W @ theta))))
        P = np.exp(h._closure_matrix @ ls)
        assert P[1] <= P[0] + 1e-12                 # child ≤ parent along every edge
        assert P[2] <= P[0] + 1e-12
        assert P[3] <= P[1] + 1e-12 and P[3] <= P[2] + 1e-12


def test_dag_head_gradient_matches_finite_difference():
    """The base autograd accumulators differentiate the DAG-closure loss correctly:
    the accumulated (∂/∂topics_repr, ∂/∂w_CK) match a central finite-difference of
    the same batch NLL to max rel err <= 1e-5 — so a new flavor needs NO hand-derived
    gradient, only its per-doc loss."""
    h = DagClosureHead(DIAMOND_PARENTS)
    rng = np.random.default_rng(1)
    K, V, C = 4, 12, 4
    n_iters = 20
    alpha = np.full(K, 1.1)
    topics_repr = rng.random((K, V)) * 0.3 + 0.01      # expElogbeta-like, positive
    w_CK = rng.standard_normal((C, K)) * 0.4           # modest -> logP away from 0
    docs = _dag_docs(seed=0, C=C, V=V)

    _loss, grad_topics, grad_wCK = h.batch_value_and_grad(
        topics_repr, w_CK, docs, alpha, K, n_iters)

    def _fd(param, grad, evaluate, eps=1e-6):
        rel = []
        for (i, j) in zip(*np.nonzero(grad)):
            pp = param.copy(); pp[i, j] += eps
            pm = param.copy(); pm[i, j] -= eps
            num = (evaluate(pp) - evaluate(pm)) / (2 * eps)
            ana = grad[i, j]
            rel.append(abs(num - ana) / max(abs(ana), abs(num), 1e-8))
        return max(rel)

    err_topics = _fd(topics_repr, grad_topics,
                     lambda p: h.batch_value(p, w_CK, docs, alpha, K, n_iters))
    err_head = _fd(w_CK, grad_wCK,
                   lambda p: h.batch_value(topics_repr, p, docs, alpha, K, n_iters))
    assert err_topics <= 1e-5, f"DAG topic-gradient rel err {err_topics:.2e} > 1e-5"
    assert err_head <= 1e-5, f"DAG head-gradient rel err {err_head:.2e} > 1e-5"


def test_dag_head_has_no_closed_form_hessian():
    """The closure product couples nodes, so there is no logistic Fisher — the base
    None is inherited, which the model reads as 'no Newton, use SGD'."""
    h = DagClosureHead(DIAMOND_PARENTS)
    assert h.batch_hessian(None, None, [], None, 0, 0) is None


def test_model_accepts_injected_head_and_newton_degrades_to_sgd():
    """OnlinePCLDA(head=DagClosureHead(...)) wires the flavor in; because the head
    emits no Fisher, local_update omits head_hess_stat even under head_optimizer=
    'newton', so update_global falls back to the SGD head step (no KeyError)."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    K, V, C = 4, 12, 4
    head = DagClosureHead(DIAMOND_PARENTS)
    model = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=1.0, grad_cavi_iters=10,
                        head_optimizer="newton", head=head)
    gp = model.initialize_global(None)
    docs = _dag_docs(seed=3, C=C, V=V)
    stats = model.local_update(docs, gp)
    assert "grad_wCK_stat" in stats and "grad_topics_stat" in stats
    assert "head_hess_stat" not in stats            # DAG head -> no closed-form Fisher
    # update_global must not KeyError on the missing Hessian; head moves off zero.
    new_gp = model.update_global(gp, stats, learning_rate=0.5)
    assert new_gp["w_CK"].shape == (C, K)
    assert not np.allclose(new_gp["w_CK"], 0.0)


def test_model_rejects_head_label_count_mismatch():
    """A head whose C disagrees with the model's C is a wiring error — fail fast."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    with pytest.raises(ValueError, match="label count must match"):
        OnlinePCLDA(K=4, vocab_size=12, C=3, weight_y=1.0,
                    head=DagClosureHead(DIAMOND_PARENTS))   # head C=4 != model C=3
