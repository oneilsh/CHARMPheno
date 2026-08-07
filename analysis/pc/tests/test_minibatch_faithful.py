"""Faithfulness gates for the document-minibatch full-batch gradient assembly.

The faithful PC model (:class:`analysis.pc.model.PCTopicModel`) is the project's
correctness ANCHOR, so a scalability change that touches the fit is only admissible
if it is provably EXACT. ``fit`` used to build the objective/gradient over the whole
training corpus at once; reverse-mode autograd through the unrolled ``pi``-inference
retains every intermediate (each of ``pi_iters`` steps holds a ``D x V`` array), so
at real-corpus scale (tens of thousands of docs) the tape is tens of GB and OOMs the
driver. The fix assembles the SAME full-batch objective and gradient by accumulating
over contiguous document minibatches — exploiting that the total loss is a plain sum
of a per-document part (``loss_x + loss_pi + loss_y``) plus document-independent
global terms (``loss_topics``, ``loss_w``, the multinomial coefficient), all over a
single global ``scale = sum(X)``. Peak tape drops to one minibatch.

These gates prove the accumulation is exact — NOT approximate:

  1. **Objective equality**: the minibatched objective value equals the full-batch
     :func:`~analysis.pc.slda_reference.loss_from_param_vec` to <= 1e-9 relative, at
     several random parameter vectors, for several minibatch sizes.
  2. **Gradient equality**: the minibatched accumulated gradient equals the
     full-batch :func:`autograd.grad` to <= 1e-8 (relative max-abs) at the same
     points, for ``doc_batch_size`` in ``{D, 64, 17}`` — including a size that does
     NOT divide ``D`` (a ragged last minibatch).
  3. **Fit-outcome parity**: a full ``fit`` with a small ``doc_batch_size`` reaches
     essentially the same final objective and heldout AUC as the single-shot
     full-batch fit (``doc_batch_size >= D``). Minor L-BFGS path differences from
     float summation order are acceptable; gross divergence is not.

The problem is deterministic given the seeds; autograd/numpy/scipy only in the core,
sklearn only here for the heldout AUC.
"""
from __future__ import annotations

import autograd
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis.pc.model import PCTopicModel
from analysis.pc.slda_reference import (
    loss_from_param_vec,
    multinomial_coef_const,
    pack_param_vec,
)


def _rand_problem(seed, D=200, V=30, K=5, C=3, off_frac=0.4):
    """A non-trivial random PC problem (D~200, C>1, a real per-cell ``label_mask``).

    Returns the count matrix, binary labels, a ``(D, C)`` observed-mask with roughly
    ``off_frac`` of the cells switched off (never an all-off column), and an
    all-labeled row mask. Every doc keeps at least one token so ``pi``-inference is
    well-posed.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(D, V)).astype(np.float64)
    X[X.sum(axis=1) == 0, 0] = 1.0
    y = rng.integers(0, 2, size=(D, C)).astype(np.float64)
    mask = (rng.random((D, C)) >= off_frac).astype(np.float64)
    for c in range(C):                       # keep every head active
        if mask[:, c].sum() == 0:
            mask[rng.integers(D), c] = 1.0
    rowmask = np.ones(D)
    return dict(X=X, y=y, mask=mask, rowmask=rowmask, D=D, V=V, K=K, C=C)


def _full_batch_value_and_grad(s, loss_kwargs, mult):
    """The reference: single-shot full-batch objective + ``autograd.grad``."""
    K, V, C = s["K"], s["V"], s["C"]

    def obj(vec):
        return loss_from_param_vec(
            vec, X_DV=s["X"], y_DC=s["y"], y_rowmask=s["rowmask"],
            label_mask=s["mask"], K=K, V=V, C=C, mult_coef_const_val=mult,
            **loss_kwargs,
        )

    grad = autograd.grad(obj)
    return lambda vec: (float(obj(vec)), np.asarray(grad(vec), dtype=np.float64))


def _minibatch_value_and_grad(s, doc_batch_size):
    """Build the model's minibatch-accumulation ``vec -> (value, grad)`` closure.

    Wires the private :meth:`PCTopicModel._make_minibatch_value_and_grad` exactly as
    ``fit`` does, so the gate exercises the real accumulation path (not a re-derived
    copy). Returns ``(value_and_grad, loss_kwargs, mult)`` so the caller can build the
    matching full-batch reference from the identical hyperparameters.
    """
    model = PCTopicModel(
        K=s["K"], C=s["C"], weight_y=5.0, lambda_w=0.01, pi_iters=25,
        doc_batch_size=doc_batch_size,
    )
    loss_kwargs = model._loss_kwargs()
    mult = multinomial_coef_const(s["X"])
    vg = model._make_minibatch_value_and_grad(
        s["X"], s["y"], s["rowmask"], s["mask"], s["V"], mult, loss_kwargs,
    )
    return vg, loss_kwargs, mult


def _rand_vecs(s, n=4):
    """``n`` random packed free vectors ``[w_KV | w_CK]`` for the equality checks."""
    out = []
    for j in range(n):
        r = np.random.default_rng(1000 + j)
        out.append(pack_param_vec(
            0.5 * r.standard_normal((s["K"], s["V"])),
            0.5 * r.standard_normal((s["C"], s["K"])),
        ))
    return out


@pytest.mark.parametrize("doc_batch_size", [200, 64, 17])
def test_minibatch_objective_equals_full_batch(doc_batch_size):
    """Minibatched objective value == full-batch loss to <= 1e-9 relative."""
    s = _rand_problem(0)
    vg, loss_kwargs, mult = _minibatch_value_and_grad(s, doc_batch_size)
    full = _full_batch_value_and_grad(s, loss_kwargs, mult)
    for vec in _rand_vecs(s):
        fv, _ = full(vec)
        mv, _ = vg(vec)
        rel = abs(mv - fv) / (abs(fv) + 1e-12)
        assert rel <= 1e-9, (
            f"minibatch obj rel error {rel:.2e} > 1e-9 (bs={doc_batch_size})"
        )


@pytest.mark.parametrize("doc_batch_size", [200, 64, 17])
def test_minibatch_gradient_equals_full_batch(doc_batch_size):
    """Minibatched accumulated gradient == full-batch autograd.grad to <= 1e-8.

    ``doc_batch_size=200`` is one batch (still routed through the global-term split,
    so it tests the split, not a trivial pass-through); 64 divides D=200 unevenly on
    the last batch; 17 never divides D — a ragged tail every time. All must match the
    single autograd tape's gradient.
    """
    s = _rand_problem(0)
    vg, loss_kwargs, mult = _minibatch_value_and_grad(s, doc_batch_size)
    full = _full_batch_value_and_grad(s, loss_kwargs, mult)
    for vec in _rand_vecs(s):
        _, fg = full(vec)
        _, mg = vg(vec)
        rel = np.max(np.abs(mg - fg)) / (np.max(np.abs(fg)) + 1e-12)
        assert rel <= 1e-8, (
            f"minibatch grad rel-maxabs {rel:.2e} > 1e-8 (bs={doc_batch_size})"
        )


def test_single_batch_ge_D_is_byte_identical_to_full_batch():
    """A ``fit`` with ``doc_batch_size >= D`` takes the original path exactly.

    The single-shot branch must be the pre-minibatch code untouched, so its fitted
    parameters and objective are bit-for-bit reproducible across two large-batch
    settings — evidence the default-size small-D tests are unperturbed.
    """
    s = _rand_problem(1, D=40, V=12, K=3, C=2)
    kw = dict(K=3, C=2, weight_y=3.0, pi_iters=20, max_iter=30, seed=0)
    m_default = PCTopicModel(doc_batch_size=2048, **kw).fit(s["X"], s["y"], label_mask=s["mask"])
    m_huge = PCTopicModel(doc_batch_size=10**9, **kw).fit(s["X"], s["y"], label_mask=s["mask"])
    assert m_default.n_doc_batches_ == 1
    assert np.array_equal(m_default.topics_, m_huge.topics_)
    assert np.array_equal(m_default.w_CK_, m_huge.w_CK_)
    assert m_default.final_obj_ == m_huge.final_obj_


# --- Fit-outcome parity (planted signal, heldout AUC) -----------------------
def _planted_corpus(seed, D, K_dom=2, sig_block=6, dom_mass=0.5, slope=24.0, n_tok=180):
    """A single-label corpus whose label is driven by one subtle predictive topic.

    ``K_dom`` dominant label-irrelevant topics carry most tokens; one subtle
    predictive topic's per-doc weight drives the binary label through a logistic
    link. Enough signal that PC recovers it, so two fit paths that are numerically
    equivalent must reach the same heldout AUC.
    """
    rng = np.random.default_rng(seed)
    K_true = K_dom + 1
    V = K_true * sig_block
    topics = np.full((K_true, V), 0.01)
    for k in range(K_true):
        topics[k, k * sig_block:(k + 1) * sig_block] += 1.0
    topics /= topics.sum(axis=1, keepdims=True)

    theta = np.zeros((D, K_true))
    theta[:, :K_dom] = rng.dirichlet(np.full(K_dom, 0.4), size=D) * dom_mass
    theta[:, K_dom] = rng.uniform(0.0, 2.0 * (1.0 - dom_mass), size=D)
    theta /= theta.sum(axis=1, keepdims=True)

    X = np.zeros((D, V))
    for d in range(D):
        X[d] = rng.multinomial(n_tok, theta[d] @ topics)
    w = theta[:, K_dom]
    p = 1.0 / (1.0 + np.exp(-slope * (w - np.median(w))))
    y = rng.binomial(1, p).astype(np.float64)
    return X, y, V


def test_fit_parity_small_batch_matches_full_batch():
    """A ``fit`` with ``doc_batch_size=32`` matches the full-batch fit's final
    objective and heldout AUC (float-summation-order path differences aside)."""
    Xtr, ytr, V = _planted_corpus(0, D=140)
    Xte, yte, _ = _planted_corpus(1, D=80)
    kw = dict(K=3, C=1, weight_y=30.0, pi_iters=40, max_iter=80, seed=0)

    m_full = PCTopicModel(doc_batch_size=10**9, **kw).fit(Xtr, ytr)
    m_mini = PCTopicModel(doc_batch_size=32, **kw).fit(Xtr, ytr)

    assert m_mini.n_doc_batches_ == int(np.ceil(140 / 32))
    # Final objective: same to a small relative tolerance (L-BFGS may take a
    # marginally different path off ~1e-13 gradient-summation differences).
    rel = abs(m_mini.final_obj_ - m_full.final_obj_) / (abs(m_full.final_obj_) + 1e-12)
    assert rel < 1e-4, f"final objective diverged (rel {rel:.2e})"

    auc_full = roc_auc_score(yte, m_full.predict_proba(Xte)[:, 0])
    auc_mini = roc_auc_score(yte, m_mini.predict_proba(Xte)[:, 0])
    assert auc_full > 0.70, f"reference fit did not recover signal (AUC {auc_full:.3f})"
    assert abs(auc_mini - auc_full) < 0.03, (
        f"heldout AUC diverged: mini={auc_mini:.4f} full={auc_full:.4f}"
    )
