"""Correctness gates for the per-cell missing-label (multi-task / index-drug)
extension of the FAITHFUL PC objective (:mod:`analysis.pc.slda_reference`,
:class:`analysis.pc.model.PCTopicModel`).

The extension lets one shared topic model carry ``C`` outcome heads while each
document is labeled for only *some* of the ``C`` outcomes: a ``(D, C)``
``label_mask`` marks the observed cells, and an unobserved cell must contribute
NOTHING to ``loss_y`` or to any parameter gradient. The Hughes antidepressant
setup is the motivating case (a patient labeled only for their initiated "index
drug", so the label matrix is almost all missing with ~one observed cell per
row), but any per-cell mask is accepted.

Three gates here, from most to least fundamental:

  1. **Masked grad-check** (the objective is the reference, so its gradient must
     be exactly right): the autograd gradient of :func:`loss_from_param_vec` at a
     small random problem with a non-trivial ``label_mask`` matches a central
     finite-difference gradient to ``< 1e-6`` relative error.
  2. **Masking is real**: permuting/corrupting the labels of the UNOBSERVED cells
     leaves the loss, its gradient, and the fitted parameters bit-for-bit
     unchanged — only observed cells enter the objective.
  3. **Composition with the per-row mask**: switching a row off with
     ``labeled_mask`` drops all of its cells regardless of ``label_mask`` (the
     effective observed set is ``labeled_mask[d] AND label_mask[d, c]``).

autograd/numpy/scipy only; deterministic given the seeds.
"""
from __future__ import annotations

import autograd
import numpy as np
import pytest

from analysis.pc.model import PCTopicModel
from analysis.pc.slda_reference import (
    calc_loss__slda,
    loss_from_param_vec,
    multinomial_coef_const,
    pack_param_vec,
)
from analysis.pc.tests._grad_utils import rel_grad_error


def _rand_problem(seed, D=6, V=8, K=3, C=3, off_frac=0.4):
    """A small random PC problem with a non-trivial per-cell ``label_mask``.

    Returns the packed free vector ``vec = [w_KV | w_CK]``, the count matrix,
    binary labels, and a ``(D, C)`` observed-mask with roughly ``off_frac`` of the
    cells switched off (but never an all-off column, so every head stays active).
    """
    rng = np.random.default_rng(seed)
    w_KV = 0.5 * rng.standard_normal((K, V))
    w_CK = 0.5 * rng.standard_normal((C, K))
    X = rng.integers(0, 5, size=(D, V)).astype(np.float64)
    # keep at least one token per doc so pi-inference is well-posed
    X[X.sum(axis=1) == 0, 0] = 1.0
    y = rng.integers(0, 2, size=(D, C)).astype(np.float64)
    mask = (rng.random((D, C)) >= off_frac).astype(np.float64)
    for c in range(C):                       # guarantee each head has >=1 obs cell
        if mask[:, c].sum() == 0:
            mask[rng.integers(D), c] = 1.0
    vec = pack_param_vec(w_KV, w_CK)
    return dict(vec=vec, X=X, y=y, mask=mask, K=K, V=V, C=C, D=D, rng=rng)


# small, fast NEF unroll for the grad-check (fidelity is not what this tests)
_LOSS_KW = dict(alpha=1.1, tau=1.1, lambda_w=0.01, weight_y=3.0, pi_iters=20)


@pytest.mark.parametrize("seed", range(4))
def test_masked_objective_grad_check(seed):
    """Autograd gradient of the masked loss matches central finite differences.

    This is the load-bearing gate: the faithful objective is our correctness
    oracle, so its gradient through both the head (``w_CK``) and the topics
    (``w_KV``, via the unrolled pi-inference) must be exactly right even when some
    cells are unobserved.
    """
    s = _rand_problem(seed)
    mult = multinomial_coef_const(s["X"])
    common = dict(
        X_DV=s["X"], y_DC=s["y"], y_rowmask=None, label_mask=s["mask"],
        K=s["K"], V=s["V"], C=s["C"], mult_coef_const_val=mult, **_LOSS_KW,
    )

    def f(vec):
        return float(loss_from_param_vec(vec, **common))

    grad = np.asarray(autograd.grad(lambda v: loss_from_param_vec(v, **common))(s["vec"]))
    err = rel_grad_error(f, grad, s["vec"], eps=1e-6)
    assert err < 1e-6, f"masked grad-check relative error {err:.2e} exceeds 1e-6"


@pytest.mark.parametrize("seed", range(4))
def test_unobserved_cells_do_not_affect_loss_or_grad(seed):
    """Corrupting labels at UNOBSERVED cells changes neither loss nor gradient.

    An unobserved cell multiplies its log-likelihood by 0, so flipping its label
    (which only flips the logistic sign) must leave the masked ``loss_y`` — and
    hence the whole loss and every parameter gradient — bit-for-bit identical.
    """
    s = _rand_problem(seed)
    mult = multinomial_coef_const(s["X"])
    common = dict(
        X_DV=s["X"], y_rowmask=None, label_mask=s["mask"],
        K=s["K"], V=s["V"], C=s["C"], mult_coef_const_val=mult, **_LOSS_KW,
    )

    y_corrupt = s["y"].copy()
    off = s["mask"] == 0.0
    y_corrupt[off] = 1.0 - y_corrupt[off]        # flip every unobserved label
    assert off.any(), "test needs at least one unobserved cell"

    v0 = float(loss_from_param_vec(s["vec"], y_DC=s["y"], **common))
    v1 = float(loss_from_param_vec(s["vec"], y_DC=y_corrupt, **common))
    assert v0 == v1, "loss changed when only unobserved labels were corrupted"

    g0 = np.asarray(autograd.grad(
        lambda v: loss_from_param_vec(v, y_DC=s["y"], **common))(s["vec"]))
    g1 = np.asarray(autograd.grad(
        lambda v: loss_from_param_vec(v, y_DC=y_corrupt, **common))(s["vec"]))
    assert np.array_equal(g0, g1), "gradient changed under unobserved-label corruption"


@pytest.mark.parametrize("seed", range(3))
def test_permuting_unobserved_labels_leaves_fit_identical(seed):
    """A full ``fit`` is invariant to relabeling the unobserved cells.

    Two models fit from the same seed on the same ``X``/``label_mask`` but with
    arbitrary garbage in the unobserved cells recover identical ``topics_`` and
    ``w_CK_`` — proof the missing labels never leak into training.
    """
    s = _rand_problem(seed, D=10, V=8, K=3, C=3)
    off = s["mask"] == 0.0

    y_a = s["y"].copy()
    y_b = s["y"].copy()
    y_b[off] = 1.0 - y_b[off]                     # differ ONLY on unobserved cells
    y_b[~off] = y_a[~off]

    fit_kw = dict(K=3, C=3, weight_y=3.0, pi_iters=20, max_iter=40, seed=1)
    m_a = PCTopicModel(**fit_kw).fit(s["X"], y_a, label_mask=s["mask"])
    m_b = PCTopicModel(**fit_kw).fit(s["X"], y_b, label_mask=s["mask"])

    assert np.allclose(m_a.topics_, m_b.topics_)
    assert np.allclose(m_a.w_CK_, m_b.w_CK_)
    # ...and a model that actually SEES the corrupted cells (all observed) differs,
    # so the invariance above is due to masking, not a degenerate problem.
    m_all = PCTopicModel(**fit_kw).fit(s["X"], y_b)          # label_mask=None
    assert not np.allclose(m_a.w_CK_, m_all.w_CK_)


@pytest.mark.parametrize("seed", range(3))
def test_row_mask_and_cell_mask_compose_by_and(seed):
    """``labeled_mask[d]`` AND ``label_mask[d, c]`` is the effective observed set.

    Turning a row off with ``labeled_mask`` must drop ALL of that row's cells,
    whatever ``label_mask`` says — i.e. the loss equals what you get by zeroing
    that entire row of ``label_mask`` instead.
    """
    s = _rand_problem(seed, D=8, V=8, K=3, C=3)
    D, C = s["D"], s["C"]
    rowmask = np.ones(D)
    off_rows = s["rng"].choice(D, size=2, replace=False)
    rowmask[off_rows] = 0.0

    # AND-composed reference: zero those rows directly in the cell mask.
    mask_zeroed = s["mask"].copy()
    mask_zeroed[off_rows, :] = 0.0

    d_rowmask = calc_loss__slda(
        _topics(s), _head(s), s["X"], s["y"], rowmask, s["mask"],
        return_dict=True, **_LOSS_KW,
    )
    d_zeroed = calc_loss__slda(
        _topics(s), _head(s), s["X"], s["y"], None, mask_zeroed,
        return_dict=True, **_LOSS_KW,
    )
    assert d_rowmask["loss_y"] == d_zeroed["loss_y"]
    assert d_rowmask["loss_ttl"] == d_zeroed["loss_ttl"]


# -- helpers to evaluate calc_loss at the raw (topics, head) of a problem ------
def _topics(s):
    from analysis.pc.slda_reference import softmax_rows, unpack_param_vec
    w_KV, _ = unpack_param_vec(s["vec"], K=s["K"], V=s["V"], C=s["C"])
    return np.asarray(softmax_rows(w_KV))


def _head(s):
    from analysis.pc.slda_reference import unpack_param_vec
    _, w_CK = unpack_param_vec(s["vec"], K=s["K"], V=s["V"], C=s["C"])
    return np.asarray(w_CK)
