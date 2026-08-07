"""Faithful autograd reference for Prediction-Constrained (PC) sLDA training.

This module mirrors, term-for-term, the objective and per-document topic
inference of the authors' public code
(``pc_toolbox/model_slda/slda_loss__autograd.py`` :func:`calc_loss__slda` and
``est_local_params__single_doc_map/calc_nef_map_pi_d_K__autograd.py``), i.e.
Hughes, Hope, Weiner, McCoy, Perlis, Sudderth & Doshi-Velez 2017/2018. It is the
correctness ORACLE for our PC work; unlike the rest of ``spark-vi`` it is allowed
``autograd`` (it is a gradient-through-inference reference, not a VI model).

The one idea that makes PC *faithful* (and different from our earlier free-pi
attempt, kept in :mod:`analysis.pc.variants`): each document's topic vector
``pi_d`` is a **generative MAP estimate from the words ONLY** — the label never
enters ``pi``-inference, so the routine is byte-for-byte identical at train and
test time. The supervised loss reshapes the *global* topics by differentiating
*through* that ``pi``-inference. We reproduce that by unrolling a FIXED number of
the authors' Natural-Exponential-Family (NEF) exponentiated-gradient steps and
letting ``autograd`` flow topics -> pi -> loss.

Symbols
-------
    D           number of documents
    V           vocabulary size
    K           number of topics
    C           number of binary labels (one logistic head per label)
    X_DV        (D, V) nonnegative word counts (dense; a doc's unique-word
                sparsity is handled implicitly — absent words have count 0 and so
                drop out of every NEF term, exactly as in the authors' sparse loop)
    topics_KV   (K, V) topic-word rows on the V-simplex (their ``topics_KV``)
    w_CK        (C, K) per-label logistic weights (their ``w_CK``); NO bias, since
                ``pi_d`` sums to 1 the bias is unidentifiable — matches the authors
    y_DC        (D, C) binary labels in {0, 1}
    y_rowmask   (D,) 1 for labeled docs, 0 for unlabeled (semi-supervised)
    label_mask  (D, C) 1 where cell (d, c) is OBSERVED, 0 where the label is
                missing (per-cell / multi-task semi-supervision; see below)

Per-cell missing labels (multi-task)
------------------------------------
The single shared topic model can carry ``C`` outcome heads while each document
is labeled for only *some* of the ``C`` outcomes — the Hughes antidepressant
setup, where a patient is labeled only for the drug they actually initiated
(their "index drug"), so the ``D x C`` label matrix is almost all missing with
about one observed cell per row. ``label_mask[d, c] == 1`` marks cell ``(d, c)``
as observed; a missing cell contributes NOTHING to ``loss_y`` or its gradient,
and — because ``pi_d`` here is label-free generative MAP — nothing to
``pi``-inference either (that coupling is structurally absent in the faithful
model). The per-cell mask composes with the per-row ``y_rowmask`` by logical AND:
the effective observed set is ``obs_dc = y_rowmask[d] * label_mask[d, c]``, so a
row switched off by ``y_rowmask`` drops all of its cells regardless of
``label_mask``. ``label_mask is None`` recovers the all-observed behavior (every
labeled row contributes all ``C`` cells).

Parametrization
---------------
For the differentiable fit we place ``topics_KV`` on the simplex via a **row
softmax** of a free real matrix ``w_KV`` (K, V). The authors instead use a
K x (V-1) transform with a ``min_eps`` floor
(``utils_diffable_transforms.tfm__2D_rows_sum_to_one``). The two agree up to
``min_eps`` ~ 1e-11 and the plan explicitly permits the softmax; see the module
``NOTE`` at :func:`pack_param_vec`. ``calc_loss__slda`` itself accepts
``topics_KV``/``w_CK`` directly, so it can also be evaluated at the authors'
exact provided parameters (used by the oracle-value stretch check).

References
----------
    Hughes, Hope, Weiner, McCoy, Perlis, Sudderth, Doshi-Velez 2017. Prediction-
        constrained training for semi-supervised models. NeurIPS ML4H / 2018 AISTATS.
"""
from __future__ import annotations

import autograd.numpy as anp
import numpy as np
from scipy.special import gammaln as _np_gammaln

# Default per-doc NEF optimization knobs, copied from the authors'
# ``calc_nef_map_pi_d_K__defaults.make_default_kwargs``. We deliberately drop
# their data-dependent early-stop / step-size-restart machinery: a fixed unroll
# is what lets autograd differentiate cleanly through pi-inference, and it makes
# the train and test routines bit-for-bit identical.
DEFAULT_PI_STEP_SIZE = 0.005
DEFAULT_PI_ITERS = 100


def make_convex_alpha_minus_1(alpha: float) -> float:
    """Map a Dirichlet concentration ``alpha`` to the NEF MAP prior coefficient.

    Mirrors the authors' ``make_convex_alpha_minus_1``: a value in ``[0, 1)`` that
    keeps the per-doc MAP problem convex. ``alpha > 1`` -> ``alpha - 1`` (the usual
    Dirichlet(alpha) log-prior slope); ``alpha <= 1`` -> ``alpha`` (their convex
    reparametrization for sparsity-inducing priors).

    Returns:
        convex_alpha_minus_1 in ``[0, 1)``.
    """
    alpha = float(alpha)
    if alpha > 1.0:
        cam1 = alpha - 1.0
    else:
        cam1 = alpha
    assert 0.0 <= cam1 < 1.0, "convex_alpha_minus_1 must lie in [0, 1)"
    return cam1


def log_logistic_sigmoid(x):
    """Numerically stable ``log(sigmoid(x)) = -log(1 + exp(-x))``, autograd-safe.

    Uses the branch-free identity ``min(x, 0) - log1p(exp(-|x|))`` so both large
    positive and large negative logits stay finite and differentiable (the
    authors carry a hand-defined VJP for the same purpose; this closed form needs
    none). Works elementwise on arrays.
    """
    return anp.minimum(x, 0.0) - anp.log1p(anp.exp(-anp.abs(x)))


def nef_map_pi_DK(
    topics_KV,
    X_DV,
    convex_alpha_minus_1,
    pi_iters: int = DEFAULT_PI_ITERS,
    pi_step_size: float = DEFAULT_PI_STEP_SIZE,
):
    """Label-free generative-MAP topic vectors for every doc, via unrolled NEF.

    Vectorized, autograd-differentiable reimplementation of the authors'
    ``calc_nef_map_pi_d_K__autograd`` run over all documents at once. Their inner
    loop works on a document's *unique* words; here we run it densely over the
    whole vocabulary — words absent from a document have count 0, so their NEF
    terms vanish and the two are algebraically identical.

    One exponentiated-gradient step (per row ``d``; ``a_m1`` = convex_alpha_minus_1)::

        M_dv   = (pi_d @ topics)_v
        g_dk   = step * ( sum_v (x_dv / M_dv) * topics_kv  +  a_m1 / (1e-9 + pi_dk) )
        g_dk  -= max_k g_dk                    # authors' overflow guard
        pi_dk *= exp(g_dk)
        pi_d  /= sum_k pi_dk                    # renormalize onto the simplex

    ``pi`` is initialized uniform (``1/K``) exactly as the authors do. We unroll a
    FIXED ``pi_iters`` steps — no convergence break, no step-size restart — so the
    map ``topics -> pi`` is a smooth, differentiable function.

    Args:
        topics_KV: (K, V) topic-word rows on the simplex.
        X_DV:      (D, V) nonnegative counts.
        convex_alpha_minus_1: NEF prior coefficient in ``[0, 1)``.
        pi_iters:  number of exponentiated-gradient iterations to unroll.
        pi_step_size: NEF step size (authors' default 0.005).

    Returns:
        Pi_DK: (D, K) doc-topic rows on the simplex.
    """
    D = X_DV.shape[0]
    K = topics_KV.shape[0]
    Pi_DK = anp.ones((D, K)) / float(K)
    a_m1 = float(convex_alpha_minus_1)
    step = float(pi_step_size)
    for _ in range(int(pi_iters)):
        # Floor M_DV: a rare word with ~0 probability under every topic (softmax
        # topic rows can underflow to exactly 0 in float64) would otherwise give
        # x_dv / 0 = inf, which cascades (inf - inf at the max-subtract) to NaN.
        # Never triggers in the well-conditioned toy-bars regime; real sparse
        # corpora hit it on the first iteration.
        M_DV = anp.maximum(anp.dot(Pi_DK, topics_KV), 1e-12)   # (D, V) word probs
        Q_DV = X_DV / M_DV                            # x_dv / M_dv
        grad_DK = step * (
            anp.dot(Q_DV, topics_KV.T)                # sum_v (x/M) topics_kv
            + a_m1 / (1e-9 + Pi_DK)
        )
        grad_DK = grad_DK - anp.max(grad_DK, axis=1, keepdims=True)
        new_Pi_DK = Pi_DK * anp.exp(grad_DK)
        # Guard the normalizer too: if a row's updated mass underflows to 0,
        # 0/0 = NaN; the floor keeps the (degenerate) row finite instead.
        Pi_DK = new_Pi_DK / anp.maximum(
            anp.sum(new_Pi_DK, axis=1, keepdims=True), 1e-300
        )
    return Pi_DK


def multinomial_coef_const(X_DV: np.ndarray) -> float:
    """Constant ``sum_d [ gammaln(1 + N_d) - sum_v gammaln(1 + x_dv) ]``.

    This is the multinomial coefficient the authors add into ``logpdf_x``. It does
    not depend on any parameter, so it shifts the loss by a constant and has zero
    gradient. We keep it (computed once, in numpy) only so that ``calc_loss__slda``
    reproduces the authors' loss *value* for the oracle-value comparison; set
    ``include_mult_coef=False`` to drop it (the from-scratch fit is unaffected
    either way).
    """
    X_DV = np.asarray(X_DV, dtype=np.float64)
    N_D = X_DV.sum(axis=1)
    return float(np.sum(_np_gammaln(1.0 + N_D) - np.sum(_np_gammaln(1.0 + X_DV), axis=1)))


def calc_loss__slda(
    topics_KV,
    w_CK,
    X_DV,
    y_DC,
    y_rowmask=None,
    label_mask=None,
    *,
    alpha: float = 1.1,
    tau: float = 1.1,
    lambda_w: float = 0.001,
    weight_x: float = 1.0,
    weight_y: float = 1.0,
    weight_pi: float = 1.0,
    pi_iters: int = DEFAULT_PI_ITERS,
    pi_step_size: float = DEFAULT_PI_STEP_SIZE,
    rescale_total_loss_by_n_tokens: bool = True,
    include_mult_coef: bool = True,
    mult_coef_const_val: float | None = None,
    return_dict: bool = False,
):
    """Total PC-sLDA loss, mirroring the authors' ``calc_loss__slda`` term-for-term.

    Semi-supervised, binary-label (per-label logistic) case. Every term keeps the
    authors' name and sign; the ``pi_d`` used everywhere is the label-free NEF-MAP
    from :func:`nef_map_pi_DK`.

    Terms (before the optional per-token rescaling)::

        loss_x      = -weight_x  * sum_d [ sum_v x_dv log(pi_d @ topics)_v  +  mult_coef_d ]
        loss_pi     = -weight_pi * sum_d sum_k a_m1 * log(1e-9 + pi_dk)
        loss_y      = -weight_y  * sum_{(d,c) observed} log_sigmoid( s_dc * (w_CK @ pi_d)_c )
                      with s_dc = sign(y_dc - 0.01)   (their binary convention)
                      and "observed" = y_rowmask[d] * label_mask[d, c]
        loss_topics = -(tau - 1) * sum_kv log topics_kv
        loss_w      =  weight_y * lambda_w * sum w_CK^2
        loss_ttl    = (loss_x + loss_y + loss_pi + loss_topics + loss_w) / scale

    ``scale = sum(X)`` when ``rescale_total_loss_by_n_tokens`` (the authors'
    default), else 1. Unlabeled docs (``y_rowmask == 0``) contribute to
    ``loss_x``/``loss_pi`` only — never to ``loss_y`` — which is the semi-supervised
    asymmetry. ``label_mask`` extends that asymmetry to the *cell* level: an
    unobserved cell ``(d, c)`` is dropped from ``loss_y`` (and thus from every
    parameter gradient) exactly as an unlabeled row is; the effective observed
    set is ``obs_dc = y_rowmask[d] * label_mask[d, c]`` (logical AND). This is the
    joint multi-task / index-drug mode: one shared topic model, ``C`` heads, each
    head trained only on the cells observed for its outcome. ``weight_y == 0``
    makes the loss (and its topic gradient) independent of the labels: the
    unsupervised LDA-MAP baseline.

    Args:
        topics_KV: (K, V) simplex rows. May be autograd-boxed.
        w_CK:      (C, K) logistic head weights. May be autograd-boxed.
        X_DV:      (D, V) counts (plain numpy).
        y_DC:      (D, C) binary labels.
        y_rowmask: (D,) 1=labeled, 0=unlabeled. None => all labeled.
        label_mask: (D, C) 1=cell observed, 0=cell missing. None => every cell of
            a labeled row observed. Composes with ``y_rowmask`` by AND.
        alpha, tau, lambda_w, weight_*: hyperparameters (authors' defaults).
        pi_iters, pi_step_size: NEF unroll controls.
        include_mult_coef: include the constant multinomial coefficient in loss_x.
        mult_coef_const_val: precomputed constant (avoids recompute in a hot loop).
        return_dict: if True, return a dict of every term (numpy floats) plus the
            inferred ``pi_DK`` and per-doc ``y_proba_DC`` — for diagnostics/tests.

    Returns:
        Scalar total loss (autograd-differentiable), or a diagnostics dict.
    """
    a_m1 = make_convex_alpha_minus_1(alpha)
    D, V = X_DV.shape
    C, K = w_CK.shape[0], w_CK.shape[1]

    if y_rowmask is None:
        y_rowmask = anp.ones(D)
    y_rowmask = anp.asarray(y_rowmask, dtype=float)

    Pi_DK = nef_map_pi_DK(
        topics_KV, X_DV, a_m1, pi_iters=pi_iters, pi_step_size=pi_step_size
    )

    # loss_x : generative multinomial neg-log-likelihood over ALL docs.
    M_DV = anp.dot(Pi_DK, topics_KV)
    # Floor before log: an all-but-zero word prob -> log(0) = -inf -> NaN loss
    # (and 0 * -inf = NaN where x_dv == 0). Same guard as the pi-inference divide.
    loss_x = -weight_x * anp.sum(X_DV * anp.log(anp.maximum(M_DV, 1e-12)))
    if include_mult_coef:
        if mult_coef_const_val is None:
            mult_coef_const_val = multinomial_coef_const(X_DV)
        loss_x = loss_x - weight_x * mult_coef_const_val

    # loss_pi : Dirichlet(alpha) MAP prior on pi over ALL docs.
    loss_pi = -weight_pi * anp.sum(a_m1 * anp.log(1e-9 + Pi_DK))

    # loss_y : per-label logistic loss over OBSERVED cells only. The observed
    # set is the per-row mask AND the per-cell mask, obs_dc = y_rowmask[d] *
    # label_mask[d, c]; an unobserved cell multiplies its log-likelihood by 0 and
    # so drops out of the loss and every parameter gradient. Both masks are
    # constant in the parameters, so multiplying is autograd-safe.
    logits_DC = anp.dot(Pi_DK, w_CK.T)                # (D, C)
    sign_DC = anp.sign(y_DC - 0.01)                   # {-1, +1}, constant in params
    ll_DC = log_logistic_sigmoid(sign_DC * logits_DC)
    obs_DC = y_rowmask[:, None]                        # (D, 1), broadcasts over C
    if label_mask is not None:
        obs_DC = obs_DC * anp.asarray(label_mask, dtype=float)   # (D, C) AND
    loss_y = -weight_y * anp.sum(obs_DC * ll_DC)

    # Global regularizers. Floor topics before log: softmax rows are positive in
    # exact arithmetic but can underflow to 0.0 in float64 for very negative
    # logits, which would make log(topics) = -inf. (loss_pi is already floored
    # by its 1e-9 offset.)
    loss_topics = -1.0 * (tau - 1.0) * anp.sum(anp.log(anp.maximum(topics_KV, 1e-300)))
    loss_w = float(weight_y) * lambda_w * anp.sum(w_CK ** 2)

    if rescale_total_loss_by_n_tokens:
        scale_ttl = float(np.sum(np.asarray(X_DV)))
    else:
        scale_ttl = 1.0

    loss_ttl = (loss_x + loss_y + loss_pi + loss_topics + loss_w) / scale_ttl

    if return_dict:
        proba_DC = 1.0 / (1.0 + np.exp(-np.asarray(logits_DC)))
        return dict(
            loss_ttl=float(loss_ttl),
            loss_x=float(loss_x) / scale_ttl,
            loss_y=float(loss_y) / scale_ttl,
            loss_pi=float(loss_pi) / scale_ttl,
            loss_topics=float(loss_topics) / scale_ttl,
            loss_w=float(loss_w) / scale_ttl,
            scale_ttl=scale_ttl,
            pi_DK=np.asarray(Pi_DK),
            y_proba_DC=proba_DC,
        )
    return loss_ttl


# ---------------------------------------------------------------------------
# Flat-vector packing for scipy.optimize (the differentiable fit).
# Layout: [ w_KV (K*V) | w_CK (C*K) ].  topics = softmax(w_KV, axis=1).
# ---------------------------------------------------------------------------

def pack_param_vec(w_KV: np.ndarray, w_CK: np.ndarray) -> np.ndarray:
    """Flatten the free reals ``(w_KV, w_CK)`` into one contiguous vector.

    NOTE (faithfulness): topics are recovered as ``softmax(w_KV, axis=1)`` — a
    K x V row-softmax — whereas the authors use a K x (V-1) transform with a
    ``min_eps`` floor. The two coincide up to ``min_eps`` ~ 1e-11 and the plan
    permits softmax; the only visible consequence is that ``loss_topics``'s
    ``log topics`` term can see marginally smaller entries under softmax. Kept
    documented rather than matched, since it does not affect the reproduction.
    """
    return np.concatenate([
        np.asarray(w_KV, dtype=np.float64).ravel(),
        np.asarray(w_CK, dtype=np.float64).ravel(),
    ])


def unpack_param_vec(vec, *, K: int, V: int, C: int):
    """Inverse of :func:`pack_param_vec`. Returns ``(w_KV, w_CK)`` (autograd-safe).

    Uses ``anp`` slicing/reshape so the same function works on plain numpy vectors
    and on autograd-boxed vectors inside the differentiated loss.
    """
    n_w = K * V
    w_KV = anp.reshape(vec[:n_w], (K, V))
    w_CK = anp.reshape(vec[n_w:n_w + C * K], (C, K))
    return w_KV, w_CK


def softmax_rows(w):
    """Row-wise softmax onto the simplex (autograd-safe)."""
    w = w - anp.max(w, axis=1, keepdims=True)
    e = anp.exp(w)
    return e / anp.sum(e, axis=1, keepdims=True)


def loss_from_param_vec(
    vec,
    *,
    X_DV,
    y_DC,
    y_rowmask,
    K: int,
    V: int,
    C: int,
    mult_coef_const_val: float,
    label_mask=None,
    **loss_kwargs,
):
    """Total loss as a function of the packed free vector — the autograd target.

    Unpacks ``vec`` -> ``(w_KV, w_CK)``, maps ``topics = softmax(w_KV)``, and calls
    :func:`calc_loss__slda`. ``autograd.grad`` of this w.r.t. ``vec`` is the fit's
    Jacobian. All data/hyperparameters are bound by the caller (closure/kwargs).
    ``label_mask`` (D, C) is forwarded unchanged, restricting ``loss_y`` to the
    observed cells (multi-task / index-drug mode); ``None`` => all cells observed.
    """
    w_KV, w_CK = unpack_param_vec(vec, K=K, V=V, C=C)
    topics_KV = softmax_rows(w_KV)
    return calc_loss__slda(
        topics_KV, w_CK, X_DV, y_DC, y_rowmask, label_mask,
        mult_coef_const_val=mult_coef_const_val,
        **loss_kwargs,
    )
