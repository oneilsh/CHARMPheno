"""OnlinePCLDA: VI-native Prediction-Constrained topic model as a VIModel.

This is the faithful VI port of ``analysis/pc``. It is built in two increments:

  * INCREMENT 1 (``weight_y == 0``) — the UNSUPERVISED SVI scaffolding. At
    ``weight_y == 0`` the Prediction-Constrained objective collapses to
    unsupervised LDA-MAP (the two-stage baseline's representation), so on this
    path ``OnlinePCLDA`` is deliberately identical to ``OnlineLDA``: the label
    never enters inference and the global step is exactly the LDA λ
    natural-gradient step. Every ``weight_y == 0`` contract method delegates its
    math to an internal ``OnlineLDA``.

  * INCREMENT 2 (``weight_y > 0``) — the SUPERVISED content, built on top of
    increment 1. A logistic head ``w_CK`` (C labels × K topics) predicts each
    doc's outcomes from its LABEL-FREE topic mix, and the label reshapes the
    GLOBAL topics by differentiating *through* that label-free doc inference.
    This is what makes the port *prediction-constrained* rather than a two-stage
    "fit topics, then a classifier on top" pipeline.

Generative model (words), identical to ``OnlineLDA``:
    theta_d ~ Dirichlet(alpha · 1_K);  z_dn ~ Cat(theta_d);  w_dn ~ Cat(beta_z)
    beta_k ~ Dirichlet(eta · 1_V)
Supervised head (analysis/pc/head.py; binary per-label logistic):
    P(y_dc = 1) = sigmoid(w_CK[c] · pi_d),   pi_d the LABEL-FREE doc-topic mix.

Variational mean field: same as ``OnlineLDA`` —
    q(beta_k) = Dirichlet(lambda_k)  (global, K×V)
    q(theta_d) = Dirichlet(gamma_d)  (local, K)

Design contract (mapping onto VIModel):
    initialize_global -> LDA globals {lambda, alpha, eta} PLUS w_CK (C×K zeros).
    local_update      -> LABEL-FREE per-doc CAVI (reuses OnlineLDA), emits
                         lambda_stats exactly as OnlineLDA. For weight_y > 0 it
                         ADDITIONALLY emits, over OBSERVED cells only, the partial
                         supervised gradients ``grad_topics_stat`` (∂loss_y/∂the
                         topic representation CAVI reads) and ``grad_wCK_stat``
                         (∂loss_y/∂w_CK), autograd-computed and unboxed to plain
                         numpy.
    update_global     -> LDA λ natural-gradient step (reuses OnlineLDA), THEN for
                         weight_y > 0 the ρ-blended non-conjugate corrections: a
                         supervised topic correction on λ and an SGD step on w_CK
                         (the OnlineSTM Γ M-step template).
    infer_local       -> the SAME label-free CAVI as local_update (train/test π
                         consistency — the faithfulness invariant).
    compute_elbo      -> unsupervised LDA ELBO (the supervised NLL is a training
                         penalty on the globals, not part of the reported bound).

THE π-ESTIMATOR (the faithfulness invariant, decided for increment 2)
---------------------------------------------------------------------
The reference (``analysis/pc``) differentiates the supervised loss through
``nef_map_pi_DK`` — a label-free NEF-MAP point estimate. This VI port instead
predicts with mean-field CAVI (``OnlineLDA.infer_local``). The invariant that
MUST hold is: the label-free π the supervised gradient differentiates through is
the SAME π the model predicts with. We therefore standardize the whole model on
CAVI: the supervised gradient differentiates through an ``autograd.numpy``
re-implementation of the SAME CAVI fixed point (``_cavi_theta_anp``), unrolled a
FIXED, short number of steps (``grad_cavi_iters``, default 20) to bound the
autograd tape. Prediction (``infer_local``) and training thus use the identical
label-free routine — the ``weight_y == 0`` path is left byte-for-byte untouched,
so increment 1's recovery/equivalence gates are unaffected. (We chose CAVI-anp
over standardizing on a NEF-MAP unroll precisely because CAVI is what increment 1
already ships, and ``autograd.scipy.special.psi`` differentiates the digamma in
the CAVI recurrence cleanly — so no estimator swap, and no re-run of increment 1.)

THE AUTOGRAD CHARTER (Option B, CONTAINED to this module)
---------------------------------------------------------
``spark-vi`` is numpy/scipy by charter. The single documented exception is this
file: the supervised gradient-through-inference of increment 2 is computed with
``autograd`` (imported ONLY here; grep-verified). The gradient is taken w.r.t.
the topic representation CAVI reads (``expElogbeta``) and ``w_CK``, then UNBOXED
to plain numpy before any statistic leaves ``local_update`` — no autograd box
ever crosses the Spark partition boundary (Spark serializes plain arrays).

References:
    Hughes, Hope, Weiner, McCoy, Perlis, Sudderth, Doshi-Velez 2017/2018.
        Prediction-Constrained topic models.
    Hoffman, Blei, Bach 2010; Hoffman, Blei, Wang, Paisley 2013 (Online/SVI LDA).
    analysis/pc/model.py (the exact in-memory oracle this port validates against).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import digamma, polygamma

# Autograd — the ONE charter exception, contained to this module (Option B).
import autograd
import autograd.numpy as anp
import autograd.scipy.special as asp

from spark_vi.core.model import VIModel
from spark_vi.models.topic.lda import OnlineLDA
from spark_vi.models.topic.types import PCDocument


# ---------------------------------------------------------------------------
# Autograd supervised loss (the per-doc PC prediction NLL, differentiated
# through the SAME label-free CAVI π the model predicts with).
# ---------------------------------------------------------------------------

def _log_sigmoid_anp(x):
    """Numerically stable ``log(sigmoid(x)) = -log(1 + exp(-x))``, autograd-safe.

    Branch-free identity ``min(x, 0) - log1p(exp(-|x|))`` — finite and
    differentiable for large-magnitude logits either sign. Ported verbatim from
    ``analysis.pc.slda_reference.log_logistic_sigmoid`` (the reference's loss_y
    uses the identical closed form).
    """
    return anp.minimum(x, 0.0) - anp.log1p(anp.exp(-anp.abs(x)))


def _cavi_theta_anp(eb_d, counts, alpha_vec, K, n_iters):
    """Label-free CAVI doc-topic mean θ_d, in ``autograd.numpy`` (differentiable).

    A fixed-``n_iters`` unroll of the SAME mean-field CAVI fixed point that
    ``OnlineLDA._cavi_doc_inference`` (and therefore ``infer_local``) computes —
    the faithfulness invariant. The digamma in the recurrence is
    ``autograd.scipy.special.psi`` (differentiable; its VJP is the trigamma
    polygamma(1)), so the map ``eb_d -> θ_d`` is a smooth function autograd can
    flow the supervised loss back through.

    Recurrence (per doc; ``eb_d = expElogbeta[:, indices]``), identical in form
    to ``_cavi_doc_inference`` but unrolled a FIXED, short number of steps from a
    deterministic init (the CAVI fixed point is init-independent; a fixed short
    unroll bounds the autograd tape and keeps the routine deterministic)::

        expElogthetad = exp(psi(gamma) - psi(sum gamma))
        phi_norm      = eb_d.T @ expElogthetad + 1e-100
        gamma         = alpha + expElogthetad * (eb_d @ (counts / phi_norm))
        ...
        theta         = gamma / sum(gamma)

    Args:
        eb_d:      (K, n_unique) topic representation for this doc's words
                   (``expElogbeta[:, indices]``); the autograd-tracked input.
        counts:    (n_unique,) word counts (plain numpy constant).
        alpha_vec: (K,) Dirichlet concentration (plain numpy constant).
        K:         number of topics.
        n_iters:   fixed unroll depth.

    Returns:
        theta: (K,) doc-topic mean on the simplex (θ = γ / Σγ).
    """
    gamma = anp.asarray(alpha_vec, dtype=float)
    for _ in range(int(n_iters)):
        expElogthetad = anp.exp(asp.psi(gamma) - asp.psi(anp.sum(gamma)))
        phi_norm = anp.dot(eb_d.T, expElogthetad) + 1e-100
        gamma = alpha_vec + expElogthetad * anp.dot(eb_d, counts / phi_norm)
    return gamma / anp.sum(gamma)


def _per_doc_sup_nll(eb_d, w_CK, counts, s, obs, alpha_vec, K, n_iters):
    """Per-document supervised prediction NLL (weight_y factored OUT).

    ``loss_y_d = - Σ_c obs_c · log σ( s_c · (w_CK[c] · π_d) )`` over this doc's
    OBSERVED cells only, with π_d the label-free CAVI mean (differentiated
    through). Mirrors the reference's ``loss_y`` per-doc term
    (``analysis.pc.slda_reference.calc_loss__slda``): ``s_c = sign(y_c - 0.01)``
    is the reference's binary convention, ``obs_c = y_rowmask · label_mask`` the
    observed-cell mask (both constants in the parameters). ``weight_y`` is applied
    once at the global step, so it is deliberately absent here.
    """
    theta = _cavi_theta_anp(eb_d, counts, alpha_vec, K, n_iters)   # (K,)
    logits = anp.dot(w_CK, theta)                                  # (C,)
    ll = _log_sigmoid_anp(s * logits)                              # (C,)
    return -anp.sum(obs * ll)


# argnum=(0, 1): gradient w.r.t. (eb_d, w_CK). Built once at import — autograd
# closures are cheap to reuse and never cross the Spark boundary (only the
# unboxed numpy results do).
_per_doc_sup_vg = autograd.value_and_grad(_per_doc_sup_nll, argnum=(0, 1))


def _grad_topics_to_lambda(grad_eb: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Map ∂loss/∂expElogbeta (topic-PROBABILITY space) → ∂loss/∂λ (Dirichlet-COUNT
    space) via the EXACT Jacobian of ``expElogbeta = exp(ψ(λ) − ψ(Σ_v λ))``.

    The supervised autograd gradient is taken w.r.t. ``expElogbeta`` (that bounds
    the per-doc tape to the doc's unique words). But ``expElogbeta`` is a
    normalized function of the ACTUAL global parameter λ, so descending λ needs the
    remaining chain-rule step. With ``ψ' = polygamma(1)`` (trigamma) and
    ``eb = expElogbeta``::

        ∂eb_kv/∂λ_kv' = eb_kv·( ψ'(λ_kv)·[v'=v] − ψ'(Σ_v λ_k) )
        ⇒ ∂loss/∂λ_kv = eb_g_kv·ψ'(λ_kv) − ψ'(Σλ_k)·Σ_v eb_g_kv ,   eb_g = eb·grad_eb

    The second (per-topic) term is the normalizer coupling the old per-cell "sign
    carries to λ" argument dropped. Omitting the whole transform left the raw
    ∂loss/∂eb — which is ~V·wy too large AND mis-directed (finite-difference:
    ~65× norm, ~0.84 direction cosine) — subtracted from λ, so the supervised
    correction degraded the topics and ratcheted Σλ; the per-cell trust region
    only masked the magnitude. This transform is finite-difference-EXACT (see
    ``test_supervised_lambda_gradient_matches_finite_difference``) and cheap: two
    trigamma evals + a per-topic sum, no extra autograd tape.
    """
    lam = np.asarray(lam, dtype=np.float64)
    eb = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    eb_g = eb * grad_eb
    return (eb_g * polygamma(1, lam)
            - polygamma(1, lam.sum(axis=1, keepdims=True))
            * eb_g.sum(axis=1, keepdims=True))


def _supervised_batch_value_and_grad(
    topics_repr: np.ndarray,
    w_CK: np.ndarray,
    rows: list,
    alpha_vec: np.ndarray,
    K: int,
    n_iters: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Accumulate (value, ∂/∂topics_repr, ∂/∂w_CK) of the supervised NLL over a batch.

    The supervised loss is a plain SUM over documents, so its gradient is the sum
    of the per-doc gradients. We take each doc's gradient w.r.t. its sliced
    ``eb_d = topics_repr[:, indices]`` (bounding the autograd tape to one doc's
    unique words) and SCATTER it back into the dense ``(K, V)`` topic-gradient at
    those columns — words absent from a doc do not enter its loss and so get zero
    gradient, exactly. Unobserved/unlabeled cells contribute 0 via the ``obs``
    mask (the semi-supervised asymmetry): a doc with no observed cell is skipped
    whole. Every returned array is plain numpy (autograd's top-level grad unboxes).

    Shared by ``local_update`` (per Spark partition) and the internal grad-check.

    Returns:
        (loss, grad_topics (K, V), grad_wCK (C, K)) — all plain numpy.
    """
    K_, V = topics_repr.shape
    C = w_CK.shape[0]
    grad_topics = np.zeros((K_, V), dtype=np.float64)
    grad_wCK = np.zeros((C, K), dtype=np.float64)
    loss = 0.0
    for doc in rows:
        obs = np.asarray(doc.label_mask, dtype=np.float64)
        if obs.sum() == 0.0:
            continue                      # no observed cell -> zero supervised contribution
        indices = doc.indices
        counts = np.asarray(doc.counts, dtype=np.float64)
        s = np.sign(np.asarray(doc.y, dtype=np.float64) - 0.01)   # {-1, +1}, constant
        eb_d = topics_repr[:, indices]                            # (K, n_unique)
        val, (g_eb, g_w) = _per_doc_sup_vg(
            eb_d, w_CK, counts, s, obs, alpha_vec, K, n_iters,
        )
        loss += float(val)
        grad_topics[:, indices] += np.asarray(g_eb, dtype=np.float64)
        grad_wCK += np.asarray(g_w, dtype=np.float64)
    return loss, grad_topics, grad_wCK


def _supervised_batch_value(
    topics_repr: np.ndarray,
    w_CK: np.ndarray,
    rows: list,
    alpha_vec: np.ndarray,
    K: int,
    n_iters: int,
) -> float:
    """Value-only supervised NLL over a batch — the finite-difference target for
    the internal grad-check (no autograd grad, so cheap to call many times)."""
    loss = 0.0
    for doc in rows:
        obs = np.asarray(doc.label_mask, dtype=np.float64)
        if obs.sum() == 0.0:
            continue
        s = np.sign(np.asarray(doc.y, dtype=np.float64) - 0.01)
        eb_d = topics_repr[:, doc.indices]
        loss += float(_per_doc_sup_nll(
            eb_d, w_CK, np.asarray(doc.counts, dtype=np.float64),
            s, obs, alpha_vec, K, n_iters,
        ))
    return loss


def _supervised_head_hessian(
    topics_repr: np.ndarray,
    w_CK: np.ndarray,
    rows: list,
    alpha_vec: np.ndarray,
    K: int,
    n_iters: int,
) -> np.ndarray:
    """Per-label NLL Hessian (Fisher information) of the logistic head over OBSERVED
    cells: ``H_c = Σ_d obs_dc · p_dc(1-p_dc) · outer(π_d, π_d)``, with ``π_d`` the SAME
    label-free CAVI mean the head gradient reads. Pairs with ``grad_wCK_stat`` (the NLL
    gradient ``g_c = Σ_d obs (p-y) π``) to form a per-iteration ridge-Newton (IRLS) head
    step: both are additive doc-sums (aggregatable via combine_stats), fixed-size
    ``(C, K, K)`` / ``(C, K)``, and — since the runner scales BOTH by corpus/batch — the
    Newton solve ``H⁻¹g`` is scale-INVARIANT (the scaling cancels). One such step per SVI
    iteration converges the logistic head on the current θ (Newton converges logistic in
    a handful of steps), which is the aggregatable way to "converge the head aggressively
    within each iteration" WITHOUT collecting raw per-doc θ to the driver.
    """
    C = w_CK.shape[0]
    H = np.zeros((C, K, K), dtype=np.float64)
    for doc in rows:
        obs = np.asarray(doc.label_mask, dtype=np.float64)
        if obs.sum() == 0.0:
            continue
        theta = _cavi_theta_anp(
            topics_repr[:, doc.indices], np.asarray(doc.counts, dtype=np.float64),
            alpha_vec, K, n_iters)                              # (K,), plain numpy
        p = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(w_CK) @ theta, -50.0, 50.0)))
        wt = obs * p * (1.0 - p)                                # (C,) observed IRLS weights
        H += wt[:, None, None] * np.outer(theta, theta)[None, :, :]
    return H


# ---------------------------------------------------------------------------
# Supervised-head seam. The head FLAVOR is a single per-doc loss ``per_doc_nll``;
# the batch gradient accumulation, the digamma-Jacobian topic transform
# (``_grad_topics_to_lambda``), and the ``update_global`` corrections are all
# head-AGNOSTIC. A flavor subclasses ``SupervisedHead`` and provides
# ``per_doc_nll`` (autograd handles ``grad_topics``/``grad_wCK``); it MAY provide a
# closed-form ``batch_hessian`` (the default returns None → the Newton head falls
# back to autograd/SGD). ``FlatLogisticHead`` is the default and delegates to the
# proven module-level free functions, so the increment-2 numbers are byte-for-byte
# unchanged. See docs/superpowers/specs/2026-08-12-pc-supervised-head-seam-design.md.
# ---------------------------------------------------------------------------


def _dag_block_fisher(topics_repr, w_CK, rows, alpha_vec, K, n_iters, closure_matrix):
    """EXACT per-node block Fisher for the DAG-closure head, aggregated over a batch:

        H[a] = I_a · outer(theta, theta),
        I_a  = (1 - p_a)^2 · Σ_{l: a ∈ closure(l)} obs_l · P_l / (1 - P_l),

    with p_a = σ(w_a·θ) the node's LOCAL sigmoid and P_l = ∏_{b∈closure(l)} p_b the
    node PROBABILITY. This is the Fisher information (expected Hessian) of the per-node
    independent-Bernoulli(P_l) likelihood — PSD, label-INDEPENDENT, and aggregatable
    (C,K,K). Unlike the flat local Fisher ``obs_a·p_a(1-p_a)``, an INTERNAL node accrues
    curvature from every observed DESCENDANT ``l`` (those with ``a ∈ closure(l)``), so
    deep nodes are conditioned, not just the frontier — the ``Mᵀ·ratio`` sum below.

    Reduces EXACTLY to ``_supervised_head_hessian`` (the flat logistic Fisher) when the
    DAG is a star (``closure(l) = {l}`` → M = I). It equals ``E_y[Hessian]`` (the
    dropped indefinite term ``−p_a(1-p_a)·Σ obs_l r_l`` has ``E_y[r_l] = 0``), which is
    the Gauss-Newton = Fisher identity for this model. The OFF-diagonal blocks
    (nodes sharing a descendant) are dropped — that is the full (C·K, C·K) Newton, a
    separate lift. Derivation + finite-difference check: tests/test_pc_dag_head.py.
    """
    C = w_CK.shape[0]
    w = np.asarray(w_CK, dtype=np.float64)
    M = np.asarray(closure_matrix, dtype=np.float64)          # (C, C), M[l,a]=1 iff a∈cl(l)
    H = np.zeros((C, K, K), dtype=np.float64)
    for doc in rows:
        obs = np.asarray(doc.label_mask, dtype=np.float64)
        if obs.sum() == 0.0:
            continue
        theta = _cavi_theta_anp(
            topics_repr[:, doc.indices], np.asarray(doc.counts, dtype=np.float64),
            alpha_vec, K, n_iters)                            # (K,) plain numpy
        z = np.clip(w @ theta, -50.0, 50.0)                   # (C,) local logits
        p = 1.0 / (1.0 + np.exp(-z))
        log_sig = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
        P = np.clip(np.exp(M @ log_sig), 1e-12, 1.0 - 1e-6)   # (C,) node probabilities
        ratio = obs * P / (1.0 - P)                           # (C,), observed cells only
        coeff = (1.0 - p) ** 2 * (M.T @ ratio)                # (C,) closure-aware I_a
        H += coeff[:, None, None] * np.outer(theta, theta)[None, :, :]
    return H


def _predict_proba_np(theta, w_CK, closure_matrix=None):
    """Per-label prediction probability for ONE doc's topic mean (plain numpy).

    ``closure_matrix is None`` -> the flat logistic ``σ(w_l·θ)``. A ``(C, C)`` closure
    indicator -> the DAG-closure PRODUCT ``∏_{a ∈ closure(l)} σ(w_a·θ)`` = ``exp(M ·
    logσ(w·θ))``. Shared by the head ``predict_proba`` methods AND the mllib transform,
    which broadcasts these arrays (never the head object, whose autograd closure is not
    picklable) into its probability UDF.
    """
    z = np.clip(np.asarray(w_CK, dtype=np.float64) @ np.asarray(theta, dtype=np.float64),
                -50.0, 50.0)
    if closure_matrix is None:
        return 1.0 / (1.0 + np.exp(-z))
    log_sig = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))       # log σ(z), stable
    return np.exp(np.asarray(closure_matrix, dtype=np.float64) @ log_sig)


class SupervisedHead:
    """A per-document supervised NLL ``loss_y(eb_d, w_CK, doc)`` plus generic
    autograd batch accumulators built on top of it.

    A head FLAVOR overrides :meth:`per_doc_nll` (a differentiable
    ``autograd.numpy`` function of the doc's topic slice ``eb_d`` and the head
    ``w_CK``); the base then differentiates it per doc and sums over the batch, so a
    new flavor needs NO gradient derivation. ``per_doc_nll`` reads π_d via the SAME
    label-free CAVI (:func:`_cavi_theta_anp`) the model predicts with — the PC
    faithfulness invariant — so ``weight_y`` stays factored out here and is applied
    once at the global step.
    """

    def per_doc_nll(self, eb_d, w_CK, counts, s, obs, alpha_vec, K, n_iters):
        raise NotImplementedError

    def __init__(self):
        # value_and_grad w.r.t. (eb_d, w_CK); built once per head (autograd closures
        # are cheap to reuse and never cross the Spark boundary — only unboxed numpy
        # results do).
        self._per_doc_vg = autograd.value_and_grad(self.per_doc_nll, argnum=(0, 1))

    def batch_value_and_grad(self, topics_repr, w_CK, rows, alpha_vec, K, n_iters):
        """Accumulate (value, ∂/∂topics_repr (K,V), ∂/∂w_CK (C,K)) over a batch.

        The supervised loss is a plain SUM over docs, so its gradient is the sum of
        per-doc gradients. Each doc's autograd grad is taken w.r.t. its sliced
        ``eb_d = topics_repr[:, indices]`` (bounding the tape to one doc's unique
        words) and SCATTERED back into the dense (K,V) at those columns; unobserved
        cells contribute 0 via ``obs`` (a no-observed-cell doc is skipped whole).
        Generic over the flavor — used by non-flat heads; :class:`FlatLogisticHead`
        short-circuits to the proven free function.
        """
        K_, V = topics_repr.shape
        C = w_CK.shape[0]
        grad_topics = np.zeros((K_, V), dtype=np.float64)
        grad_wCK = np.zeros((C, K), dtype=np.float64)
        loss = 0.0
        for doc in rows:
            obs = np.asarray(doc.label_mask, dtype=np.float64)
            if obs.sum() == 0.0:
                continue
            indices = doc.indices
            counts = np.asarray(doc.counts, dtype=np.float64)
            s = np.sign(np.asarray(doc.y, dtype=np.float64) - 0.01)
            eb_d = topics_repr[:, indices]
            val, (g_eb, g_w) = self._per_doc_vg(
                eb_d, w_CK, counts, s, obs, alpha_vec, K, n_iters)
            loss += float(val)
            grad_topics[:, indices] += np.asarray(g_eb, dtype=np.float64)
            grad_wCK += np.asarray(g_w, dtype=np.float64)
        return loss, grad_topics, grad_wCK

    def batch_value(self, topics_repr, w_CK, rows, alpha_vec, K, n_iters):
        """Value-only supervised NLL over a batch (the finite-difference target)."""
        loss = 0.0
        for doc in rows:
            obs = np.asarray(doc.label_mask, dtype=np.float64)
            if obs.sum() == 0.0:
                continue
            s = np.sign(np.asarray(doc.y, dtype=np.float64) - 0.01)
            eb_d = topics_repr[:, doc.indices]
            loss += float(self.per_doc_nll(
                eb_d, w_CK, np.asarray(doc.counts, dtype=np.float64),
                s, obs, alpha_vec, K, n_iters))
        return loss

    def batch_hessian(self, topics_repr, w_CK, rows, alpha_vec, K, n_iters):
        """Closed-form per-label NLL Hessian if the flavor has one, else None.

        None signals the Newton/IRLS head to fall back (autograd Hessian or the SGD
        head step). :class:`FlatLogisticHead` provides the logistic Fisher info.
        """
        return None

    def predict_proba(self, theta: np.ndarray, w_CK: np.ndarray) -> np.ndarray:
        """Per-label P(y_l = 1) for one doc's topic mean ``theta`` (K,) — plain numpy
        (no autograd; prediction needs no gradient). Default = the flat C-way logistic
        ``σ(w_l·θ)``; structured flavors (e.g. the DAG-closure head) override."""
        return _predict_proba_np(theta, w_CK, None)


class FlatLogisticHead(SupervisedHead):
    """Default flavor: C INDEPENDENT logistic heads, ``σ(w_c·π)`` — Hughes' flat
    PC head. Delegates to the proven module-level free functions so the increment-2
    numbers are byte-for-byte unchanged, and exposes the closed-form logistic Fisher
    that powers the Newton/IRLS head (ADR 0039)."""

    def per_doc_nll(self, eb_d, w_CK, counts, s, obs, alpha_vec, K, n_iters):
        return _per_doc_sup_nll(eb_d, w_CK, counts, s, obs, alpha_vec, K, n_iters)

    def batch_value_and_grad(self, topics_repr, w_CK, rows, alpha_vec, K, n_iters):
        return _supervised_batch_value_and_grad(
            topics_repr, w_CK, rows, alpha_vec, K, n_iters)

    def batch_value(self, topics_repr, w_CK, rows, alpha_vec, K, n_iters):
        return _supervised_batch_value(topics_repr, w_CK, rows, alpha_vec, K, n_iters)

    def batch_hessian(self, topics_repr, w_CK, rows, alpha_vec, K, n_iters):
        return _supervised_head_hessian(
            topics_repr, w_CK, rows, alpha_vec, K, n_iters)


class DagClosureHead(SupervisedHead):
    """Label-side HIERARCHY head for an ontology DAG (e.g. Mondo case-finding).

    Generalizes HSLDA's tree parent-gating (Perotte 2011, ICD-9) to a DAG. Each
    label ``l`` is a node with an is-a CLOSURE — ``l`` plus all its ancestors, each
    counted ONCE even under diamonds (multiple paths to a shared ancestor). A node
    fires only if its whole closure fires::

        log P(node_l = 1) = Σ_{a ∈ closure(l)} log σ(w_a · π)

    so ``P(child) ≤ P(parent)`` by construction (the extra closure terms are ≤ 0) —
    the monotone is-a consistency Mondo needs, baked into TRAINING rather than
    enforced post-hoc. The per-doc NLL over OBSERVED cells is::

        loss_y_d = Σ_l obs_l · [ −y_l·log P_l − (1−y_l)·log(1−P_l) ] .

    It is a smooth function of ``(eb_d, w_CK)``, so the base autograd accumulators
    yield the topic + head gradients with NO new derivation (the point of the seam).
    The head is converged by a Newton/IRLS step over the EXACT per-node block Fisher
    (:meth:`batch_hessian` → :func:`_dag_block_fisher`), which conditions internal
    nodes via their observed descendants and reduces to the flat logistic Fisher on a
    star DAG; only the off-diagonal (full (C·K)² Newton) coupling is left out.

    ``closure_parents[l]`` lists the DIRECT parent label indices of node ``l`` (a
    root has none); the closure is formed here, diamond-safe. Ids are integer label
    indices in the SAME ``[0, C)`` space as ``w_CK``'s rows — the engine stays
    domain-agnostic (no concept ids).
    """

    def __init__(self, closure_parents):
        parents = [tuple(int(p) for p in ps) for ps in closure_parents]
        C = len(parents)
        # Diamond-safe ancestral closure via memoized DFS (acyclic is-a DAG); each
        # ancestor lands in the set once regardless of how many paths reach it.
        _clo: list = [None] * C
        def _closure(node: int):
            if _clo[node] is not None:
                return _clo[node]
            acc = {node}
            for p in parents[node]:
                acc |= _closure(p)
            _clo[node] = acc
            return acc
        M = np.zeros((C, C), dtype=np.float64)
        for l in range(C):
            for a in _closure(l):
                M[l, a] = 1.0
        self.C = C
        self._parents = parents
        self._closure_matrix = M          # (C, C): M[l, a] == 1 iff a ∈ closure(l)
        super().__init__()                # builds _per_doc_vg from self.per_doc_nll

    def per_doc_nll(self, eb_d, w_CK, counts, s, obs, alpha_vec, K, n_iters):
        theta = _cavi_theta_anp(eb_d, counts, alpha_vec, K, n_iters)     # (K,)
        ls = _log_sigmoid_anp(anp.dot(w_CK, theta))                      # (C,) log σ(w_a·π), < 0
        logP = anp.dot(self._closure_matrix, ls)                         # (C,) log P(node_l=1), < 0
        # Cap logP just below 0 so P ≤ 1 − 1e-6: as a saturated node drives P_l → 1,
        # log(1 − P_l) → −∞ and its gradient → −∞ (and 0·(−∞) = NaN poisons even the
        # y=1 cells). The cap bounds the negative-cell gradient — mirrors the P clip in
        # predict_proba / _dag_block_fisher — with no effect away from saturation.
        logP = anp.minimum(logP, -1e-6)
        y = (s + 1.0) / 2.0                                              # {0,1} from sign(y − 0.01)
        # log(1 − P) = log(−expm1(logP)); logP < 0 strictly, so −expm1(logP) ∈ (0, 1).
        log1mP = anp.log(-anp.expm1(logP))                              # (C,)
        per_label = -(y * logP + (1.0 - y) * log1mP)                    # (C,) NLL
        return anp.sum(obs * per_label)

    def batch_hessian(self, topics_repr, w_CK, rows, alpha_vec, K, n_iters):
        """EXACT per-node block Fisher (Gauss-Newton) for the closure-coupled head:
        ``H[a] = (1-p_a)^2 · Σ_{l: a∈closure(l)} obs_l·P_l/(1-P_l) · θθᵀ`` — see
        :func:`_dag_block_fisher`. Paired in ``update_global`` with the exact coupled
        gradient (``grad_wCK``, autograd), the per-node ridge solve ``H_a⁻¹ g_a`` is a
        true Newton/IRLS step (ADR 0039), converging the head where a single RM-SGD
        step per iteration cannot (insight 0065). PSD, label-independent, aggregatable
        (C,K,K), scale-invariant, and exactly the flat logistic Fisher on a star DAG.
        Unlike the flat local Fisher it conditions INTERNAL nodes via their observed
        descendants. Off-diagonal coupling (the full (C·K)² Newton) is the only piece
        not captured.
        """
        return _dag_block_fisher(
            topics_repr, w_CK, rows, alpha_vec, K, n_iters, self._closure_matrix)

    def predict_proba(self, theta: np.ndarray, w_CK: np.ndarray) -> np.ndarray:
        """Per-node P(node_l = 1) = ∏_{a ∈ closure(l)} σ(w_a·θ), the closure PRODUCT
        (not the local σ(w_l·θ)). Monotone: P(child) ≤ P(parent). Plain numpy."""
        return _predict_proba_np(theta, w_CK, self._closure_matrix)


class OnlinePCLDA(VIModel):
    """Prediction-Constrained LDA fittable by VIRunner with mini-batch SVI.

    At ``weight_y == 0`` (increment 1) this behaves exactly like ``OnlineLDA`` —
    same recovery, same ELBO — because every contract method delegates its LDA
    math to an internal ``OnlineLDA`` and adds only the (inert) head. The
    equivalence is by construction: it is the exact same code path.

    At ``weight_y > 0`` (increment 2) the supervised content attaches:
    ``local_update`` additionally emits the autograd supervised partial gradients
    over observed cells; ``update_global`` applies a ρ-blended supervised topic
    correction to λ and an SGD step to the head. The unsupervised λ step stays
    closed-form; the head + topic correction are the gradient pieces the runner's
    Robbins-Monro ρ_t damps (the OnlineSTM Γ M-step template).

    Parameters mirror ``OnlineLDA`` plus:
        C:        number of binary outcome heads (rows of ``w_CK``); default 1.
        weight_y: PC prediction-loss weight (the PC dial). 0.0 = unsupervised
                  LDA-MAP; > 0 turns on the supervised correction.
        lambda_w: L2 ridge on the head weights (scaled by weight_y), authors'
                  default 0.001.
        grad_cavi_iters: fixed CAVI unroll depth for the differentiated π
                  (bounds the autograd tape; default 20).
        head_lr_scale: extra multiplier on the head SGD step size — the RM ↔
                  weight_y decoupling knob (see the class note below). Default 1.0.
        topic_trust: trust-region fraction for the supervised topic correction on
                  λ. Each λ cell's per-iteration supervised change is clipped to
                  ``±topic_trust · λ[k,v]`` (a per-CELL relative cap on the current
                  λ, which already carries the corpus scaling, so the cap is
                  scale-invariant). This keeps ``λ_new ≥ (1-topic_trust)·λ_unsup >
                  0`` — no cell nears the Dirichlet floor where digamma(λ) and the
                  ELBO explode — and Σλ cannot run away, making ``weight_y`` a robust
                  dial that cannot diverge λ for any corpus size / ``weight_y`` /
                  ``tau0`` (see the "Topic correction: space/scale" note on
                  ``update_global``). Default 0.1. Analogous to ``head_lr_scale``
                  for the head, but a HARD per-cell cap rather than a linear scale,
                  because the topic gradient lives in a DIFFERENT space than λ.
        weight_y_warmup_iters: linearly ramp the effective weight_y from 0 to
                  weight_y over this many global steps (0 = no warmup). Damps the
                  early-iteration head/topic-correction shock when a large
                  weight_y meets the aggressive early ρ_t.

    RM ↔ weight_y coupling (design risk 4)
    --------------------------------------
    One ρ_t damps λ, the supervised topic correction, AND the head SGD. A large
    ``weight_y`` with an aggressive early ρ_t can shove the head across the
    logistic's saturated tail in a single step (the STM softmax-saturation
    failure mode). ``head_lr_scale`` scales ONLY the head step, and
    ``weight_y_warmup_iters`` ramps the whole supervised weight in so the first,
    largest ρ_t steps carry little supervised signal. Both default to the no-op
    setting; reach for them if the ELBO/AUC trace shows the head diverging.

    The λ (topic) correction has its OWN, mandatory guard — a trust region
    (``topic_trust``), NOT merely a ρ-blend. The supervised topic gradient lives
    in topic-PROBABILITY space (``expElogbeta`` ~ O(1/V)) while λ lives in
    Dirichlet-COUNT space (Σλ ~ corpus tokens); the two are decades apart in
    magnitude, and the gradient arrives corpus-SUMMED (and corpus-scaled by the
    runner), so a bare subtraction ``λ − ρ·wy·gT`` scales with corpus size and
    diverges λ (observed at 33k docs: Σλ doubling per iter, ELBO → -1e32). The
    trust region caps the per-iteration supervised change to a fixed fraction of
    the (unconditionally stable) unsupervised λ step, making ``weight_y`` a dial
    that cannot diverge at any scale. See ``update_global``.
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        C: int = 1,
        weight_y: float = 0.0,
        lambda_w: float = 0.001,
        grad_cavi_iters: int = 20,
        head_lr_scale: float = 1.0,
        topic_trust: float = 0.1,
        weight_y_warmup_iters: int = 0,
        head_optimizer: str = "sgd",
        head_lr: float = 0.05,
        head_newton_ridge: float = 1e-2,
        head_l2: float = 1e-3,
        topic_support: "list[np.ndarray] | None" = None,
        alpha: float | np.ndarray | None = None,
        eta: float | None = None,
        optimize_alpha: bool = False,
        optimize_eta: bool = False,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-3,
        random_seed: int | None = None,
        head: "SupervisedHead | None" = None,
        topic_engine: "OnlineLDA | None" = None,
    ) -> None:
        if C < 1:
            raise ValueError(f"C must be >= 1, got {C}")
        if weight_y < 0:
            raise ValueError(f"weight_y must be >= 0, got {weight_y}")
        if lambda_w < 0:
            raise ValueError(f"lambda_w must be >= 0, got {lambda_w}")
        if grad_cavi_iters < 1:
            raise ValueError(f"grad_cavi_iters must be >= 1, got {grad_cavi_iters}")
        if head_lr_scale <= 0:
            raise ValueError(f"head_lr_scale must be > 0, got {head_lr_scale}")
        if topic_trust <= 0:
            raise ValueError(f"topic_trust must be > 0, got {topic_trust}")
        if weight_y_warmup_iters < 0:
            raise ValueError(
                f"weight_y_warmup_iters must be >= 0, got {weight_y_warmup_iters}"
            )
        if head_optimizer not in ("sgd", "newton"):
            raise ValueError(
                f"head_optimizer must be 'sgd' or 'newton', got {head_optimizer!r}"
            )
        if head_lr <= 0:
            raise ValueError(f"head_lr must be > 0, got {head_lr}")
        if head_newton_ridge < 0:
            raise ValueError(f"head_newton_ridge must be >= 0, got {head_newton_ridge}")

        # The topic engine. Every LDA global (λ, α, η) and every topic-side update
        # is owned by this delegate, so at weight_y == 0 OnlinePCLDA IS the delegate
        # on the numbers — the increment-1 equivalence gate holds by construction.
        # OnlineLDA validates K/vocab_size/alpha/eta/gamma_shape/cavi_* for us;
        # PCDocument is duck-compatible with the BOWDocument its local_update/
        # infer_local consume (both only touch .indices/.counts).
        #
        # Gated-PC composition (topic-side seam): a caller may INJECT a pre-built
        # OnlineLDA subclass — e.g. GatedOnlineLDA, whose gated E-step welds each
        # node's topics to its DAG subtree's documents. Because the head is topic-
        # engine-agnostic (it reads only global_params["lambda"]/alpha/K to form the
        # label-free θ it shapes and predicts on), the DAG-gated topic-side seam and
        # the supervised label-side head compose by construction — they touch
        # different seams. When injected, K/V come from the engine and the LDA-
        # building kwargs (K, alpha, eta, optimize_*, gamma_shape, cavi_*) are the
        # engine's own; passing a conflicting K here is a caller error.
        if topic_engine is not None:
            if not isinstance(topic_engine, OnlineLDA):
                raise TypeError(
                    "topic_engine must be an OnlineLDA (or subclass, e.g. "
                    f"GatedOnlineLDA); got {type(topic_engine).__name__}"
                )
            if topic_engine.V != int(vocab_size):
                raise ValueError(
                    f"topic_engine.V ({topic_engine.V}) != vocab_size ({vocab_size})"
                )
            self._lda = topic_engine
        else:
            self._lda = OnlineLDA(
                K=K,
                vocab_size=vocab_size,
                alpha=alpha,
                eta=eta,
                optimize_alpha=optimize_alpha,
                optimize_eta=optimize_eta,
                gamma_shape=gamma_shape,
                cavi_max_iter=cavi_max_iter,
                cavi_tol=cavi_tol,
                random_seed=random_seed,
            )
        self.K = self._lda.K
        self.V = self._lda.V
        self.C = int(C)
        self.weight_y = float(weight_y)
        self.lambda_w = float(lambda_w)
        self.grad_cavi_iters = int(grad_cavi_iters)
        self.head_lr_scale = float(head_lr_scale)
        self.topic_trust = float(topic_trust)
        self.weight_y_warmup_iters = int(weight_y_warmup_iters)
        # Head optimizer. 'sgd' (default) = the RM-damped step ρ·head_lr_scale·wy·g
        # (one first-order step per SVI iteration — provably-in-practice too slow to
        # converge the coupled head against a moving θ; see insight 0065). 'newton' =
        # a per-iteration ridge-Newton (IRLS) step that CONVERGES the logistic head on
        # the current θ — the settled head fix (ADR 0039).
        self.head_optimizer = str(head_optimizer)
        self.head_lr = float(head_lr)
        # 'newton' head: relative ridge (fraction of mean(diag(H))) that conditions the
        # per-label IRLS solve. AUC is scale-invariant to head magnitude, so this only
        # stabilizes the solve; it does not bias the head DIRECTION. head_lr doubles as
        # the Newton damping (step fraction; ~0.5-1.0 for newton, since one damped
        # Newton step per iter already converges the logistic head on the current θ).
        self.head_newton_ridge = float(head_newton_ridge)
        # 'newton' head: FIXED ABSOLUTE L2 prior on w_CK == Hughes's lambda_w. It is the
        # ridge on the CORPUS-SUMMED head gradient, so at the head fixed point |w_CK| ~
        # |g|/head_l2 — NOT scaled by n_docs. (Recommended default 1e-3, Hughes's value,
        # data-independent.) Two roles, both essential:
        #   (1) BLOWUP GUARD. The relative ridge (head_newton_ridge·mean(diag H)) vanishes
        #       as p(1−p)→0 once PC's shaping makes the topics separable, leaving the
        #       logistic MLE at infinity so |w_CK| runs away (observed 3.4e11); any
        #       positive absolute L2 keeps it finite.
        #   (2) CALIBRATED SHAPING. Shaping strength ∝ |w_CK|, so the ridge magnitude sets
        #       how hard PC shapes. Hughes's weak lambda_w=1e-3 lets |w| grow large enough
        #       to shape strongly (reference |w|~105, topics-LR~0.87) while staying finite.
        #
        # HISTORY / CALIBRATION (do not re-break): head_l2 was briefly applied PER-DOC
        # (ridge = head_l2·n_docs), which made head_l2=1e-3 act like lambda_w~0.84 — ~840x
        # (= n_docs) too strong. That throttled |w| to ~5 and collapsed shaping to
        # topics-LR ~0.53, which was mis-attributed to a joint-vs-alternating optimization
        # gap. The joint-vs-alternating de-risk (manual_pc_joint_vs_alternating) REFUTED
        # that — a reference block-ALTERNATING fit reaches topics-LR 0.874 — and the recal
        # sweep (manual_pc_head_l2_recalibration) showed the sweet spot is the ABSOLUTE
        # ridge ~lambda_w (topics-LR ~0.93, |w| finite). Hence: absolute, not ×n_docs.
        # Default 1e-3 (= Hughes lambda_w); the good basin is wide (~1e-4..1e-2, topics-LR
        # 0.91-0.96). 0.0 disables it -> relative-ridge-only, which BLOWS UP on the
        # separable topics PC creates (|w|=3.4e11) — only set 0.0 to reproduce that.
        self.head_l2 = float(head_l2)
        # LOCALIZED head (topic-side hierarchy in the HEAD's support, ADR 0042 done
        # right): per-node topic support — node c's logistic reads ONLY topic_support[c]
        # (its gated block + ancestors' blocks + background, e.g. DagLayout.allowed(c)),
        # not all K. w_c stays 0 off-support (inits at 0; the Newton solve updates only
        # the support sub-block), so the per-node Fisher/solve is O(|support|^3) not
        # O(K^3) — the whole-Mondo scale fix (insight 0071). None = dense (read all K,
        # the shipped behavior). Only the 'newton' head honors this; validated at the
        # 41-anchor scale before whole-Mondo (exp 0089).
        if topic_support is None:
            self._topic_support = None
        else:
            if len(topic_support) != self.C:
                raise ValueError(
                    f"topic_support has {len(topic_support)} entries but C={self.C}")
            self._topic_support = [np.asarray(s, dtype=np.intp) for s in topic_support]
        # Driver-side global-step counter, used only for weight_y warmup. Bumped
        # once per update_global call (the runner drives that single-threaded on
        # the driver), so it is a faithful iteration index without threading t
        # through the VIModel contract.
        self._update_calls = 0
        # Supervised-head flavor (the label-side seam). Default = the flat C-way
        # logistic head (Hughes). A DAG-closure head (Mondo, label-side hierarchy)
        # slots in here without touching the SVI math or the increment-1 gate.
        self._head = head if head is not None else FlatLogisticHead()
        head_C = getattr(self._head, "C", None)
        if head_C is not None and int(head_C) != self.C:
            raise ValueError(
                f"head has C={head_C} but the model has C={self.C}; the head's "
                "label count must match the number of outcome heads")

    # Convenience passthroughs so callers/tests can read the LDA hypers off the
    # PC model without reaching into the delegate.
    @property
    def alpha(self) -> np.ndarray:
        return self._lda.alpha

    @property
    def eta(self) -> float:
        return self._lda.eta

    @property
    def random_seed(self) -> int | None:
        return self._lda.random_seed

    # -- VIModel contract ---------------------------------------------------

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """LDA globals {lambda, alpha, eta} PLUS the seeded logistic head w_CK.

        The LDA globals come verbatim from ``OnlineLDA.initialize_global`` (same
        random-gamma λ, same α/η seeding), so a PC fit and an LDA fit started
        from the same seed share the identical starting λ. ``w_CK`` is seeded to
        zeros — the maximum-entropy head, contributing nothing to prediction —
        exactly as the reference inits its head (analysis/pc/model.py
        ``_init_param_vec``: "the head w_CK inits at zero"), and mirroring how
        ``OnlineSTM`` seeds Γ = 0. At weight_y == 0 it is never touched; at
        weight_y > 0 the supervised SGD moves it off zero from here.
        """
        gp = self._lda.initialize_global(data_summary)
        gp["w_CK"] = np.zeros((self.C, self.K), dtype=np.float64)
        return gp

    def _expElogbeta_from_lambda(self, lam) -> np.ndarray:
        """The concatenated (K, V) topic representation the (gated) CAVI reads.

        Single-domain: λ is a single (K, V) array — the same row-normalized
        ``exp(ψ(λ) − ψ(Σ_v λ))`` ``OnlineLDA.local_update`` forms (byte-identical).
        Multi-domain: λ is a per-domain dict ``{m: (K, V_m)}`` — delegate to the
        gated engine's ``_assemble_expElogbeta`` so each domain is normalized on
        ITS OWN vocab and the blocks are concatenated in domain order (a domain's
        rows are never normalized against another domain's mass)."""
        if isinstance(lam, dict):
            return self._lda._assemble_expElogbeta(lam)
        return np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

    def _corrected_lambda_block(self, grad_eb, lam_pre, lam_unsup, rho, wy):
        """One block's supervised topic correction: transform ∂loss/∂expElogbeta
        (topic-PROBABILITY space) to ∂loss/∂λ (Dirichlet-COUNT space) at the
        pre-step λ, then the per-cell trust-region-capped descent step off the
        just-taken unsupervised λ. The whole single-domain correction body
        (lines below), factored so update_global can apply it per domain block."""
        grad_topics = _grad_topics_to_lambda(grad_eb, lam_pre)
        raw_corr = rho * wy * grad_topics
        cell_cap = self.topic_trust * lam_unsup
        corr = np.clip(raw_corr, -cell_cap, cell_cap)
        return np.maximum(lam_unsup - corr, 1e-30)

    def local_update(
        self,
        rows: Iterable[PCDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition — LABEL-FREE inference (y never shapes π).

        At weight_y == 0 this is byte-for-byte ``OnlineLDA.local_update``: run the
        label-free CAVI per doc and accumulate ``lambda_stats`` (+ ELBO scalars,
        + optional ``e_log_theta_sum``). No supervised statistic is emitted.

        At weight_y > 0 it ALSO accumulates — for OBSERVED cells only
        (``obs_dc = label_mask[d, c]``; the shim folds the per-row y_rowmask into
        label_mask) — the partial supervised gradients:

          * ``grad_topics_stat`` (K, V) = Σ_d ∂loss_y_d / ∂expElogbeta, and
          * ``grad_wCK_stat``    (C, K) = Σ_d ∂loss_y_d / ∂w_CK,

        where ``expElogbeta`` is the SAME topic representation the label-free CAVI
        reads (so the differentiated π equals the predicted π), and ``loss_y_d`` is
        the per-doc prediction NLL with weight_y factored out (applied at the
        global step). Both are dense additive plain-numpy arrays the default
        combine_stats sums; the autograd boxes are unboxed here and never cross the
        partition boundary. The label-free CAVI (and thus ``lambda_stats``) is
        UNCHANGED by this addition — only extra keys join the returned dict.
        """
        if self.weight_y == 0.0:
            # Pure unsupervised path — delegate wholesale so the numbers are
            # identical to OnlineLDA. PCDocument.indices/.counts are all the
            # delegate reads; y/label_mask ride along untouched.
            return self._lda.local_update(rows, global_params)

        # Supervised path. Materialize the partition so the LDA stats and the
        # supervised partial stats read the same y/label_mask-carrying rows.
        rows = list(rows)
        stats = self._lda.local_update(rows, global_params)

        lam = global_params["lambda"]
        w_CK = np.asarray(global_params["w_CK"], dtype=np.float64)
        alpha_vec = np.asarray(global_params["alpha"], dtype=np.float64)
        # The topic representation CAVI reads (identical to lda.local_update).
        # Multi-domain: λ is a per-domain dict; fuse to the concatenated (K, V)
        # the shared gated CAVI consumes (each domain row-normalized on its vocab).
        expElogbeta = self._expElogbeta_from_lambda(lam)

        _loss, grad_topics, grad_wCK = self._head.batch_value_and_grad(
            expElogbeta, w_CK, rows, alpha_vec, self.K, self.grad_cavi_iters,
        )
        stats["grad_topics_stat"] = grad_topics
        stats["grad_wCK_stat"] = grad_wCK
        if self.head_optimizer == "newton":
            # Additive per-label NLL Hessian (Fisher info) for the ridge-Newton head
            # step; sums across partitions via the delegate's combine_stats like every
            # other dense stat. grad_wCK_stat is the paired gradient. None when the
            # flavor has no closed form (the DAG head → autograd/SGD fallback).
            hess = self._head.batch_hessian(
                expElogbeta, w_CK, rows, alpha_vec, self.K, self.grad_cavi_iters)
            if hess is not None:
                stats["head_hess_stat"] = hess
        return stats

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """M-step at rho_t — LDA λ natural-gradient step, then the PC corrections.

        At weight_y == 0 the LDA globals {lambda, alpha, eta} are updated verbatim
        by ``OnlineLDA.update_global`` and ``w_CK`` passes through unchanged.

        At weight_y > 0, AFTER the unsupervised λ step, apply the non-conjugate
        corrections (the OnlineSTM Γ ridge-M-step template, damped by the runner's
        RM ρ_t). With ``wy`` the (warmup-scaled) effective weight_y,
        ``N`` = ``target_stats["n_docs"]`` the corpus-equivalent doc count:

          (b) supervised topic correction on λ, PROPERLY TRANSFORMED then damped.
              ``grad_topics_stat`` is Σ_d ∂loss_y_d/∂expElogbeta — a gradient in
              topic-PROBABILITY space, because the autograd tape is bounded at
              ``expElogbeta`` (the doc-sliced topic representation). But
              ``expElogbeta = exp(ψ(λ) − ψ(Σλ))`` is a normalized function of the
              ACTUAL global parameter λ, so the descent step on λ needs the
              remaining chain-rule factor. :func:`_grad_topics_to_lambda` applies
              the EXACT digamma-Jacobian to produce the true ``gT = ∂loss_y/∂λ``
              (Dirichlet-COUNT space), evaluated at the λ the gradient was taken at.
              Then the faithful descent step, lightly trust-region-damped:
                  raw  = ρ · wy · gT                          (gT now in λ-space)
                  corr = clip(raw, −topic_trust·λ_unsup, +topic_trust·λ_unsup)
                  λ ← λ_unsup − corr                          (λ_unsup = new_lam)

              History / why this matters: the transform was MISSING — the raw
              ∂loss/∂expElogbeta (~V·wy too large AND ~33° mis-directed;
              finite-difference: ~65× norm, ~0.84 direction cosine) was subtracted
              directly from λ. That mis-transformed, persistently-biased push
              degraded the supervised topics (heldout AUC BELOW the unsupervised
              two-stage baseline) and ratcheted Σλ ~260× over 200 iters — a
              compounding runaway the per-cell cap bounds per-step but (like STM's
              analogous non-conjugate M-step, ADR 0034) cannot stop. With the
              correct λ-space gradient the correction is naturally count-scaled and
              correctly directed, so the runaway does not arise; the trust region is
              retained only as a light per-cell safety (guaranteeing
              ``λ_new ≥ (1-topic_trust)·λ_unsup > 0`` so no cell nears the
              ``digamma`` floor), no longer the load-bearing stabilizer. ``wy``
              remains the dial; below the cap the step is the exact faithful
              ρ·wy·∂loss_y/∂λ.
          (c) head SGD. The head has NO ``(1-ρ)`` shrinkage and no corpus-scale
              anchor, so a corpus-SUMMED gradient makes the un-damped step grow
              with corpus size and run the logistic head off to its saturated tail
              (the RM ↔ weight_y coupling, risk 4 — observed as a sign-flipped head
              at large corpus × weight_y). We therefore use the per-doc MEAN head
              gradient ``gW`` = ``grad_wCK_stat`` / N: scale-invariant, minibatch-
              unbiased, and mirroring the reference's per-token loss rescale. Plus
              the ridge ∂/∂w_CK loss_w = wy·λ_w·2·w:
                  w_CK ← w_CK  −  ρ · head_lr_scale · wy · ( gW + λ_w · 2 · w_CK )

        λ's unsupervised part stays closed-form; the head + correction are the only
        gradient pieces. Signs follow the reference's loss convention (loss_y is a
        NEGATIVE-log-likelihood; we minimize, hence the minus). ``head_lr_scale``
        and ``weight_y_warmup_iters`` further decouple the head from ρ_t if needed.
        """
        new_gp = self._lda.update_global(global_params, target_stats, learning_rate)

        if self.weight_y == 0.0:
            # Head unchanged at weight_y == 0 (stays at its zero seed).
            new_gp["w_CK"] = global_params["w_CK"]
            return new_gp

        self._update_calls += 1
        wy = self._effective_weight_y()
        rho = float(learning_rate)

        n_docs = float(target_stats.get("n_docs", np.array(1.0)))
        inv_n = 1.0 / max(n_docs, 1.0)
        # Topic correction: the autograd stat is ∂loss/∂expElogbeta (topic-
        # PROBABILITY space); transform it to the true ∂loss/∂λ (Dirichlet-COUNT
        # space) via the exact digamma-Jacobian (:func:`_grad_topics_to_lambda`)
        # BEFORE it touches λ. The Jacobian is evaluated at the λ the gradient was
        # taken at (``global_params["lambda"]``, the pre-step value local_update
        # read). This is the fix for the mis-transformed correction that made the
        # supervised topics degrade (finite-difference-verified exact).
        grad_topics_eb = np.asarray(target_stats["grad_topics_stat"], dtype=np.float64)
        # Head: per-doc MEAN (scale-invariant; the un-shrunk head has no corpus anchor).
        grad_wCK = np.asarray(target_stats["grad_wCK_stat"], dtype=np.float64) * inv_n
        w_CK = np.asarray(global_params["w_CK"], dtype=np.float64)

        # (b) supervised topic correction on λ, PER-CELL trust-region-capped
        # (:meth:`_corrected_lambda_block`). The raw descent step ρ·wy·gT lives in
        # topic-PROBABILITY space and is corpus-summed/scaled, so subtracting it
        # directly from λ (Dirichlet-COUNT space) diverges λ at scale. The block clips
        # the per-cell correction to ±topic_trust · λ_unsup[k,v] (a relative cap on the
        # JUST-TAKEN unsupervised λ). Consequences: (i) λ_new ≥ (1-topic_trust)·λ_unsup
        # > 0, so no cell nears the digamma floor; (ii) Σλ cannot run away
        # geometrically; (iii) the cap is relative to the true per-iteration λ state
        # (which already carries the runner's corpus scaling), so it is scale-INVARIANT
        # — weight_y is a dial that saturates rather than diverges. Below the cap the
        # step is exactly ρ·wy·gT.
        #
        # MULTI-DOMAIN: λ is a per-domain dict {m:(K,V_m)} and grad_topics_stat is the
        # CONCATENATED (K, V_total) ∂loss/∂expElogbeta. Split it into per-domain blocks
        # (``_split_to_domains``) and correct EACH block against its OWN domain's λ —
        # so _grad_topics_to_lambda's normalizer (Σ_v over axis 1) runs over that
        # domain's vocab only. A pooled transform over the fused vocab would normalize
        # a domain's rows against every other domain's mass (wrong). This scatter is
        # the one new engine piece multi-domain PC needs: the per-node gate then lets
        # each disease node's topic block specialize toward its predictive domain.
        lam_pre = global_params["lambda"]
        lam_unsup = new_gp["lambda"]
        if isinstance(lam_pre, dict):
            grad_blocks = self._lda._split_to_domains(grad_topics_eb)
            new_gp["lambda"] = {
                m: self._corrected_lambda_block(
                    grad_blocks[m], lam_pre[m], lam_unsup[m], rho, wy)
                for m in lam_pre
            }
        else:
            new_gp["lambda"] = self._corrected_lambda_block(
                grad_topics_eb, lam_pre, lam_unsup, rho, wy)

        # (c) head step. Two optimizers:
        #   'sgd'    — the RM-damped step ρ·head_lr_scale·wy·g (default; unchanged).
        #   'newton' — a per-iteration ridge-Newton (IRLS) step that CONVERGES the
        #              logistic head on the current θ (sgd takes ONE noisy gradient
        #              step per SVI iter and does not converge the head against a moving
        #              θ — heldout AUC ≈ chance, w_CK ⊥ the batch-LR direction, invariant
        #              to lr; insight 0065). g and H are corpus-scaled additive doc-sums,
        #              so H⁻¹g is SCALE-INVARIANT (the corpus/batch factor cancels) and
        #              needs no raw θ on the driver. This also feeds the topic correction
        #              a VALID head signal each iter (the correction's ∂loss_y/∂θ flows
        #              through w_CK).

        if self.head_optimizer == "newton" and "head_hess_stat" in target_stats:
            # Per-label ridge-Newton: w_c ← w_c − head_lr · (H_c + λI)⁻¹ (g_c + λ w_c).
            # Gated on the stat's presence: a head flavor without a closed-form Fisher
            # (e.g. DagClosureHead) emits no head_hess_stat, so 'newton' gracefully
            # degrades to the SGD step below rather than KeyError-ing.
            # g_c is the corpus-scaled NLL gradient sum (grad_wCK_stat), H_c its Fisher
            # info (head_hess_stat). Ridge = a FIXED ABSOLUTE L2 prior (head_l2, Hughes's
            # lambda_w: the ridge on the CORPUS-SUMMED head gradient, so |w| ~ |g|/head_l2
            # — see __init__) PLUS head_newton_ridge·mean(diag H_c) (a numerical
            # conditioner only). head_lr damps the step (~0.5-1.0 for newton).
            g_CK = np.asarray(target_stats["grad_wCK_stat"], dtype=np.float64)
            H_CKK = np.asarray(target_stats["head_hess_stat"], dtype=np.float64)
            new_w = w_CK.copy()
            for c in range(self.C):
                # LOCALIZED head: solve only over node c's topic support (its gated
                # block + ancestors); w_c stays 0 off-support (new_w is a copy of the
                # 0-initialized w_CK). Numerically identical to solving the full K×K
                # Fisher's support sub-block, at O(|support|^3) — the Mondo scale fix.
                sup = None if self._topic_support is None else self._topic_support[c]
                Hc = H_CKK[c] if sup is None else H_CKK[c][np.ix_(sup, sup)]
                wc = w_CK[c] if sup is None else w_CK[c][sup]
                gc = g_CK[c] if sup is None else g_CK[c][sup]
                d = self.K if sup is None else len(sup)
                ridge = (self.head_l2
                         + self.head_newton_ridge * (float(np.trace(Hc)) / d) + 1e-10)
                A = Hc + ridge * np.eye(d)
                b = gc + ridge * wc
                try:
                    delta = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(A, b, rcond=None)[0]
                if sup is None:
                    new_w[c] = w_CK[c] - self.head_lr * delta
                else:
                    new_w[c, sup] = w_CK[c, sup] - self.head_lr * delta
            new_gp["w_CK"] = new_w
            return new_gp

        # 'sgd' head: one RM-damped first-order step (grad + ridge ∂loss_w/∂w).
        head_grad = grad_wCK + self.lambda_w * 2.0 * w_CK
        new_gp["w_CK"] = w_CK - rho * self.head_lr_scale * wy * head_grad
        return new_gp

    def _effective_weight_y(self) -> float:
        """weight_y after the optional linear warmup ramp (design risk 4).

        Ramps 0 -> weight_y across ``weight_y_warmup_iters`` global steps so the
        aggressive early ρ_t does not shove the head across the logistic's
        saturated tail; a no-op (returns weight_y) once warmup is exhausted or when
        ``weight_y_warmup_iters == 0``. ``_update_calls`` is 1 on the first
        supervised step.
        """
        if self.weight_y_warmup_iters <= 0:
            return self.weight_y
        frac = min(1.0, self._update_calls / float(self.weight_y_warmup_iters))
        return self.weight_y * frac

    def combine_stats(
        self,
        a: dict[str, np.ndarray],
        b: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Elementwise-sum suff-stat dicts — the LDA delegate's (default) combiner.

        The LDA stats and the increment-2 supervised partial stats
        (``grad_topics_stat``, ``grad_wCK_stat``) are all dense additive arrays, and
        the default VIModel combiner sums over the union of keys, so a partition
        that emitted the supervised keys and one that (all-unobserved) still emits
        them as zeros combine cleanly.
        """
        return self._lda.combine_stats(a, b)

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """Unsupervised LDA ELBO (doc-likelihood + doc KL − global β KL).

        The supervised NLL is a penalty on the GLOBAL parameters (it reshapes λ and
        trains the head), not part of the variational bound on the word data, so the
        reported ELBO stays the unsupervised bound — a clean, monotone-ish trace to
        watch alongside the head/AUC diagnostics for the RM ↔ weight_y coupling.
        """
        return self._lda.compute_elbo(global_params, aggregated_stats)

    def infer_local(self, row: PCDocument, global_params: dict[str, np.ndarray]):
        """Per-doc label-free CAVI — the IDENTICAL routine to local_update.

        This is the faithfulness invariant: train-time and test-time π come from
        the same label-free E-step (there is no train/test representation
        mismatch), mirroring ``OnlineLDA.infer_local``. The differentiated CAVI in
        ``_cavi_theta_anp`` is a short autograd unroll of this SAME fixed point, so
        the π the supervised gradient reshapes topics through equals the π predicted
        here. The head ``w_CK`` is not read here; the head-derived per-label
        probability is produced by the shim's ``predictProbability``.
        """
        return self._lda.infer_local(row, global_params)

    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """LDA per-iter summary plus the head magnitude, so the RM ↔ weight_y
        coupling (a head running away to the saturated logistic tail) is visible in
        the fit log."""
        base = self._lda.iteration_summary(global_params)
        w = np.asarray(global_params["w_CK"])
        return f"{base}, |w_CK|max={np.abs(w).max():.3g}, weight_y={self.weight_y:g}"

    def get_metadata(self) -> dict[str, Any]:
        """Shape constants for VIResult round-trip — K, V, and C heads."""
        md = self._lda.get_metadata()
        md["C"] = self.C
        return md

    def iteration_diagnostics(
        self, global_params: dict[str, np.ndarray],
    ) -> dict[str, float | np.ndarray]:
        """LDA concentration traces (α, η) plus the head-weight magnitude trace."""
        diag = self._lda.iteration_diagnostics(global_params)
        if self.weight_y != 0.0:
            diag["w_CK_absmax"] = float(np.abs(global_params["w_CK"]).max())
        return diag
