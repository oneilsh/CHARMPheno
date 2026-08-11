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
        head_beta1: float = 0.9,
        head_beta2: float = 0.999,
        head_eps: float = 1e-8,
        alpha: float | np.ndarray | None = None,
        eta: float | None = None,
        optimize_alpha: bool = False,
        optimize_eta: bool = False,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-3,
        random_seed: int | None = None,
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
        if head_optimizer not in ("sgd", "adam"):
            raise ValueError(
                f"head_optimizer must be 'sgd' or 'adam', got {head_optimizer!r}"
            )
        if head_lr <= 0:
            raise ValueError(f"head_lr must be > 0, got {head_lr}")
        if not (0.0 <= head_beta1 < 1.0 and 0.0 <= head_beta2 < 1.0):
            raise ValueError("head_beta1/head_beta2 must lie in [0, 1)")
        if head_eps <= 0:
            raise ValueError(f"head_eps must be > 0, got {head_eps}")

        # The unsupervised LDA engine. Every LDA global (λ, α, η) and every LDA
        # update is owned by this delegate, so at weight_y == 0 OnlinePCLDA IS
        # OnlineLDA on the numbers — the increment-1 equivalence gate holds by
        # construction. OnlineLDA validates K/vocab_size/alpha/eta/gamma_shape/
        # cavi_* for us; PCDocument is duck-compatible with the BOWDocument its
        # local_update/infer_local consume (both only touch .indices/.counts).
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
        # Head optimizer. 'sgd' (default) = the RM-damped step ρ·head_lr_scale·wy·g.
        # 'adam' = a per-parameter adaptive step on the head, DECOUPLED from ρ and
        # weight_y (Adam self-normalizes the gradient scale, so both cancel). This
        # is the two-timescale fix: the non-conjugate head gets its own adaptive
        # rate rather than sharing the topics' single Robbins-Monro schedule — the
        # structure Hughes et al. (AISTATS 2018) used (Adam) to keep the coupled
        # PC objective from landing in a mis-directed-head local optimum.
        self.head_optimizer = str(head_optimizer)
        self.head_lr = float(head_lr)
        self.head_beta1 = float(head_beta1)
        self.head_beta2 = float(head_beta2)
        self.head_eps = float(head_eps)
        # Driver-side global-step counter, used only for weight_y warmup. Bumped
        # once per update_global call (the runner drives that single-threaded on
        # the driver), so it is a faithful iteration index without threading t
        # through the VIModel contract.
        self._update_calls = 0

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
        if self.head_optimizer == "adam":
            # First/second-moment buffers for the Adam head step; global-only
            # (updated on the driver in update_global), so they never cross the
            # Spark stats boundary and combine_stats is unaffected.
            gp["w_CK_m"] = np.zeros((self.C, self.K), dtype=np.float64)
            gp["w_CK_v"] = np.zeros((self.C, self.K), dtype=np.float64)
        return gp

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
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        _loss, grad_topics, grad_wCK = _supervised_batch_value_and_grad(
            expElogbeta, w_CK, rows, alpha_vec, self.K, self.grad_cavi_iters,
        )
        stats["grad_topics_stat"] = grad_topics
        stats["grad_wCK_stat"] = grad_wCK
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
            if self.head_optimizer == "adam":
                # Carry the (untouched) Adam buffers so the global-params key set
                # stays constant across iterations for combine/resume. Lazy-init to
                # zeros if a warm-start checkpoint (e.g. an sgd phase 1) lacks them.
                zc = np.zeros((self.C, self.K), dtype=np.float64)
                new_gp["w_CK_m"] = global_params.get("w_CK_m", zc)
                new_gp["w_CK_v"] = global_params.get("w_CK_v", zc.copy())
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
        grad_topics = _grad_topics_to_lambda(
            grad_topics_eb, global_params["lambda"])
        # Head: per-doc MEAN (scale-invariant; the un-shrunk head has no corpus anchor).
        grad_wCK = np.asarray(target_stats["grad_wCK_stat"], dtype=np.float64) * inv_n
        w_CK = np.asarray(global_params["w_CK"], dtype=np.float64)

        # (b) supervised topic correction on λ, PER-CELL trust-region-capped. The
        # raw descent step ρ·wy·gT lives in topic-PROBABILITY space and is
        # corpus-summed/scaled, so subtracting it directly from λ (Dirichlet-COUNT
        # space) diverges λ at scale. We clip the per-cell correction to
        # ±topic_trust · λ_unsup[k,v] (a relative cap on the JUST-TAKEN
        # unsupervised λ), so every cell moves by at most a topic_trust fraction of
        # its own value. Consequences: (i) λ_new ≥ (1-topic_trust)·λ_unsup > 0, so
        # no cell is ever driven toward the floor where digamma(λ) — and hence the
        # ELBO's global β-KL — explodes; (ii) Σλ cannot run away geometrically;
        # (iii) the cap is relative to the true per-iteration λ state (which already
        # carries the runner's corpus scaling), so it is scale-INVARIANT — weight_y
        # is a dial that saturates rather than diverges. Sign (descend loss_y) is
        # preserved per cell; below the cap the step is exactly ρ·wy·gT.
        lam_unsup = new_gp["lambda"]
        raw_corr = rho * wy * grad_topics
        cell_cap = self.topic_trust * lam_unsup
        corr = np.clip(raw_corr, -cell_cap, cell_cap)
        # Floor kept as a belt-and-suspenders invariant (unreachable given the cap):
        # keeps λ a valid strictly-positive Dirichlet pseudocount.
        new_gp["lambda"] = np.maximum(lam_unsup - corr, 1e-30)

        # (c) head step: mean data gradient + ridge. Two optimizers:
        #   'sgd'  — the RM-damped step ρ·head_lr_scale·wy·g (default; unchanged).
        #   'adam' — a per-parameter adaptive step, DECOUPLED from ρ and wy. Adam
        #            normalizes by the running gradient RMS, so the wy factor on
        #            head_grad cancels in m̂/√v̂ and ρ is replaced by Adam's own
        #            step dynamics: the head runs on its OWN (fast) timescale rather
        #            than sharing the topics' single Robbins-Monro schedule. This is
        #            the two-timescale remedy for the coupled-objective failure mode
        #            where 10 shared-θ heads under minibatch noise drive w_CK into a
        #            mis-directed local optimum (heldout head AUC ≈ chance while a
        #            batch LR on the SAME topics predicts) — the instability Hughes
        #            et al. (AISTATS 2018) avoided by optimizing {φ, η} with Adam.
        head_grad = grad_wCK + self.lambda_w * 2.0 * w_CK
        if self.head_optimizer == "adam":
            b1, b2, eps = self.head_beta1, self.head_beta2, self.head_eps
            # Lazy-init the moment buffers: a WARM START (or resume) seeds the global
            # params from a saved checkpoint, replacing initialize_global's output, so
            # a checkpoint written under head_optimizer='sgd' (e.g. the unsupervised
            # phase-1 warm-up) carries no w_CK_m/w_CK_v. Default them to zeros — the
            # standard Adam cold-start — so switching to 'adam' at warm start works.
            m_prev = global_params.get("w_CK_m")
            v_prev = global_params.get("w_CK_v")
            if m_prev is None:
                m_prev = np.zeros_like(w_CK)
            if v_prev is None:
                v_prev = np.zeros_like(w_CK)
            m = b1 * m_prev + (1.0 - b1) * head_grad
            v = b2 * v_prev + (1.0 - b2) * (head_grad * head_grad)
            t = self._update_calls  # >= 1 (bumped above)
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)
            new_gp["w_CK"] = w_CK - self.head_lr * m_hat / (np.sqrt(v_hat) + eps)
            new_gp["w_CK_m"] = m
            new_gp["w_CK_v"] = v
        else:
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
