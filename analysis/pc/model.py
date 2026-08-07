"""Faithful Prediction-Constrained topic model (the primary reference wrapper).

A small sklearn-flavored ``fit`` / ``transform`` / ``predict_proba`` class around
the autograd objective in :mod:`analysis.pc.slda_reference`, reproducing Hughes,
Hope, Weiner, McCoy, Perlis, Sudderth & Doshi-Velez 2017/2018.

What makes it *faithful* (vs. the free-pi :class:`analysis.pc.variants.PCTopicModelFreePi`):

  * **Per-doc ``pi`` is label-free generative MAP, always.** Both ``fit`` (inside
    the loss, recomputed as topics change) and ``transform`` obtain ``pi`` from the
    *identical* unrolled NEF routine (:func:`analysis.pc.slda_reference.nef_map_pi_DK`)
    using words only. There is NO train/test representation mismatch — that
    equality is the whole point. The label never shapes ``pi``; it reshapes the
    global ``topics`` by differentiating through ``pi``-inference.

  * **Optimization.** ``fit`` minimizes the total PC loss over the global
    parameters ``(w_KV, w_CK)`` with ``scipy.optimize.minimize(method="L-BFGS-B")``
    and an ``autograd``-computed Jacobian, from a seeded init. ``weight_y == 0``
    drops the label entirely => an unsupervised LDA-MAP fit (the two-stage
    baseline's representation).

  * **Binary logistic head, one weight row per label** (``w_CK``, C labels), no
    bias (``pi`` sums to 1, so a bias is unidentifiable) — matching the authors.
    ``predict_proba`` returns ``P(y=1)`` per label = ``sigmoid(w_CK @ pi)``.

  * **Determinism.** Init is a seeded numpy Generator; refits reproduce bit-for-bit.

``autograd``/numpy/scipy only; id-agnostic; no clinical knowledge.
"""
from __future__ import annotations

import autograd
import numpy as np
from scipy.optimize import minimize

from analysis.pc.slda_reference import (
    DEFAULT_PI_ITERS,
    DEFAULT_PI_STEP_SIZE,
    calc_loss__slda,
    loss_from_param_vec,
    make_convex_alpha_minus_1,
    multinomial_coef_const,
    nef_map_pi_DK,
    pack_param_vec,
    softmax_rows,
    unpack_param_vec,
)


def _as_y_DC(y: np.ndarray, C: int) -> np.ndarray:
    """Coerce labels to a ``(D, C)`` float array of 0/1.

    Accepts a 1D ``(D,)`` binary vector (single label, ``C == 1``) or an already
    2D ``(D, C)`` array.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] != C:
        raise ValueError(f"label array has {y.shape[1]} columns, expected C={C}")
    return y


class PCTopicModel:
    """Faithful flat Prediction-Constrained topic model (in-memory reference).

    Parameters
    ----------
    K : int
        Number of topics.
    C : int, default 1
        Number of binary labels (one logistic head row each). ``C == 1`` is the
        single-label case used by the synthetic gate and toy-bars oracle.
    weight_y : float, default 1.0
        Weight on the prediction loss (the authors' PC dial). ``weight_y == 0`` =>
        unsupervised LDA-MAP (the two-stage baseline representation).
    alpha : float, default 1.1
        Dirichlet concentration on ``pi`` (authors' default). Enters both the NEF
        MAP prior and ``loss_pi``, identically at train and test.
    tau : float, default 1.1
        Dirichlet concentration on topics (``loss_topics`` prior).
    lambda_w : float, default 0.001
        L2 penalty on the head weights (scaled by ``weight_y``, authors' default).
    weight_x : float, default 1.0
        Weight on the generative word likelihood.
    weight_pi : float, default 1.0
        Weight on the ``pi`` Dirichlet MAP prior term.
    pi_iters : int, default 100
        FIXED number of NEF exponentiated-gradient iterations to unroll for
        ``pi``-inference (see the module note on the unroll choice). Identical at
        train and test.
    pi_step_size : float, default 0.005
        NEF step size (authors' default).
    rescale_by_n_tokens : bool, default True
        Divide the total loss by the token count (authors' default).
    max_iter : int, default 500
        L-BFGS-B iteration cap for ``fit``.
    seed : int, default 0
        Seed for the random parameter init. Fits are deterministic given seed+data.
    init_scale : float, default 0.5
        Std of the Gaussian init for ``w_KV`` (topic-word logits). The head ``w_CK``
        inits at zero.

    Fitted attributes
    -----------------
    topics_ : (K, V) topic-word simplex rows.
    w_CK_ : (C, K) logistic head weights.
    Pi_ : (D, K) fitted train doc-topic proportions (label-free MAP).
    """

    def __init__(
        self,
        K: int,
        C: int = 1,
        weight_y: float = 1.0,
        alpha: float = 1.1,
        tau: float = 1.1,
        lambda_w: float = 0.001,
        weight_x: float = 1.0,
        weight_pi: float = 1.0,
        pi_iters: int = DEFAULT_PI_ITERS,
        pi_step_size: float = DEFAULT_PI_STEP_SIZE,
        rescale_by_n_tokens: bool = True,
        max_iter: int = 500,
        seed: int = 0,
        init_scale: float = 0.5,
    ) -> None:
        self.K = int(K)
        self.C = int(C)
        self.weight_y = float(weight_y)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.lambda_w = float(lambda_w)
        self.weight_x = float(weight_x)
        self.weight_pi = float(weight_pi)
        self.pi_iters = int(pi_iters)
        self.pi_step_size = float(pi_step_size)
        self.rescale_by_n_tokens = bool(rescale_by_n_tokens)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.init_scale = float(init_scale)

    # -- internal ----------------------------------------------------------
    def _loss_kwargs(self) -> dict:
        """Hyperparameter kwargs shared by every ``calc_loss__slda`` call."""
        return dict(
            alpha=self.alpha,
            tau=self.tau,
            lambda_w=self.lambda_w,
            weight_x=self.weight_x,
            weight_y=self.weight_y,
            weight_pi=self.weight_pi,
            pi_iters=self.pi_iters,
            pi_step_size=self.pi_step_size,
            rescale_total_loss_by_n_tokens=self.rescale_by_n_tokens,
        )

    def _init_param_vec(self, V: int) -> np.ndarray:
        """Seeded packed init: Gaussian ``w_KV`` (breaks topic symmetry), zero head."""
        rng = np.random.default_rng(self.seed)
        w_KV = self.init_scale * rng.standard_normal((self.K, V))
        w_CK = np.zeros((self.C, self.K))
        return pack_param_vec(w_KV, w_CK)

    # -- fit ---------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labeled_mask: np.ndarray | None = None,
    ) -> "PCTopicModel":
        """Fit global ``(w_KV, w_CK)`` by L-BFGS-B on the faithful PC loss.

        ``labeled_mask is None`` => every doc labeled. When ``weight_y == 0`` the
        label and ``labeled_mask`` are ignored (unsupervised LDA-MAP).
        """
        X = np.asarray(X, dtype=np.float64)
        D, V = X.shape
        self.D_, self.V_ = D, V
        y_DC = _as_y_DC(y, self.C)

        if labeled_mask is None:
            y_rowmask = np.ones(D)
        else:
            y_rowmask = np.asarray(labeled_mask, dtype=float)

        mult_const = multinomial_coef_const(X)
        x0 = self._init_param_vec(V)

        loss_kwargs = self._loss_kwargs()

        def objective(vec):
            return loss_from_param_vec(
                vec, X_DV=X, y_DC=y_DC, y_rowmask=y_rowmask,
                K=self.K, V=V, C=self.C,
                mult_coef_const_val=mult_const,
                **loss_kwargs,
            )

        grad_fn = autograd.grad(objective)

        self.init_obj_ = float(objective(x0))
        res = minimize(
            lambda v: float(objective(v)),
            x0,
            jac=lambda v: np.asarray(grad_fn(v), dtype=np.float64),
            method="L-BFGS-B",
            options=dict(maxiter=self.max_iter),
        )
        self.result_ = res
        self.final_obj_ = float(res.fun)
        self.n_iter_ = int(res.nit)

        w_KV, w_CK = unpack_param_vec(res.x, K=self.K, V=V, C=self.C)
        self.topics_ = np.asarray(softmax_rows(w_KV))
        self.w_CK_ = np.asarray(w_CK)
        self.Pi_ = np.asarray(
            nef_map_pi_DK(
                self.topics_, X, make_convex_alpha_minus_1(self.alpha),
                pi_iters=self.pi_iters, pi_step_size=self.pi_step_size,
            )
        )
        return self

    # -- transform (label-free NEF-MAP; IDENTICAL routine to train) --------
    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """Infer doc-topic proportions for new docs with ``topics_`` frozen.

        Runs the SAME label-free unrolled NEF-MAP used inside ``fit`` — no label,
        no train/test mismatch. Returns ``Pi_new`` (N, K) on the simplex.
        """
        if not hasattr(self, "topics_"):
            raise RuntimeError("transform called before fit")
        X_new = np.asarray(X_new, dtype=np.float64)
        N, V = X_new.shape
        if V != self.V_:
            raise ValueError(f"vocab mismatch: fit V={self.V_}, got {V}")
        return np.asarray(
            nef_map_pi_DK(
                self.topics_, X_new, make_convex_alpha_minus_1(self.alpha),
                pi_iters=self.pi_iters, pi_step_size=self.pi_step_size,
            )
        )

    # -- predict -----------------------------------------------------------
    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        """Per-label ``P(y=1) = sigmoid(w_CK @ pi)`` for new docs. Shape (N, C)."""
        Pi_new = self.transform(X_new)
        logits = Pi_new @ self.w_CK_.T          # (N, C)
        return 1.0 / (1.0 + np.exp(-logits))

    # -- diagnostics -------------------------------------------------------
    def loss_terms(self, X: np.ndarray, y: np.ndarray, labeled_mask=None) -> dict:
        """Return every loss term (as numpy floats) at the fitted parameters.

        Convenience for tests/reports; evaluates :func:`calc_loss__slda` with
        ``return_dict=True`` at ``topics_``/``w_CK_``.
        """
        X = np.asarray(X, dtype=np.float64)
        y_DC = _as_y_DC(y, self.C)
        D = X.shape[0]
        y_rowmask = np.ones(D) if labeled_mask is None else np.asarray(labeled_mask, float)
        return calc_loss__slda(
            self.topics_, self.w_CK_, X, y_DC, y_rowmask,
            return_dict=True, **self._loss_kwargs(),
        )
