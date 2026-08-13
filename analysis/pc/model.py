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
    global_terms_from_param_vec,
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


def _as_label_mask_DC(label_mask, D: int, C: int) -> np.ndarray | None:
    """Coerce a per-cell observed-mask to a ``(D, C)`` float 0/1 array, or None.

    ``None`` passes through unchanged (all cells observed). A 1D ``(D,)`` mask for
    the single-label case ``C == 1`` is promoted to a column. Any other shape that
    is not exactly ``(D, C)`` is a caller error.
    """
    if label_mask is None:
        return None
    m = np.asarray(label_mask, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    if m.shape != (D, C):
        raise ValueError(
            f"label_mask has shape {m.shape}, expected (D, C) = {(D, C)}"
        )
    return m


class PCTopicModel:
    """Faithful flat Prediction-Constrained topic model (in-memory reference).

    Parameters
    ----------
    K : int
        Number of topics.
    C : int, default 1
        Number of binary labels/outcome heads (one logistic head row each).
        ``C == 1`` is the single-label case used by the synthetic gate and
        toy-bars oracle; ``C > 1`` with a per-cell ``label_mask`` in :meth:`fit`
        is the joint multi-task / index-drug mode (one shared topic model, ``C``
        heads, each head trained only on its outcome's observed cells).
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
    doc_batch_size : int, default 2048
        Document-minibatch size for assembling the FULL-batch objective and
        gradient inside ``fit``. Reverse-mode autograd through the unrolled
        ``pi``-inference retains every intermediate (each of ``pi_iters`` steps
        holds ``D x V`` arrays), so differentiating the whole corpus at once needs
        a tape ~ ``D x pi_iters x V`` — tens of GB at real-corpus scale. Because
        the loss is a plain SUM over documents plus a handful of global terms,
        ``fit`` instead partitions the ``D`` training docs into contiguous
        minibatches of this size, differentiates each minibatch's per-document loss
        separately, and ACCUMULATES value+gradient; the global terms
        (``loss_topics``, ``loss_w``, the multinomial coefficient) and the single
        ``/ scale`` are applied once outside the loop. The result is the exact same
        full-batch objective/gradient handed to L-BFGS-B, with peak tape bounded to
        one minibatch (~ ``doc_batch_size x pi_iters x V``). When
        ``doc_batch_size >= D`` the fit takes the original single-shot full-batch
        path, byte-for-byte identical to before this knob existed (so small-``D``
        tests and the oracle are unchanged). Does NOT change the objective's math
        or the optimizer — only how the gradient is assembled.
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
        doc_batch_size: int = 2048,
        seed: int = 0,
        init_scale: float = 0.5,
        fit_mode: str = "joint",
        alt_rounds: int = 30,
        alt_block_maxiter: int = 50,
        alt_tol: float = 1e-6,
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
        self.doc_batch_size = int(doc_batch_size)
        if self.doc_batch_size < 1:
            raise ValueError("doc_batch_size must be a positive integer")
        self.seed = int(seed)
        self.init_scale = float(init_scale)
        # Optimization mode — the joint-vs-alternating isolation dial.
        #   'joint'       (default, faithful): L-BFGS-B over the CONCATENATED
        #                 (w_KV, w_CK) vector, so the quasi-Newton curvature model
        #                 spans the topic<->head cross-block. This is what Hughes /
        #                 the oracle do; unchanged from before this knob existed.
        #   'alternating': block-coordinate — repeatedly L-BFGS the topic block with
        #                 the head fixed, then the head block with topics fixed. Holds
        #                 EVERYTHING else identical (same objective, pi-MAP, L2, init,
        #                 full-batch, same L-BFGS solver) so the ONLY difference from
        #                 'joint' is whether the optimizer sees the coupled vector or
        #                 alternates over blocks — isolating the online OnlinePCLDA
        #                 scheme's alternating structure at reference convergence.
        if fit_mode not in ("joint", "alternating"):
            raise ValueError(f"fit_mode must be 'joint' or 'alternating', got {fit_mode!r}")
        self.fit_mode = str(fit_mode)
        self.alt_rounds = int(alt_rounds)
        self.alt_block_maxiter = int(alt_block_maxiter)
        self.alt_tol = float(alt_tol)

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

    def _make_minibatch_value_and_grad(
        self, X, y_DC, y_rowmask, label_mask_DC, V, mult_const, loss_kwargs,
    ):
        """Build the ``vec -> (value, grad)`` closure for the minibatch fit path.

        Returns a callable that reproduces the FULL-batch objective and gradient
        of :func:`~analysis.pc.slda_reference.loss_from_param_vec` exactly, but
        assembles them by accumulating over contiguous document minibatches so the
        autograd tape never exceeds one minibatch. The decomposition it exploits::

            loss_ttl = ( sum_d [loss_x_d + loss_pi_d + loss_y_d]      # per-doc
                         + loss_topics + loss_w                       # global
                         - weight_x * mult_coef_const ) / scale       # global const

        Per-minibatch calls set ``include_global_terms=False``,
        ``include_mult_coef=False`` and ``rescale_total_loss_by_n_tokens=False`` so
        each returns ONLY its documents' ``loss_x + loss_pi + loss_y`` (unscaled,
        the minibatch's own token sum never enters). The global terms are
        differentiated once via
        :func:`~analysis.pc.slda_reference.global_terms_from_param_vec`; the
        multinomial coefficient is a parameter-free constant (zero gradient) added
        once; and the whole accumulated value+gradient is divided by the single
        GLOBAL ``scale = sum(X)`` (or 1 when ``rescale_by_n_tokens`` is off) — never
        a per-minibatch scale. Both masks are sliced per minibatch so the observed
        set is unchanged.

        The returned callable is suitable for ``scipy.optimize.minimize(...,
        jac=True)``: it returns ``(float value, float64 gradient)``.
        """
        D = X.shape[0]
        bs = self.doc_batch_size
        bounds = [(i, min(i + bs, D)) for i in range(0, D, bs)]
        scale = float(np.sum(X)) if self.rescale_by_n_tokens else 1.0

        # Per-doc kwargs: strip the token rescale (applied once, globally) and both
        # the global regularizers and the multinomial constant (added once, below).
        perdoc_kwargs = dict(loss_kwargs)
        perdoc_kwargs["rescale_total_loss_by_n_tokens"] = False
        perdoc_kwargs["include_global_terms"] = False
        perdoc_kwargs["include_mult_coef"] = False

        def _perdoc_loss(vec, lo, hi):
            mb = None if label_mask_DC is None else label_mask_DC[lo:hi]
            return loss_from_param_vec(
                vec, X_DV=X[lo:hi], y_DC=y_DC[lo:hi], y_rowmask=y_rowmask[lo:hi],
                label_mask=mb, K=self.K, V=V, C=self.C,
                mult_coef_const_val=0.0, **perdoc_kwargs,
            )

        perdoc_vg = autograd.value_and_grad(_perdoc_loss)
        global_vg = autograd.value_and_grad(global_terms_from_param_vec)
        # Parameter-free constant part of loss_x (zero gradient), added once.
        mult_term = -float(self.weight_x) * float(mult_const)
        global_kwargs = dict(
            K=self.K, V=V, C=self.C, tau=self.tau,
            lambda_w=self.lambda_w, weight_y=self.weight_y,
        )

        def value_and_grad(vec):
            vec = np.asarray(vec, dtype=np.float64)
            total_val = 0.0
            total_grad = np.zeros_like(vec)
            for lo, hi in bounds:
                v, g = perdoc_vg(vec, lo, hi)
                total_val += float(v)
                total_grad += np.asarray(g, dtype=np.float64)
            gv, gg = global_vg(vec, **global_kwargs)
            total_val += float(gv) + mult_term
            total_grad += np.asarray(gg, dtype=np.float64)
            return total_val / scale, total_grad / scale

        return value_and_grad

    # -- fit ---------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labeled_mask: np.ndarray | None = None,
        label_mask: np.ndarray | None = None,
    ) -> "PCTopicModel":
        """Fit global ``(w_KV, w_CK)`` by L-BFGS-B on the faithful PC loss.

        Parameters
        ----------
        X : (D, V) nonnegative counts.
        y : (D, C) binary labels (a 1D (D,) vector is the single-label ``C == 1``
            case). Values at *unobserved* cells are ignored, so any placeholder is
            fine there.
        labeled_mask : (D,) or None
            Per-ROW semi-supervision. ``None`` => every row labeled. A row with
            ``labeled_mask[d] == 0`` contributes to ``loss_x``/``loss_pi`` (its
            words still shape the shared topics) but never to ``loss_y``.
        label_mask : (D, C) or None
            Per-CELL semi-supervision for joint multi-task fitting — ``True``/1
            marks cell ``(d, c)`` as observed. This is the index-drug mode: one
            shared topic model, ``C`` outcome heads, each head trained only on the
            cells observed for its outcome, so an almost-all-missing label matrix
            (about one observed cell per row) still trains every head off the
            shared representation. ``None`` => all cells of a labeled row observed.
            The two masks compose by logical AND — the effective observed set is
            ``obs(d, c) = labeled_mask[d] AND label_mask[d, c]`` — so switching a
            row off with ``labeled_mask`` drops all of its cells whatever
            ``label_mask`` says.

        When ``weight_y == 0`` the labels and both masks are ignored entirely
        (unsupervised LDA-MAP). Backward compatible: with ``label_mask=None`` the
        objective is byte-for-byte the previous per-row behavior.
        """
        X = np.asarray(X, dtype=np.float64)
        D, V = X.shape
        self.D_, self.V_ = D, V
        y_DC = _as_y_DC(y, self.C)

        if labeled_mask is None:
            y_rowmask = np.ones(D)
        else:
            y_rowmask = np.asarray(labeled_mask, dtype=float)
        label_mask_DC = _as_label_mask_DC(label_mask, D, self.C)

        mult_const = multinomial_coef_const(X)
        x0 = self._init_param_vec(V)

        loss_kwargs = self._loss_kwargs()

        if self.fit_mode == "alternating":
            return self._fit_alternating(
                x0, X, y_DC, y_rowmask, label_mask_DC, V, mult_const, loss_kwargs
            )

        if self.doc_batch_size >= D:
            # --- Single-shot full-batch path (unchanged; byte-for-byte the pre-
            # minibatch behavior). Used whenever the whole corpus fits one batch,
            # so small-D tests and the oracle are numerically identical. ----------
            def objective(vec):
                return loss_from_param_vec(
                    vec, X_DV=X, y_DC=y_DC, y_rowmask=y_rowmask,
                    label_mask=label_mask_DC,
                    K=self.K, V=V, C=self.C,
                    mult_coef_const_val=mult_const,
                    **loss_kwargs,
                )

            grad_fn = autograd.grad(objective)
            self.init_obj_ = float(objective(x0))
            self.n_doc_batches_ = 1
            res = minimize(
                lambda v: float(objective(v)),
                x0,
                jac=lambda v: np.asarray(grad_fn(v), dtype=np.float64),
                method="L-BFGS-B",
                options=dict(maxiter=self.max_iter),
            )
        else:
            # --- Document-minibatch accumulation path (real-corpus scale). --------
            # The full loss is (sum over docs of per-doc terms + global terms) /
            # scale. We differentiate each contiguous doc minibatch's per-doc loss
            # separately and accumulate value+grad, then add the (document-
            # independent) global terms and the constant multinomial coefficient
            # ONCE and divide by the single GLOBAL scale. This yields the EXACT
            # full-batch objective/gradient while bounding the autograd tape to one
            # minibatch. See PCTopicModel.doc_batch_size.
            value_and_grad = self._make_minibatch_value_and_grad(
                X, y_DC, y_rowmask, label_mask_DC, V, mult_const, loss_kwargs
            )
            self.init_obj_ = float(value_and_grad(x0)[0])
            self.n_doc_batches_ = int(np.ceil(D / self.doc_batch_size))
            res = minimize(
                value_and_grad,
                x0,
                jac=True,
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

    def _fit_alternating(
        self, x0, X, y_DC, y_rowmask, label_mask_DC, V, mult_const, loss_kwargs
    ) -> "PCTopicModel":
        """Block-coordinate fit: alternate L-BFGS over the topic and head blocks.

        Isolates the joint-vs-alternating axis. ``_make_minibatch_value_and_grad``
        yields the EXACT full-batch ``value_and_grad(vec)`` (one batch when
        ``doc_batch_size >= D``), so each block sub-problem simply reuses it and
        slices the gradient to its own coordinates — the partial derivative w.r.t. a
        block IS that slice, exactly what L-BFGS-B needs. Same objective / pi-MAP /
        L2 / init / solver as the joint path; only the coupling is severed.
        """
        D = X.shape[0]
        n_w = self.K * V                                   # split: [w_KV | w_CK]
        full_vg = self._make_minibatch_value_and_grad(
            X, y_DC, y_rowmask, label_mask_DC, V, mult_const, loss_kwargs
        )
        vec = np.asarray(x0, dtype=np.float64).copy()
        self.init_obj_ = float(full_vg(vec)[0])
        self.n_doc_batches_ = int(np.ceil(D / self.doc_batch_size))

        def block_vg(sub, sl):
            v = vec.copy()
            v[sl] = sub
            val, g = full_vg(v)
            return float(val), np.asarray(g[sl], dtype=np.float64)

        topic_sl, head_sl = slice(0, n_w), slice(n_w, None)
        prev = self.init_obj_
        total_iters = 0
        obj_trace = [prev]
        for _ in range(self.alt_rounds):
            rt = minimize(lambda s: block_vg(s, topic_sl), vec[topic_sl], jac=True,
                          method="L-BFGS-B", options=dict(maxiter=self.alt_block_maxiter))
            vec[topic_sl] = rt.x
            rh = minimize(lambda s: block_vg(s, head_sl), vec[head_sl], jac=True,
                          method="L-BFGS-B", options=dict(maxiter=self.alt_block_maxiter))
            vec[head_sl] = rh.x
            cur = float(full_vg(vec)[0])
            total_iters += int(rt.nit) + int(rh.nit)
            obj_trace.append(cur)
            if abs(prev - cur) <= self.alt_tol * max(1.0, abs(prev)):
                prev = cur
                break
            prev = cur

        self.result_ = None
        self.final_obj_ = prev
        self.n_iter_ = total_iters
        self.alt_obj_trace_ = obj_trace
        w_KV, w_CK = unpack_param_vec(vec, K=self.K, V=V, C=self.C)
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
    def loss_terms(
        self, X: np.ndarray, y: np.ndarray, labeled_mask=None, label_mask=None
    ) -> dict:
        """Return every loss term (as numpy floats) at the fitted parameters.

        Convenience for tests/reports; evaluates :func:`calc_loss__slda` with
        ``return_dict=True`` at ``topics_``/``w_CK_``. ``label_mask`` (D, C)
        restricts ``loss_y`` to the observed cells, AND-composed with the per-row
        ``labeled_mask`` exactly as in :meth:`fit`.
        """
        X = np.asarray(X, dtype=np.float64)
        y_DC = _as_y_DC(y, self.C)
        D = X.shape[0]
        y_rowmask = np.ones(D) if labeled_mask is None else np.asarray(labeled_mask, float)
        label_mask_DC = _as_label_mask_DC(label_mask, D, self.C)
        return calc_loss__slda(
            self.topics_, self.w_CK_, X, y_DC, y_rowmask, label_mask_DC,
            return_dict=True, **self._loss_kwargs(),
        )
