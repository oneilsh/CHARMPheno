"""PC-family VARIANT wrapper: the free-pi ``PCTopicModelFreePi`` with an
sklearn-flavored ``fit`` / ``transform`` / ``predict_proba`` surface.

.. warning::

    This is **NOT** Hughes et al.'s algorithm. At *train* time it gives each
    document a FREE per-doc ``pi`` (``u_d``) that the label loss is allowed to
    reshape directly — so a doc's train ``pi`` is label-informed while its *test*
    ``pi`` (from label-free :meth:`transform`) is not, reintroducing exactly the
    train/test representation mismatch that weakens sLDA. The faithful reference
    (:class:`analysis.pc.model.PCTopicModel`) instead infers ``pi`` by label-free
    generative MAP identically at train and test, and reshapes the *global*
    topics through that inference.

    We keep this class deliberately: it is the natural **seed for the future
    VI-native port** (the label-shaped E-step is what a supervised-VI runner
    computes cheaply in ``local_update``), and its factored objective in
    :mod:`analysis.pc.objective` is the composable head + generative split. Use
    the faithful ``PCTopicModel`` for correctness work; use this only as the
    VI-port seed / free-pi contrast.

This is the in-memory variant API. It owns no math beyond wiring: ``fit``
optimizes the PC Lagrangian (:func:`analysis.pc.objective.pc_objective`) by
full-batch L-BFGS-B over the packed unconstrained parameters ``(w, u, eta, b)``
from a seeded random init; ``transform`` runs label-free heldout pi inference
(``beta_`` held FIXED, generative-only per-doc objective); ``predict_proba``
reads the class distribution off the head applied to the inferred topic
frequencies.

Design decisions worth knowing:

  * **Head init at zero.** ``eta`` and ``b`` start at 0 (topic-word ``w`` and
    doc-topic ``u`` start at seeded Gaussian noise to break topic symmetry).
    Because the PC objective's eta/b gradient is identically 0 when ``lam == 0``
    (see the objective docstring), a ``lam == 0`` fit leaves the head at zero,
    so ``predict_proba`` is *exactly* uniform (chance) and the fit reduces to an
    unsupervised LDA-MAP representation. That is the two-stage baseline's
    representation — reuse this class with ``lam=0``.

  * **Transform is label-free by construction.** ``transform`` never takes ``y``
    and never touches ``eta_``/``b_``: it minimizes only the generative term over
    a fresh doc's ``u_d`` with ``beta_`` frozen. This is the MAP topic estimate
    you would get at prediction time, when no label exists. Optimization is done
    jointly over all new docs' ``u`` in one L-BFGS-B call — docs are independent
    given ``beta_``, so the joint optimum equals the per-doc optima, at less
    Python overhead.

  * **Determinism.** All randomness is a seeded ``numpy`` Generator; refitting
    with the same seed and inputs reproduces the fit bit-for-bit.

numpy/scipy only; id-agnostic; no clinical knowledge.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from analysis.pc.generative import generative_neg_loglik
from analysis.pc.objective import (
    _softmax_vjp,
    pack_params,
    pc_objective,
    unpack_params,
)


def _generative_only_value_grad(
    u_flat: np.ndarray,
    *,
    beta: np.ndarray,
    X: np.ndarray,
    alpha: float,
    N: int,
    K: int,
) -> tuple[float, np.ndarray]:
    """Value and grad (w.r.t. flat ``u``) of the generative-only objective with
    ``beta`` held fixed — the per-doc heldout-inference objective, vectorized
    over all ``N`` docs. No labels enter here."""
    Pi = softmax(u_flat.reshape(N, K), axis=1)
    val, _grad_beta, grad_Pi = generative_neg_loglik(beta, Pi, X, alpha)
    grad_u = _softmax_vjp(Pi, grad_Pi)
    return float(val), grad_u.ravel()


class PCTopicModelFreePi:
    """Free-pi PC-family variant (NOT Hughes' algorithm; see module docstring).

    Parameters
    ----------
    K : int
        Number of topics.
    C : int
        Number of classes for the supervised head.
    lam : float
        PC multiplier scaling the prediction constraint. ``lam == 0`` gives an
        unsupervised LDA-MAP fit (the two-stage baseline representation).
    alpha : float, default 1.0
        Symmetric Dirichlet concentration on the doc-topic proportions. Used
        identically in ``fit`` and in heldout ``transform``. ``alpha == 1`` => no
        prior.
    max_iter : int, default 500
        L-BFGS-B iteration cap for ``fit``.
    transform_max_iter : int, default 500
        L-BFGS-B iteration cap for heldout ``transform`` inference.
    seed : int, default 0
        Seed for the random parameter init (and the transform init). Fits are
        deterministic given the seed and inputs.
    init_scale : float, default 0.5
        Std of the Gaussian init for ``w`` (topic-word) and ``u`` (doc-topic).
        The head (``eta``, ``b``) always inits at zero.

    Fitted attributes
    -----------------
    beta_ : (K, V) topic-word simplex rows.
    eta_ : (C, K) head weights.
    b_ : (C,) head bias.
    Pi_ : (D, K) fitted train doc-topic proportions.
    """

    def __init__(
        self,
        K: int,
        C: int,
        lam: float,
        alpha: float = 1.0,
        max_iter: int = 500,
        transform_max_iter: int = 500,
        seed: int = 0,
        init_scale: float = 0.5,
    ) -> None:
        self.K = int(K)
        self.C = int(C)
        self.lam = float(lam)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.transform_max_iter = int(transform_max_iter)
        self.seed = int(seed)
        self.init_scale = float(init_scale)

    # -- fit ---------------------------------------------------------------
    def _init_params(self, D: int, V: int) -> np.ndarray:
        """Seeded packed init: Gaussian ``w``/``u``, zero head."""
        rng = np.random.default_rng(self.seed)
        w = self.init_scale * rng.standard_normal((self.K, V))
        u = self.init_scale * rng.standard_normal((D, self.K))
        eta = np.zeros((self.C, self.K))
        b = np.zeros(self.C)
        return pack_params(w, u, eta, b)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        labeled_mask: np.ndarray | None = None,
    ) -> "PCTopicModelFreePi":
        """Fit by full-batch L-BFGS-B on the PC Lagrangian.

        ``labeled_mask is None`` => every doc is labeled. When ``lam == 0`` the
        label is ignored entirely (unsupervised LDA-MAP).
        """
        X = np.asarray(X, dtype=np.float64)
        D, V = X.shape
        self.D_, self.V_ = D, V
        y = np.asarray(y)
        if labeled_mask is None:
            labeled_mask = np.ones(D, dtype=bool)
        else:
            labeled_mask = np.asarray(labeled_mask, dtype=bool)

        x0 = self._init_params(D, V)

        def fun(flat: np.ndarray) -> tuple[float, np.ndarray]:
            return pc_objective(
                flat, X=X, y=y, labeled_mask=labeled_mask,
                K=self.K, C=self.C, lam=self.lam, alpha=self.alpha,
            )

        self.init_obj_ = float(fun(x0)[0])
        res = minimize(
            fun, x0, jac=True, method="L-BFGS-B",
            options=dict(maxiter=self.max_iter),
        )
        self.result_ = res
        self.final_obj_ = float(res.fun)
        self.n_iter_ = int(res.nit)

        w, u, eta, b = unpack_params(res.x, K=self.K, V=V, D=D, C=self.C)
        self.beta_ = softmax(w, axis=1)
        self.eta_ = np.array(eta)
        self.b_ = np.array(b)
        self.Pi_ = softmax(u, axis=1)
        return self

    # -- transform (label-free heldout inference) --------------------------
    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """Infer doc-topic proportions for new docs with ``beta_`` frozen.

        Minimizes the generative-only objective over the new docs' ``u`` — no
        label is used or needed. Returns ``Pi_new`` (N, K) on the simplex.
        """
        if not hasattr(self, "beta_"):
            raise RuntimeError("transform called before fit")
        X_new = np.asarray(X_new, dtype=np.float64)
        N, V = X_new.shape
        if V != self.V_:
            raise ValueError(f"vocab mismatch: fit V={self.V_}, got {V}")

        # Seeded init, distinct from the fit seed stream but reproducible.
        rng = np.random.default_rng(self.seed + 1)
        u0 = self.init_scale * rng.standard_normal((N, self.K))

        res = minimize(
            lambda uf: _generative_only_value_grad(
                uf, beta=self.beta_, X=X_new, alpha=self.alpha, N=N, K=self.K
            ),
            u0.ravel(),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.transform_max_iter),
        )
        return softmax(res.x.reshape(N, self.K), axis=1)

    # -- predict -----------------------------------------------------------
    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        """Class probabilities: ``softmax(transform(X_new) @ eta_.T + b_)``."""
        Pi_new = self.transform(X_new)
        logits = Pi_new @ self.eta_.T + self.b_
        return softmax(logits, axis=1)
