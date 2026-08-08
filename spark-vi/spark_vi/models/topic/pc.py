"""OnlinePCLDA: VI-native Prediction-Constrained topic model as a VIModel.

This is INCREMENT 1 of the faithful VI port of ``analysis/pc`` — the
UNSUPERVISED SVI scaffolding (``weight_y == 0``). At ``weight_y == 0`` the
Prediction-Constrained objective collapses to unsupervised LDA-MAP (the
two-stage baseline's representation), so ``OnlinePCLDA`` is deliberately
identical to ``OnlineLDA`` on this path: the label never enters inference and
the global step is exactly the LDA λ natural-gradient step. A logistic head
``w_CK`` (C labels × K topics) is SEEDED (zeros) but UNUSED — carried so the
global-parameter shape is stable when increment 2 attaches the supervised head
SGD + topic correction (mirroring how ``OnlineSTM`` seeds Γ and refines it in a
ρ-blended non-conjugate M-step).

Generative model (words), identical to ``OnlineLDA``:
    theta_d ~ Dirichlet(alpha · 1_K);  z_dn ~ Cat(theta_d);  w_dn ~ Cat(beta_z)
    beta_k ~ Dirichlet(eta · 1_V)
Supervised head (present, inert at weight_y == 0; see analysis/pc/head.py):
    P(y_dc = 1) = sigmoid(w_CK[c] · pi_d),   pi_d the label-free doc-topic mix.

Variational mean field: same as ``OnlineLDA`` —
    q(beta_k) = Dirichlet(lambda_k)  (global, K×V)
    q(theta_d) = Dirichlet(gamma_d)  (local, K)

Design contract (mapping onto VIModel):
    initialize_global -> LDA globals {lambda, alpha, eta} PLUS w_CK (C×K zeros).
    local_update      -> LABEL-FREE per-doc CAVI (reuses OnlineLDA), emits
                         lambda_stats exactly as OnlineLDA; emits NO supervised
                         stats at weight_y == 0. Increment-2 seam marked inline.
    update_global     -> LDA λ natural-gradient step (reuses OnlineLDA); the head
                         stays at init. Increment-2 seam marked inline.
    infer_local       -> the SAME label-free CAVI as local_update (train/test π
                         consistency — the faithfulness invariant).
    compute_elbo      -> unsupervised LDA ELBO (fine for increment 1).

Numpy/scipy only (charter): the supervised gradient-through-inference of
increment 2 is hand-coded numpy, never autograd. Increment 1 needs no gradient.

References:
    Hughes, Hope, Weiner, McCoy, Perlis, Sudderth, Doshi-Velez 2017/2018.
        Prediction-Constrained topic models.
    Hoffman, Blei, Bach 2010; Hoffman, Blei, Wang, Paisley 2013 (Online/SVI LDA).
    analysis/pc/model.py (the exact in-memory oracle this increment validates
        against at weight_y == 0).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from spark_vi.core.model import VIModel
from spark_vi.models.topic.lda import OnlineLDA
from spark_vi.models.topic.types import PCDocument


class OnlinePCLDA(VIModel):
    """Prediction-Constrained LDA fittable by VIRunner with mini-batch SVI.

    Increment 1 (``weight_y == 0``): the unsupervised SVI path. Behaves exactly
    like ``OnlineLDA`` — same recovery, same ELBO — because on this path every
    contract method delegates its LDA math to an internal ``OnlineLDA`` and adds
    only the (inert) head. The equivalence is by construction, not by
    coincidence: it is the exact same code path, which is the whole point of the
    ``weight_y == 0`` faithfulness gate.

    The supervised content (``weight_y > 0``) is increment 2 and is NOT built
    here. Every place it will attach is marked with an ``INCREMENT 2 SEAM``
    comment and, if reached, raises ``NotImplementedError`` — so a caller who
    sets ``weight_y > 0`` gets a clear failure rather than a silently wrong fit.
    Zero supervised risk.

    Parameters mirror ``OnlineLDA`` plus:
        C:        number of binary outcome heads (rows of ``w_CK``); default 1.
        weight_y: PC prediction-loss weight. MUST be 0.0 in increment 1;
                  any other value raises NotImplementedError at the supervised
                  seam. Carried so the shim's ``weightY`` param is fully wired.
    """

    def __init__(
        self,
        K: int,
        vocab_size: int,
        C: int = 1,
        weight_y: float = 0.0,
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
        ``OnlineSTM`` seeds Γ = 0. At weight_y == 0 it is never touched.
        """
        gp = self._lda.initialize_global(data_summary)
        gp["w_CK"] = np.zeros((self.C, self.K), dtype=np.float64)
        return gp

    def local_update(
        self,
        rows: Iterable[PCDocument],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one Spark partition — LABEL-FREE (y never enters here).

        At weight_y == 0 this is byte-for-byte ``OnlineLDA.local_update``: run
        the label-free CAVI per doc, accumulate ``lambda_stats`` (+ the ELBO
        scalars and, if optimize_alpha, ``e_log_theta_sum``). No supervised
        statistic is emitted, so the default ``combine_stats`` sums exactly the
        LDA suff-stats and ``update_global`` reduces to the LDA λ step.

        INCREMENT 2 SEAM (weight_y > 0): here — for observed cells only,
        ``obs_dc = label_mask[d, c]`` — accumulate the partial head gradient
        ``Σ_d ∂loss_y/∂w_CK`` and the partial supervised topic gradient
        ``Σ_d ∂loss_y/∂topics`` (both dense additive arrays the default
        combine_stats will sum). The label-free CAVI above is UNCHANGED by that
        addition; only extra keys join the returned dict. Not built here.
        """
        if self.weight_y == 0.0:
            # Pure unsupervised path — delegate wholesale so the numbers are
            # identical to OnlineLDA. PCDocument.indices/.counts are all the
            # delegate reads; y/label_mask ride along untouched.
            return self._lda.local_update(rows, global_params)

        # INCREMENT 2 SEAM: materialize the partition so both the LDA stats and
        # the supervised partial stats read the same y/label_mask-carrying rows.
        rows = list(rows)
        stats = self._lda.local_update(rows, global_params)
        raise NotImplementedError(
            "OnlinePCLDA supervised local stats (weight_y > 0) are increment 2; "
            "increment 1 supports weight_y == 0 (unsupervised) only. "
            f"partial LDA stats already computed for {len(stats)} keys."
        )

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """M-step at rho_t — LDA λ natural-gradient step; head stays at init.

        At weight_y == 0 the LDA globals {lambda, alpha, eta} are updated
        verbatim by ``OnlineLDA.update_global``
        (``(1-ρ)λ + ρ(η + expElogβ · lambda_stats)``, plus the optional α/η
        Newton steps), and ``w_CK`` is passed through unchanged — the head never
        moves off its zero seed.

        INCREMENT 2 SEAM (weight_y > 0): after the unsupervised λ step, attach
        (b) the supervised topic correction
        ``λ ← λ − ρ·weight_y·(∂loss_y/∂topics stats)`` and (c) the head SGD
        ``w_CK ← w_CK − ρ·(∂loss_y/∂w_CK + weight_y·λ_w·2·w_CK)`` (the
        ``OnlineSTM`` Γ ridge-M-step template, damped by the runner's RM ρ_t).
        λ's unsupervised part stays closed-form; the head + correction are the
        only gradient pieces. Not built here.
        """
        new_gp = self._lda.update_global(global_params, target_stats, learning_rate)

        if self.weight_y != 0.0:
            # INCREMENT 2 SEAM: supervised λ correction + head SGD go here,
            # reading the supervised partial stats emitted by local_update.
            raise NotImplementedError(
                "OnlinePCLDA supervised global correction + head SGD "
                "(weight_y > 0) are increment 2; increment 1 supports "
                "weight_y == 0 (unsupervised) only."
            )

        # Head unchanged at weight_y == 0 (stays at its zero seed).
        new_gp["w_CK"] = global_params["w_CK"]
        return new_gp

    def combine_stats(
        self,
        a: dict[str, np.ndarray],
        b: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Elementwise-sum suff-stat dicts — the LDA delegate's combiner.

        At weight_y == 0 the emitted stats are exactly LDA's dense arrays, so
        the default elementwise sum is correct. Increment 2's supervised partial
        stats are also dense additive arrays, so this stays valid there too.
        """
        return self._lda.combine_stats(a, b)

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """Unsupervised LDA ELBO (doc-likelihood + doc KL − global β KL).

        Fine for increment 1: at weight_y == 0 the objective IS the unsupervised
        bound. Increment 2 adds the supervised ``−weight_y·loss_y`` term.
        """
        return self._lda.compute_elbo(global_params, aggregated_stats)

    def infer_local(self, row: PCDocument, global_params: dict[str, np.ndarray]):
        """Per-doc label-free CAVI — the IDENTICAL routine to local_update.

        This is the faithfulness invariant: train-time and test-time π come from
        the same label-free E-step (there is no train/test representation
        mismatch), mirroring ``OnlineLDA.infer_local`` and the reference's shared
        ``nef_map_pi_DK``. The head ``w_CK`` is NOT read here — a probability
        column from the head (``sigmoid(w_CK · θ)``) is increment 2.
        """
        return self._lda.infer_local(row, global_params)

    def iteration_summary(self, global_params: dict[str, np.ndarray]) -> str:
        """LDA per-iter summary plus the (inert) head magnitude, for a glance
        at whether increment 2 has started moving the head."""
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
        """LDA concentration traces (α, η). The head is omitted while inert."""
        return self._lda.iteration_diagnostics(global_params)
