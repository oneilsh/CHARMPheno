"""VIModel: the base class model authors subclass.

See docs/architecture/SPARK_VI_FRAMEWORK.md#the-vimodel-base-class for the
contract's design rationale. The three required methods correspond to the
three slots in the standard distributed VI iteration:

    lambda_{t+1} = (1 - rho_t) * lambda_t  +  rho_t * lambda_hat(sum_p s_p)
    ^--- update_global                        ^--- aggregated local_updates
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np


class VIModel(ABC):
    """Base class for models fittable by VIRunner.

    Subclasses implement initialize_global, local_update, and update_global.
    Optional hooks — combine_stats, compute_elbo, has_converged — have
    sensible defaults.
    """

    @abstractmethod
    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        """Return starting values of the global variational parameters.

        data_summary: optional model-defined summary produced by the framework
            in a pre-pass (e.g., vocabulary size). Models that need nothing
            can ignore this argument.
        """

    @abstractmethod
    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """E-step on one data partition.

        rows: iterable over the partition's local rows.
        global_params: current global variational parameters.
        returns: dict of additive sufficient statistics (or gradient contributions).
        """

    @abstractmethod
    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        """M-step: apply the natural-gradient update with stepsize rho_t.

        target_stats are aggregated sufficient statistics already pre-scaled
        to form the natural-gradient target. A model should compute
        lambda_hat = prior_natural_params + target_stats and interpolate
        against the current global_params via learning_rate. In mini-batch
        mode the runner has multiplied by corpus_size / batch_size so the
        same arithmetic produces an unbiased estimate of the full-corpus
        target; in full-batch mode the values equal the raw aggregated stats.
        Models do not need to know which mode is active.
        """

    # Optional overrides ----------------------------------------------------

    def combine_stats(
        self,
        a: dict[str, np.ndarray],
        b: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Elementwise-sum two sufficient-statistic dicts.

        Default implementation is correct for models whose statistics live
        in dense NumPy arrays (most exponential-family VI models). Override
        for sparse or structured statistics (see RISKS_AND_MITIGATIONS.md).
        """
        keys = set(a) | set(b)
        out: dict[str, np.ndarray] = {}
        for k in keys:
            if k in a and k in b:
                out[k] = np.asarray(a[k]) + np.asarray(b[k])
            elif k in a:
                out[k] = np.asarray(a[k])
            else:
                out[k] = np.asarray(b[k])
        return out

    def compute_elbo(
        self,
        global_params: dict[str, np.ndarray],
        aggregated_stats: dict[str, np.ndarray],
    ) -> float:
        """ELBO surrogate for diagnostics; override for a real bound.

        aggregated_stats here are the raw aggregated sufficient statistics
        over the current iteration's data — not pre-scaled to a corpus-level
        target. Use them for ELBO terms that represent observed data evidence.
        Note this differs from the target_stats passed to update_global.

        Default returns NaN, which callers treat as 'ELBO not available'.

        ELBO term placement pattern
        ---------------------------
        When an ELBO term depends on per-record local state — γ_d, per-doc
        auxiliary normalizers, sample-level expectations, etc. — that is
        already in scope inside local_update, accumulate that term *there*
        as a scalar entry in the returned suff-stats dict. The runner sums
        it across partitions via combine_stats; compute_elbo just reads the
        scalar back out.

        Trying to recover such a term inside compute_elbo is a trap: you'd
        have to either re-run local_update from scratch, or stash per-record
        state into a corpus-sized array (memory blowup on large corpora).

        compute_elbo is the right home for ELBO terms that depend only on
        global_params — typically global Dirichlet/Gaussian KL terms — which
        are cheap to evaluate once on the driver.

        See VanillaLDA for a worked example: doc_loglik_sum and
        doc_theta_kl_sum are accumulated in local_update; the global
        Dirichlet KL on β is the only term computed in compute_elbo.
        """
        return float("nan")

    def has_converged(
        self,
        elbo_trace: list[float],
        convergence_tol: float,
    ) -> bool:
        """Default: converged when the relative ELBO improvement falls below tol.

        Returns False until at least two finite ELBO values are present.
        """
        if len(elbo_trace) < 2:
            return False
        prev, curr = elbo_trace[-2], elbo_trace[-1]
        if not (np.isfinite(prev) and np.isfinite(curr)):
            return False
        denom = max(abs(prev), 1e-12)
        return abs(curr - prev) / denom < convergence_tol

    def infer_local(self, row, global_params: dict[str, np.ndarray]):
        """Optional capability: per-row variational posterior under fixed global params.

        Models with local latent variables (LDA, HDP) override this to enable
        VIRunner.transform. Models without (e.g. CountingModel) leave it
        unimplemented.

        MUST be a pure function of (row, global_params). No dependence on
        instance state from training. This invariant keeps a future MLlib
        Estimator/Transformer compatibility shim mechanical.

        Default raises NotImplementedError naming the concrete subclass.
        Silent fallback would mask a real user error.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement local inference. "
            f"Models without per-row latent variables cannot be used with "
            f"VIRunner.transform()."
        )
