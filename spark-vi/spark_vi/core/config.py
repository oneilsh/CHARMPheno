"""Training-loop hyperparameters for VIRunner.

See docs/architecture/SPARK_VI_FRAMEWORK.md#viconfig for the design rationale.
Hoffman, Blei, Wang, Paisley 2013 ("Stochastic Variational Inference") set
the tau0 / kappa conventions for the Robbins-Monro step size schedule.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VIConfig:
    """Immutable configuration for the VI training loop.

    Attributes:
        max_iterations: hard upper bound on training iterations.
        learning_rate_tau0: delay parameter in rho_t = (tau0 + t)^-kappa, where
            t indexes iterations from 1 per Hoffman et al. 2013. The runner
            indexes iterations from 0 internally, so it evaluates the
            equivalent (tau0 + t + 1)^-kappa; both forms yield the same step
            sizes.
        learning_rate_kappa: decay exponent. Validation accepts (0, 1]; the
            Robbins-Monro convergence guarantee (sum rho_t = inf and
            sum rho_t^2 < inf) holds only for kappa in (0.5, 1]. Values in
            (0, 0.5] are permitted for experimentation but are not guaranteed
            to converge.
        convergence_tol: relative ELBO improvement threshold for early stop.
        checkpoint_interval: if set, write checkpoint every N iterations.

    Why these defaults: tau0=1.0, kappa=0.7 is Hoffman et al. 2013's common
    choice for stochastic VI on text corpora; it biases toward faster initial
    progress at the cost of larger late-iteration updates. Models processing
    full batches should override kappa (see RISKS_AND_MITIGATIONS.md).
    """
    max_iterations: int = 100
    learning_rate_tau0: float = 1.0
    learning_rate_kappa: float = 0.7
    convergence_tol: float = 1e-4
    checkpoint_interval: int | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.learning_rate_tau0 <= 0.0:
            raise ValueError(f"learning_rate_tau0 must be > 0, got {self.learning_rate_tau0}")
        if not (0.0 < self.learning_rate_kappa <= 1.0):
            raise ValueError(
                f"learning_rate_kappa must be in (0, 1], got {self.learning_rate_kappa}"
            )
        if self.convergence_tol <= 0.0:
            raise ValueError(f"convergence_tol must be > 0, got {self.convergence_tol}")
        if self.checkpoint_interval is not None and self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be None or >= 1, got {self.checkpoint_interval}"
            )
