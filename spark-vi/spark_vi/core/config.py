"""Training-loop hyperparameters for VIRunner.

See docs/architecture/SPARK_VI_FRAMEWORK.md#viconfig for the design rationale.
Hoffman, Blei, Wang, Paisley 2013 ("Stochastic Variational Inference") set
the tau0 / kappa conventions for the Robbins-Monro step size schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VIConfig:
    """Immutable configuration for the VI training loop.

    Attributes:
        max_iterations: hard upper bound on training iterations *per fit
            invocation*. On a resume (VIRunner.fit(..., resume_from=path)
            or shim resumeFrom), this is the number of *additional* iters
            to run, not a target total — i.e. resuming a 50-iter
            checkpoint with max_iterations=30 yields 80 total iters.
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
        checkpoint_interval: if set, the runner writes a VIResult to
            checkpoint_dir every N iterations during fit() (overwriting the
            previous checkpoint — last one is the only one needed for resume).
            Must be set together with checkpoint_dir or not at all.
        checkpoint_dir: target directory for auto-checkpoints. The runner
            writes via spark_vi.io.export.save_result, producing the same
            human-inspectable manifest.json + params/*.npy layout used for
            final-result exports. Must be set together with checkpoint_interval.
        mini_batch_fraction: if set, the runner samples this fraction of the
            input RDD per iteration (with replacement by default), matching
            the OnlineLDAOptimizer subsamplingRate convention. Must lie in
            (0, 1]. None means "use the full RDD every iteration" (legacy
            full-batch behavior). Without sampling, the Robbins-Monro decaying
            step size will eventually stall full-batch updates before
            convergence; mini-batching is what makes the schedule appropriate.
        sample_with_replacement: passed through to RDD.sample. Default True
            matches MLlib OnlineLDAOptimizer and the standard SVI assumption
            that each iteration draws an i.i.d. batch.
        random_seed: optional root seed for the per-iteration sample seeds.
            If None, sampling is non-reproducible across runs. Even when set,
            Spark partition ordering may introduce small floating-point
            variation; reproducibility is best-effort, not bit-exact.

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
    checkpoint_dir: Path | str | None = None
    mini_batch_fraction: float | None = None
    sample_with_replacement: bool = True
    random_seed: int | None = None

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
        if (self.checkpoint_interval is None) != (self.checkpoint_dir is None):
            raise ValueError(
                "checkpoint_interval and checkpoint_dir must both be set or both "
                f"be None; got interval={self.checkpoint_interval!r}, "
                f"dir={self.checkpoint_dir!r}"
            )
        if self.mini_batch_fraction is not None and not (
            0.0 < self.mini_batch_fraction <= 1.0
        ):
            raise ValueError(
                f"mini_batch_fraction must be None or in (0, 1], "
                f"got {self.mini_batch_fraction}"
            )
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise ValueError(
                f"random_seed must be None or int, got {type(self.random_seed).__name__}"
            )
