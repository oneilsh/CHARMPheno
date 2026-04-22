"""VIRunner: the training-loop driver for distributed variational inference.

Each iteration executes the canonical distributed-VI step:

    1. Broadcast current global params to all partitions.
    2. mapPartitions: each worker runs model.local_update and emits stats.
    3. treeAggregate: sum stats across partitions (via model.combine_stats).
    4. Driver: model.update_global with Robbins-Monro learning rate.
    5. Record ELBO; test convergence.

The MLlib `OnlineLDAOptimizer` uses an identical pattern; see
docs/architecture/SPARK_VI_FRAMEWORK.md for references.
"""
from __future__ import annotations

import logging
from typing import Any

from pyspark import RDD

from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult

log = logging.getLogger(__name__)


class VIRunner:
    """Drives a VIModel through iterations of distributed VI on a Spark RDD."""

    def __init__(self, model: VIModel, config: VIConfig | None = None) -> None:
        if not isinstance(model, VIModel):
            raise TypeError(f"model must be a VIModel subclass, got {type(model).__name__}")
        self.model = model
        self.config = config if config is not None else VIConfig()

    def fit(self, data_rdd: RDD, data_summary: Any | None = None) -> VIResult:
        """Run the distributed VI loop until convergence or max_iterations."""
        model = self.model
        cfg = self.config

        global_params = model.initialize_global(data_summary)
        elbo_trace: list[float] = []
        sc = data_rdd.context
        prior_bcast = None
        converged = False

        for t in range(cfg.max_iterations):
            # 1. Broadcast current global params.
            bcast = sc.broadcast(global_params)

            # 2 & 3. Distributed E-step + aggregate.
            def _local(rows, _bcast=bcast, _model=model):
                return [_model.local_update(rows, _bcast.value)]

            stats_seq = data_rdd.mapPartitions(_local).collect()
            aggregated = stats_seq[0]
            for more in stats_seq[1:]:
                aggregated = model.combine_stats(aggregated, more)

            # 4. M-step (Robbins-Monro step size).
            # Hoffman et al. 2013 index t from 1 so the first rho is (tau0+1)^-kappa
            # rather than (tau0)^-kappa — the latter can collapse to 1.0 and
            # force a full jump on the first step (see SVI paper §3).
            rho_t = (cfg.learning_rate_tau0 + t + 1) ** -cfg.learning_rate_kappa
            global_params = model.update_global(global_params, aggregated, learning_rate=rho_t)

            # 5. ELBO + convergence.
            elbo = model.compute_elbo(global_params, aggregated)
            elbo_trace.append(float(elbo))

            # Unpersist the *previous* broadcast so we don't leak them.
            # See RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
            if prior_bcast is not None:
                prior_bcast.unpersist(blocking=False)
            prior_bcast = bcast

            if model.has_converged(elbo_trace, cfg.convergence_tol):
                converged = True
                log.info("Converged at iteration %d (ELBO=%.6f)", t + 1, elbo)
                # One-more unpersist for the final broadcast.
                prior_bcast.unpersist(blocking=False)
                prior_bcast = None
                return VIResult(
                    global_params=global_params,
                    elbo_trace=elbo_trace,
                    n_iterations=t + 1,
                    converged=True,
                    metadata={"model_class": type(model).__name__},
                )

        # Hit max_iterations without convergence.
        if prior_bcast is not None:
            prior_bcast.unpersist(blocking=False)
        return VIResult(
            global_params=global_params,
            elbo_trace=elbo_trace,
            n_iterations=cfg.max_iterations,
            converged=False,
            metadata={"model_class": type(model).__name__},
        )
