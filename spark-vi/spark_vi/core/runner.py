"""VIRunner: the training-loop driver for distributed variational inference.

Each iteration executes the canonical distributed-VI step:

    1. Optionally sample a mini-batch from the input RDD (with replacement).
    2. Broadcast current global params to all partitions.
    3. mapPartitions: each worker runs model.local_update and emits stats.
    4. treeReduce: tree-shaped sum of stats across partitions
       (via model.combine_stats), keeping driver memory bounded to a single
       per-partition stats dict rather than the sum of all of them.
    5. Driver: pre-scale the aggregated stats to a corpus-equivalent target
       (corpus_size / batch_size) when in mini-batch mode, then call
       model.update_global with Robbins-Monro learning rate.
    6. Record ELBO (raw, not pre-scaled); test convergence.

The MLlib `OnlineLDAOptimizer` uses an equivalent pattern (with
treeAggregate); see docs/architecture/SPARK_VI_FRAMEWORK.md and
docs/decisions/0005-mini-batch-sampling.md for references.
"""
from __future__ import annotations

import logging
import random
from typing import Any

from pyspark import RDD, StorageLevel

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

    def fit(
        self,
        data_rdd: RDD,
        data_summary: Any | None = None,
        start_iteration: int = 0,
    ) -> VIResult:
        """Run the distributed VI loop until convergence or max_iterations.

        start_iteration offsets the Robbins-Monro step counter for resumed
        runs: after a checkpoint at iteration k, pass start_iteration=k so
        the first post-resume rho matches what a continuous run would use.
        """
        model = self.model
        cfg = self.config

        global_params = model.initialize_global(data_summary)
        elbo_trace: list[float] = []
        sc = data_rdd.context
        prior_bcast = None
        converged = False

        # If mini-batching is enabled, count the corpus once and seed the RNG
        # used to derive per-iteration sample seeds. corpus_size matches the
        # MLlib OnlineLDAOptimizer convention of using corpus_size / batch_size
        # as the natural-gradient scale.
        if cfg.mini_batch_fraction is not None:
            corpus_size = data_rdd.count()
            rng = random.Random(cfg.random_seed)
        else:
            corpus_size = None
            rng = None

        for step in range(cfg.max_iterations):
            t = start_iteration + step

            # 1. Sample a mini-batch (or use the full RDD).
            if cfg.mini_batch_fraction is not None:
                batch_rdd = data_rdd.sample(
                    withReplacement=cfg.sample_with_replacement,
                    fraction=cfg.mini_batch_fraction,
                    seed=rng.randint(0, 2 ** 31 - 1),
                )
                # Cache the sampled RDD so count() and mapPartitions don't each
                # recompute the sample lineage. Without persistence, sampling
                # would be triggered twice per iteration.
                batch_rdd = batch_rdd.persist(StorageLevel.MEMORY_AND_DISK)
                batch_size = batch_rdd.count()
                if batch_size == 0:
                    batch_rdd.unpersist(blocking=False)
                    log.info("Iteration %d skipped: empty mini-batch", step + 1)
                    continue
                stats_scale = float(corpus_size) / float(batch_size)
            else:
                batch_rdd = data_rdd
                stats_scale = 1.0

            # 2. Broadcast current global params.
            bcast = sc.broadcast(global_params)

            # 3 & 4. Distributed E-step + aggregate.
            def _local(rows, _bcast=bcast, _model=model):
                return [_model.local_update(rows, _bcast.value)]

            # treeReduce is the tree-shaped tree-aggregate; driver memory holds
            # one merged stats dict, not the per-partition list. Requires
            # combine_stats to be associative + commutative (already required
            # by the VIModel contract for additive sufficient statistics).
            aggregated = batch_rdd.mapPartitions(_local).treeReduce(model.combine_stats)

            # 5. Pre-scale aggregated stats to form the natural-gradient target.
            # In mini-batch mode this multiplies each ndarray by corpus / batch
            # so the model's update_global sees an unbiased corpus-equivalent.
            # In full-batch mode (stats_scale == 1.0) the dict is passed through
            # unchanged.
            if stats_scale != 1.0:
                target_stats = {k: v * stats_scale for k, v in aggregated.items()}
            else:
                target_stats = aggregated

            # M-step (Robbins-Monro step size).
            # Hoffman et al. 2013 index t from 1 so the first rho is (tau0+1)^-kappa
            # rather than (tau0)^-kappa — the latter can collapse to 1.0 and
            # force a full jump on the first step (see SVI paper §3).
            rho_t = (cfg.learning_rate_tau0 + t + 1) ** -cfg.learning_rate_kappa
            global_params = model.update_global(global_params, target_stats, learning_rate=rho_t)

            # 6. ELBO + convergence. Pass the *raw* aggregated stats (not the
            # pre-scaled target_stats) so ELBO terms representing observed
            # data evidence stay correct.
            elbo = model.compute_elbo(global_params, aggregated)
            elbo_trace.append(float(elbo))

            # Unpersist the *previous* broadcast so we don't leak them.
            # See RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
            if prior_bcast is not None:
                prior_bcast.unpersist(blocking=False)
            prior_bcast = bcast

            # Free the cached batch RDD before the next iteration's sample.
            if cfg.mini_batch_fraction is not None:
                batch_rdd.unpersist(blocking=False)

            if model.has_converged(elbo_trace, cfg.convergence_tol):
                converged = True
                log.info("Converged at iteration %d (ELBO=%.6f)", step + 1, elbo)
                # One-more unpersist for the final broadcast.
                prior_bcast.unpersist(blocking=False)
                prior_bcast = None
                return VIResult(
                    global_params=global_params,
                    elbo_trace=elbo_trace,
                    n_iterations=step + 1,
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
