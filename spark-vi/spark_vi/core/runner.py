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
    6. Record ELBO (raw, not pre-scaled); auto-checkpoint if configured;
       test convergence.

The MLlib `OnlineLDAOptimizer` uses an equivalent pattern (with
treeAggregate); see docs/architecture/SPARK_VI_FRAMEWORK.md and
docs/decisions/0005-mini-batch-sampling.md for references. Auto-checkpoint
and resume_from semantics are described in
docs/decisions/0006-unified-persistence-format.md.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

from pyspark import RDD, StorageLevel

from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult
from spark_vi.diagnostics.persist import assert_persisted
from spark_vi.io.export import load_result, save_result

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
        resume_from: Path | str | None = None,
        on_iteration: Callable[[int, dict, list[float]], None] | None = None,
    ) -> VIResult:
        """Run the distributed VI loop until convergence or max_iterations.

        Parameters:
            data_rdd: input RDD to train on.
            data_summary: optional pre-pass metadata for model.initialize_global.
            start_iteration: offsets the Robbins-Monro step counter. Internal —
                callers wanting to resume a checkpointed run should prefer
                resume_from, which sets this automatically.
            resume_from: if set, load a previously-saved VIResult from this
                path (written by save_result or by the runner's own
                auto-checkpoint mechanism) and continue training. The loaded
                global_params replace what model.initialize_global would have
                returned; the loaded elbo_trace seeds this run's trace; and
                start_iteration is set to the loaded result's n_iterations so
                the Robbins-Monro schedule matches a continuous run.
            on_iteration: optional diagnostic callback invoked after each
                iteration as `fn(iter_num, global_params, elbo_trace)`.
                Kwarg-on-fit rather than a method on VIModel because the
                callback is per-invocation observation, not model state —
                models stay diagnostic-free; each fit can opt in differently.
                The callback runs on the driver in the fit's hot path; keep
                it cheap or throttle with a modulo. Must not mutate
                global_params — the same dict feeds the next iteration's
                broadcast. The runner does not defensive-copy (deep-copy of
                a (K, V) lambda every iter is too expensive for a diagnostic
                path); document-the-contract is the chosen tradeoff.
                Exceptions are caught and logged so a buggy diagnostic
                doesn't kill the fit.
        """
        model = self.model
        cfg = self.config

        if resume_from is not None:
            loaded = load_result(resume_from)
            global_params = loaded.global_params
            elbo_trace: list[float] = list(loaded.elbo_trace)
            start_iteration = loaded.n_iterations
            log.info(
                "Resuming from %s (n_iterations=%d, converged=%s)",
                resume_from, loaded.n_iterations, loaded.converged,
            )
        else:
            global_params = model.initialize_global(data_summary)
            elbo_trace = []

        # Strict precondition: data_rdd must be cached. Loop-heavy training
        # otherwise re-executes the upstream lineage (e.g. a BigQuery scan)
        # every iteration. See spark_vi.diagnostics.persist for the rationale
        # behind raising vs. logging here.
        assert_persisted(data_rdd, name="VIRunner.data_rdd")

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
            t_iter_start = time.perf_counter()

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
            # Default-arg closure capture (`_bcast=bcast, _model=model`) is the
            # Spark-safe convention for shipping closures to executors. Python
            # would otherwise capture `bcast` and `model` as free variables via
            # `__closure__`, which leaves them subject to two failure modes:
            # (1) if the enclosing scope mutates the name between definition
            # and pickling, the closure picks up the mutated value; (2)
            # cloudpickle's handling of deeply-nested lexical scopes has been
            # historically inconsistent. Default args are bound at function-
            # definition time and stored in `__defaults__`, which is pinned-by-
            # value and pickles cleanly. Same idiom used in transform() below.
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

            # Per-iteration progress line. INFO level so it can be surfaced
            # by configuring `spark_vi` to INFO without firehosing root.
            iter_dt = time.perf_counter() - t_iter_start
            batch_str = (f"batch={batch_size}" if cfg.mini_batch_fraction
                         else "full-batch")
            log.info(
                "iter %d/%d: ELBO=%.4f, %s, rho=%.4f, %.1fs",
                step + 1, cfg.max_iterations, elbo, batch_str, rho_t, iter_dt,
            )
            # Model-defined summary, emitted as one log.info per line so any
            # configured log formatter prefix (e.g. "[driver]   ") gets reapplied
            # to each line. Empty / missing => skipped.
            model_str = model.iteration_summary(global_params)
            for line in model_str.splitlines():
                if line:
                    log.info(line)

            # Diagnostic callback (model-agnostic; whatever the caller wants
            # to do with global_params). Catch + log so a buggy callback
            # doesn't kill the fit — the model itself is the load-bearing
            # work, the diagnostic is incidental.
            if on_iteration is not None:
                try:
                    on_iteration(step + 1, global_params, elbo_trace)
                except Exception as exc:
                    log.warning("on_iteration callback raised %r — continuing fit",
                                exc)

            # 7. Auto-checkpoint (if configured). Writes a VIResult to
            # cfg.checkpoint_dir every checkpoint_interval iterations,
            # overwriting the previous checkpoint. Done after the global-params
            # update so the on-disk checkpoint reflects the current loop state.
            if (
                cfg.checkpoint_interval is not None
                and (step + 1) % cfg.checkpoint_interval == 0
            ):
                interim = VIResult(
                    global_params=global_params,
                    elbo_trace=list(elbo_trace),
                    n_iterations=t + 1,
                    converged=False,
                    metadata={
                        "model_class": type(model).__name__,
                        "checkpoint": True,
                    },
                )
                save_result(interim, cfg.checkpoint_dir)

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
            n_iterations=start_iteration + cfg.max_iterations,
            converged=False,
            metadata={"model_class": type(model).__name__},
        )

    def transform(self, data_rdd: RDD, global_params: dict[str, Any]) -> RDD:
        """Apply trained global params to infer per-row posteriors.

        One pass over the RDD: broadcasts global_params, calls
        model.infer_local on each row, returns the resulting RDD. No reduce,
        no global update, no checkpoint.

        For models that don't implement infer_local, the per-row map raises
        NotImplementedError when collected.
        """
        sc = data_rdd.context
        bcast = sc.broadcast(global_params)
        model = self.model

        # Default-arg closure-capture pattern; see explanation in fit().
        def _infer(row, _bcast=bcast, _model=model):
            return _model.infer_local(row, _bcast.value)

        try:
            return data_rdd.map(_infer)
        finally:
            # Eager unpersist matches fit()'s broadcast discipline. The
            # returned RDD captures bcast in the closure, so this is safe:
            # Spark resolves bcast.value at task launch time, which has
            # already happened (or will be re-broadcast lazily) when the
            # caller materializes the RDD.
            #
            # Note: if the caller chains .map / .filter and triggers an
            # action much later, the broadcast may already be unpersisted.
            # That is acceptable for transform — Spark re-broadcasts on
            # demand. Long-lived inference pipelines should call
            # .persist() on the returned RDD.
            bcast.unpersist(blocking=False)
