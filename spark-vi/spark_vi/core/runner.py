"""VIRunner: the training-loop driver for distributed variational inference.

Each iteration executes the canonical distributed-VI step:

    1. Optionally sample a mini-batch from the input RDD (with replacement).
    2. Broadcast current global params to all partitions.
    3. mapPartitions: each worker runs model.local_update and emits stats.
    4. treeReduce: tree-shaped sum of stats across partitions
       (via model.combine_stats), keeping driver memory bounded to a single
       per-partition stats dict rather than the sum of all of them.
       Steps 2-4 (and the batch count) run under `retry_spark_action`, so a
       job aborted by a preemption wave is re-run rather than killing the fit.
    5. Driver: pre-scale the aggregated stats to a corpus-equivalent target
       (corpus_size / batch_size) when in mini-batch mode, then call
       model.update_global with Robbins-Monro learning rate.
    6. Record ELBO (raw, not pre-scaled); auto-checkpoint at the configured
       interval; test convergence. When `cfg.checkpoint_dir` is set, the
       runner additionally performs a single final save before returning
       (on either convergence or max-iterations) so the directory is
       authoritative as the post-fit artifact regardless of where the
       iteration count falls relative to `checkpoint_interval`.

The MLlib `OnlineLDAOptimizer` uses an equivalent pattern (with
treeAggregate); see docs/architecture/SPARK_VI_FRAMEWORK.md and
docs/decisions/0005-mini-batch-sampling.md for references. Auto-checkpoint
and resume_from semantics are described in
docs/decisions/0006-unified-persistence-format.md.

Steps 5-6 and all the bookkeeping live in `spark_vi.core.loop.run_svi_loop`,
shared with the in-memory `LocalRunner` (ADR 0051 §2); this module supplies only
the Spark E-step (steps 1-4).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from pyspark import RDD, StorageLevel

from spark_vi.core.broadcast import destroy_broadcast as _destroy_broadcast
from spark_vi.core.config import VIConfig
from spark_vi.core.loop import run_svi_loop
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult
from spark_vi.core.retry import retry_spark_action
from spark_vi.diagnostics.persist import assert_persisted

log = logging.getLogger(__name__)


_AGG_DEPTH_THRESHOLD_BYTES = 128 * 1024 * 1024


def _agg_depth(global_params: dict) -> int:
    """treeReduce depth for the stats aggregate, sized from the params payload.

    Each per-partition sufficient-stats partial has (at least) the dense shape
    of the global params — for the topic models, a lambda-shaped (K, V) dict —
    so the params' ndarray bytes are a driver-side proxy for the partial size
    the reduce will ship. At depth d over P partitions the driver receives
    ~P^(1/d) partials in one burst: fine at 41 MB/partial (K=444), too much for
    an 8 GB driver heap at 355 MB/partial (K~3,800, V~11,600) under depth 2.
    Above the threshold, depth 3 trades one extra executor combine round for a
    ~P^(1/6)-fold smaller driver burst.
    Recurses because multi-domain lambda is a dict-of-arrays inside the dict.
    """
    def _nbytes(v) -> int:
        if isinstance(v, dict):
            return sum(_nbytes(x) for x in v.values())
        return int(getattr(v, "nbytes", 0))

    return 3 if _nbytes(global_params) > _AGG_DEPTH_THRESHOLD_BYTES else 2


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
        """Run the distributed VI loop until convergence (full-batch) or max_iterations.

        The same distinction sets the M-step STEP SIZE: full batch uses rho = 1
        (batch variational EM), mini-batch uses the decaying Robbins-Monro
        rho_t = (tau0 + t + 1)^-kappa (see `spark_vi.core.loop`); so
        learning_rate_tau0/kappa have NO effect on a full-batch fit.

        Mini-batch fits never early-stop — the per-minibatch ELBO is too noisy for
        a valid convergence test, so they run the full max_iterations, matching
        Spark MLlib's online LDA. Only full-batch fits use the ELBO relative-change
        early-stop. "Full-batch" here means cfg.mini_batch_fraction is None; a set
        fraction is always treated as mini-batch — including 1.0, since
        RDD.sample(fraction=1.0) is still a stochastic (non-identity) draw, so its
        ELBO is a sampled quantity. Use None (not 1.0) for full-batch + early-stop.

        The returned VIResult's `metadata` is the merge of
        `model.get_metadata()` with runner-set keys (`model_class`, and
        `checkpoint` for interim saves); runner-set keys win on conflict so
        downstream code can trust the canonical names.

        Per-iteration values from `model.iteration_diagnostics(global_params)`
        accumulate into `VIResult.diagnostic_traces` (one list per key) and
        are also logged each iter as a compact key=value line.

        Final-save guarantee: when `cfg.checkpoint_dir` is set, the runner
        calls `save_result` once before each return path (convergence and
        max-iterations) so the directory is the authoritative post-fit
        artifact regardless of where the iteration count lands relative to
        `checkpoint_interval`. The interim-save loop continues to fire on
        the interval; both interim and final saves write to the same
        directory and overwrite the previous contents.

        Parameters:
            data_rdd: input RDD to train on.
            data_summary: optional model-defined summary, forwarded unchanged
                to model.initialize_global.
            start_iteration: offsets the Robbins-Monro step counter. Internal —
                callers wanting to resume a checkpointed run should prefer
                resume_from, which sets this automatically.
            resume_from: if set, load a previously-saved VIResult from this
                path (written by save_result or by the runner's own
                auto-checkpoint mechanism) and continue training. The loaded
                global_params replace what model.initialize_global would have
                returned; the loaded elbo_trace and diagnostic_traces continue;
                start_iteration is set to the loaded result's n_iterations so
                the Robbins-Monro schedule matches a continuous run; and the
                minibatch seed stream resumes where the checkpoint left it, so
                checkpoint + resume is the same fit as one uninterrupted run
                (given random_seed).
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

        # Strict precondition: data_rdd must be cached. Loop-heavy training
        # otherwise re-executes the upstream lineage (e.g. a BigQuery scan)
        # every iteration. See spark_vi.diagnostics.persist for the rationale
        # behind raising vs. logging here.
        assert_persisted(data_rdd, name="VIRunner.data_rdd")
        sc = data_rdd.context

        # corpus_size matches the MLlib OnlineLDAOptimizer convention of
        # using corpus_size / batch_size as the natural-gradient scale.
        corpus_size = (data_rdd.count() if cfg.mini_batch_fraction is not None
                       else None)

        def _estep(global_params, seed):
            # 1. Sample a mini-batch (or use the full RDD). The sample is a pure
            # function of (data_rdd, seed), so a retried action that recomputes
            # lost partitions from lineage gets the same batch.
            if seed is not None:
                batch_rdd = data_rdd.sample(
                    withReplacement=cfg.sample_with_replacement,
                    fraction=cfg.mini_batch_fraction,
                    seed=seed,
                )
                # Cache the sampled RDD so count() and mapPartitions don't each
                # recompute the sample lineage.
                batch_rdd = batch_rdd.persist(StorageLevel.MEMORY_AND_DISK)
            else:
                batch_rdd = data_rdd

            def _aggregate():
                # 2. Broadcast current global params — INSIDE the retried action,
                # so a retry re-broadcasts instead of reusing blocks that may have
                # died with the executors (retry_spark_action's contract).
                bcast = sc.broadcast(global_params)

                # Default-arg capture binds bcast and model by value at definition
                # time (stored in __defaults__), the Spark-safe closure idiom.
                def _local(rows, _bcast=bcast, _model=model):
                    return [_model.local_update(rows, _bcast.value)]

                # 3 & 4. Distributed E-step + tree-shaped aggregate: the driver
                # holds one merged stats dict, not the per-partition list.
                # Requires combine_stats to be associative + commutative (already
                # required by the VIModel contract for additive statistics).
                # Depth is sized from the stats payload (see _agg_depth): each
                # per-partition partial is a DENSE lambda-shaped dict (~355 MB at
                # K~3,800 x V~11,600), and depth 3 moves one more combine round
                # onto the executors so fewer of them land on the driver at once.
                try:
                    return batch_rdd.mapPartitions(_local).treeReduce(
                        model.combine_stats, depth=_agg_depth(global_params))
                finally:
                    # DESTROY the broadcast once the aggregate has returned (or
                    # failed), not unpersist: unpersist frees only the executor
                    # copies, while the driver's block-manager copy and its
                    # pickled temp file on local disk survive until destroy(). At
                    # ~355 MB of λ per iteration that fills the master's disk over
                    # a long fit. See RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
                    _destroy_broadcast(bcast)

            # Each Spark action is retried across preemption waves (a fresh job
            # resets the per-taskset excludeOnFailure lists); deterministic and
            # application-level failures still fail fast. See spark_vi.core.retry.
            try:
                if seed is not None:
                    batch_size = retry_spark_action(batch_rdd.count,
                                                    label="fit batch count")
                    if batch_size == 0:
                        return None, 0
                else:
                    batch_size = None
                aggregated = retry_spark_action(_aggregate, label="fit E-step")
            finally:
                # Free the cached batch RDD before the next iteration's sample.
                if seed is not None:
                    batch_rdd.unpersist(blocking=False)
            return aggregated, batch_size

        return run_svi_loop(
            model, cfg, _estep,
            corpus_size=corpus_size,
            data_summary=data_summary,
            start_iteration=start_iteration,
            resume_from=resume_from,
            on_iteration=on_iteration,
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

        # Do NOT eagerly unpersist bcast here. The returned RDD captures it in
        # the closure, so the broadcast must live as long as the RDD is used.
        # Unlike fit() — whose per-iteration broadcasts are destroyed AFTER
        # that iteration's action has read them — a lazy transform has no action
        # of its own, so unpersisting now frees nothing and forces a re-broadcast
        # on every downstream action. Spark's ContextCleaner reclaims the
        # broadcast when the returned RDD is garbage-collected, matching how
        # MLlib's own models manage their transform broadcasts.
        return data_rdd.map(_infer)
