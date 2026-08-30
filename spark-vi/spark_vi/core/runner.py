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
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pyspark import RDD, StorageLevel

from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult
from spark_vi.diagnostics.persist import assert_persisted
from spark_vi.io.export import load_result, save_result

log = logging.getLogger(__name__)


def _runner_metadata(model, **extra):
    """Merge model.get_metadata() with runner-set keys.

    Runner-set keys (passed as kwargs) win over model-supplied keys.
    Used at all three VIResult-construction sites so the precedence
    rule lives in one place and downstream code can rely on
    `metadata['model_class']` being the actual class name even if a
    misconfigured model returns a conflicting key.
    """
    return {**model.get_metadata(), "model_class": type(model).__name__, **extra}


_AGG_DEPTH_THRESHOLD_BYTES = 128 * 1024 * 1024


def _agg_depth(global_params: dict) -> int:
    """treeReduce depth for the stats aggregate, sized from the params payload.

    Each per-partition sufficient-stats partial has (at least) the dense shape
    of the global params — for the topic models, a lambda-shaped (K, V) dict —
    so the params' ndarray bytes are a driver-side proxy for the partial size
    the reduce will ship. At depth d over P partitions the driver receives
    ~P^(1/d) partials in one burst: fine at 41 MB/partial (K=444), fatal at
    355 MB/partial (whole-Mondo K~3,800, exp 0104 smoke: driver-JVM heap OOM at
    iteration 11 under depth 2 / 8g). Above the threshold, depth 3 trades one
    extra executor combine round for a ~P^(1/6)-fold smaller driver burst.
    Recurses because multi-domain lambda is a dict-of-arrays inside the dict.
    """
    def _nbytes(v) -> int:
        if isinstance(v, dict):
            return sum(_nbytes(x) for x in v.values())
        return int(getattr(v, "nbytes", 0))

    return 3 if _nbytes(global_params) > _AGG_DEPTH_THRESHOLD_BYTES else 2


def _destroy_broadcast(bcast) -> None:
    """Fully release a per-iteration broadcast: executor blocks, the driver
    block-manager copy, AND the driver-local pickled temp file.

    ``unpersist`` frees only the executor copies; the other two survive until
    ``destroy`` and accumulate at one λ-sized pickle per fit iteration — which at
    whole-Mondo scale (~355 MB each) filled the master's disk over a 100-iteration
    fit and killed the run's post-fit readout with ENOSPC (exp 0104, 2026-08-30).
    Cleanup must never outrank the fit itself, so failures degrade to unpersist
    and a log line rather than raising."""
    try:
        bcast.destroy()
    except Exception as exc:
        try:
            bcast.unpersist(blocking=False)
        except Exception:
            pass
        log.warning("broadcast destroy fell back to unpersist: %s", exc)


def _fmt_diagnostic(value: object) -> str:
    """Compact formatter for one iteration_diagnostics value.

    Scalars (Python or NumPy) format to ~4 sig figs. 0-d ndarrays are treated
    as scalars. 1-D arrays show up to 6 elements with 4 sig figs each;
    higher-dim arrays are flattened then truncated (no shape prefix). Longer
    arrays get an ellipsis. Anything else falls back to repr — diagnostic log
    lines are best-effort and shouldn't crash the fit.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return f"{float(value):.4g}"
        flat = value.ravel()
        if not np.issubdtype(flat.dtype, np.number):
            # Non-numeric arrays (e.g. string label arrays from topic_block_labels)
            # fall back to repr rather than crashing on float() conversion.
            return repr(flat[:6].tolist()) + (", ..." if flat.size > 6 else "")
        head = ", ".join(f"{float(x):.4g}" for x in flat[:6])
        suffix = ", ..." if flat.size > 6 else ""
        return f"[{head}{suffix}]"
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{float(value):.4g}"
    return repr(value)


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
        warm_start_from: Path | str | None = None,
        on_iteration: Callable[[int, dict, list[float]], None] | None = None,
    ) -> VIResult:
        """Run the distributed VI loop until convergence (full-batch) or max_iterations.

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
            warm_start_from: if set, load a previously-saved VIResult's
                global_params and use them as this fit's INITIAL global params —
                but with a FRESH iteration counter (start_iteration is left at
                the passed value, default 0) and a FRESH (empty) elbo_trace.
                This is the warm-INIT counterpart to resume_from and is
                deliberately DISTINCT from it: resume_from CONTINUES the
                Robbins-Monro counter (so rho_t is already small on the first
                resumed step, correct for continuing an in-progress schedule),
                whereas warm_start_from RESETS the counter so rho_t restarts
                near rho_0 and the schedule can actually move parameters that
                begin at a fresh init. The motivating use is the unsupervised
                warm-start protocol (Hughes et al.): fit phase 1 at weight_y == 0
                to learn topics (leaving the head at its zero init), then
                warm_start_from that checkpoint for a supervised phase 2 whose
                head must train against a full, undecayed rho schedule — a
                decayed (resume-style) rho would leave the head barely able to
                move. The caller owns the semantic content of the loaded params:
                for the PC model a weight_y == 0 phase-1 checkpoint yields warm
                topics (lambda) AND a fresh (zero) head by construction, since
                the unsupervised path never moves w_CK off its zero seed. Mutually
                exclusive with resume_from (one continues the counter, the other
                resets it — supplying both is a contradiction and raises).
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

        if resume_from is not None and warm_start_from is not None:
            raise ValueError(
                "resume_from and warm_start_from are mutually exclusive: "
                "resume_from CONTINUES the Robbins-Monro iteration counter "
                "(rho already decayed), while warm_start_from RESETS it (fresh "
                "rho schedule). Pass at most one."
            )

        if resume_from is not None:
            loaded = load_result(resume_from)
            global_params = loaded.global_params
            elbo_trace: list[float] = list(loaded.elbo_trace)
            start_iteration = loaded.n_iterations
            log.info(
                "Resuming from %s (n_iterations=%d, converged=%s)",
                resume_from, loaded.n_iterations, loaded.converged,
            )
        elif warm_start_from is not None:
            # Warm-INIT (distinct from resume): adopt the saved global params as
            # this fit's starting point but keep start_iteration (default 0) and
            # an empty elbo_trace, so the Robbins-Monro schedule restarts from
            # rho_0 rather than continuing the saved run's decayed schedule.
            loaded = load_result(warm_start_from)
            global_params = loaded.global_params
            elbo_trace = []
            log.info(
                "Warm-starting from %s (loaded global params as init; FRESH "
                "iteration counter t=%d, so rho restarts near rho_0 — distinct "
                "from resume, which would continue the decayed schedule)",
                warm_start_from, start_iteration,
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

        # Per-iteration trajectories of model-supplied scalars/small-arrays.
        # Populated below from model.iteration_diagnostics(); empty if the
        # model doesn't override the default.
        diagnostic_traces: dict[str, list[Any]] = {}

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
            # Depth is sized from the stats payload (see _agg_depth): each
            # per-partition partial is a DENSE lambda-shaped dict, so at
            # whole-Mondo scale (K~3,800 x V~11,600 = ~355 MB float64) a
            # depth-2 tree still lands ~sqrt(P) such partials on the driver
            # JVM in one burst per iteration — which is exactly how exp 0104's
            # first smoke OOM'd an 8g driver heap at iteration 11. Depth 3
            # pushes one more combine round onto the executors and cuts the
            # driver burst to ~P^(1/3) partials, for one extra shuffle round
            # that is noise next to the E-step.
            aggregated = batch_rdd.mapPartitions(_local).treeReduce(
                model.combine_stats, depth=_agg_depth(bcast.value))

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

            # Per-iteration diagnostics: accumulate into traces and emit a
            # compact key=value log line. Default impl returns {}, so models
            # that don't opt in pay nothing here.
            diagnostics = model.iteration_diagnostics(global_params)
            if diagnostics:
                # We don't .copy() the values — models must return fresh objects per
                # call (don't mutate-and-return the same array across iterations).
                for key, value in diagnostics.items():
                    diagnostic_traces.setdefault(key, []).append(value)
                parts = [f"{k}={_fmt_diagnostic(v)}" for k, v in diagnostics.items()]
                log.info("  diagnostics: " + ", ".join(parts))

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
                    metadata=_runner_metadata(model, checkpoint=True),
                    diagnostic_traces={k: list(v) for k, v in diagnostic_traces.items()},
                )
                save_result(interim, cfg.checkpoint_dir)

            # DESTROY the *previous* broadcast so we don't leak it — and destroy,
            # not unpersist, because unpersist frees the EXECUTOR copies only.
            # pyspark writes every broadcast's pickle to a temp file on the
            # DRIVER's local disk and keeps a driver block-manager copy, and both
            # survive until destroy(). At whole-Mondo scale that is ~355 MB of λ
            # per fit iteration: exp 0104's 100-iteration record fit (2026-08-30)
            # leaked the master's whole disk and its post-fit readout died on
            # `[Errno 28] No space left on device` from `sc.broadcast` itself —
            # the same lifecycle bug already fixed on the readout side
            # (analysis/cloud/distributed_readout._destroy_broadcast). Destroy is
            # safe here for the same reason the deferred-unpersist was: iteration
            # t's aggregate has already RETURNED before iteration t+1 reaches
            # this line, so nothing references the prior broadcast any more.
            # See RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
            if prior_bcast is not None:
                _destroy_broadcast(prior_bcast)
            prior_bcast = bcast

            # Free the cached batch RDD before the next iteration's sample.
            if cfg.mini_batch_fraction is not None:
                batch_rdd.unpersist(blocking=False)

            # Early-stop only in FULL-BATCH mode. In mini-batch SVI the
            # per-iteration ELBO is computed on a fresh random sub-sample, so
            # its iteration-to-iteration variance makes a two-point
            # relative-change test fire on coincidental closeness, not true
            # convergence (repeatedly observed: false stops at iter 67 and 157
            # of a 200-iter budget). Spark MLlib's reference online LDA
            # (OnlineLDAOptimizer) has NO convergence tolerance and runs exactly
            # maxIterations for this reason; we match it — mini-batch fits run
            # the full budget, full-batch fits keep the valid corpus-wide check.
            if (cfg.mini_batch_fraction is None
                    and model.has_converged(elbo_trace, cfg.convergence_tol)):
                converged = True
                log.info("Converged at iteration %d (ELBO=%.6f)", step + 1, elbo)
                # One-more destroy for the final broadcast.
                _destroy_broadcast(prior_bcast)
                prior_bcast = None
                result = VIResult(
                    global_params=global_params,
                    elbo_trace=elbo_trace,
                    n_iterations=t + 1,
                    converged=True,
                    metadata=_runner_metadata(model),
                    diagnostic_traces=diagnostic_traces,
                )
                # Final-save guarantee: when checkpoint_dir is set, the
                # post-fit directory is authoritative regardless of where
                # the interval boundary falls.
                if cfg.checkpoint_dir is not None:
                    save_result(result, cfg.checkpoint_dir)
                return result

        # Hit max_iterations without convergence.
        if prior_bcast is not None:
            _destroy_broadcast(prior_bcast)
        result = VIResult(
            global_params=global_params,
            elbo_trace=elbo_trace,
            n_iterations=start_iteration + cfg.max_iterations,
            converged=False,
            metadata=_runner_metadata(model),
            diagnostic_traces=diagnostic_traces,
        )
        if cfg.checkpoint_dir is not None:
            save_result(result, cfg.checkpoint_dir)
        return result

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
        # Unlike fit() — whose per-iteration broadcasts are unpersisted AFTER
        # that iteration's action has read them — a lazy transform has no action
        # of its own, so unpersisting now frees nothing and forces a re-broadcast
        # on every downstream action. Spark's ContextCleaner reclaims the
        # broadcast when the returned RDD is garbage-collected, matching how
        # MLlib's own models manage their transform broadcasts.
        return data_rdd.map(_infer)
