"""The SVI iteration loop shared by both engines (ADR 0051 §2).

`VIRunner` (Spark) and `LocalRunner` (in-memory) differ only in how one
iteration's E-step is computed: which rows form the minibatch and where
`model.local_update` runs. Everything else -- the Robbins-Monro step size,
the corpus/batch pre-scaling of the sufficient statistics, the ELBO trace,
diagnostics, the on_iteration hook, auto-checkpoint, the full-batch early stop
and the final save -- lives here once, so the two engines cannot drift apart.

The engine supplies an `estep(global_params, seed)` callable returning
`(aggregated_stats, batch_size)`; `aggregated_stats is None` means the sampled
minibatch was empty and the iteration is skipped (the step counter still
advances, as it always has in VIRunner). `seed` is the per-iteration sample
seed, drawn here from `random.Random(cfg.random_seed)` so both engines consume
the same seed stream; in full-batch mode it is None.

Pure Python + NumPy: importing this module does not import pyspark.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult

log = logging.getLogger("spark_vi.core.runner")

EStep = Callable[[dict, "int | None"], "tuple[dict | None, int]"]


def runner_metadata(model, **extra):
    """Merge model.get_metadata() with runner-set keys.

    Runner-set keys (passed as kwargs) win over model-supplied keys, so
    downstream code can rely on `metadata['model_class']` being the actual
    class name even if a misconfigured model returns a conflicting key.
    """
    return {**model.get_metadata(), "model_class": type(model).__name__, **extra}


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


def run_svi_loop(
    model: VIModel,
    cfg: VIConfig,
    estep: EStep,
    *,
    corpus_size: int | None,
    data_summary: Any | None = None,
    start_iteration: int = 0,
    resume_from: Path | str | None = None,
    on_iteration: Callable[[int, dict, list[float]], None] | None = None,
) -> VIResult:
    """Run the SVI loop; see VIRunner.fit for the user-facing contract.

    `corpus_size` is the number of rows in the full corpus (required in
    mini-batch mode, ignored in full-batch mode): the aggregated minibatch
    stats are pre-scaled by corpus_size / batch_size so update_global sees an
    unbiased corpus-equivalent target (the MLlib OnlineLDAOptimizer convention).
    """
    # Lazy import to break a module-load cycle (fix 2cb3c45c, ported with ADR
    # 0051): spark_vi.io.export imports spark_vi.core.result, which runs
    # spark_vi.core.__init__ -> runner -> this module, so a module-level import here
    # fails for any consumer that imports export BEFORE core (e.g. a post-hoc reader
    # doing `from spark_vi.io.export import load_result` first).
    from spark_vi.io.export import load_result, save_result

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

    minibatch = cfg.mini_batch_fraction is not None
    if minibatch and corpus_size is None:
        raise ValueError("corpus_size is required in mini-batch mode")
    rng = random.Random(cfg.random_seed) if minibatch else None

    # Per-iteration trajectories of model-supplied scalars/small-arrays.
    # Populated below from model.iteration_diagnostics(); empty if the
    # model doesn't override the default.
    diagnostic_traces: dict[str, list[Any]] = {}

    def _result(n_iterations: int, converged: bool, **meta) -> VIResult:
        return VIResult(
            global_params=global_params,
            elbo_trace=list(elbo_trace),
            n_iterations=n_iterations,
            converged=converged,
            metadata=runner_metadata(model, **meta),
            diagnostic_traces={k: list(v) for k, v in diagnostic_traces.items()},
        )

    for step in range(cfg.max_iterations):
        t = start_iteration + step
        t_iter_start = time.perf_counter()

        # 1-4. Sample a minibatch and aggregate its sufficient statistics.
        seed = rng.randint(0, 2 ** 31 - 1) if minibatch else None
        aggregated, batch_size = estep(global_params, seed)
        if aggregated is None:
            log.info("Iteration %d skipped: empty mini-batch", step + 1)
            continue

        # 5. Pre-scale aggregated stats to form the natural-gradient target.
        # In mini-batch mode this multiplies each ndarray by corpus / batch
        # so the model's update_global sees an unbiased corpus-equivalent.
        # In full-batch mode the dict is passed through unchanged.
        stats_scale = float(corpus_size) / float(batch_size) if minibatch else 1.0
        if stats_scale != 1.0:
            target_stats = {k: v * stats_scale for k, v in aggregated.items()}
        else:
            target_stats = aggregated

        # M-step step size. MINI-BATCH uses the decaying Robbins-Monro
        # schedule; FULL BATCH uses rho = 1 (batch variational EM).
        #
        # Why full batch must NOT decay: with the whole corpus every iteration,
        # `target_stats` is a DETERMINISTIC map T(params) of the current
        # parameters, so update_global's `new = (1 - rho)·old + rho·T(old)` moves
        # by rho·‖T(old) - old‖. As rho_t -> 0 the parameters freeze while the
        # fixed-point residual is still large, so (a) the fit lands short of the
        # batch-VB fixed point at any finite budget and (b) the relative-ELBO early
        # stop fires on the vanishing step rather than on convergence. A finite-
        # budget defect (sum_t rho_t diverges for kappa < 1), which is why it went
        # unnoticed. Measured on a 1600-doc gated synthetic at 200 iterations
        # (commit 3f71de55's message): placement MRR 0.856 / top2 0.828 damped
        # vs 0.910 / 0.988 at rho = 1; early stop at iteration 15 with 760 nats
        # still to gain.
        # rho = 1 is also the only full-batch regime the SVI-vs-Gibbs gates
        # validate (tests/_stm_synth.fit_gated_svi_local: "full-batch lr=1.0").
        # (Fix 3f71de55, from a sibling branch; ported with ADR 0051.)
        #
        # Mini-batch: Hoffman et al. 2013 index t from 1 so the first rho is
        # (tau0+1)^-kappa rather than (tau0)^-kappa — the latter can collapse to
        # 1.0 and force a full jump on the first step (see SVI paper §3).
        #
        # optimize_alpha / optimize_eta models apply this same rate to their
        # Newton step, so a full-batch fit takes an UNDAMPED Newton step.
        rho_t = (1.0 if not minibatch else
                 (cfg.learning_rate_tau0 + t + 1) ** -cfg.learning_rate_kappa)
        global_params = model.update_global(global_params, target_stats, learning_rate=rho_t)

        # 6. ELBO. Pass the *raw* aggregated stats (not the pre-scaled
        # target_stats) so ELBO terms representing observed data evidence
        # stay correct.
        elbo = model.compute_elbo(global_params, aggregated)
        elbo_trace.append(float(elbo))

        # Per-iteration progress line. INFO level so it can be surfaced
        # by configuring `spark_vi` to INFO without firehosing root.
        iter_dt = time.perf_counter() - t_iter_start
        batch_str = f"batch={batch_size}" if minibatch else "full-batch"
        log.info(
            "iter %d/%d: ELBO=%.4f, %s, rho=%.4f, %.1fs",
            step + 1, cfg.max_iterations, elbo, batch_str, rho_t, iter_dt,
        )
        # Model-defined summary, emitted as one log.info per line so any
        # configured log formatter prefix (e.g. "[driver]   ") gets reapplied
        # to each line. Empty / missing => skipped.
        for line in model.iteration_summary(global_params).splitlines():
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
                log.warning("on_iteration callback raised %r — continuing fit", exc)

        # 7. Auto-checkpoint (if configured). Writes a VIResult to
        # cfg.checkpoint_dir every checkpoint_interval iterations,
        # overwriting the previous checkpoint. Done after the global-params
        # update so the on-disk checkpoint reflects the current loop state.
        if (cfg.checkpoint_interval is not None
                and (step + 1) % cfg.checkpoint_interval == 0):
            save_result(_result(t + 1, False, checkpoint=True), cfg.checkpoint_dir)

        # Early-stop only in FULL-BATCH mode. In mini-batch SVI the
        # per-iteration ELBO is computed on a fresh random sub-sample, so
        # its iteration-to-iteration variance makes a two-point
        # relative-change test fire on coincidental closeness, not true
        # convergence (repeatedly observed: false stops at iter 67 and 157
        # of a 200-iter budget). Spark MLlib's reference online LDA
        # (OnlineLDAOptimizer) has NO convergence tolerance and runs exactly
        # maxIterations for this reason; we match it — mini-batch fits run
        # the full budget, full-batch fits keep the valid corpus-wide check.
        if not minibatch and model.has_converged(elbo_trace, cfg.convergence_tol):
            log.info("Converged at iteration %d (ELBO=%.6f)", step + 1, elbo)
            result = _result(t + 1, True)
            # Final-save guarantee: when checkpoint_dir is set, the post-fit
            # directory is authoritative regardless of where the interval
            # boundary falls.
            if cfg.checkpoint_dir is not None:
                save_result(result, cfg.checkpoint_dir)
            return result

    # Hit max_iterations without convergence.
    result = _result(start_iteration + cfg.max_iterations, False)
    if cfg.checkpoint_dir is not None:
        save_result(result, cfg.checkpoint_dir)
    return result
