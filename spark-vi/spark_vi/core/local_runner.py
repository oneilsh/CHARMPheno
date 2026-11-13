"""LocalRunner: the in-memory engine for the same VIModel contract (ADR 0051 §2).

A corpus often fits in a large machine's memory, and a cluster costs a spin-up; LocalRunner runs a `VIModel` over a Python sequence of rows with no Spark.
It drives the exact loop VIRunner drives (`spark_vi.core.loop.run_svi_loop`: the
Robbins-Monro schedule, corpus/batch pre-scaling, ELBO trace, diagnostics,
checkpoint, resume), and its E-step calls the same `local_update` /
`combine_stats` the executors call. Model-held schedules such as the DAG gate's
`gate_iter` ride `global_params`, so they advance identically.

Fidelity: a local fit is the Spark fit *in distribution*, not bit for bit. The
minibatch is drawn the way `RDD.sample` draws it -- each row independently, with
`Poisson(fraction)` copies when `cfg.sample_with_replacement` (Spark's
PoissonSampler, the default) and with probability `fraction` otherwise (Spark's
BernoulliSampler) -- but from a NumPy stream, not Spark's per-partition XORShift
streams, so the drawn batches differ. Full-batch fits (`mini_batch_fraction=None`)
agree with Spark up to floating-point summation order.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from spark_vi.core.config import VIConfig
from spark_vi.core.loop import run_svi_loop
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult


def sample_rows(n: int, fraction: float, with_replacement: bool,
                seed: int) -> np.ndarray:
    """Row indices of one minibatch, drawn as RDD.sample draws them.

    Each of the n rows is included independently: Poisson(fraction) times with
    replacement, once with probability `fraction` without. The batch size is
    therefore random (mean n * fraction), which is why the loop rescales by the
    realized corpus/batch ratio rather than 1 / fraction.
    """
    rng = np.random.default_rng(seed)
    if with_replacement:
        return np.repeat(np.arange(n), rng.poisson(fraction, size=n))
    return np.flatnonzero(rng.random(n) < fraction)


class LocalRunner:
    """Drives a VIModel through SVI iterations over an in-memory corpus."""

    def __init__(self, model: VIModel, config: VIConfig | None = None,
                 chunk_size: int | None = None, processes: int = 1) -> None:
        """`chunk_size` splits each batch into chunks of at most that many rows,
        each through its own `local_update`, merged by `combine_stats` -- the
        local analogue of partitions. It bounds the per-call working set of
        models whose local_update builds batch-sized arrays; None means one call
        per batch (or, with `processes > 1`, one chunk per process). Additive stats
        make the result independent of the split up to floating-point summation
        order.

        `processes` runs the chunks' `local_update` calls in that many worker
        processes (0 = one per CPU), the local analogue of executors. Workers are
        FORKED once per E-step after the iteration's `broadcast_params`, so the
        model, the shipped parameters and the corpus are shared copy-on-write, not
        pickled; only each chunk's statistics travel back. They are merged in chunk
        order. Forking is used on Linux only (macOS lists 'fork' but it is unsafe
        there with system frameworks loaded, and Windows lacks it): elsewhere, or
        with one chunk, it runs serially.

        Reproducibility with `processes > 1`. Before forking, the parent draws one
        seed per chunk from the global NumPy RNG and each worker seeds `np.random`
        with its chunk's seed before `local_update`. So:
        * a model whose randomness is content-seeded (`random_seed` set, e.g. the
          gamma init keyed by the document) gives, for a given `chunk_size`, a fit
          bit-identical to `processes=1`;
        * a model that draws from the global `np.random` (`random_seed=None`) is
          reproducible given `np.random`'s state (e.g. after `np.random.seed(0)`),
          but differs from the serial fit, whose chunks share one stream.
        The per-chunk seeds are drawn only when the E-step actually forks."""
        if not isinstance(model, VIModel):
            raise TypeError(f"model must be a VIModel subclass, got {type(model).__name__}")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be None or >= 1, got {chunk_size}")
        if processes is None or int(processes) < 0:
            raise ValueError(f"processes must be >= 0 (0 = one per CPU), got {processes}")
        self.model = model
        self.config = config if config is not None else VIConfig()
        self.chunk_size = chunk_size
        self.processes = int(processes) or (os.cpu_count() or 1)

    def _chunks(self, n: int) -> list[tuple[int, int]]:
        if n == 0:
            return []
        if self.chunk_size is not None:
            step = self.chunk_size
        else:
            step = -(-n // self.processes) if self.processes > 1 else n
        return [(lo, min(lo + step, n)) for lo in range(0, n, step)]

    def _aggregate(self, rows: Sequence, global_params: dict) -> dict:
        model = self.model
        params = model.broadcast_params(global_params)    # once, as VIRunner does
        spans = self._chunks(len(rows))
        n_workers = min(self.processes, len(spans))
        if n_workers > 1 and _can_fork():
            parts = _forked_local_updates(model, params, rows, spans, n_workers)
        else:
            # iter(): local_update sees an iterator, as under mapPartitions.
            parts = (model.local_update(iter(rows[lo:hi]), params) for lo, hi in spans)
        agg = None
        for stats in parts:                    # chunk order: deterministic merge
            agg = stats if agg is None else model.combine_stats(agg, stats)
        return agg

    def fit(
        self,
        data: Sequence,
        data_summary: Any | None = None,
        start_iteration: int = 0,
        resume_from: Path | str | None = None,
        on_iteration: Callable[[int, dict, list[float]], None] | None = None,
    ) -> VIResult:
        """Run the SVI loop over `data`; same contract as `VIRunner.fit`.

        `data` is any indexable sequence of rows (the row type the model's
        local_update expects, e.g. BOWDocument); it is held as a list.
        """
        cfg = self.config
        rows = data if isinstance(data, list) else list(data)
        if not rows:
            raise ValueError("LocalRunner.fit: empty corpus")
        minibatch = cfg.mini_batch_fraction is not None

        def _estep(global_params, seed):
            if seed is None:
                return self._aggregate(rows, global_params), len(rows)
            idx = sample_rows(len(rows), cfg.mini_batch_fraction,
                              cfg.sample_with_replacement, seed)
            if idx.size == 0:
                return None, 0
            batch = [rows[i] for i in idx]
            return self._aggregate(batch, global_params), len(batch)

        return run_svi_loop(
            self.model, cfg, _estep,
            corpus_size=len(rows) if minibatch else None,
            data_summary=data_summary,
            start_iteration=start_iteration,
            resume_from=resume_from,
            on_iteration=on_iteration,
        )

    def transform(self, data: Sequence, global_params: dict[str, Any]) -> list:
        """Per-row posteriors via model.infer_local (eager, unlike VIRunner's lazy RDD)."""
        return [self.model.infer_local(row, global_params) for row in data]


# --------------------------------------------------------------------------- #
# Forked E-step workers                                                        #
# --------------------------------------------------------------------------- #
# What the forked workers read: set in the parent immediately before the pool forks,
# so every worker inherits it copy-on-write (nothing array-shaped is pickled to them).
# A module global, so two runners in two threads would clobber each other's state:
# `_FORK_LOCK` serializes set -> fork -> clear (the workers have their copy once forked).
_FORK_STATE: tuple | None = None
_FORK_LOCK = threading.Lock()


def _can_fork() -> bool:
    """Fork the E-step only on Linux, where 'fork' is the safe default start method."""
    return (sys.platform.startswith("linux")
            and "fork" in multiprocessing.get_all_start_methods())


def _fork_local_update(task):
    # Seed the global NumPy RNG per chunk: a model without a content seed draws from
    # it, and forked children would otherwise all inherit the parent's stream (the
    # same numbers in every chunk). The seed comes from the parent's RNG, so the fit
    # is reproducible given the parent's state.
    (lo, hi), seed = task
    np.random.seed(seed)
    model, params, rows = _FORK_STATE
    return model.local_update(iter(rows[lo:hi]), params)


def _forked_local_updates(model, params, rows, spans, n_workers, seeds=None) -> list:
    """`local_update` over each span in `n_workers` forked processes, results in span
    order. The pool lives for one E-step: forking after this iteration's
    `broadcast_params` is what lets the workers see it without a copy. `seeds`: one
    `np.random` seed per span; by default drawn here from the parent's `np.random`."""
    import warnings
    global _FORK_STATE
    if seeds is None:
        seeds = np.random.randint(0, 2**32 - 1, size=len(spans), dtype=np.uint64)
    tasks = [(span, int(sd)) for span, sd in zip(spans, seeds)]
    with _FORK_LOCK:
        _FORK_STATE = (model, params, rows)
        try:
            ctx = multiprocessing.get_context("fork")
            with warnings.catch_warnings():
                # Python warns when a process with threads forks (here, py4j's
                # threads when a SparkSession is live): a child may inherit a lock
                # another thread held. The workers only run the model's numpy E-step
                # on inherited arrays and return its statistics; they never touch
                # py4j, logging or another thread's state, so the inherited locks are
                # never taken.
                warnings.filterwarnings("ignore", category=DeprecationWarning,
                                        message=".*multi-threaded, use of fork.*")
                with ctx.Pool(n_workers) as pool:
                    return pool.map(_fork_local_update, tasks, chunksize=1)
        finally:
            _FORK_STATE = None
