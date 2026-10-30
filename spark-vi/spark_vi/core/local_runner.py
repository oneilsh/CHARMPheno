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
                 chunk_size: int | None = None) -> None:
        """`chunk_size` splits each batch into chunks of at most that many rows,
        each through its own `local_update`, merged by `combine_stats` -- the
        local analogue of partitions. It bounds the per-call working set of
        models whose local_update builds batch-sized arrays; None means one call
        per batch. Additive stats make the result independent of the split."""
        if not isinstance(model, VIModel):
            raise TypeError(f"model must be a VIModel subclass, got {type(model).__name__}")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be None or >= 1, got {chunk_size}")
        self.model = model
        self.config = config if config is not None else VIConfig()
        self.chunk_size = chunk_size

    def _aggregate(self, rows: Sequence, global_params: dict) -> dict:
        model = self.model
        step = self.chunk_size or max(len(rows), 1)
        agg = None
        for lo in range(0, len(rows), step):
            # iter(): local_update sees an iterator, as under mapPartitions.
            stats = model.local_update(iter(rows[lo:lo + step]), global_params)
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
