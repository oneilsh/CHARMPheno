"""Batched multi-head logistic regression — re-export shim.

The solver moved to its canonical home in spark-vi
(:mod:`spark_vi.models.topic.batched_lr`) so the co-fit head (which lives in
``spark_vi.models.topic.pc``) and the post-hoc readout (this analysis layer)
share ONE source of truth instead of two diverging copies of a
numerically-critical solver (plan D1, 2026-09-07). ``analysis -> spark_vi`` is
the legal dependency direction, so this module is now a thin re-export: every
existing ``from analysis.pc.batched_lr import ...`` keeps working, byte-for-byte,
against the lifted implementation.

The full "why" (the objective, the standardization fold, why L-BFGS not Newton,
the stats seam, node masking/freezing) lives in the canonical module's docstring
and is not duplicated here.
"""
from __future__ import annotations

from spark_vi.models.topic.batched_lr import (  # noqa: F401
    fold_standardization,
    make_inmemory_stats_fn,
    solve_batched_lr,
    standardization_moments,
    standardized_grad_from_raw,
    unfold_standardization,
)

__all__ = [
    "standardization_moments",
    "fold_standardization",
    "unfold_standardization",
    "standardized_grad_from_raw",
    "make_inmemory_stats_fn",
    "solve_batched_lr",
]
