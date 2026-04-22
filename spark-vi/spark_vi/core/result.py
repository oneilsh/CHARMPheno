"""VIResult: immutable record of a completed VI training run.

See docs/architecture/SPARK_VI_FRAMEWORK.md#viresult-and-model-export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VIResult:
    """Outcome of a VIRunner.fit call.

    Attributes:
        global_params: fitted global variational parameters, keyed by name.
        elbo_trace: per-iteration ELBO values (or surrogate, if ELBO is unavailable).
        n_iterations: how many iterations actually ran.
        converged: whether convergence criterion was met (vs. max_iterations hit).
        metadata: free-form dict (model class name, timestamps, git sha, ...).
    """
    global_params: dict[str, np.ndarray]
    elbo_trace: list[float]
    n_iterations: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_elbo(self) -> float | None:
        """Last ELBO value, or None if the trace is empty."""
        return self.elbo_trace[-1] if self.elbo_trace else None
