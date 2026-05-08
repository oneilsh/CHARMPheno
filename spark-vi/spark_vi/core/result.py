"""VIResult: immutable record of a VI training run (completed or in-progress).

The same dataclass round-trips through save_result/load_result for both
"final fit outcome" and "interim checkpoint written during a long run."
converged=True indicates a finished, converged run; converged=False covers
both runs that exhausted max_iterations without converging and interim
checkpoints written by VIRunner's auto-checkpoint mechanism.

See docs/architecture/SPARK_VI_FRAMEWORK.md#viresult-and-model-export and
docs/decisions/0006-unified-persistence-format.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VIResult:
    """Training-run state (completed or in-progress).

    A VIResult records both completed VIRunner.fit results and interim
    checkpoints. The same dataclass round-trips through save_result/load_result
    for either use case.

    Attributes:
        global_params: fitted global variational parameters, keyed by name.
        elbo_trace: per-iteration ELBO values (or surrogate, if ELBO is unavailable).
        n_iterations: how many iterations actually ran. For an interim checkpoint,
            this is the iteration count at the moment the checkpoint was written.
        converged: True iff the convergence criterion was met. False covers both
            "ran out of iterations" and "this is an interim checkpoint."
        metadata: free-form dict (model class name, timestamps, git sha, ...).
            Auto-checkpoints stamp metadata["checkpoint"] = True.
        diagnostic_traces: per-iteration trajectories of optional model-supplied
            scalar/array diagnostics (e.g. hyperparameter evolution). Keyed by
            diagnostic name; values are lists of floats or small numpy arrays
            accumulated over the fit, each with length n_iterations.
    """
    global_params: dict[str, np.ndarray]
    elbo_trace: list[float]
    n_iterations: int
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostic_traces: dict[str, list[float | np.ndarray]] = field(default_factory=dict)

    @property
    def final_elbo(self) -> float | None:
        """Last ELBO value, or None if the trace is empty."""
        return self.elbo_trace[-1] if self.elbo_trace else None
