"""Online HDP topic model — Wang/Paisley/Blei 2011, AISTATS.

Implements the algorithm via the spark_vi `VIModel` contract: per-doc CAVI
on workers (`local_update`), natural-gradient SVI step on the driver
(`update_global`).

Notation. We follow Wang's reference-code convention (also used by intel-spark):
  T = corpus-level truncation (paper's K)
  K = doc-level truncation    (paper's T)
The AISTATS paper inverts these letters; see
docs/architecture/TOPIC_STATE_MODELING.md "Notation" for the rationale.

References:
  - Wang, Paisley, Blei (2011). "Online Variational Inference for the
    Hierarchical Dirichlet Process." AISTATS. Eqs 15-18 give doc-CAVI;
    Eqs 22-27 give the natural-gradient SVI step.
  - Wang's reference Python implementation:
    https://github.com/blei-lab/online-hdp (onlinehdp.py).
  - intel-spark TopicModeling Scala port (cited for confirmation only;
    we explicitly diverge from its driver-side `chunk.collect()` E-step).
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.special import digamma, gammaln

from spark_vi.core.model import VIModel


# ---------------------------------------------------------------------------
# Module-private math helpers (pure functions, easy to unit-test in isolation).
# ---------------------------------------------------------------------------


def _log_normalize_rows(M: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise log-normalize.

    Returns log(softmax(M, axis=1)). Subtracts the per-row max before
    exponentiating to avoid overflow on large positive entries; for very
    negative entries np.exp underflows to 0, which is benign.
    """
    row_max = M.max(axis=1, keepdims=True)
    shifted = M - row_max
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_norm


# Stub OnlineHDP class — methods filled in by later tasks.
class OnlineHDP(VIModel):
    """Stub during incremental implementation; see Task 6 onwards."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_topics: int = 150,
        eta: float = 0.01,
        alpha: float = 1.0,
        omega: float = 1.0,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab_size = int(vocab_size)
        self.max_topics = int(max_topics)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.omega = float(omega)

    def initialize_global(self, data_summary: Any | None) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 7.")

    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 8.")

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is being built; see Task 9.")
