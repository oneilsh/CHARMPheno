"""Online Hierarchical Dirichlet Process topic model (STUB).

This is a placeholder with the intended public signature so `charmpheno` can
depend on a stable name during bootstrap. The real implementation — a PySpark
port of Hoffman/Wang/Blei/Paisley stochastic VI for the HDP, patterned after
Spark MLlib's OnlineLDAOptimizer and the intel-spark TopicModeling Scala
implementation — is its own follow-on spec.

See docs/architecture/SPARK_VI_FRAMEWORK.md#online-hdp-topic-model and
docs/architecture/TOPIC_STATE_MODELING.md for the target contract.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from spark_vi.core.model import VIModel


class OnlineHDP(VIModel):
    """Stub OnlineHDP; real implementation deferred to a dedicated spec."""

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
        raise NotImplementedError(
            "OnlineHDP is stubbed during bootstrap. See the follow-on spec in "
            "docs/superpowers/specs/ for the real implementation."
        )

    def local_update(
        self,
        rows: Iterable[Any],
        global_params: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is stubbed during bootstrap.")

    def update_global(
        self,
        global_params: dict[str, np.ndarray],
        target_stats: dict[str, np.ndarray],
        learning_rate: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("OnlineHDP is stubbed during bootstrap.")
