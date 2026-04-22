"""CharmPhenoHDP: clinical wrapper around the generic spark_vi OnlineHDP.

The wrapper adds the clinical/OMOP layer on top of the generic topic model:
concept-vocabulary handling, downstream export hooks, phenotype labels
(when the underlying OnlineHDP has converged). Bootstrap leaves the .fit()
path propagating the underlying stub's NotImplementedError so callers get
a clear signal that the framework is wired but the model math is not yet
implemented.

See docs/architecture/TOPIC_STATE_MODELING.md for the clinical design.
"""
from __future__ import annotations

from typing import Any

from pyspark import RDD
from spark_vi.core import VIConfig, VIResult, VIRunner
from spark_vi.models import OnlineHDP


class CharmPhenoHDP:
    """Thin clinical wrapper around `spark_vi.models.OnlineHDP`.

    Args:
        vocab_size: number of distinct concept_ids in the working vocabulary.
        max_topics: HDP truncation level (upper bound on discovered topics).
        eta, alpha, omega: hyperparameters passed through to OnlineHDP.
    """

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
        self.model = OnlineHDP(
            vocab_size=self.vocab_size,
            max_topics=self.max_topics,
            eta=eta,
            alpha=alpha,
            omega=omega,
        )

    def fit(
        self,
        data_rdd: RDD,
        config: VIConfig | None = None,
        data_summary: Any | None = None,
    ) -> VIResult:
        """Fit the underlying OnlineHDP on an RDD of documents.

        Raises NotImplementedError until the real OnlineHDP lands (see
        the follow-on spec in docs/superpowers/specs/).
        """
        runner = VIRunner(self.model, config=config)
        return runner.fit(data_rdd, data_summary=data_summary)
