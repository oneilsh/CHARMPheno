"""CharmPhenoHDP: clinical wrapper around the generic spark_vi OnlineHDP.

The wrapper adds the clinical/OMOP layer on top of the generic topic model:
concept-vocabulary handling, downstream export hooks, phenotype labels
(when the underlying OnlineHDP has converged).

See docs/architecture/TOPIC_STATE_MODELING.md for the clinical design.
"""
from __future__ import annotations

from typing import Any

from pyspark import RDD
from spark_vi.core import VIConfig, VIResult, VIRunner
from spark_vi.models import OnlineHDP


class CharmPhenoHDP:
    """Thin clinical wrapper around `spark_vi.models.OnlineHDP`.

    Constructor arg names use clinical-user-facing terms (max_topics,
    max_doc_topics) which translate to the inner model's T (corpus
    truncation) and K (doc truncation). All other args pass through.

    Args:
      vocab_size: number of distinct concept_ids in the working vocabulary.
      max_topics: HDP corpus-level truncation (upper bound on discovered
        topics).
      max_doc_topics: HDP doc-level truncation (upper bound on topics per
        visit).
      eta: topic-word Dirichlet concentration.
      alpha: doc-level stick concentration.
      gamma: corpus-level stick concentration.
      gamma_shape: shape parameter for the Gamma init of λ.
      cavi_max_iter: hard cap on doc-CAVI iterations per doc.
      cavi_tol: relative ELBO convergence threshold for doc-CAVI early
        termination.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        max_topics: int = 150,
        max_doc_topics: int = 15,
        eta: float = 0.01,
        alpha: float = 1.0,
        gamma: float = 1.0,
        gamma_shape: float = 100.0,
        cavi_max_iter: int = 100,
        cavi_tol: float = 1e-4,
    ) -> None:
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        self.vocab_size = int(vocab_size)
        self.max_topics = int(max_topics)
        self.max_doc_topics = int(max_doc_topics)
        self.model = OnlineHDP(
            T=self.max_topics,
            K=self.max_doc_topics,
            vocab_size=self.vocab_size,
            alpha=alpha,
            gamma=gamma,
            eta=eta,
            gamma_shape=gamma_shape,
            cavi_max_iter=cavi_max_iter,
            cavi_tol=cavi_tol,
        )

    def fit(
        self,
        data_rdd: RDD,
        config: VIConfig | None = None,
        data_summary: Any | None = None,
    ) -> VIResult:
        """Fit the underlying OnlineHDP on an RDD of documents."""
        runner = VIRunner(self.model, config=config)
        result = runner.fit(data_rdd, data_summary=data_summary)
        self._fitted_globals = result.global_params
        return result

    def transform(self, data_rdd: RDD) -> RDD:
        """Per-doc frozen-globals inference.

        Requires `.fit()` to have populated `self._fitted_globals`.
        Maps each input row through `OnlineHDP.infer_local` and yields
        (input_row, theta) pairs where theta is the length-T topic
        proportion vector for that doc.

        Stub during bootstrap; fully wired once VIRunner.transform lands
        for HDP. Until then, this method runs infer_local in driver-side
        Python — adequate for small held-out sets but not for production.
        """
        if not hasattr(self, "_fitted_globals"):
            raise RuntimeError(
                "CharmPhenoHDP.transform requires fit() first; "
                "_fitted_globals is unset."
            )
        globals_bcast = data_rdd.context.broadcast(self._fitted_globals)
        model = self.model

        def _per_partition(rows):
            g = globals_bcast.value
            for row in rows:
                out = model.infer_local(row, g)
                yield (row, out["theta"])

        return data_rdd.mapPartitions(_per_partition)
