"""Row types for topic models.

BOWDocument is the bag-of-words representation consumed by OnlineLDA and
OnlineHDP. Sparse-vector content; the type exists to make the
VIModel-contract input self-documenting and anchors the MLlib shim's
DataFrame→RDD conversion (spark_vi.mllib.topic._common._vector_to_bow_document).

Lives under spark_vi.models.topic.types rather than spark_vi.core.types
because BOWDocument is topic-specific — non-topic models (e.g. a future
factor-analysis VIModel) won't consume it. Generic framework primitives
live in spark_vi.core; topic-specific types live alongside the topic
models that consume them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BOWDocument:
    """Bag-of-words document.

    Invariants (callers' responsibility — not enforced at construction):
      indices: sorted int32 array of token indices, all in [0, vocab_size).
      counts: float64 array with len(counts) == len(indices), all > 0.
      length: int total tokens (sum of counts).
    """
    indices: np.ndarray
    counts: np.ndarray
    length: int

    @classmethod
    def from_spark_row(cls, row, features_col: str = "features") -> "BOWDocument":
        """Construct from a row whose `features` column is a SparseVector.

        Accepts both pyspark.sql.Row and dict-like objects. Coerces dtypes
        for downstream numpy arithmetic.
        """
        sv = row[features_col]
        return cls(
            indices=np.asarray(sv.indices, dtype=np.int32),
            counts=np.asarray(sv.values, dtype=np.float64),
            length=int(sv.values.sum()),
        )
