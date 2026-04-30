"""Canonical row types shared across spark_vi models.

BOWDocument is the bag-of-words representation consumed by topic-style
models (VanillaLDA, future OnlineHDP). Sparse-vector content; the type
exists to make the contract self-documenting and to anchor a future MLlib
Estimator/Transformer compatibility shim.
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
