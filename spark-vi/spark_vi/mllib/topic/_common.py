"""Topic-shim-specific helpers (shared between the LDA and HDP shims).

Generic shim infrastructure (persistence Params, _PersistableModel) lives
in spark_vi.mllib._common. This module holds the helpers that depend on
the topic row type (BOWDocument) and so don't generalize to non-topic
shims.
"""
from __future__ import annotations

import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector, Vector

from spark_vi.models.topic.types import BOWDocument


def _vector_to_bow_document(v: Vector) -> BOWDocument:
    """Convert a pyspark.ml.linalg Vector to a BOWDocument.

    SparseVector indices/values pass through. DenseVectors are sparsified
    (nonzero entries only) so the downstream CAVI loop sees the same shape
    of input regardless of the producer (CountVectorizer emits Sparse,
    user-constructed inputs may be Dense).
    """
    if isinstance(v, SparseVector):
        indices = np.asarray(v.indices, dtype=np.int32)
        counts = np.asarray(v.values, dtype=np.float64)
    elif isinstance(v, DenseVector):
        values = np.asarray(v.values, dtype=np.float64)
        nz = np.nonzero(values)[0].astype(np.int32)
        indices = nz
        counts = values[nz]
    else:
        raise TypeError(
            f"_vector_to_bow_document expected Sparse/DenseVector, got {type(v).__name__}"
        )
    return BOWDocument(indices=indices, counts=counts, length=int(counts.sum()))
