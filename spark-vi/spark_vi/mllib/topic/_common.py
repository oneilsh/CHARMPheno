"""Topic-shim-specific helpers (shared between the LDA and HDP shims).

Generic shim infrastructure (persistence Params, _PersistableModel) lives
in spark_vi.mllib._common. This module holds the helpers that depend on
the topic row type (BOWDocument) and so don't generalize to non-topic
shims.
"""
from __future__ import annotations

import numpy as np
from pyspark.ml.linalg import DenseVector, SparseVector, Vector

from spark_vi.models.topic.types import BOWDocument, STMDocument


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


def _vector_to_stm_document(
    row,
    features_col: str = "features",
    covariates_col: str = "covariates",
) -> STMDocument:
    """Construct an STMDocument from a row with both a BOW vector and a covariate vector.

    Accepts pyspark.sql.Row and dict-like objects. The covariate vector
    must be a DenseVector (or numpy-coercible array); covariates_col
    cannot be sparse for STM (every doc has a complete x vector).
    """
    sv = row[features_col]
    cov = row[covariates_col]
    return STMDocument(
        indices=np.asarray(sv.indices, dtype=np.int32),
        counts=np.asarray(sv.values, dtype=np.float64),
        length=int(sv.values.sum()),
        x=np.asarray(cov, dtype=np.float64),
    )
