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


def _concat_domain_features(vectors, sizes):
    """Concatenate per-domain sparse vectors into the engine's single id space.

    The engine stores one topic-word matrix per domain but consumes ONE
    concatenated token-id space, with a token's domain recovered by
    `searchsorted(domain_bounds, w)`. Domain m's local ids therefore shift by
    `sum(sizes[:m])`. Because each domain's ids are already ascending and the
    domains are laid out in order, the concatenated ids are GLOBALLY sorted,
    which is what the E-step's `expElogbeta[:, indices]` gather assumes.

    Raises ValueError naming the domain and both widths if a vector disagrees
    with the established layout: the layout is derived once per fit, and a row
    that silently re-lays-out the vocabulary would corrupt the fit with no
    symptom (SP3a design).

    Equally, `len(vectors)` must equal `len(sizes)`: a bare zip TRUNCATES to the
    shorter, silently dropping a trailing domain — the same invisible re-layout,
    reachable wherever the columns and the widths do not arrive together (e.g. a
    model carrying featuresCols and domainBounds as separate Params).
    """
    vectors, sizes = list(vectors), list(sizes)
    if len(vectors) != len(sizes):
        raise ValueError(
            f"_concat_domain_features got {len(vectors)} vector(s) for "
            f"{len(sizes)} domain size(s); the counts must match or a trailing "
            f"domain would be silently dropped")
    idx_parts, cnt_parts, offset = [], [], 0
    for m, (v, width) in enumerate(zip(vectors, sizes)):
        if int(v.size) != int(width):
            raise ValueError(
                f"featuresCols domain {m} vector size {int(v.size)} != expected "
                f"{int(width)} (layout derived from the first row); every row must "
                f"use the same per-domain vocabulary widths")
        bow = _vector_to_bow_document(v)
        idx_parts.append(bow.indices.astype(np.int64) + offset)
        cnt_parts.append(bow.counts)
        offset += int(width)
    indices = np.concatenate(idx_parts).astype(np.int32) if idx_parts else np.empty(0, np.int32)
    counts = np.concatenate(cnt_parts) if cnt_parts else np.empty(0, np.float64)
    return indices, counts


def _vector_to_stm_document(
    row,
    features_col: str = "features",
    covariates_col: str = "covariates",
    group_col: str | None = None,
) -> STMDocument:
    """Construct an STMDocument from a row with a BOW vector and a covariate vector.

    When group_col is set, row[group_col] supplies the doc's gating group(s):
    a scalar value becomes a singleton frozenset; a list/tuple value stores all
    members. When None, groups is empty (background-only / gating off).

    Accepts pyspark.sql.Row and dict-like objects. The features_col vector
    may be Sparse or Dense (both are sparsified to nonzero entries). The
    covariate vector must be a DenseVector (or numpy-coercible array);
    covariates_col cannot be sparse for STM (every doc has a complete x vector).
    """
    bow = _vector_to_bow_document(row[features_col])
    cov = row[covariates_col]
    if group_col is None:
        groups = frozenset()
    else:
        val = row[group_col]
        groups = (frozenset(str(v) for v in val)
                  if isinstance(val, (list, tuple))
                  else frozenset({str(val)}))
    return STMDocument(
        indices=bow.indices,
        counts=bow.counts,
        length=bow.length,
        x=np.asarray(cov, dtype=np.float64),
        groups=groups,
    )
