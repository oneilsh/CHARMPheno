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


@dataclass(frozen=True, slots=True)
class STMDocument:
    """Structural Topic Model document.

    Extends BOWDocument with a per-document covariate vector x.
    The engine never learns what x means — only its shape and dtype.

    Invariants (callers' responsibility — not enforced at construction):
      indices: sorted int32 array of token indices, all in [0, vocab_size).
      counts:  float64 array with len(counts) == len(indices), all > 0.
      length:  int total tokens (sum of counts).
      x:       float64 array of shape (P,) — the doc's covariate vector.
      groups:  frozenset[str] of group labels (empty = background only / gating off).
    """
    indices: np.ndarray
    counts: np.ndarray
    length: int
    x: np.ndarray
    groups: frozenset = frozenset()


@dataclass(frozen=True, slots=True)
class PCDocument:
    """Prediction-Constrained topic-model document.

    Extends BOWDocument with a per-document outcome vector ``y`` (C binary
    labels, one per logistic head) and a per-cell observed mask ``label_mask``
    (C,), mirroring how STMDocument extends it with a covariate vector ``x``.
    The C outcome heads share one topic model; ``label_mask[c] == 1`` marks
    cell (doc, c) as an observed training label for head c, so an
    almost-all-missing label matrix (index-drug mode) still trains every head
    off the shared representation.

    Increment 1 (weight_y == 0, the unsupervised SVI scaffolding) carries y and
    label_mask on every row but never READS them — the label-free CAVI E-step
    and the LDA λ natural-gradient step are outcome-blind, exactly as the
    faithful reference's weight_y == 0 path is. The fields are present now so
    the row type is stable when increment 2 attaches the supervised head
    gradient + topic correction (which read y/label_mask for the observed
    cells only).

    Invariants (callers' responsibility — not enforced at construction):
      indices:    sorted int32 array of token indices, all in [0, vocab_size).
      counts:     float64 array with len(counts) == len(indices), all > 0.
      length:     int total tokens (sum of counts).
      y:          float64 array of shape (C,) — the doc's 0/1 outcome labels.
      label_mask: float64 array of shape (C,) — 1 where cell (doc, c) is an
                  observed training label, 0 where unobserved. All-zero = the
                  doc's words shape the shared topics but no head trains on it.
    """
    indices: np.ndarray
    counts: np.ndarray
    length: int
    y: np.ndarray
    label_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class GatedBOWDocument:
    """Bag-of-words document tagged with a DAG frontier for gated topic training.

    Mirrors STMDocument's `groups` gating precedent, but the gate is a set of DAG
    node ids (the doc's frontier = most-specific attested nodes) rather than covariate
    groups. GatedOnlineLDA restricts each training doc's variational E-step to
    DagLayout.allowed_set(frontier). Empty frontier = ungated (full-K), used for
    held-out fold-in at deployment.

    Invariants (callers' responsibility — not enforced at construction):
      indices: sorted int32 array of token indices, all in [0, vocab_size).
      counts:  float64 array with len(counts) == len(indices), all > 0.
      length:  int total tokens (sum of counts).
      frontier: frozenset[int] of DAG node ids (empty = ungated).
    """
    indices: np.ndarray
    counts: np.ndarray
    length: int
    frontier: frozenset = frozenset()
