"""countTransform: per-token BOW count damping before the spectral seed + fit.

Raw occurrence counts make the anchor/evidence criteria track utilization VOLUME
(a code recorded every visit counts N times), which is what let a demographic
burst signal (pregnancy codes on prenatal visits) out-anchor a node's phenotype
in exp 0114. 'binary' collapses each token to per-doc presence; 'log1p' damps it.
These tests pin the transform itself (Spark-free) and that it reaches the fit
column selection.
"""
import numpy as np

from spark_vi.models.topic.types import PCDocument, GatedPCDocument
from spark_vi.mllib.topic.pc import _transform_counts


def _pc(counts):
    c = np.asarray(counts, dtype=np.float64)
    return PCDocument(indices=np.arange(len(c), dtype=np.int32), counts=c,
                      length=int(c.sum()), y=np.zeros(2), label_mask=np.zeros(2))


def test_binary_collapses_counts_to_presence_and_recomputes_length():
    d = _pc([5.0, 1.0, 20.0])                 # a bursty (per-visit) code at 20
    out = _transform_counts(d, "binary")
    assert np.array_equal(out.counts, [1.0, 1.0, 1.0])
    assert out.length == 3
    assert np.array_equal(out.indices, d.indices)      # untouched
    assert d.counts[2] == 20.0                          # original unmutated (frozen)


def test_log1p_damps_without_flattening():
    d = _pc([1.0, 20.0])
    out = _transform_counts(d, "log1p")
    assert np.allclose(out.counts, np.log1p([1.0, 20.0]))
    assert out.counts[1] > out.counts[0]               # keeps some ordering
    assert out.length == int(round(float(np.log1p([1.0, 20.0]).sum())))


def test_none_is_identity():
    d = _pc([5.0, 1.0, 20.0])
    assert _transform_counts(d, "none") is d


def test_preserves_type_and_frontier_for_gated_doc():
    c = np.asarray([3.0, 7.0], dtype=np.float64)
    g = GatedPCDocument(indices=np.arange(2, dtype=np.int32), counts=c, length=10,
                        y=np.zeros(1), label_mask=np.zeros(1), frontier=frozenset({4}))
    out = _transform_counts(g, "binary")
    assert isinstance(out, GatedPCDocument)
    assert out.frontier == frozenset({4})              # gate frontier survives
    assert np.array_equal(out.counts, [1.0, 1.0])


def test_bad_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="none|binary|log1p"):
        _transform_counts(_pc([1.0]), "sqrt")


def test_estimator_accepts_and_defaults_count_transform():
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    assert OnlinePCLDAEstimator().getOrDefault("countTransform") == "none"
    assert OnlinePCLDAEstimator(countTransform="binary").getOrDefault(
        "countTransform") == "binary"
