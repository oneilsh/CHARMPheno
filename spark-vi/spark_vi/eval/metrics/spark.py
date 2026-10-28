"""Per-key metrics over an RDD of scored columns, with no driver collect of cells
(ADR 0051 §3).

Input: an RDD of `(key, score (n,), y (n,))` column pieces, any number per key (a
partition emits its slice of each column; the producer decides what a key is: a
node, a (view, node) pair, one pooled detection column). Each piece is tallied on
its executor, the tallies merge by key in the shuffle (`reduceByKey` combines
map-side first), and each key's metrics and, for the keys in `curve_keys`, its
thinned ROC/PR curve are computed where the merged tally lands. The driver receives
one small dict per key.

Nothing array-shaped rides a task closure (ADR 0047): the closure carries the
targets, `topk_frac`, `max_points` and the (small) set of curve keys.
"""
from __future__ import annotations

from spark_vi.eval.metrics import core


def per_key_metrics(pieces, *, recall_targets=(), fdr_targets=(), topk_frac=0.01,
                    curve_keys=(), max_points=300, num_partitions=None):
    """`{key: metrics dict (+ "curve" for curve_keys)}` from an RDD of
    `(key, score, y)` pieces; see `core.node_metrics` for the definitions."""
    rt = tuple(float(r) for r in recall_targets)
    fq = tuple(float(q) for q in fdr_targets)
    tk, mp = float(topk_frac), int(max_points)
    ck = frozenset(curve_keys)

    def _eval(kv, _rt=rt, _fq=fq, _tk=tk, _mp=mp, _ck=ck):
        key, t = kv
        m = core.node_metrics(t, recall_targets=_rt, fdr_targets=_fq, topk_frac=_tk)
        if key in _ck:
            m["curve"] = core.curve(t, max_points=_mp)
        return key, m

    tallies = (pieces.map(lambda kys: (kys[0], core.tally(kys[1], kys[2])))
               .reduceByKey(core.merge, numPartitions=num_partitions))
    return dict(tallies.map(_eval).collect())


def per_key_calibration(pieces, *, n_bins=10, num_partitions=None):
    """`{key: ECE}` from an RDD of `(key, p, y)` pieces (`core.calibration_bins`)."""
    nb = int(n_bins)

    def _add(a, b):
        return tuple(x + z for x, z in zip(a, b))

    bins = (pieces.map(lambda kpy, _nb=nb: (kpy[0], core.calibration_bins(
        kpy[1], kpy[2], _nb))).reduceByKey(_add, numPartitions=num_partitions))
    return {k: core.ece_from_bins(b) for k, b in bins.collect()}
