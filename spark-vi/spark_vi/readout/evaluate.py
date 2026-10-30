"""Evaluate the readout heads on the test split without a driver collect (ADR 0051
step 4).

The scored test split is a set of SCORE BLOCKS, one per partition (Spark) or one per
chunk (local): dicts with `ids` (n,) int64, `proba` (n, C) float32, `y` and `mask`
(n, C) uint8, plus any extra per-row arrays a caller attaches (the closure product
under "stacked", a prior-evidence matrix, ...). `Blocks` wraps either an RDD of them
or a list, and every function here runs on both through `spark_vi.eval.metrics`: the
blocks emit per-column pieces, the pieces are tallied and merged by key, and only
per-column results reach the driver. Nothing (D, C)-shaped ever does.

The results keep the schema the readout has always written (`ranking`, `pr`,
`detection`, `per_node`), so every reader of `results_readout*.json` is unchanged.
"""
from __future__ import annotations

import functools
import itertools

import numpy as np

from spark_vi.eval.metrics import local as mlocal
from spark_vi.eval.metrics import report


class Blocks:
    """Score blocks on Spark (an RDD) or in memory (a list), one interface."""

    def __init__(self, blocks, *, bcasts=None):
        self.blocks = blocks
        self.is_rdd = hasattr(blocks, "mapPartitions")
        # Broadcasts the (lazy) blocks still read; destroyed by `unpersist`.
        self._bcasts = list(bcasts or [])

    def map(self, fn):
        """New blocks, `fn` applied to each (on the executors for an RDD)."""
        if self.is_rdd:
            return Blocks(self.blocks.map(fn), bcasts=self._bcasts)
        return Blocks([fn(b) for b in self.blocks])

    def persist(self):
        if self.is_rdd:
            from pyspark import StorageLevel
            self.blocks.persist(StorageLevel.MEMORY_AND_DISK)
        return self

    def unpersist(self):
        """Release the cached blocks and every broadcast they were built from."""
        if self.is_rdd:
            self.blocks.unpersist()
        for b in self._bcasts:
            _destroy(b)
        self._bcasts = []

    def broadcast(self, value):
        """A value the block functions read: a Spark broadcast on an RDD (nothing
        array-shaped rides a closure, ADR 0047), the value itself locally. Read it
        with `_val`."""
        if self.is_rdd:
            return self.blocks.context.broadcast(value)
        return value

    def metrics(self, piece_fn, **kw):
        """`{key: metrics}` over the pieces `piece_fn(block)` yields."""
        if self.is_rdd:
            from spark_vi.eval.metrics import spark as mspark
            return mspark.per_key_metrics(self.blocks.flatMap(piece_fn), **kw)
        return mlocal.per_key_metrics(
            itertools.chain.from_iterable(piece_fn(b) for b in self.blocks), **kw)

    def calibration(self, piece_fn, *, n_bins=10):
        if self.is_rdd:
            from spark_vi.eval.metrics import spark as mspark
            return mspark.per_key_calibration(self.blocks.flatMap(piece_fn),
                                              n_bins=n_bins)
        return mlocal.per_key_calibration(
            itertools.chain.from_iterable(piece_fn(b) for b in self.blocks),
            n_bins=n_bins)

    def reduce(self, fn, combine):
        """`combine` over `fn(block)` for every block (a treeReduce on Spark)."""
        if self.is_rdd:
            return self.blocks.map(fn).treeReduce(combine)
        return functools.reduce(combine, (fn(b) for b in self.blocks))

    def max_by_unit(self, fn):
        """`{unit: (max score, max y)}` over the `(unit, score, y)` rows `fn(block)`
        yields: pooling rows onto a coarser unit (documents onto persons)."""
        def _mx(a, b):
            return (max(a[0], b[0]), max(a[1], b[1]))
        if self.is_rdd:
            return dict(self.blocks.flatMap(fn).map(lambda r: (r[0], (r[1], r[2])))
                        .reduceByKey(_mx).collect())
        out = {}
        for b in self.blocks:
            for u, s, y in fn(b):
                out[u] = _mx(out[u], (s, y)) if u in out else (s, y)
        return out


def _val(v):
    return v.value if hasattr(v, "value") and hasattr(v, "unpersist") else v


def _destroy(v):
    if hasattr(v, "destroy") and hasattr(v, "unpersist"):
        from spark_vi.core.broadcast import destroy_broadcast
        destroy_broadcast(v)


def with_stacked(blocks, M, *, src="proba", dst="stacked"):
    """Blocks with `dst` = the closure product of `src` (`readout.stacked`), computed
    where each block lives. `M` is the (C, C) closure matrix."""
    from spark_vi.readout.stacked import closure_operator, stacked_proba

    bM = blocks.broadcast(closure_operator(M))

    def _add(b, _bM=bM, _src=src, _dst=dst):
        return {**b, _dst: stacked_proba(b[_src], _val(_bM))}
    out = blocks.map(_add)
    if blocks.is_rdd:
        out._bcasts.append(bM)
    return out


def column_metrics(blocks, *, score="proba", mask="mask", columns=None,
                   recall_targets=(), fdr_targets=(), topk_frac=0.01,
                   curve_columns=(), max_points=300):
    """`{c: core.node_metrics}` for each column over its rows with `mask` set (every
    row when `mask` is None); the `curve_columns` also carry a thinned ROC/PR
    "curve" (`core.curve`)."""
    cols = None if columns is None else tuple(int(c) for c in columns)

    def _pieces(b, _s=score, _m=mask, _cols=cols):
        return mlocal.column_pieces(b[_s], b["y"], None if _m is None else b[_m],
                                    columns=_cols)
    return blocks.metrics(_pieces, recall_targets=recall_targets,
                          fdr_targets=fdr_targets, topk_frac=topk_frac,
                          curve_keys=tuple(int(c) for c in curve_columns),
                          max_points=max_points)


def ranking_and_pr(per_col, C, *, recall_targets, fdr_targets, min_count=0,
                   skip_constant=False):
    """The `_bundle_masked` ranking block and the per-node PR block from one pass of
    column metrics."""
    empty = {"n_pos": 0, "n_neg": 0, "n_distinct": 0, "auc": None, "ap": None,
             "par": {}, "raf": {}}
    per_label = {c: report.score_record(per_col.get(c, empty), min_count=min_count,
                                        skip_constant=skip_constant)
                 for c in range(int(C))}
    per_pr = {c: report.pr_record(per_col.get(c, empty), min_count=min_count)
              for c in range(int(C))}
    return ({"per_label": per_label, "macro": report.macro(per_label)},
            {"per_node": per_pr,
             "macro": report.pr_macro(per_pr, recall_targets, fdr_targets)})


def detection(blocks, recall_targets, *, score="proba", columns="disease",
              unit_divisor=None):
    """Case vs background on one pooled column: per row the max over the
    INFORMATIVE disease columns (node 0 is the root; `columns="root"` uses the root
    column alone), the foreground indicator the root's label. Constant columns are
    excluded (a constant column in the max can pin it, and the AUC at 0.5). With
    `unit_divisor`, rows pool onto units `ids // unit_divisor` by max (documents onto
    persons), so a person with several documents is one case."""
    def _range(b, _s=score):
        p = b[_s]
        if p.shape[0] == 0:
            return None
        return (p.min(axis=0), p.max(axis=0))

    def _comb(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))

    rng = blocks.reduce(_range, _comb)
    if rng is None or rng[0].shape[0] < (1 if columns == "root" else 2):
        return {"skipped": "empty or single-node"}
    lo, hi = rng
    if columns == "root":
        sel = np.array([0])
        n_const = int(hi[0] == lo[0])
    else:
        sel = np.arange(1, lo.shape[0])
        n_const = int(np.sum(hi[1:] == lo[1:]))
    info = sel[hi[sel] > lo[sel]]
    if info.size == 0:
        return {"skipped": "all disease-node scores are constant",
                "n_constant_nodes": n_const}
    binfo = blocks.broadcast(info)

    def _row(b, _s=score, _i=binfo):
        return b[_s][:, _val(_i)].max(axis=1).astype(np.float64), b["y"][:, 0]

    if unit_divisor is None:
        per = blocks.metrics(lambda b: [("det",) + _row(b)],
                             recall_targets=recall_targets)
        grain = "document"
    else:
        div = int(unit_divisor)

        def _units(b, _div=div):
            s, y = _row(b)
            return zip((np.asarray(b["ids"], dtype=np.int64) // _div).tolist(),
                       s.tolist(), y.tolist())
        pooled = blocks.max_by_unit(_units)
        s = np.fromiter((v[0] for v in pooled.values()), dtype=np.float64)
        y = np.fromiter((v[1] for v in pooled.values()), dtype=np.float64)
        per = mlocal.per_key_metrics([("det", s, y)], recall_targets=recall_targets)
        grain = "person"
    _destroy(binfo)
    m = per.get("det")
    if m is None or m["n_pos"] == 0 or m["n_neg"] == 0:
        return {"skipped": "degenerate foreground indicator",
                "n_constant_nodes": n_const}
    n = m["n_pos"] + m["n_neg"]
    return {"skipped": None, "prevalence": m["n_pos"] / n, "auc": m["auc"],
            "ap": m["ap"], "n_constant_nodes": n_const, "grain": grain,
            "n_units": int(n), "par": dict(m["par"])}


def readout_metrics(blocks, C, *, recall_targets, fdr_targets, min_count=0,
                    skip_constant=False, unit_divisor=None, score="proba",
                    curve_columns=(), max_points=300):
    """The readout block: `{ranking, pr, detection, per_node}` (the schema
    `results_readout*.json` has always carried), from `score` over each node's
    `mask` rows. `per_node` keeps the scored nodes' auc / ap / n_pos and the top-k
    screening fields, for splitting the macro by node rarity. With `curve_columns`
    the block also carries `curves`: `{c: thinned ROC/PR points}` for those nodes
    (None for a single-class one)."""
    per_col = column_metrics(blocks, score=score, recall_targets=recall_targets,
                             fdr_targets=fdr_targets, curve_columns=curve_columns,
                             max_points=max_points)
    ranking, pr = ranking_and_pr(per_col, C, recall_targets=recall_targets,
                                 fdr_targets=fdr_targets, min_count=min_count,
                                 skip_constant=skip_constant)
    det = detection(blocks, recall_targets, score=score, unit_divisor=unit_divisor)
    per_node = {}
    for c, rl in ranking["per_label"].items():
        if rl.get("skipped") is None and rl.get("auc") is not None:
            per_node[c] = {"auc": float(rl["auc"]),
                           "ap": None if rl.get("ap") is None else float(rl["ap"]),
                           "n_pos": int(pr["per_node"][c]["n_pos"]),
                           "prec_at_k": rl.get("prec_at_k"),
                           "recall_at_k": rl.get("recall_at_k"),
                           "lift_at_k": rl.get("lift_at_k"),
                           "topk_frac": rl.get("topk_frac")}
    out = {"ranking": ranking["macro"], "pr": pr["macro"], "detection": det,
           "per_node": per_node}
    if curve_columns:
        out["curves"] = {int(c): per_col.get(int(c), {}).get("curve")
                         for c in curve_columns}
    return out


def calibration_by_group(blocks, groups, *, scores=("proba",), n_bins=10):
    """`{(score, group): ECE}` pooled over every row and every column of each group
    (`groups` = `{name: [columns]}`): the marginal calibration, not a cohort's."""
    g = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
    bg = blocks.broadcast(g)

    def _pieces(b, _g=bg, _scores=tuple(scores)):
        for name, cols in _val(_g).items():
            yy = b["y"][:, cols].astype(np.float64).ravel()
            for s in _scores:
                yield (s, name), b[s][:, cols].astype(np.float64).ravel(), yy
    out = blocks.calibration(_pieces, n_bins=n_bins)
    _destroy(bg)
    return out
