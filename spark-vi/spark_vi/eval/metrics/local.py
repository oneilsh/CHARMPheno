"""The in-memory backend: the same per-key metrics as `spark.per_key_metrics`, from an
iterable of `(key, score, y)` pieces held in this process (ADR 0051 §3)."""
from __future__ import annotations

from spark_vi.eval.metrics import core


def per_key_metrics(pieces, *, recall_targets=(), fdr_targets=(), topk_frac=0.01,
                    curve_keys=(), max_points=300):
    """`{key: metrics dict (+ "curve" for curve_keys)}`; see `core.node_metrics`."""
    tallies = {}
    for key, s, y in pieces:
        t = core.tally(s, y)
        tallies[key] = core.merge(tallies[key], t) if key in tallies else t
    out = {}
    for key, t in tallies.items():
        m = core.node_metrics(t, recall_targets=recall_targets,
                              fdr_targets=fdr_targets, topk_frac=topk_frac)
        if key in set(curve_keys):
            m["curve"] = core.curve(t, max_points=max_points)
        out[key] = m
    return out


def per_key_calibration(pieces, *, n_bins=10):
    """`{key: ECE}` from `(key, p, y)` pieces (`core.calibration_bins`)."""
    bins = {}
    for key, p, y in pieces:
        b = core.calibration_bins(p, y, n_bins)
        bins[key] = tuple(x + z for x, z in zip(bins[key], b)) if key in bins else b
    return {k: core.ece_from_bins(b) for k, b in bins.items()}


def column_pieces(score, y, keep=None, *, columns=None, key=lambda c: c):
    """`(key(c), score[rows, c], y[rows, c])` per column of dense (n, C) arrays, rows
    = `keep[:, c]` (all rows when `keep` is None). The piece producer the readout
    uses on each partition's block, and on the whole matrix locally."""
    cols = range(score.shape[1]) if columns is None else columns
    for c in cols:
        if keep is None:
            yield key(c), score[:, c], y[:, c]
        else:
            r = keep[:, c].astype(bool)
            yield key(c), score[r, c], y[r, c]
