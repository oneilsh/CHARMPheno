"""Exact ranking metrics from score TALLIES (ADR 0051 §3). Pure NumPy.

A tally of one scored column is its distinct scores in DESCENDING order with the
number of positives and negatives at each: `(s (G,) float64, pos (G,) int64,
neg (G,) int64)`. Every metric the readout reports is a function of the tally, and
tallies MERGE (concatenate, re-group equal scores, add the counts). That is what
makes the metrics exact without a driver collect: a Spark backend tallies each
partition's cells per key, merges by key (`spark.per_key_metrics`), and evaluates
each key's metrics on the executor that owns it; only the per-key results reach the
driver. A tally is never larger than the column (G <= n), so what one executor holds
is one key's column, not the matrix.

Definitions, all equal to scikit-learn's on the same column (the oracle tests):

* **AUC** (Mann-Whitney, ties at half credit = `roc_auc_score`). Each negative at a
  score level is outranked by every positive above it and ties with the positives at
  its level: `AUC = sum_g neg_g (tp_g - pos_g / 2) / (P N)`, with `tp_g` the
  cumulative positives through level g.
* **AP** (`average_precision_score`, the step sum, no interpolation):
  `AP = sum_g (pos_g / P) prec_g`, `prec_g = tp_g / (tp_g + fp_g)`.
* **precision@recall t**: the largest precision over the levels with recall >= t,
  counting the curve's (recall 0, precision 1) end as sklearn's PR curve does.
* **recall@FDR q**: the largest recall over the levels with precision >= 1 - q; 0.0
  when no level qualifies (the (0, 1) end always does).
* **top-k screening**: k = ceil(frac n) (at least 1); precision, recall and lift of
  the top k. When the k-th place falls inside a tie, the tied rows count at their
  mean positive rate: the EXPECTED value under random tie-breaking, which needs no
  row order. (The pre-ADR-0051 numpy scorer broke ties by row order; the two agree
  whenever the cut does not split a tie.)

`calibration_bins` / `ece_from_bins` are the same idea for the equal-width-bin ECE:
per-bin (count, sum p, sum y) add across partitions.
"""
from __future__ import annotations

import math

import numpy as np

EMPTY = (np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))


def _group(s, pos, neg):
    """Re-group equal scores and return the tally in descending score order."""
    if s.size == 0:
        return EMPTY
    u, inv = np.unique(s, return_inverse=True)
    p = np.bincount(inv, weights=pos, minlength=u.size)
    n = np.bincount(inv, weights=neg, minlength=u.size)
    return (u[::-1].copy(), np.rint(p[::-1]).astype(np.int64),
            np.rint(n[::-1]).astype(np.int64))


def tally(score, y):
    """Tally one column: `score` (n,) and binary `y` (n,)."""
    s = np.asarray(score, dtype=np.float64).ravel()
    yy = np.asarray(y).ravel().astype(bool)
    return _group(s, yy.astype(np.float64), (~yy).astype(np.float64))


def merge(a, b):
    """The tally of the union of two columns' rows."""
    if a[0].size == 0:
        return b
    if b[0].size == 0:
        return a
    return _group(np.concatenate([a[0], b[0]]),
                  np.concatenate([a[1], b[1]]).astype(np.float64),
                  np.concatenate([a[2], b[2]]).astype(np.float64))


def node_metrics(t, *, recall_targets=(), fdr_targets=(), topk_frac=0.01):
    """Every metric of one tallied column. Single-class columns get `auc`/`ap` None
    and NaN precision@recall / recall@FDR (undefined, as the scorers always
    reported them); `n_pos`, `n_neg` and `n_distinct` are always set."""
    s, pos, neg = t
    P, N = int(pos.sum()), int(neg.sum())
    out = {"n_pos": P, "n_neg": N, "n_distinct": int(s.size),
           "topk_frac": float(topk_frac)}
    if P == 0 or N == 0:
        out.update(auc=None, ap=None, prec_at_k=None, recall_at_k=None,
                   lift_at_k=None,
                   par={float(r): float("nan") for r in recall_targets},
                   raf={float(q): float("nan") for q in fdr_targets})
        return out
    tp = np.cumsum(pos).astype(np.float64)
    fp = np.cumsum(neg).astype(np.float64)
    prec = tp / (tp + fp)
    rec = tp / P
    out["auc"] = float(np.sum(neg * (tp - pos / 2.0)) / (float(P) * float(N)))
    out["ap"] = float(np.sum(pos * prec) / P)
    par = {}
    for r in recall_targets:
        m = rec >= r
        best = float(np.max(prec[m])) if m.any() else float("nan")
        par[float(r)] = max(best, 1.0) if r <= 0 else best
    out["par"] = par
    raf = {}
    for q in fdr_targets:
        m = prec >= (1.0 - q)
        raf[float(q)] = float(np.max(rec[m])) if m.any() else 0.0
    out["raf"] = raf
    n = P + N
    k = min(n, max(1, int(math.ceil(float(topk_frac) * n))))
    cum = tp + fp
    g = int(np.searchsorted(cum, k))                  # first level reaching k
    size = float(pos[g] + neg[g])
    tpk = (tp[g] - pos[g]) + (k - (cum[g] - size)) * pos[g] / size
    prev = P / n
    out["prec_at_k"] = float(tpk / k)
    out["recall_at_k"] = float(tpk / P)
    out["lift_at_k"] = float(tpk / k / prev)
    return out


def curve(t, *, max_points=300):
    """ROC and PR points of one tallied column, thinned to about `max_points`
    levels (evenly by level index, both ends kept): `{threshold, tpr, fpr,
    precision, recall}` lists, highest threshold first. None for a single-class
    column."""
    s, pos, neg = t
    P, N = int(pos.sum()), int(neg.sum())
    if P == 0 or N == 0:
        return None
    tp = np.cumsum(pos).astype(np.float64)
    fp = np.cumsum(neg).astype(np.float64)
    idx = np.unique(np.rint(np.linspace(0, s.size - 1, min(int(max_points), s.size)))
                    .astype(np.int64))
    return {"threshold": s[idx].tolist(), "tpr": (tp[idx] / P).tolist(),
            "fpr": (fp[idx] / N).tolist(),
            "precision": (tp[idx] / (tp[idx] + fp[idx])).tolist(),
            "recall": (tp[idx] / P).tolist()}


def calibration_bins(p, y, n_bins=10):
    """Per equal-width bin of `p` in [0, 1]: (count, sum p, sum y), each (n_bins,).
    Bin edges as `np.digitize` over `linspace(0, 1, n_bins + 1)`, p = 1 in the last."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, int(n_bins) - 1)
    return (np.bincount(idx, minlength=n_bins).astype(np.float64),
            np.bincount(idx, weights=p, minlength=n_bins),
            np.bincount(idx, weights=y, minlength=n_bins))


def ece_from_bins(bins):
    """Expected calibration error sum_b (n_b / n) |mean p_b - mean y_b| from merged
    `calibration_bins`; None when empty."""
    cnt, sp, sy = bins
    n = float(cnt.sum())
    if n == 0:
        return None
    return float(np.sum(np.abs(sp - sy)) / n)
