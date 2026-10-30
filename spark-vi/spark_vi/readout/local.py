"""Local backend of the readout: the same kernels over rows held in memory.

A readout pass on Spark is a scan of the whole scored frame (exp 0163: 22-65 s per
pass, 700-900 passes per solve). When the split fits in one machine's memory --
always for a top-m truncated θ, which is ~3 KB per row at m=256 -- `LocalReadoutData`
holds it once and every pass is the same numpy kernel over that list
(`readout.kernels`), so a local solve agrees with the Spark one up to floating-point
summation order (ADR 0051 §2). Build it from numpy arrays (`from_arrays`, no Spark
at all) or from a scored DataFrame (`from_spark`: the executors pack and truncate
the rows, the driver collects the compact form).

Each row is held as `(id, features, y_idx, m_idx)`: features dense (K,) or, under
top-m truncation, the `(idx (m,), val (m,))` pair; `y_idx` the positive nodes;
`m_idx` the observed nodes, or None for "every node observed".
"""
from __future__ import annotations

import numpy as np

from spark_vi.readout.kernels import (
    dense_triples, densify_lean_blocks, lean_eval_kernel, moments_kernel,
    moments_to_mu_sd, sparse_moments_kernel, sparse_stats_kernel, stats_kernel,
    to_array, topm_sparse,
)

# Rows per lean-eval block: bounds the transient (n, C) float32 a block holds.
_LEAN_BLOCK_ROWS = 4096


def _compact_row(row_id, theta, label, mask, topm):
    """One row -> `(id, features, y_idx, m_idx)`; truncation happens here, once."""
    theta = np.asarray(theta, dtype=np.float64)
    feats = topm_sparse(theta, topm) if int(topm) > 0 else theta
    mask = np.asarray(mask)
    m_idx = None if mask.all() else np.flatnonzero(mask).astype(np.int32)
    return (int(row_id), feats, np.flatnonzero(np.asarray(label)).astype(np.int32),
            m_idx)


def _spark_compact_rows(rows, id_col, topic_col, label_col, mask_col, topm):
    """Executor-side: Spark Rows -> compact local rows (truncated when topm > 0)."""
    for r in rows:
        yield _compact_row(r[id_col], to_array(r[topic_col]), to_array(r[label_col]),
                           to_array(r[mask_col]), topm)


class LocalStatsFn:
    """`stats_fn(W_std, b_std, node_mask=None) -> (loss (C,), gW_std (C,K), gb (C,))`
    over the held rows: `SparkStatsFn.__call__` without the treeAggregate. A
    context manager for symmetry with the Spark backend (there is nothing to free)."""

    def __init__(self, data, mu, sd, *, fold_standardization, standardized_grad_from_raw):
        self.data = data
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self._fold = fold_standardization
        self._fold_grad = standardized_grad_from_raw

    def __call__(self, W_std, b_std, node_mask=None):
        d = self.data
        V, b_raw = self._fold(np.asarray(W_std, dtype=np.float64),
                              np.asarray(b_std, dtype=np.float64), self.mu, self.sd)
        keep = None
        if node_mask is not None:
            keep = np.asarray(node_mask, dtype=bool)
            if keep.shape != (d.C,):
                raise ValueError(f"node_mask shape {keep.shape} != ({d.C},)")
            if keep.all():
                keep = None
        if d.topm > 0:
            loss, g_raw, s = sparse_stats_kernel(d._packed(), V, b_raw, d.C, d.K,
                                                 node_mask=keep)
        else:
            loss, g_raw, s = stats_kernel(dense_triples(d._packed(), d.C), V, b_raw,
                                          d.C, d.K, node_mask=keep)
        return loss, self._fold_grad(g_raw, s, self.mu, self.sd), s

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LocalReadoutData:
    """One split of scored rows in memory, as `readout.heads.fit_heads` reads it."""

    def __init__(self, rows, C, K, *, topm=0):
        self.rows = list(rows)
        self.C, self.K, self.topm = int(C), int(K), int(topm)

    @classmethod
    def from_arrays(cls, theta, label, mask=None, ids=None, *, topm=0):
        """`theta (D,K)`, `label (D,C)`, `mask (D,C)` (None = all observed)."""
        theta = np.asarray(theta, dtype=np.float64)
        label = np.asarray(label)
        D, K = theta.shape
        C = label.shape[1]
        mask = np.ones((D, C), dtype=np.uint8) if mask is None else np.asarray(mask)
        ids = np.arange(D) if ids is None else np.asarray(ids)
        rows = [_compact_row(ids[d], theta[d], label[d], mask[d], topm)
                for d in range(D)]
        return cls(rows, C, K, topm=topm)

    @classmethod
    def from_spark(cls, df, C, K, *, topm=0, topic_col="topicDistribution",
                   label_col="label", mask_col="labelMask", id_col="doc_key"):
        """Collect a scored frame in compact form: the executors pack (and truncate)
        each row, so the driver receives O(m + nnz) per row, not O(K + C)."""
        from spark_vi.core.retry import retry_spark_action

        cols = (id_col, topic_col, label_col, mask_col)

        def _pack(rows, _cols=cols, _m=int(topm)):
            return _spark_compact_rows(rows, *_cols, _m)

        rows = retry_spark_action(
            lambda: df.select(*cols).rdd.mapPartitions(_pack).collect(),
            label="local readout collect")
        return cls(rows, C, K, topm=topm)

    def _packed(self):
        """The rows in `kernels.pack_rows` form: `(theta, obs, y_obs)`, or
        `(idx, val, obs, y_obs)` under truncation."""
        all_obs = np.arange(self.C, dtype=np.int32)
        for _id, feats, y_idx, m_idx in self.rows:
            obs = all_obs if m_idx is None else m_idx
            y_obs = np.isin(obs, y_idx).astype(np.float64)
            if self.topm > 0:
                yield (feats[0], feats[1], obs, y_obs)
            else:
                yield (feats, obs, y_obs)

    def moments(self, *, eps=1e-12):
        if self.topm > 0:
            s1, s2, n_obs, n_pos = sparse_moments_kernel(self._packed(), self.C, self.K)
        else:
            s1, s2, n_obs, n_pos = moments_kernel(dense_triples(self._packed(), self.C),
                                                  self.C, self.K)
        mu, sd = moments_to_mu_sd(s1, s2, n_obs, eps=eps)
        return mu, sd, n_obs, n_pos

    def stats_fn(self, mu, sd, *, fold_standardization, standardized_grad_from_raw):
        return LocalStatsFn(self, mu, sd, fold_standardization=fold_standardization,
                            standardized_grad_from_raw=standardized_grad_from_raw)

    def score_blocks(self, V, b_raw, *, degenerate=None, const=None, extra=None):
        """The held rows scored, as one in-memory `readout.evaluate.Blocks` block.

        `extra` ({name: per-row values, in row order}) carries further columns into
        the block, as `SparkReadoutData.score_blocks(extra_cols=)` carries Spark
        columns: the held rows keep only the readout's own fields, so the values
        come from the caller."""
        from spark_vi.readout.evaluate import Blocks

        proba, y, mask, ids = self.lean_proba(V, b_raw, degenerate=degenerate,
                                              const=const)
        blk = {"ids": np.asarray(ids, dtype=np.int64), "proba": proba, "y": y,
               "mask": mask}
        for name, vals in (extra or {}).items():
            vals = list(vals)
            if len(vals) != len(self.rows):
                raise ValueError(f"extra[{name!r}] has {len(vals)} values for "
                                 f"{len(self.rows)} rows")
            blk[name] = vals
        return Blocks([blk])

    def lean_proba(self, V, b_raw, *, degenerate=None, const=None):
        """Score the held rows: `(proba (D,C) f32, y u8, mask u8, ids)`, the same
        arrays `readout.spark.collect_lean_proba` returns."""
        C = self.C
        V = np.ascontiguousarray(V, dtype=np.float64)
        b_raw = np.ascontiguousarray(b_raw, dtype=np.float64)

        def _quads(chunk):
            for row_id, feats, y_idx, m_idx in chunk:
                label = np.zeros(C)
                label[y_idx] = 1.0
                if m_idx is None:
                    mask = np.ones(C)
                else:
                    mask = np.zeros(C)
                    mask[m_idx] = 1.0
                yield (row_id, feats, label, mask)

        blocks = [lean_eval_kernel(_quads(self.rows[s:s + _LEAN_BLOCK_ROWS]), C, V, b_raw)
                  for s in range(0, len(self.rows), _LEAN_BLOCK_ROWS)]
        proba, y, mask, ids = densify_lean_blocks(blocks, C)
        if degenerate is not None and bool(np.any(degenerate)):
            deg = np.asarray(degenerate, dtype=bool)
            proba[:, deg] = np.asarray(const, dtype=np.float32)[deg]
        return proba, y, mask, ids
