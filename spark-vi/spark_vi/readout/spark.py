"""Spark backend of the readout: the kernels in `readout.kernels` behind
`mapPartitions` + `treeAggregate`, over a DataFrame of scored rows.

Input frame contract: a features column (`topicDistribution` by default; a Spark ML
Vector or array of length K), `label` and `labelMask` (arrays of C doubles), and,
for the lean collect, an int64 row-id column. On frozen features the C per-node
problems are independent K-dim convex fits that SHARE every data pass, so one
`treeAggregate` replaces C separate Spark ML jobs (`SparkStatsFn`), and the same
independence lets a pass be restricted to the heads the solver still needs
(`node_mask=`). ADR 0047 closure doctrine: nothing array-shaped rides a task
closure (parameters go by a fresh broadcast per pass, destroyed in `finally`), and
treeAggregate zeros are None sentinels.

`SparkReadoutData` is this backend's half of the interface `readout.heads.fit_heads`
drives; `readout.local.LocalReadoutData` is the other.
"""
from __future__ import annotations

import logging

import numpy as np

from spark_vi.core.broadcast import destroy_broadcast
from spark_vi.core.retry import _error_first_line, retry_spark_action
from spark_vi.readout.kernels import (
    _COVERAGE_BINS, coverage_from_accum, dense_triples, densify_lean_blocks,
    lean_eval_kernel, moments_kernel, moments_to_mu_sd, pack_rows, row_quads,
    row_sparse, row_triples, sparse_moments_kernel, sparse_stats_kernel,
    stats_kernel, to_array, topm_coverage_kernel,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Spark wiring (cluster-covered; thin shells over the kernels above).         #
# --------------------------------------------------------------------------- #
def theta_topm_coverage(scored_df, K, *, ms=(64, 128, 256, 512),
                        topic_col="topicDistribution", n_bins=_COVERAGE_BINS,
                        depth=2, q=0.10, excess=False):
    """One distributed pass measuring how much θ mass a top-m truncation keeps.
    `excess=True` measures the mass above the per-doc prior floor
    (`topm_coverage_kernel`).

    Returns `{m: (mean_coverage, p10_coverage)}`. This is the MEASUREMENT that has
    to precede any use of the sparse-θ path: the lever's entire premise is that a
    Dirichlet(α=0.5) posterior mean over thousands of topics is concentrated, and a
    premise about the data is not something a kernel can assume. The p10 is the
    number that matters — a mean of 0.98 with a p10 of 0.4 means the truncation is
    fine for the typical doc and destroys the tail, and the tail is what the rare
    nodes are fit on.

    Cheap by construction (see `topm_coverage_kernel`): one O(K) partition per doc
    and a fixed `(len(ms), n_bins)` int64 accumulator per partition, so this costs a
    fraction of one L-BFGS pass and nothing D-sized reaches the driver.
    """
    ms = tuple(int(m) for m in ms)

    def _local(rows, _col=topic_col, _ms=ms, _nb=int(n_bins), _ex=bool(excess)):
        return [topm_coverage_kernel((to_array(r[_col]) for r in rows), _ms, _nb,
                                      excess=_ex)]

    def _combine(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    zero = (np.zeros(len(ms)), np.zeros((len(ms), int(n_bins)), dtype=np.int64), 0)
    # `.rdd` is taken INSIDE the retried closure: on a DataFrame whose cached
    # transform has not materialized yet, the `javaToPython` conversion behind
    # `.rdd` can itself run Spark jobs — and this coverage pass is the FIRST
    # action of the whole readout, so a preemption abort lands there first.
    sums, hist, n = retry_spark_action(
        lambda: scored_df.select(topic_col).rdd.mapPartitions(_local).treeAggregate(
            zero, _combine, _combine, depth=int(depth)),
        label="theta top-m coverage")
    return coverage_from_accum(sums, hist, n, ms, q=q)


def mass_grid(K, *, start=16):
    """The candidate truncation widths a mass target is resolved over: powers of
    two from `start` up to (and excluding) K, then K itself — so the answer can be
    "no truncation" when the mass target is not met below full width."""
    K = int(K)
    ms = []
    m = int(start)
    while m < K:
        ms.append(m)
        m *= 2
    ms.append(K)
    return tuple(ms)


def choose_topm_for_mass(coverage, q, *, K=None):
    """The smallest m whose p10 θ-mass coverage reaches `q`; `0` (= full K, no
    truncation) when none does or when the only such m is K itself.

    `coverage` is `theta_topm_coverage`'s `{m: (mean, p10)}`. The p10 is the
    criterion, not the mean: a truncation is judged on the WORST decile of
    documents — the tail the rare nodes are fit on — exactly as the coverage
    pass's own docstring argues. Pure, so the rule is unit-tested against made-up
    coverage tables; the Spark pass that produces the table is
    `resolve_theta_topm_by_mass`."""
    q = float(q)
    ok = sorted(int(m) for m, (_mean, p10) in coverage.items() if float(p10) >= q)
    if not ok:
        return 0
    m = ok[0]
    if K is not None and m >= int(K):
        return 0
    return m


def resolve_theta_topm_by_mass(scored_df, K, q, *, topic_col="topicDistribution",
                               depth=2, grid=None, excess=True):
    """Resolve a MASS target to a top-m width on the train frame (truncate by
    mass, not by count).

    `--readout-theta-topm 256` kept 84% of the topics at K=306 and 17% at K=1498;
    at a ribbon-scale K it would keep ~5%, and whether that is a cheap
    reparameterization or a lobotomy depends on the data, so a fixed count is the
    wrong knob. This measures coverage over `mass_grid(K)` in ONE distributed pass
    (`theta_topm_coverage`) and picks the smallest width whose p10 coverage meets
    `q` (`choose_topm_for_mass`). The chosen m then flows through every existing
    `theta_topm=` seam unchanged — the kernels stay rectangular (every doc keeps the
    same m), which is what makes them exact.

    Returns `(m, coverage)`; `m == 0` means no width below K met the target and the
    readout runs dense.

    The coverage is of the mass ABOVE THE PRIOR FLOOR (`excess=True`, the
    default): raw θ coverage on a wide gated layout is
    dominated by α's floor over the allowed topics and resolved q=0.99 to full K
    (p10 0.117 at m=256) although the token assignments are concentrated. See
    `topm_coverage_kernel`."""
    K = int(K)
    ms = tuple(grid) if grid is not None else mass_grid(K)
    cov = theta_topm_coverage(scored_df, K, ms=ms, topic_col=topic_col, depth=depth,
                              excess=excess)
    return choose_topm_for_mass(cov, q, K=K), cov


def masked_moments(scored_df, C, K, *, topic_col="topicDistribution",
                   label_col="label", mask_col="labelMask", eps=1e-12, depth=2,
                   topm=0):
    """One distributed pass for the per-node masked standardization moments.

    The sklearn oracle standardizes each node's features on THAT node's
    observed train rows, so the reparameterization needs (C,K) means and sds. This
    is the one-time aggregate that produces them; the result (~174 MB at C=K=3,300)
    is driver-side numpy and is then held fixed for the whole fit.

    Returns `(mu (C,K), sd (C,K), n_obs (C,), n_pos (C,))`. `n_obs`/`n_pos` come
    free from the same pass and are what the caller needs to identify the degenerate
    (empty or single-class observed set) nodes — they must get
    the same constant-prediction fallback as `_lr_proba_per_label_masked`.

    `treeAggregate` (rather than `treeReduce`) so an empty corpus returns correctly
    shaped zeros instead of raising.

    `topm > 0` computes the moments of the TRUNCATED features — mandatory whenever
    the fit runs truncated, since a standardization derived from full θ describes a
    design matrix the solver never sees (module docstring, consistency rule).
    """
    C, K = int(C), int(K)
    topm = int(topm)

    def _local(rows, _cols=(topic_col, label_col, mask_col), _C=C, _K=K, _m=topm):
        if _m > 0:
            return [sparse_moments_kernel(row_sparse(rows, *_cols, _m), _C, _K)]
        return [moments_kernel(row_triples(rows, *_cols), _C, _K)]

    # None-sentinel zero, same reason as SparkStatsFn's pass zero: a dense
    # zeroValue ((C,K)×2 ≈ 217 MB here) rides the task closure, which pyspark
    # auto-broadcasts over ~1 MB into an internal broadcast whose pickled temp
    # file nothing ever unlinks. One-time per readout rather than per-pass, but
    # the same leak class — keep every treeAggregate closure under the threshold.
    def _combine(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])

    # Retried: the moments pass is a pure sum over a lineage-recomputable
    # projection, and losing it to a preemption wave costs the whole fit (it is
    # the standardization every later pass is expressed in). `.rdd` is taken
    # inside the closure — `javaToPython` can run jobs of its own on an
    # unmaterialized frame (see theta_topm_coverage).
    out = retry_spark_action(
        lambda: scored_df.select(topic_col, label_col, mask_col)
        .rdd.mapPartitions(_local).treeAggregate(
            None, _combine, _combine, depth=int(depth)),
        label="masked moments")
    sum_theta, sum_theta_sq, n_obs, n_pos = (
        (np.zeros((C, K)), np.zeros((C, K)), np.zeros(C), np.zeros(C))
        if out is None else out)
    mu, sd = moments_to_mu_sd(sum_theta, sum_theta_sq, n_obs, eps=eps)
    return mu, sd, n_obs, n_pos


class SparkStatsFn:
    """Callable `stats_fn(W_std, b_std, node_mask=None)
    -> (loss_data (C,), gW_std (C,K), gb (C,))`.

    The per-pass seam: ONE `treeAggregate` returns all C heads' gradients, so a
    batched L-BFGS iteration costs one data scan regardless of C — versus Spark ML's
    C sequential jobs.

    `node_mask` is the SECOND economy, and at C in the thousands the bigger one. A
    batched L-BFGS iteration is one pass only when every node accepts its first
    trial step; with C≈3,800 independent Armijo tests sharing the pass, some node
    is nearly always still backtracking (a run measured ~26 full passes per
    iteration for it). The solver therefore names the nodes whose stats
    it does not already hold, and the kernels skip every other node's cells — so a
    deep straggler costs a pass over ITS rows, not over all 56M cells. Masked-out
    rows of the result are exactly zero (the contract: the caller owns merging
    them with the values it holds), and an all-True mask is normalized to `None`
    so the unmasked path stays literally the pre-mask code.

    Lifecycle, deliberately explicit:

      * The projected `(theta, obs_idx, y_obs)` RDD is built and **persisted once**
        at construction (`MEMORY_AND_DISK`: L-BFGS re-reads it every pass, and
        re-deriving it means re-scanning the scored DataFrame — which on the cluster
        is a Parquet/BigQuery read, not a cheap in-memory projection). A `count()`
        materializes it up front so the first pass measures the fit, not the read.
      * A **fresh broadcast per call**, DESTROYED before returning. The parameters
        change every L-BFGS pass, so a long-lived broadcast would only pile up ~87 MB
        (C=K=3,300) of stale blocks per executor, and an unpersist-only lifecycle
        leaks the driver's copy and pickled temp file every pass. The
        broadcast/destroy pair lives INSIDE the retried closure
        (`retry_spark_action`), so a preemption-wave job abort costs one pass and a
        backoff sleep rather than the whole solve — see that helper for why the
        retry is idempotent rather than hopeful.
      * `close()` (or `with make_spark_stats_fn(...) as stats_fn:`) unpersists the
        cached RDD. The caller owns that — an un-closed instance leaks executor
        storage for the life of the SparkContext.

    The standardized↔raw fold is INJECTED (`fold_standardization`,
    `standardized_grad_from_raw` from `readout.solver`) so this module
    never imports the solver; executors only ever see the folded raw-space (V, b_raw).
    """

    def __init__(self, scored_df, C, K, mu, sd, *, fold_standardization,
                 standardized_grad_from_raw, topic_col="topicDistribution",
                 label_col="label", mask_col="labelMask", depth=2,
                 storage_level=None, topm=0):
        from pyspark import StorageLevel

        self.C, self.K = int(C), int(K)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self._fold = fold_standardization
        self._fold_grad = standardized_grad_from_raw
        self._depth = int(depth)
        # Truncation is baked into the PERSISTED projection, so the pass cost, the
        # cache footprint and the moments the caller passed in all describe one
        # design matrix. Re-deriving it per pass would be the same arithmetic at
        # K-wide cost, which would defeat the lever entirely.
        self._topm = int(topm)
        cols = (topic_col, label_col, mask_col)

        def _pack(rows, _cols=cols, _m=self._topm):
            return pack_rows(rows, *_cols, topm=_m)

        level = StorageLevel.MEMORY_AND_DISK if storage_level is None else storage_level

        def _project():
            # `.rdd` inside the retried closure, as in `theta_topm_coverage`: on an
            # unmaterialized frame `javaToPython` can run jobs of its own.
            rdd = scored_df.select(*cols).rdd.mapPartitions(_pack).persist(level)
            try:
                rdd.count()                     # materialize before the first pass
            except Exception:
                rdd.unpersist()
                raise
            return rdd

        self._rdd = retry_spark_action(_project, label="packed projection")
        self._closed = False

    def __call__(self, W_std, b_std, node_mask=None):
        V, b_raw = self._fold(np.asarray(W_std, dtype=np.float64),
                              np.asarray(b_std, dtype=np.float64), self.mu, self.sd)
        V = np.ascontiguousarray(V, dtype=np.float64)
        b_raw = np.ascontiguousarray(b_raw, dtype=np.float64)
        keep = None
        if node_mask is not None:
            keep = np.ascontiguousarray(np.asarray(node_mask, dtype=bool))
            if keep.shape != (self.C,):
                raise ValueError(
                    f"node_mask shape {keep.shape} != ({self.C},)")
            if keep.all():
                keep = None                     # nothing to skip: the plain pass
        sc = self._rdd.context
        C, K = self.C, self.K

        # The treeAggregate zero is a None SENTINEL, not a dense (C,K) tuple.
        # zeroValue is pickled into the task closure, and pyspark AUTO-BROADCASTS
        # any closure over ~1 MB — an internal broadcast no destroy() of ours ever
        # reaches, whose pickled temp file sits under the app's pyspark-* dir
        # until context shutdown. A dense zero here is ~108 MB of closure per
        # pass, one leaked temp file per pass (ADR 0047's addendum, leak #2):
        # periodic GC cleans the JVM half, nothing cleans the file.
        def _combine(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

        zero = None

        def _pass(_m=self._topm):
            # The broadcast is created INSIDE the retried closure. A retry is here
            # precisely because executors died, and their copies of the broadcast
            # blocks died with them; a fresh broadcast per attempt keeps the
            # "parameters for THIS pass, then gone" lifecycle exact instead of
            # carrying a half-torn-down one into the retry. ~87 MB per attempt is
            # noise next to the pass it feeds.
            #
            # The mask rides IN the parameter broadcast rather than in a second
            # one: it changes on every trial exactly as (V, b_raw) do, and it is
            # (C,) bools against ~87 MB of parameters, so a separate broadcast
            # would only add a round trip per pass.
            bcast = sc.broadcast((V, b_raw, keep))
            try:
                def _local(packed, _b=bcast, _C=C, _K=K, _m=_m):
                    V_, b_, keep_ = _b.value
                    if _m > 0:
                        return [sparse_stats_kernel(packed, V_, b_, _C, _K,
                                                     node_mask=keep_)]
                    return [stats_kernel(dense_triples(packed, _C), V_, b_, _C,
                                          _K, node_mask=keep_)]

                return self._rdd.mapPartitions(_local).treeAggregate(
                    zero, _combine, _combine, depth=self._depth)
            finally:
                # DESTROY per attempt, not unpersist. `unpersist` removes the
                # EXECUTOR copies only; the driver keeps both its block-manager
                # copy AND the pickled temp file pyspark wrote under the Spark
                # temp dir when the broadcast was created — and those survive
                # until `destroy`. At one fresh ~117 MB broadcast per pass, an
                # unpersist-only lifecycle leaks ~117 MB of driver-local disk per
                # pass (~53 GB over a 450-pass solve, enough to end it with ENOSPC
                # from `sc.broadcast` itself).
                # A destroyed broadcast is unusable afterwards, which is exactly
                # the contract here: the next attempt/pass builds a fresh one.
                destroy_broadcast(bcast)

        # THE hot seam: L-BFGS calls this once per line-search trial, so a
        # multi-hour solve is overwhelmingly likely to be inside this action when
        # a preemption wave lands.
        out = retry_spark_action(
            _pass,
            label=f"stats pass ({'all' if keep is None else int(keep.sum())}/{C} "
                  "nodes)")
        # None only if every partition was empty — the cached projection is
        # materialized non-empty in __init__, so this is belt-and-braces for the
        # sentinel zero, not a reachable state in a real solve.
        loss, g_raw, s = ((np.zeros(C), np.zeros((C, K)), np.zeros(C))
                          if out is None else out)
        gW_std = self._fold_grad(g_raw, s, self.mu, self.sd)
        return loss, gW_std, s

    def close(self):
        """Unpersist the cached projection. Idempotent, and SAFE ON A DEAD CONTEXT.

        This runs on the error path too (`with` exit), and when the failure is the
        SparkContext itself dying — the cluster manager failing the whole
        application — the JVM-side unpersist NPEs, and an exception here
        REPLACES the real one mid-propagation. Cleanup of a dead context's blocks
        is moot (the block managers died with it), so a failed unpersist is logged
        and swallowed, never raised.
        """
        if not self._closed:
            self._closed = True
            try:
                self._rdd.unpersist()
            except Exception as exc:
                log.warning("packed-projection unpersist skipped "
                      f"({_error_first_line(exc)}) — harmless if the SparkContext "
                      "is shutting down")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_spark_stats_fn(scored_df, C, K, mu, sd, *, fold_standardization,
                        standardized_grad_from_raw, topic_col="topicDistribution",
                        label_col="label", mask_col="labelMask", depth=2,
                        storage_level=None, topm=0):
    """Build the batched-L-BFGS stats oracle over `scored_df`. See `SparkStatsFn`.

    Returned object is callable as `stats_fn(W_std, b_std, node_mask=None)` and
    MUST be `close()`d (or used as a context manager) to release the persisted
    projection.

    `topm > 0` fits on top-m truncated θ; `mu`/`sd` must then come from a
    `masked_moments` call with the SAME `topm`.
    """
    return SparkStatsFn(
        scored_df, C, K, mu, sd,
        fold_standardization=fold_standardization,
        standardized_grad_from_raw=standardized_grad_from_raw,
        topic_col=topic_col, label_col=label_col, mask_col=mask_col,
        depth=depth, storage_level=storage_level, topm=topm)



def collect_lean_proba(scored_df, C, V=None, b_raw=None, *, degenerate=None,
                       const=None, score_col="topicDistribution", label_col="label",
                       mask_col="labelMask", id_col="doc_key", theta_topm=0):
    """LEAN readout collect: `(proba (D,C) f32, y u8, mask u8, id_order)`.

    Not on the readout's production path since ADR 0051 step 4 (the metrics read
    `score_blocks` where they live); kept for inspecting and aligning small frames.

    The test split's per-node probabilities and labels at ~6 bytes per (row, node)
    cell (`lean_eval_kernel` / `densify_lean_blocks`): no features and no float64
    (D,C) arrays reach the driver. With `(V, b_raw)` the executors score the raw
    features with the fitted heads; with `V=None` an already-computed per-row (C,)
    probability column (`score_col`) is packed as-is.

    `degenerate`/`const` apply the fallback for nodes whose observed TRAIN set was
    empty or single-class: those columns are the lone class value (0.0 when nothing
    was observed), as `fit_heads` returns them.

    `theta_topm` must MATCH the fit that produced `(V, b_raw)`: the coefficients
    belong to the truncated design matrix. It is ignored on the `V is None` branch.

    `id_col` names an int64 column unique per row (the clinical drivers synthesize
    a doc key under that name)."""
    C = int(C)
    theta_topm = int(theta_topm)
    cols = (id_col, score_col, label_col, mask_col)
    sc = scored_df.sparkSession.sparkContext

    def _collect():
        if V is None:
            bcast = None

            def _block(rows, _C=C, _cols=cols):
                return [lean_eval_kernel(row_quads(rows, *_cols), _C)]
        else:
            bcast = sc.broadcast((np.ascontiguousarray(V, dtype=np.float64),
                                  np.ascontiguousarray(b_raw, dtype=np.float64)))

            def _block(rows, _b=bcast, _C=C, _cols=cols, _m=theta_topm):
                V_, b_ = _b.value
                return [lean_eval_kernel(row_quads(rows, *_cols, topm=_m), _C, V_, b_)]

        try:
            return scored_df.select(*cols).rdd.mapPartitions(_block).collect()
        finally:
            if bcast is not None:
                destroy_broadcast(bcast)

    blocks = retry_spark_action(_collect, label="lean eval collect")
    proba, y, mask, ids = densify_lean_blocks(blocks, C)
    if degenerate is not None and bool(np.any(degenerate)):
        deg = np.asarray(degenerate, dtype=bool)
        proba[:, deg] = np.asarray(const, dtype=np.float32)[deg]
    return proba, y, mask, ids


class SparkReadoutData:
    """One split of scored rows on Spark, as `readout.heads.fit_heads` reads it.

    Thin and lazy: nothing is persisted until `stats_fn` builds its packed
    projection (closed by the solve's `with` block). `topm` fixes the truncation
    width every pass reads; `coverage` measures it on the untruncated features."""

    can_measure_coverage = True

    def __init__(self, df, C, K, *, topm=0, topic_col="topicDistribution",
                 label_col="label", mask_col="labelMask", id_col="doc_key",
                 depth=None):
        self.df = df
        self.C, self.K, self.topm = int(C), int(K), int(topm)
        self.cols = dict(topic_col=topic_col, label_col=label_col, mask_col=mask_col)
        self.id_col = id_col
        # Same driver-burst sizing rule as spark_vi.core.runner._agg_depth: the
        # stats partial is (C,K) float64 x2 (loss/grad + moments).
        self.depth = (int(depth) if depth is not None
                      else 3 if 2 * self.C * self.K * 8 > 128 * 1024 * 1024 else 2)

    def coverage(self, ms=(64, 128, 256, 512), *, excess=False):
        return theta_topm_coverage(self.df, self.K, ms=ms,
                                   topic_col=self.cols["topic_col"],
                                   depth=self.depth, excess=excess)

    def moments(self, *, eps=1e-12):
        return masked_moments(self.df, self.C, self.K, eps=eps, depth=self.depth,
                              topm=self.topm, **self.cols)

    def stats_fn(self, mu, sd, *, fold_standardization, standardized_grad_from_raw):
        return SparkStatsFn(self.df, self.C, self.K, mu, sd,
                            fold_standardization=fold_standardization,
                            standardized_grad_from_raw=standardized_grad_from_raw,
                            depth=self.depth, topm=self.topm, **self.cols)

    def score_blocks(self, V, b_raw, *, degenerate=None, const=None, extra_cols=()):
        """The split scored by the heads as persisted `readout.evaluate.Blocks`, left
        on the executors; `unpersist()` releases them and the heads broadcast."""
        from spark_vi.readout.evaluate import Blocks

        rdd, bcast = score_blocks(
            self.df, self.C, V, b_raw, degenerate=degenerate, const=const,
            score_col=self.cols["topic_col"], label_col=self.cols["label_col"],
            mask_col=self.cols["mask_col"], id_col=self.id_col, theta_topm=self.topm,
            extra_cols=extra_cols)
        return Blocks(rdd, bcasts=[bcast]).persist()

    def lean_proba(self, V, b_raw, *, degenerate=None, const=None):
        return collect_lean_proba(
            self.df, self.C, V, b_raw, degenerate=degenerate, const=const,
            score_col=self.cols["topic_col"], label_col=self.cols["label_col"],
            mask_col=self.cols["mask_col"], id_col=self.id_col,
            theta_topm=self.topm)


def score_blocks(scored_df, C, V, b_raw, *, degenerate=None, const=None,
                 score_col="topicDistribution", label_col="label",
                 mask_col="labelMask", id_col="doc_key", theta_topm=0, extra_cols=()):
    """The test split scored by the heads as an RDD of SCORE BLOCKS, one per
    partition, kept on the executors (`readout.evaluate`; ADR 0051 step 4):
    `{"ids", "proba" (n,C) f32, "y" (n,C) u8, "mask" (n,C) u8}` plus one list per
    `extra_cols` name (that column's per-row values). The same kernel as
    `collect_lean_proba` (and the same fallback for degenerate heads), without the
    collect. The broadcast of the heads is returned for the caller to destroy once
    the blocks are no longer read: `(rdd, bcast)`."""
    C = int(C)
    deg = (np.zeros(C, dtype=bool) if degenerate is None
           else np.asarray(degenerate, dtype=bool))
    cst = np.zeros(C, dtype=np.float32) if const is None else np.asarray(
        const, dtype=np.float32)
    sc = scored_df.sparkSession.sparkContext
    bcast = sc.broadcast((np.ascontiguousarray(V, dtype=np.float64),
                          np.ascontiguousarray(b_raw, dtype=np.float64), deg, cst))
    cols = (id_col, score_col, label_col, mask_col)
    extra = tuple(extra_cols)

    def _block(rows, _b=bcast, _C=C, _cols=cols, _m=int(theta_topm), _ex=extra):
        rows = list(rows)
        if not rows:
            return []
        V_, b_, d_, c_ = _b.value
        blk = lean_eval_kernel(row_quads(rows, *_cols, topm=_m), _C, V_, b_)
        proba, y, mask, ids = densify_lean_blocks([blk], _C)
        if d_.any():
            proba[:, d_] = c_[d_]
        out = {"ids": np.asarray(ids, dtype=np.int64), "proba": proba, "y": y,
               "mask": mask}
        for name in _ex:
            out[name] = [r[name] for r in rows]
        return [out]

    rdd = scored_df.select(*(cols + extra)).rdd.mapPartitions(_block)
    return rdd, bcast
