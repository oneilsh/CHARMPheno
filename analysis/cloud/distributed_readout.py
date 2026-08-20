"""Distributed per-node readout: executor-side stats for the batched multi-head LR.

This is the Spark half of the plan in
`docs/superpowers/plans/2026-08-20-distributed-readout-plan.md` ("Distributed
per-node readout — driver-safe at whole-Mondo scale"). The plan's blocker is that
`gated_pc_cloud._collect_theta_labels` pulls per-doc θ (D,K) AND dense per-doc
label/mask (D,C) to the driver: ~8 GB + ~16 GB at K≈C≈3,300 over a 300k-doc
cohort, and `readout_sample_frac` is the wrong lever because uniform row sampling
guts exactly the rare tail the conditional/VOI positioning cares about.

What this module supplies (plan §Design steps 1–3), and nothing else:

  * `masked_moments`   — the ONE-TIME per-node masked mean/var pass that defines
    the standardization reparameterization (plan §1: "Standardization comes from
    one masked mean/var aggregate ((C,K)×2 ≈ 174 MB, one-time)").
  * `make_spark_stats_fn` — the per-L-BFGS-pass `treeAggregate` seam: broadcast
    the current parameters, return per-node (summed log-loss, gradient) for ALL C
    heads from ONE data pass. This is the plan's key structural claim — on FROZEN
    θ the C per-node problems are independent K-dim convex fits that SHARE every
    data pass, so one pass replaces C separate Spark ML jobs.
  * `_lean_eval_kernel` — plan §2.1's LEAN driver collect: per doc, the dense
    float32 P(node) row plus the label/mask as INDEX lists, so the whole existing
    driver eval stack runs unchanged on ~6 bytes/cell instead of the driver path's
    24 (θ float64 + label/mask float64).
  * `score_cells_df` / `per_node_metric_rows` — plan §3's exact distributed eval:
    explode only the observed `(node, y, p)` cells (16 bytes/cell, not a K-wide
    feature row) and `groupBy(node)`; the driver receives (C,)-sized metric tables.
    The ESCAPE HATCH for when D_te x C outgrows the driver, not the default path.

**Layering / what this module deliberately does NOT know.** The solver and the
standardized↔raw fold live in `analysis/pc/batched_lr.py` (Package A). This module
speaks ONLY raw-θ coordinates and takes `fold_standardization` /
`standardized_grad_from_raw` as INJECTED callables, so the Spark layer never
imports the solver (and the solver stays unit-testable without Spark). The fold is
a fixed affine reparameterization — plan §1: "the fitted W is then mapped back to
raw-θ coordinates, so scoring needs no scaler" — which is why the executors only
ever need (V, b_raw).

**Repo convention (`gated_pc_cloud.py`): Spark wiring is cluster-covered, pure
numpy partition kernels are unit-tested.** Everything numeric here is therefore a
pure function of numpy arrays (`_moments_kernel`, `_stats_kernel`,
`_score_cells_kernel`); the Spark functions are thin `mapPartitions` +
`treeAggregate` shells around them, following the `mapPartitions(_local)` +
tree-combine idiom used throughout `spark_vi/mllib/topic/stm.py`.

**Executor-side imports.** `per_node_metric_rows` imports
`analysis.pc.evaluate._score_label` INSIDE the per-group scoring function on purpose:
the metric semantics (degenerate-column skip, `min_count` masking) must be the
same object the driver readout uses so they cannot drift (plan §"What must NOT
change"). That makes `analysis.pc.evaluate` — and hence `sklearn` (and, via its
module-level `PCTopicModel` import, `scipy`/`autograd`) — an executor dependency
for the eval step; ship `analysis/` on `--py-files` alongside the spark-vi and
charmpheno zips, or hoist `_score_label` into a leaf module first.

Input DataFrame contract (as produced by `gated_pc_cloud._collect_theta_labels`'s
source frame): `topicDistribution` a Spark ML dense Vector of length K, `label`
and `labelMask` arrays of C doubles.
"""
from __future__ import annotations

import numpy as np

# Scores are clipped to this magnitude before the sigmoid/log-loss, matching
# `gated_pc_cloud._converge_localized_head`'s `np.clip(z, -50, 50)`. σ(±50) is
# 1∓2e-22, so the clip is a no-op for any non-separating parameter vector; it
# exists to keep `exp` from overflowing on a run-away head (the |w_CK|=273
# blowup). NOTE the clip flattens the loss beyond ±50, so the returned gradient
# is the gradient of the UNCLIPPED loss evaluated at the clipped score — exact
# wherever it matters, and the same compromise the engine head already makes.
# Package A's in-memory reference (`batched_lr.make_inmemory_stats_fn`) uses an
# unclipped branch-stable sigmoid instead: the two agree to ~1e-22 in `p` and in
# every gradient, and can disagree in the reported LOSS only for a node driven
# past |z| = 50 (where the clipped objective is the bounded log-loss). Metrics —
# the plan's A/B equality gate — read `p`, so they are unaffected.
_SCORE_CLIP = 50.0

# Mask-density cutoff for the per-row score computation in `_stats_kernel` /
# `_score_cells_kernel`. `V[obs] @ theta` costs |obs|·K flops PLUS a gather that
# copies |obs|·K float64s out of V; the full `V @ theta` costs C·K flops with no
# copy and runs in one BLAS call. Break-even is therefore around |obs| ≈ C/2 —
# below it the gather wins (plan §1: closure-mask mode touches "tens" of cells per
# doc out of C≈3,300, i.e. density ~0.01), above it the contiguous matvec wins.
_DENSE_MASK_FRACTION = 0.5


# --------------------------------------------------------------------------- #
# Pure numpy partition kernels (unit-tested; no SparkSession).                 #
# --------------------------------------------------------------------------- #
def _moments_kernel(rows, C, K):
    """Accumulate the per-node MASKED first/second θ moments + label counts.

    `rows` is an iterable of `(theta (K,), label (C,), mask (C,))` float arrays —
    one document. Node c accumulates document d's θ only when `mask[c]` is truthy,
    i.e. over exactly the rows the per-node LR is fit on. That masking is the whole
    point: the readout's sklearn oracle (`_lr_proba_per_label_masked`) standardizes
    "on that node's observed train rows (leak-free)", so a shared corpus-wide
    mean/var would silently change the fitted objective. Plan §1 buys this with ONE
    extra pass and (C,K)×2 driver memory rather than C separate scans.

    Returns `(sum_theta (C,K), sum_theta_sq (C,K), n_obs (C,), n_pos (C,))` — raw
    sufficient statistics, so partitions combine by plain addition and the mean/var
    reduction happens once on the driver (`masked_moments`).
    """
    C, K = int(C), int(K)
    sum_theta = np.zeros((C, K), dtype=np.float64)
    sum_theta_sq = np.zeros((C, K), dtype=np.float64)
    n_obs = np.zeros(C, dtype=np.float64)
    n_pos = np.zeros(C, dtype=np.float64)
    for theta, label, mask in rows:
        obs = np.flatnonzero(mask)
        if obs.size == 0:
            continue
        theta_sq = theta * theta
        if obs.size == C:
            # Dense mask: broadcast-add over the whole (C,K) block, no gather.
            sum_theta += theta
            sum_theta_sq += theta_sq
            n_obs += 1.0
            n_pos += label
        else:
            # `obs` is strictly increasing (flatnonzero), so the fancy-index `+=`
            # has no repeated targets and the un-buffered semantics are correct.
            sum_theta[obs] += theta
            sum_theta_sq[obs] += theta_sq
            n_obs[obs] += 1.0
            n_pos[obs] += label[obs]
    return sum_theta, sum_theta_sq, n_obs, n_pos


def _row_scores(theta, obs, V, b_raw, C):
    """Raw-space scores `z = V[c]·θ + b_raw[c]` for one document's observed nodes.

    Split out because it is the single hot line of both `_stats_kernel` and
    `_score_cells_kernel`, and because the sparse-vs-dense choice (see
    `_DENSE_MASK_FRACTION`) is the difference between the plan's cheap `closure`
    mask mode ("each doc touches only its observed cells (tens) → cheap") and its
    heavy `full` mode ("a (C,K)@(K,) matvec per doc"). Returns a length-`|obs|`
    array aligned with `obs`.
    """
    if obs.size >= _DENSE_MASK_FRACTION * C:
        return (V @ theta)[obs] + b_raw[obs]
    return V[obs] @ theta + b_raw[obs]


def _stats_kernel(rows, V, b_raw, C, K):
    """Per-node DATA-TERM stats of the batched multi-head logistic, in RAW θ space.

    `rows` as in `_moments_kernel`; `V (C,K)`, `b_raw (C,)` are the raw-θ scoring
    parameters (Package A's `fold_standardization` output). For every OBSERVED cell
    (d, c) — `mask[c]` truthy — with `z = V[c]·θ_d + b_raw[c]` and `p = σ(z)`:

        loss[c]  += log(1 + e^z) − y·z          (= −log p if y=1, −log(1−p) if y=0)
        g_raw[c] += (p − y)·θ_d
        s[c]     += (p − y)

    `loss` is SUMMED (not averaged) over the node's observed cells because the
    sklearn oracle this must replicate uses `C=1` scaling: summed log-loss + ½‖w‖²
    with an unpenalized intercept (plan §1, "Formulation must replicate the sklearn
    oracle"). Averaging here would silently rescale the L2 the solver adds.

    The loss is written as `logaddexp(0, z) − y·z` rather than `−y·log p −
    (1−y)·log(1−p)`: the latter evaluates `log(0)` as soon as σ saturates (which
    rare nodes do immediately — their observed set is nearly all-negative), while
    `logaddexp` is exact in both tails. `p` itself is only ever used inside `p − y`,
    where saturation is harmless.

    `s` is the intercept gradient and `g_raw` the raw-θ coefficient gradient; the
    caller folds them to standardized space via Package A's
    `standardized_grad_from_raw` (which needs exactly this `(g_raw, s)` pair,
    because d z/d W_std = (θ − μ)/σ = θ/σ − μ/σ separates into the two sums).

    Returns `(loss (C,), g_raw (C,K), s (C,))`.
    """
    C, K = int(C), int(K)
    V = np.asarray(V, dtype=np.float64)
    b_raw = np.asarray(b_raw, dtype=np.float64)
    loss = np.zeros(C, dtype=np.float64)
    g_raw = np.zeros((C, K), dtype=np.float64)
    s = np.zeros(C, dtype=np.float64)
    for theta, label, mask in rows:
        obs = np.flatnonzero(mask)
        if obs.size == 0:
            continue
        z = np.clip(_row_scores(theta, obs, V, b_raw, C), -_SCORE_CLIP, _SCORE_CLIP)
        y = label[obs]
        p = 1.0 / (1.0 + np.exp(-z))
        r = p - y
        loss[obs] += np.logaddexp(0.0, z) - y * z
        s[obs] += r
        if obs.size == C:
            g_raw += r[:, None] * theta
        else:
            g_raw[obs] += r[:, None] * theta
    return loss, g_raw, s


def _score_cells_kernel(rows, V, b_raw, C):
    """Yield `(node, y, p)` for every OBSERVED cell of every row — the eval explode.

    Plan §3: per-node metrics need only that node's `(y, p)` pairs (16 bytes/cell),
    never the K-wide feature row, so this is the ONLY thing that has to leave the
    executors for eval. Same sparse-vs-dense score reuse as `_stats_kernel`.
    Generator (not a list) so a partition's cells stream into the shuffle writer
    instead of materializing per-partition.
    """
    C = int(C)
    V = np.asarray(V, dtype=np.float64)
    b_raw = np.asarray(b_raw, dtype=np.float64)
    for theta, label, mask in rows:
        obs = np.flatnonzero(mask)
        if obs.size == 0:
            continue
        z = np.clip(_row_scores(theta, obs, V, b_raw, C), -_SCORE_CLIP, _SCORE_CLIP)
        p = 1.0 / (1.0 + np.exp(-z))
        y = label[obs]
        for j in range(obs.size):
            yield int(obs[j]), float(y[j]), float(p[j])


def _lean_eval_kernel(rows, C, V=None, b_raw=None):
    """Pack one partition of scored docs into the LEAN eval block (plan §3, v2.1).

    `rows` is an iterable of `(doc_id, score, label (C,), mask (C,))`, where `score`
    is the raw θ (K,) when `(V, b_raw)` are given — then `p = σ(clip(V·θ + b_raw))`,
    the same arithmetic as `_score_cells_kernel` — or an ALREADY-computed per-doc
    (C,) probability when `V is None` (the co-fit head's `probability` column, which
    needs the same lean treatment but no fit).

    Returns ONE block per partition, not one record per row:

        `(ids (n,) int64, P (n,C) float32, y_idx int32, y_ptr (n+1,) int64,
          m_idx int32 | None, m_ptr (n+1,) int64 | None)`

    Three things about that shape are load-bearing for the plan's driver budget:

      * **float32 p, index-list y/mask.** The driver eval needs the FULL (D_te,C)
        probability matrix (`detection_readout` takes a per-doc max over all nodes,
        `conditional_readout` scores against an all-ones mask), so p cannot be
        sparsified — but float32 halves it, and its ~1e-7 resolution is orders below
        the ~5e-4 solver-vs-sklearn disagreement the A/B gate measures. `label`/
        `labelMask` ARE sparse (closure membership), so they travel as indices and
        densify into uint8 on the driver: 6 bytes/cell all in, vs the 24 the driver
        path pays for float64 label+mask alone.
      * **One block per partition, not one record per row.** A per-row record would
        add ~100 bytes of Python object overhead per doc and pickle C floats
        individually; a stacked numpy block pickles as one buffer and lets the driver
        `del` each partition's memory as it densifies it.
      * **`m_idx is None` = "every cell observed".** `--label-mask-mode full` (the
        default) makes the mask all-ones for every doc, where an index list would be
        C int32s per row — larger than the float32 probabilities it accompanies.

    Rows are emitted in partition order and carry `doc_id`, so the driver can align
    two independently collected paths (the A/B equality gate) without a shuffle.
    """
    C = int(C)
    if V is not None:
        V = np.asarray(V, dtype=np.float64)
        b_raw = np.asarray(b_raw, dtype=np.float64)
    ids, P = [], []
    y_parts, y_ptr = [], [0]
    m_parts, m_ptr = [], [0]
    for doc_id, score, label, mask in rows:
        if V is None:
            p = np.asarray(score, dtype=np.float32)
        else:
            z = np.clip(V @ score + b_raw, -_SCORE_CLIP, _SCORE_CLIP)
            p = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        ids.append(doc_id)
        P.append(p)
        yi = np.flatnonzero(label).astype(np.int32)
        y_parts.append(yi)
        y_ptr.append(y_ptr[-1] + int(yi.size))
        # Dense rows keep a `None` placeholder rather than an arange, so the
        # all-ones-mask case never materializes n*C indices in executor memory.
        nnz = int(np.count_nonzero(mask))
        mi = None if nnz == C else np.flatnonzero(mask).astype(np.int32)
        m_parts.append(mi)
        m_ptr.append(m_ptr[-1] + nnz)
    n = len(ids)
    P = np.stack(P) if P else np.zeros((0, C), dtype=np.float32)
    y_idx = (np.concatenate(y_parts) if y_parts
             else np.zeros(0, dtype=np.int32)).astype(np.int32, copy=False)
    if all(mi is None for mi in m_parts):
        m_idx, m_ptr_arr = None, None           # every doc observes every node
    else:
        dense = np.arange(C, dtype=np.int32)
        m_idx = np.concatenate([dense if mi is None else mi for mi in m_parts])
        m_ptr_arr = np.asarray(m_ptr, dtype=np.int64)
    return (np.asarray(ids, dtype=np.int64).reshape(n),
            P, y_idx, np.asarray(y_ptr, dtype=np.int64), m_idx, m_ptr_arr)


def moments_to_mu_sd(sum_theta, sum_theta_sq, n_obs, *, eps=1e-12):
    """Reduce accumulated moments to `(mu (C,K), sd (C,K))`, population (ddof=0).

    ddof=0 because the oracle is `sklearn.preprocessing.StandardScaler`, which
    divides by n — using the sample std would put the distributed fit on a slightly
    different objective than the driver readout it must equal at the plan's
    cardiovascular A/B gate.

    This is the driver-side reduction of `_moments_kernel`'s sums, and it is the
    distributed twin of Package A's `analysis.pc.batched_lr.standardization_moments`
    — it must agree with it cell-for-cell, so its zero-variance semantics are copied
    deliberately (not imported: the Spark layer stays solver-free):

      * `eps` is a DETECTION threshold, not a floor. A feature constant on node c's
        observed rows is flagged by the RELATIVE test `sd <= eps*max(1, |mu|)` (so
        it survives features whose scale is far from 1) and its `sd` is replaced by
        **1.0**, as `sklearn`'s `_handle_zeros_in_scale` does. Substituting the
        literal `eps` instead would be equivalent in exact arithmetic but ruinous in
        floating point: the standardized gradient reaches the solver as
        `(g_raw − s·mu)/sd`, a cancellation whose rounding residual is ~1e-16·n·|mu|,
        and dividing that by 1e-12 manufactures a ~1e-4·n gradient the node can
        never converge below.
      * A node with NO observed rows gets `mu = 0`, `sd = 1` — identity moments, and
        its stats are all-zero anyway, so the solver no-ops on it.
    """
    n_obs = np.asarray(n_obs, dtype=np.float64)
    n = np.maximum(n_obs, 1.0)[:, None]
    mu = np.asarray(sum_theta, dtype=np.float64) / n
    var = np.asarray(sum_theta_sq, dtype=np.float64) / n - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))          # clamp fp cancellation, not signal
    sd = np.where(sd <= float(eps) * np.maximum(1.0, np.abs(mu)), 1.0, sd)
    empty = n_obs <= 0
    if empty.any():
        mu = mu.copy()
        mu[empty] = 0.0
        sd[empty] = 1.0
    return mu, sd


# --------------------------------------------------------------------------- #
# Row adapters: Spark Row -> the numpy triples the kernels consume.            #
# --------------------------------------------------------------------------- #
def _to_array(v):
    """Spark ML Vector / array<double> column value -> float64 ndarray."""
    to_array = getattr(v, "toArray", None)
    if to_array is not None:                     # DenseVector / SparseVector
        return np.asarray(to_array(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _row_triples(rows, topic_col, label_col, mask_col):
    """Stream `(theta, label, mask)` float64 triples straight off Spark Rows.

    Used by the SINGLE-pass aggregations (`masked_moments`, `score_cells_df`),
    where paying the dense (C,) conversion once per row is cheaper than building a
    compact form to throw away.
    """
    for row in rows:
        yield (_to_array(row[topic_col]),
               _to_array(row[label_col]),
               _to_array(row[mask_col]))


def _row_quads(rows, id_col, score_col, label_col, mask_col):
    """Stream `(doc_id, score, label, mask)` off Spark Rows for `_lean_eval_kernel`.

    `score_col` is θ or an already-computed probability vector depending on the
    caller; both arrive as Spark ML Vectors, so the same `_to_array` serves.
    """
    for row in rows:
        yield (row[id_col], _to_array(row[score_col]),
               _to_array(row[label_col]), _to_array(row[mask_col]))


def _pack_partition(rows, topic_col, label_col, mask_col):
    """Spark Rows -> compact `(theta (K,), obs_idx (n,) int32, y_obs (n,))` tuples.

    The form the REPEATED-pass path caches. The plan's fit-side wall is that dense
    per-doc `label`/`labelMask` are ~53 KB/row at C≈3,300 while the closure mask
    observes only tens of cells; caching the dense C-vectors would re-import that
    wall into the readout's own persisted RDD. Storing `(obs_idx, y_obs)` makes the
    cached row O(nnz) instead of O(C), which is the same economy plan §5 wants for
    the corpus itself.
    """
    for row in rows:
        mask = _to_array(row[mask_col])
        obs = np.flatnonzero(mask).astype(np.int32)
        label = _to_array(row[label_col])
        yield (_to_array(row[topic_col]), obs, label[obs])


def _dense_triples(packed, C):
    """Rehydrate `(theta, label, mask)` dense triples from `_pack_partition` output.

    Two scratch (C,) buffers are RECYCLED across rows and cleared only at the
    touched indices, so per-row cost is O(nnz) rather than O(C) — at C≈3,300 and
    tens of observed cells that is a ~100x difference in the per-pass overhead, and
    L-BFGS pays it on every pass. The buffers are yielded by reference: the kernels
    in this module consume each triple fully before advancing the iterator (they
    only read `label`/`mask` into per-row temporaries), which is what makes this
    safe. Any new consumer that RETAINS the yielded arrays must copy them.
    """
    C = int(C)
    label = np.zeros(C, dtype=np.float64)
    mask = np.zeros(C, dtype=np.float64)
    for theta, obs, y_obs in packed:
        mask[obs] = 1.0
        label[obs] = y_obs
        yield theta, label, mask
        mask[obs] = 0.0
        label[obs] = 0.0


# --------------------------------------------------------------------------- #
# Spark wiring (cluster-covered; thin shells over the kernels above).         #
# --------------------------------------------------------------------------- #
def masked_moments(scored_df, C, K, *, topic_col="topicDistribution",
                   label_col="label", mask_col="labelMask", eps=1e-12, depth=2):
    """One distributed pass for the per-node masked standardization moments.

    Plan §1: the sklearn oracle standardizes each node's features on THAT node's
    observed train rows, so the reparameterization needs (C,K) means and sds. This
    is the one-time aggregate that produces them; the result (~174 MB at C=K=3,300)
    is driver-side numpy and is then held fixed for the whole fit.

    Returns `(mu (C,K), sd (C,K), n_obs (C,), n_pos (C,))`. `n_obs`/`n_pos` come
    free from the same pass and are what the caller needs to identify the degenerate
    (single-class observed set) nodes the plan's open question flags — they must get
    the same constant-prediction fallback as `_lr_proba_per_label_masked`.

    `treeAggregate` (rather than `treeReduce`) so an empty corpus returns correctly
    shaped zeros instead of raising, mirroring `_collect_theta_labels`'s empty-df
    contract.
    """
    C, K = int(C), int(K)
    rdd = scored_df.select(topic_col, label_col, mask_col).rdd

    def _local(rows, _cols=(topic_col, label_col, mask_col), _C=C, _K=K):
        return [_moments_kernel(_row_triples(rows, *_cols), _C, _K)]

    def _combine(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])

    zero = (np.zeros((C, K)), np.zeros((C, K)), np.zeros(C), np.zeros(C))
    sum_theta, sum_theta_sq, n_obs, n_pos = rdd.mapPartitions(_local).treeAggregate(
        zero, _combine, _combine, depth=int(depth))
    mu, sd = moments_to_mu_sd(sum_theta, sum_theta_sq, n_obs, eps=eps)
    return mu, sd, n_obs, n_pos


class SparkStatsFn:
    """Callable `stats_fn(W_std, b_std) -> (loss_data (C,), gW_std (C,K), gb (C,))`.

    The plan's per-pass seam: ONE `treeAggregate` returns all C heads' gradients, so
    a batched L-BFGS iteration costs one data scan regardless of C — versus Spark
    ML's C sequential jobs, "the wrong granularity" (plan §Design).

    Lifecycle, deliberately explicit:

      * The projected `(theta, obs_idx, y_obs)` RDD is built and **persisted once**
        at construction (`MEMORY_AND_DISK`: L-BFGS re-reads it every pass, and
        re-deriving it means re-scanning the scored DataFrame — which on the cluster
        is a Parquet/BigQuery read, not a cheap in-memory projection). A `count()`
        materializes it up front so the first pass measures the fit, not the read.
      * A **fresh broadcast per call**, unpersisted (blocking) before returning. The
        parameters change every L-BFGS pass, so a long-lived broadcast would only
        pile up ~87 MB (C=K=3,300) of stale blocks per executor; the plan sizes this
        broadcast as "same order as the existing λ broadcast" precisely once per pass.
      * `close()` (or `with make_spark_stats_fn(...) as stats_fn:`) unpersists the
        cached RDD. The caller owns that — an un-closed instance leaks executor
        storage for the life of the SparkContext.

    The standardized↔raw fold is INJECTED (`fold_standardization`,
    `standardized_grad_from_raw` from `analysis/pc/batched_lr.py`) so this module
    never imports the solver; executors only ever see the folded raw-space (V, b_raw).
    """

    def __init__(self, scored_df, C, K, mu, sd, *, fold_standardization,
                 standardized_grad_from_raw, topic_col="topicDistribution",
                 label_col="label", mask_col="labelMask", depth=2,
                 storage_level=None):
        from pyspark import StorageLevel

        self.C, self.K = int(C), int(K)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self._fold = fold_standardization
        self._fold_grad = standardized_grad_from_raw
        self._depth = int(depth)
        cols = (topic_col, label_col, mask_col)

        def _pack(rows, _cols=cols):
            return _pack_partition(rows, *_cols)

        self._rdd = scored_df.select(*cols).rdd.mapPartitions(_pack)
        self._rdd = self._rdd.persist(
            StorageLevel.MEMORY_AND_DISK if storage_level is None else storage_level)
        self._rdd.count()                       # materialize before the first pass
        self._closed = False

    def __call__(self, W_std, b_std):
        V, b_raw = self._fold(np.asarray(W_std, dtype=np.float64),
                              np.asarray(b_std, dtype=np.float64), self.mu, self.sd)
        V = np.ascontiguousarray(V, dtype=np.float64)
        b_raw = np.ascontiguousarray(b_raw, dtype=np.float64)
        sc = self._rdd.context
        bcast = sc.broadcast((V, b_raw))
        try:
            C, K = self.C, self.K

            def _local(packed, _b=bcast, _C=C, _K=K):
                V_, b_ = _b.value
                return [_stats_kernel(_dense_triples(packed, _C), V_, b_, _C, _K)]

            def _combine(a, b):
                return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

            zero = (np.zeros(C), np.zeros((C, K)), np.zeros(C))
            loss, g_raw, s = self._rdd.mapPartitions(_local).treeAggregate(
                zero, _combine, _combine, depth=self._depth)
        finally:
            bcast.unpersist(blocking=True)
        gW_std = self._fold_grad(g_raw, s, self.mu, self.sd)
        return loss, gW_std, s

    def close(self):
        """Unpersist the cached projection. Idempotent."""
        if not self._closed:
            self._rdd.unpersist()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_spark_stats_fn(scored_df, C, K, mu, sd, *, fold_standardization,
                        standardized_grad_from_raw, topic_col="topicDistribution",
                        label_col="label", mask_col="labelMask", depth=2,
                        storage_level=None):
    """Build the batched-L-BFGS stats oracle over `scored_df`. See `SparkStatsFn`.

    Returned object is callable as `stats_fn(W_std, b_std)` and MUST be `close()`d
    (or used as a context manager) to release the persisted projection.
    """
    return SparkStatsFn(
        scored_df, C, K, mu, sd,
        fold_standardization=fold_standardization,
        standardized_grad_from_raw=standardized_grad_from_raw,
        topic_col=topic_col, label_col=label_col, mask_col=mask_col,
        depth=depth, storage_level=storage_level)


def score_cells_df(scored_df, V, b_raw, C, *, topic_col="topicDistribution",
                   label_col="label", mask_col="labelMask"):
    """Explode the fitted model's OBSERVED test cells to `[node, y, p]`.

    Plan §2–3: broadcast the fitted (raw-space) parameters, emit `P(node c)` only
    for the cells eval needs, no collect. The output is 16 bytes of payload per
    observed cell — even the root node (positive for every doc) is an O(D)-row
    group of a few MB, which is why v1's per-node skew guard is moot here.

    Returns a DataFrame `node int, y double, p double`, one row per observed
    (doc, node) cell.
    """
    from pyspark.sql.types import (DoubleType, IntegerType, StructField,
                                   StructType)

    C = int(C)
    sc = scored_df.sparkSession.sparkContext
    bcast = sc.broadcast((np.ascontiguousarray(V, dtype=np.float64),
                          np.ascontiguousarray(b_raw, dtype=np.float64)))
    cols = (topic_col, label_col, mask_col)

    def _local(rows, _b=bcast, _C=C, _cols=cols):
        V_, b_ = _b.value
        return _score_cells_kernel(_row_triples(rows, *_cols), V_, b_, _C)

    schema = StructType([StructField("node", IntegerType(), False),
                         StructField("y", DoubleType(), False),
                         StructField("p", DoubleType(), False)])
    cells = scored_df.select(*cols).rdd.mapPartitions(_local)
    # The broadcast is kept alive by the returned DataFrame's lineage (it is lazy),
    # so it is NOT unpersisted here — it dies with the SparkContext or when the
    # caller drops the frame. It is (C,K)+(C,) floats, the same ~87 MB the fit
    # already broadcasts once per pass.
    return scored_df.sparkSession.createDataFrame(cells, schema)


def _score_group(node, y, p, min_count):
    """One node's metric record, delegating verbatim to the driver readout's scorer.

    Shipped to the executors by BOTH grouping engines below. The scorer is
    `analysis.pc.evaluate._score_label` ITSELF, not a re-implementation, because the
    plan's "What must NOT change" clause makes metric equality with the driver
    readout the correctness gate; a second copy of the degenerate/`min_count` skip
    rules is exactly how that gate would rot. It is a pure function of
    `(y_true, proba, min_count)`, so it is safe to ship.
    """
    from analysis.pc.evaluate import _score_label

    rec = _score_label(y, p, min_count=int(min_count))
    return (int(node),
            None if rec["auc"] is None else float(rec["auc"]),
            None if rec["ap"] is None else float(rec["ap"]),
            int(rec["n_pos"]), int(rec["n_neg"]), rec["skipped"])


def per_node_metric_rows(cells_df, C, *, min_count=0, engine="rdd"):
    """Per-node AUC/AP over the exploded cells — `_bundle_masked` semantics, distributed.

    Plan §3 ("Eval — exact, distributed, no subsampling"): group the cells by node
    and score each node's `(y, p)` pairs with the driver readout's own
    `_score_label` (see `_score_group`). Nothing D-sized reaches the driver — the
    result is a (C,)-sized metric table.

    `engine`:
      * `"rdd"` (default) — `groupByKey` on the cell RDD. Chosen as the default
        because it needs neither Arrow nor pandas on the executors: the repo's own
        local-Spark fixtures run with `spark.sql.execution.arrow.pyspark.enabled`
        off, and Spark 3.5's bundled Arrow cannot allocate direct buffers under a
        JDK 21 driver at all. Grouping is safe here for the reason plan §3 gives —
        a cell is 16 bytes of payload, so even the root node (positive for every
        doc) is an O(D)-row, few-MB group; v1's skew guard existed to bound
        (rows × K) design matrices, which this path never materializes.
      * `"pandas"` — `groupBy(node).applyInPandas(...)`, the Arrow-batched
        equivalent. Same numbers; prefer it on a cluster where Arrow works, since it
        streams record batches instead of building a Python list per group.

    Nodes with NO observed test cell never appear in `cells_df` (nothing to group),
    but `_bundle_masked` still emits a record for every `c in range(C)`; those are
    filled in driver-side by calling the same `_score_label` on empty arrays, which
    reproduces its all-positive/degenerate record exactly.

    Returns `{c: {"auc", "ap", "n_pos", "n_neg", "skipped"}}` — the mapping
    `_bundle_masked` puts under `"per_label"`, so `analysis.pc.evaluate._macro` can
    be applied to it unchanged.

    Requires `analysis.pc.evaluate` to be importable ON THE EXECUTORS (see module
    docstring).
    """
    C = int(C)
    min_count = int(min_count)
    if engine == "rdd":
        def _pairs(row):
            return int(row["node"]), (float(row["y"]), float(row["p"]))

        def _score(kv, _mc=min_count):
            node, pairs = kv
            arr = np.asarray(list(pairs), dtype=np.float64).reshape(-1, 2)
            return _score_group(node, arr[:, 0], arr[:, 1], _mc)

        rows = cells_df.rdd.map(_pairs).groupByKey().map(_score).collect()
    elif engine == "pandas":
        from pyspark.sql.types import (DoubleType, IntegerType, LongType,
                                       StringType, StructField, StructType)
        schema = StructType([StructField("node", IntegerType(), False),
                             StructField("auc", DoubleType(), True),
                             StructField("ap", DoubleType(), True),
                             StructField("n_pos", LongType(), False),
                             StructField("n_neg", LongType(), False),
                             StructField("skipped", StringType(), True)])

        def _metrics(pdf, _mc=min_count):
            import pandas as pd
            rec = _score_group(pdf["node"].iloc[0], pdf["y"].to_numpy(),
                               pdf["p"].to_numpy(), _mc)
            return pd.DataFrame([dict(zip(
                ("node", "auc", "ap", "n_pos", "n_neg", "skipped"), rec))])

        rows = [tuple(r) for r in
                cells_df.groupBy("node").applyInPandas(_metrics, schema).collect()]
    else:
        raise ValueError(f"per_node_metric_rows: unknown engine {engine!r}")

    from analysis.pc.evaluate import _score_label

    empty = np.zeros(0, dtype=np.float64)
    per_node = {c: _score_label(empty, empty, min_count=min_count)
                for c in range(C)}
    for node, auc, ap, n_pos, n_neg, skipped in rows:
        per_node[int(node)] = {"auc": auc, "ap": ap, "n_pos": int(n_pos),
                               "n_neg": int(n_neg), "skipped": skipped}
    return per_node
