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
    data pass, so one pass replaces C separate Spark ML jobs. The same
    independence lets a pass be RESTRICTED (`node_mask=`) to the heads the solver
    still needs, which is what keeps a deep line search from re-reading all 56M
    cells for the handful of nodes still backtracking.
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

**Top-m sparse θ (the θ-WIDTH lever).** At whole-Mondo scale the cost of a data
pass is not flops but memory traffic: the exp 0104 smoke (C=3,820, K=3,827) moves
56.2M observed cells × a K-wide dense θ dot product ≈ 1.7 TB per pass, ~65s, and a
60-iteration solve is hours. But per-doc θ is a Dirichlet(α=0.5) posterior mean over
3,827 topics and is highly concentrated, so keeping only each doc's largest `m`
entries cuts that traffic by K/m. `theta_topm_coverage` MEASURES the concentration
(it is never assumed — the mass a truncation drops is reported before any fit uses
it), and the `topm=` argument on the ingest adapters applies it.

Two properties make this a well-defined estimator rather than an approximation:

  * **Truncated, NOT renormalized.** `x_trunc` equals `x` elementwise on the kept
    entries and 0 elsewhere; the kept mass is left at whatever it is. Renormalizing
    (dividing by the kept mass) would apply a DIFFERENT per-doc scale factor to each
    row, which is not an affine reparameterization of the feature space — the
    standardization fold (`mu`, `sd` per node, folded into raw-θ coordinates) would
    still be algebraically valid but would describe a design matrix whose rows were
    each rescaled by a doc-specific constant, silently changing the fitted model and
    the meaning of `V`. Plain truncation keeps the feature map a fixed coordinate
    projection, which the fold handles exactly.
  * **Consistency: truncation happens ONCE, at pack/ingest time.** Moments, stats,
    scoring and the lean eval must all see the SAME truncated features, or the fit is
    solving one problem and being scored on another. That is why `topm` lives on
    `_pack_partition` / `_row_sparse` / `_row_quads` — the row adapters — and not
    inside the kernels: every kernel downstream of an adapter sees the truncated
    design matrix as its data, and the estimator is exactly "the readout on top-m θ".
    A coordinate kept for some of node c's docs and not others is fine; that is what
    the truncated design matrix IS, and the moments computed on it agree with it.

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

# Documents per vectorization chunk on the sparse-θ path. A chunk holds (n, m)
# int32 indices + (n, m) float64 values = 12 bytes per (doc, kept topic): 12.6 MB
# at n=4096, m=256, which is executor-safe next to the partition's own rows. The
# chunk exists so the kernel can group a partition's cells BY NODE (see
# `_use_by_node`) — grouping needs the rows materialized, streaming does not.
_TOPM_CHUNK_ROWS = 4096
# Histogram resolution for the θ mass-coverage p10. 1000 bins over [0,1] is a
# 1e-3 quantile resolution for a diagnostic whose decision threshold ("is 0.95 of
# the mass in the top 256?") lives at the second decimal — and it makes the
# aggregate a fixed (len(ms), 1000) int64 accumulator instead of a collect.
_COVERAGE_BINS = 1000


# --------------------------------------------------------------------------- #
# Pure numpy partition kernels (unit-tested; no SparkSession).                 #
# --------------------------------------------------------------------------- #
def _topm_coverage_kernel(thetas, ms, n_bins=_COVERAGE_BINS):
    """Accumulate per-doc top-m θ MASS COVERAGE into (sums, histogram, n).

    `thetas` is an iterable of (K,) arrays. For each doc and each `m`, coverage is
    `sum(top-m entries) / sum(all entries)` — the fraction of that doc's θ mass a
    top-m truncation KEEPS. It is the one number that decides whether the sparse-θ
    path is a cheap reparameterization or a lobotomy, so it is measured, never
    assumed.

    Returns raw accumulators — `(sums (M,), hist (M, n_bins) int64, n int)` — so
    partitions combine by plain addition and the driver reduces once
    (`coverage_from_accum`). The histogram is how the 10th percentile comes back
    from a distributed pass without collecting a per-doc value: a p10 needs an order
    statistic, and a fixed-width histogram gives it to 1/n_bins in ONE pass and
    O(M*n_bins) memory instead of O(D).

    Cost is one O(K) `partition` at the largest m plus an O(m log m) sort of the
    survivors — deliberately not a full O(K log K) sort per doc, because this runs
    on every doc of the train split and is supposed to be noise next to the fit.

    A doc with no mass at all (sum <= 0; θ is a posterior mean on the simplex, so
    this does not occur in practice) counts as coverage 1.0 — vacuously all of
    nothing is kept.
    """
    ms = [int(m) for m in ms]
    M = len(ms)
    sums = np.zeros(M, dtype=np.float64)
    hist = np.zeros((M, int(n_bins)), dtype=np.int64)
    rows = np.arange(M)
    n = 0
    for theta in thetas:
        t = np.asarray(theta, dtype=np.float64)
        K = int(t.shape[0])
        n += 1
        total = float(t.sum())
        if total <= 0.0:
            sums += 1.0
            hist[rows, int(n_bins) - 1] += 1
            continue
        mmax = min(max(ms), K)
        top = np.partition(t, K - mmax)[K - mmax:] if mmax < K else t.copy()
        top.sort()                              # ascending, mmax entries
        csum = np.cumsum(top[::-1])             # csum[j] = mass of the top j+1
        cov = np.array([csum[min(m, K) - 1] / total for m in ms])
        sums += cov
        b = np.clip((cov * n_bins).astype(np.int64), 0, int(n_bins) - 1)
        hist[rows, b] += 1
    return sums, hist, n


def coverage_from_accum(sums, hist, n, ms, *, q=0.10):
    """Reduce `_topm_coverage_kernel` accumulators to `{m: (mean, p_q)}`.

    The quantile is NEAREST-RANK on the binned values — the smallest bin whose
    cumulative count reaches `ceil(q*n)` — and the reported value is that bin's LEFT
    edge, i.e. the estimate rounds DOWN. Rounding down is the right direction for a
    coverage floor: the number is used to answer "how much mass does the WORST decile
    of documents keep", and an optimistic p10 is the only way this diagnostic could
    mislead. Accuracy is therefore `[p_q - 1/n_bins, p_q]`.
    """
    ms = [int(m) for m in ms]
    sums = np.asarray(sums, dtype=np.float64)
    hist = np.asarray(hist, dtype=np.int64)
    n = int(n)
    n_bins = int(hist.shape[1])
    out = {}
    for j, m in enumerate(ms):
        if n <= 0:
            out[m] = (float("nan"), float("nan"))
            continue
        target = int(np.ceil(q * n))
        b = int(np.searchsorted(np.cumsum(hist[j]), max(target, 1)))
        out[m] = (float(sums[j] / n), float(min(b, n_bins - 1) / n_bins))
    return out
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


def _stats_kernel(rows, V, b_raw, C, K, node_mask=None):
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

    `node_mask` (a (C,) bool array, or `None` for "every node") is the solver's
    seam for skipping nodes whose stats it already holds — a frozen head, or one
    that already accepted its step this iteration (`batched_lr.solve_batched_lr`).
    It is applied by DROPPING those cells from each row's observed set, so the
    pass costs what the masked-in cells cost and the masked-out rows of the
    returned arrays are exactly zero (never touched), which is the contract the
    solver merges against.

    Returns `(loss (C,), g_raw (C,K), s (C,))`.
    """
    C, K = int(C), int(K)
    V = np.asarray(V, dtype=np.float64)
    b_raw = np.asarray(b_raw, dtype=np.float64)
    keep = None if node_mask is None else np.asarray(node_mask, dtype=bool)
    loss = np.zeros(C, dtype=np.float64)
    g_raw = np.zeros((C, K), dtype=np.float64)
    s = np.zeros(C, dtype=np.float64)
    for theta, label, mask in rows:
        obs = np.flatnonzero(mask)
        if keep is not None and obs.size:
            # Filter the row's observed nodes, not the outputs: dropping the cell
            # is what makes the score, the loss and the scatter-add all skip it.
            obs = obs[keep[obs]]
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


# --------------------------------------------------------------------------- #
# Sparse (top-m θ) partition kernels. Same estimator, truncated design matrix.  #
# --------------------------------------------------------------------------- #
# These are separate functions rather than branches inside the dense kernels for
# two reasons: the dense path stays literally untouched (so `topm=0` is
# byte-identical, which is what the default-off flag promises), and the sparse loop
# has a different SHAPE — it chunks a partition and regroups its cells by node,
# which has no meaning in the dense per-row form. Their contract is exactness, not
# approximation: fed the same truncated θ, each returns what its dense twin returns
# for the densified truncation, to the last ulp of the same summation order where
# the order is the same and to ~1e-12 where BLAS re-associates.
#
# **Why the accumulating kernels regroup a chunk's cells BY NODE.** The obvious
# sparse loop is per-row: for doc d, `V[np.ix_(obs, idx)] @ val`. That is a 2-D
# fancy gather striding across |obs| different 30 KB rows of V (and, for the
# gradient, a 2-D scatter back into a 117 MB array), and it re-pays that stride for
# every doc. Grouping the chunk's cells by node instead makes each trip
# `V[c][IDX[rows]]` — a 1-D gather out of ONE contiguous row that stays in L2 for
# all of node c's docs — and turns the gradient into one `bincount` per node into a
# 30 KB buffer. Same element count, and measured 2.5-9x faster across every shape
# tried (whole-Mondo C=K=3,827 at m=256: 5.9x under a full mask, 2.5-2.6x under a
# closure mask; small-C fixtures: 3.5-9x). There is no measured regime where the
# per-row form wins, so it is not kept as a branch — only `_sparse_score_cells_kernel`
# stays per-row, and for an unrelated reason (see its docstring).
def _topm_sparse(theta, topm):
    """`theta (K,)` -> `(idx (m,) int32, val (m,))`, its `m` largest entries.

    `argpartition` (O(K)) rather than a sort (O(K log K)) — this runs once per doc
    per ingest. The indices are then SORTED ascending, which costs O(m log m) and
    buys two things: gathers out of `V[c]` walk memory forward, and the packed cache
    is a deterministic function of θ (argpartition's tie order is not contractual).
    Values are the ORIGINAL θ entries — not renormalized; see the module docstring.
    """
    t = np.asarray(theta, dtype=np.float64)
    K = int(t.shape[0])
    m = int(topm)
    if m >= K:
        return np.arange(K, dtype=np.int32), t
    idx = np.argpartition(t, K - m)[K - m:]
    idx.sort()
    return idx.astype(np.int32, copy=False), t[idx]


def _sparse_chunks(packed, chunk_rows=_TOPM_CHUNK_ROWS):
    """`(idx, val, obs, y_obs)` stream -> chunked `(IDX, VAL, ptr, node, y)` blocks.

    `IDX (n,m) int32` / `VAL (n,m) float64` stack the chunk's truncated θ (every doc
    keeps exactly the same `m`, so they stack rectangularly — that is what lets the
    per-node path gather with a single fancy index). The cells are stored flat and
    CSR-style: `node`/`y` concatenated in row order, `ptr (n+1,)` delimiting each
    doc's run — so `_node_groups` can regroup them and `_sparse_score_cells_kernel`
    can slice them per doc, off the same buffers.
    """
    IDX, VAL, obs_l, y_l = [], [], [], []

    def _finish():
        n = len(IDX)
        lens = np.fromiter((o.size for o in obs_l), dtype=np.int64, count=n)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lens, out=ptr[1:])
        node = (np.concatenate(obs_l) if n else np.zeros(0, dtype=np.int32))
        y = (np.concatenate(y_l) if n else np.zeros(0, dtype=np.float64))
        return (np.stack(IDX), np.stack(VAL), ptr,
                node.astype(np.int64, copy=False),
                y.astype(np.float64, copy=False))

    for idx, val, obs, y_obs in packed:
        IDX.append(idx)
        VAL.append(val)
        obs_l.append(np.asarray(obs))
        y_l.append(np.asarray(y_obs, dtype=np.float64))
        if len(IDX) >= int(chunk_rows):
            yield _finish()
            IDX, VAL, obs_l, y_l = [], [], [], []
    if IDX:
        yield _finish()


def _node_groups(node):
    """Cells regrouped by node: yields `(c, cell_positions (n_c,) int64)`, ascending.

    `cell_positions` index into the chunk's flat cell arrays; `_cell_rows` turns them
    back into ROW ids, which is how a node's `(n_c, m)` θ block is gathered. The
    argsort is STABLE so a node's docs stay in partition order — the accumulation
    order of a node's sums is then a function of the data alone, not of numpy's
    sort tie-breaking, which is what keeps a re-run bit-reproducible.
    """
    order = np.argsort(node, kind="stable")
    ns = node[order]
    cuts = np.flatnonzero(np.concatenate(([True], ns[1:] != ns[:-1])))
    ends = np.concatenate((cuts[1:], [ns.size]))
    for st, en in zip(cuts, ends):
        yield int(ns[st]), order[st:en]


def _cell_rows(ptr, positions):
    """Flat cell positions -> the row each belongs to (inverse of `ptr`)."""
    return np.searchsorted(ptr, positions, side="right") - 1


def _sparse_moments_kernel(packed, C, K, chunk_rows=_TOPM_CHUNK_ROWS):
    """`_moments_kernel` on truncated θ: moments of the SAME features the fit sees.

    Only KEPT entries accumulate; a coordinate node c never keeps has `sum_theta = 0`
    and `sum_theta_sq = 0`, so `moments_to_mu_sd` reads it as a zero-variance column,
    hands it `sd = 1.0`, and its standardized column is identically 0 on the fitting
    rows — the coordinate is inert for that node, which is exactly the truth about
    the truncated design matrix. That is the consistency rule doing its job: the
    standardization describes the truncated features, not the original ones.

    Cells are regrouped by node for the reason given above the sparse block, and the
    scatter-adds go through `bincount` for the reason given in `_sparse_stats_kernel`.
    """
    C, K = int(C), int(K)
    sum_theta = np.zeros((C, K), dtype=np.float64)
    sum_theta_sq = np.zeros((C, K), dtype=np.float64)
    n_obs = np.zeros(C, dtype=np.float64)
    n_pos = np.zeros(C, dtype=np.float64)
    for IDX, VAL, ptr, node, y in _sparse_chunks(packed, chunk_rows):
        if node.size == 0:
            continue
        for c, pos in _node_groups(node):
            rows = _cell_rows(ptr, pos)
            flat = IDX[rows].ravel()
            v = VAL[rows]
            sum_theta[c] += np.bincount(flat, weights=v.ravel(), minlength=K)
            sum_theta_sq[c] += np.bincount(flat, weights=(v * v).ravel(),
                                           minlength=K)
            n_obs[c] += rows.size
            n_pos[c] += float(y[pos].sum())
    return sum_theta, sum_theta_sq, n_obs, n_pos


def _sparse_stats_kernel(packed, V, b_raw, C, K, chunk_rows=_TOPM_CHUNK_ROWS,
                         node_mask=None):
    """`_stats_kernel` on truncated θ — the hot kernel the whole lever exists for.

    Per observed cell the score is `z = V[c, idx]·val + b_raw[c]`: m terms instead of
    K, and (the part that actually matters at whole-Mondo) m float64s of V pulled
    through cache instead of K. Everything else — the clip, the `logaddexp` loss
    form, the SUMMED (not averaged) reduction, the `(g_raw, s)` split the fold
    consumes — is identical to the dense kernel, because it must be: this is the same
    estimator on a different design matrix, not a different estimator.

    One trip per (node, chunk): a single `(n_c, m)` gather out of the contiguous
    `V[c]` row plus one row-wise dot serves ALL of node c's docs in the chunk. See
    the note above this block for why that beats the per-doc form 2.5-9x.

    `node_mask` (see `_stats_kernel`) is nearly free here BECAUSE the cells are
    already grouped by node: a masked-out head is one skipped group, so its
    gather, dot, `logaddexp` and `bincount` — all of the per-cell work — never
    run. What survives is the chunk's own `argsort`, which is O(cells) bookkeeping
    against O(cells·m) arithmetic, so the pass cost tracks the masked-in cells.
    """
    C, K = int(C), int(K)
    V = np.asarray(V, dtype=np.float64)
    b_raw = np.asarray(b_raw, dtype=np.float64)
    keep = None if node_mask is None else np.asarray(node_mask, dtype=bool)
    loss = np.zeros(C, dtype=np.float64)
    g_raw = np.zeros((C, K), dtype=np.float64)
    s = np.zeros(C, dtype=np.float64)
    for IDX, VAL, ptr, node, y_all in _sparse_chunks(packed, chunk_rows):
        if node.size == 0:
            continue
        for c, pos in _node_groups(node):
            if keep is not None and not keep[c]:
                continue
            rows = _cell_rows(ptr, pos)
            idx, val = IDX[rows], VAL[rows]
            y = y_all[pos]
            # `einsum` (not `(A*B).sum(1)`) so the row-wise dot does not
            # materialize an (n_c, m) product before reducing it.
            z = np.clip(np.einsum("ij,ij->i", V[c][idx], val) + b_raw[c],
                        -_SCORE_CLIP, _SCORE_CLIP)
            p = 1.0 / (1.0 + np.exp(-z))
            r = p - y
            loss[c] += float(np.sum(np.logaddexp(0.0, z) - y * z))
            s[c] += float(r.sum())
            # A doc's kept indices are unique, but the SAME index recurs across the
            # docs of one node, so a buffered `+=` would drop contributions.
            # `bincount` is the vectorized unbuffered scatter (`np.add.at` is the
            # same semantics an order of magnitude slower).
            g_raw[c] += np.bincount(idx.ravel(),
                                    weights=(val * r[:, None]).ravel(),
                                    minlength=K)
    return loss, g_raw, s


def _sparse_score_cells_kernel(packed, V, b_raw, C, chunk_rows=_TOPM_CHUNK_ROWS):
    """`_score_cells_kernel` on truncated θ. Per-ROW only, deliberately.

    This is a single pass whose output is one Python tuple per observed cell — the
    tuple construction and the shuffle write dominate by an order of magnitude, so
    regrouping the arithmetic by node would buy nothing and would scramble the
    emission order the dense kernel's tests pin. Rows come out doc-major, cell-major
    within a doc, exactly as the dense twin emits them.
    """
    C = int(C)
    V = np.asarray(V, dtype=np.float64)
    b_raw = np.asarray(b_raw, dtype=np.float64)
    for IDX, VAL, ptr, node, y_all in _sparse_chunks(packed, chunk_rows):
        for d in range(int(IDX.shape[0])):
            a, b = int(ptr[d]), int(ptr[d + 1])
            if a == b:
                continue
            obs = node[a:b]
            z = np.clip(V[np.ix_(obs, IDX[d])] @ VAL[d] + b_raw[obs],
                        -_SCORE_CLIP, _SCORE_CLIP)
            p = 1.0 / (1.0 + np.exp(-z))
            y = y_all[a:b]
            for j in range(obs.size):
                yield int(obs[j]), float(y[j]), float(p[j])


def _lean_eval_kernel(rows, C, V=None, b_raw=None):
    """Pack one partition of scored docs into the LEAN eval block (plan §3, v2.1).

    `rows` is an iterable of `(doc_id, score, label (C,), mask (C,))`, where `score`
    is the raw θ (K,) when `(V, b_raw)` are given — then `p = σ(clip(V·θ + b_raw))`,
    the same arithmetic as `_score_cells_kernel` — or an ALREADY-computed per-doc
    (C,) probability when `V is None` (the co-fit head's `probability` column, which
    needs the same lean treatment but no fit).

    Under top-m truncation `score` arrives as the `(idx, val)` pair `_row_quads`
    produces, and the matvec becomes a `(C, m)` column gather out of V — the same
    K/m traffic cut the fit gets, applied to the test split. The eval must see the
    truncated features for the same reason the fit does: the model was fit on them.

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
            if isinstance(score, tuple):        # (idx, val): top-m truncated θ
                idx, val = score
                z = V[:, idx] @ val + b_raw
            else:
                z = V @ score + b_raw
            z = np.clip(z, -_SCORE_CLIP, _SCORE_CLIP)
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


def _row_sparse(rows, topic_col, label_col, mask_col, topm):
    """Spark Rows -> `(idx (m,) int32, val (m,), obs (n,) int32, y_obs (n,))` tuples.

    The truncated twin of `_row_triples` for the SINGLE-pass sparse consumers
    (`masked_moments`, `score_cells_df`). It supersedes rather than extends
    `_row_triples`: every sparse kernel wants the O(nnz) cell form (it is the shape
    `_sparse_chunks` stacks), so there is no sparse consumer left that wants dense
    (C,) label/mask vectors, and `_row_triples` stays dense-only and unchanged.

    This — with `_pack_partition(topm=)` and `_row_quads(topm=)` — is the ONLY place
    truncation happens, which is the consistency rule from the module docstring: one
    feature map at ingest, every kernel downstream sees it.
    """
    for row in rows:
        mask = _to_array(row[mask_col])
        obs = np.flatnonzero(mask).astype(np.int32)
        label = _to_array(row[label_col])
        idx, val = _topm_sparse(_to_array(row[topic_col]), topm)
        yield idx, val, obs, label[obs]


def _row_quads(rows, id_col, score_col, label_col, mask_col, topm=0):
    """Stream `(doc_id, score, label, mask)` off Spark Rows for `_lean_eval_kernel`.

    `score_col` is θ or an already-computed probability vector depending on the
    caller; both arrive as Spark ML Vectors, so the same `_to_array` serves.

    `topm > 0` truncates the score to its top-m `(idx, val)` pair — valid ONLY when
    `score_col` is θ and the kernel is scoring it with `(V, b_raw)`. An
    already-computed probability column is not a feature vector and must never be
    truncated; `_collect_lean_proba` enforces that by passing `topm` only on the
    branch that fits.
    """
    for row in rows:
        score = _to_array(row[score_col])
        if int(topm) > 0:
            score = _topm_sparse(score, topm)
        yield (row[id_col], score,
               _to_array(row[label_col]), _to_array(row[mask_col]))


def _pack_partition(rows, topic_col, label_col, mask_col, topm=0):
    """Spark Rows -> compact `(theta (K,), obs_idx (n,) int32, y_obs (n,))` tuples.

    The form the REPEATED-pass path caches. The plan's fit-side wall is that dense
    per-doc `label`/`labelMask` are ~53 KB/row at C≈3,300 while the closure mask
    observes only tens of cells; caching the dense C-vectors would re-import that
    wall into the readout's own persisted RDD. Storing `(obs_idx, y_obs)` makes the
    cached row O(nnz) instead of O(C), which is the same economy plan §5 wants for
    the corpus itself.

    `topm > 0` switches the cached row to the truncated sparse form
    `(idx (m,) int32, val (m,), obs_idx, y_obs)` — a 4-tuple, so a consumer cannot
    confuse the two shapes — and shrinks the CACHE as well as the traffic: at
    K=3,827 and m=256 the θ half of a persisted row drops from 30 KB to 3 KB, which
    is the difference between an L-BFGS whose passes re-read from disk and one that
    stays in memory. Truncation is paid ONCE here, not on every pass.
    """
    topm = int(topm)
    for row in rows:
        mask = _to_array(row[mask_col])
        obs = np.flatnonzero(mask).astype(np.int32)
        label = _to_array(row[label_col])
        theta = _to_array(row[topic_col])
        if topm > 0:
            idx, val = _topm_sparse(theta, topm)
            yield (idx, val, obs, label[obs])
        else:
            yield (theta, obs, label[obs])


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
# Driver-side retry around a Spark ACTION (spot-VM survivability).            #
# --------------------------------------------------------------------------- #
def _spark_job_failure_types():
    """The exception types a failed Spark JOB surfaces on the driver.

    Resolved lazily (and defensively) rather than imported at module scope, for
    the same reason every other pyspark import here is function-local: this file
    ships to executors as a bare top-level .py and must import on a worker that
    has no gateway. If py4j is somehow absent the tuple comes back EMPTY, which
    makes `except ():` match nothing — a missing dependency degrades to "no
    retry", never to "retry every exception, including our own bugs".

    `Py4JJavaError` is what an RDD action raises when the JVM job dies (a job
    abort, an unrecoverable stage failure, a lost driver-side call);
    `PySparkException` is the pyspark-3.x wrapper the SQL/DataFrame paths raise
    for the same underlying JVM error, and costs nothing to include.
    """
    types = []
    try:
        from py4j.protocol import Py4JJavaError
        types.append(Py4JJavaError)
    except Exception:                            # pragma: no cover - no py4j
        pass
    try:
        from pyspark.errors import PySparkException
        types.append(PySparkException)
    except Exception:                            # pragma: no cover - old pyspark
        pass
    return tuple(types)


def _error_first_line(exc):
    """First line of a Spark exception, safely.

    `str(Py4JJavaError)` round-trips to the JVM to fetch the stack trace, which
    can itself raise if the gateway is the thing that died — and this runs on the
    failure path, where a second exception would replace a recoverable job abort
    with an unrecoverable driver crash. So: try the real message, fall back to
    the py4j-side `errmsg`, then to `repr`.
    """
    try:
        text = str(exc)
    except Exception:                            # pragma: no cover - dead gateway
        try:
            text = getattr(exc, "errmsg", "") or repr(exc)
        except Exception:
            text = exc.__class__.__name__
    for line in str(text).splitlines():
        if line.strip():
            return line.strip()
    return exc.__class__.__name__


def _retry_spark_action(fn, *, attempts=4, base_sleep_s=60, label=""):
    """Run one Spark ACTION, retrying a failed job after a backoff sleep.

    **Why this is sound here, and not a general "retry everything" wrapper.**
    Every kernel in this module is a PURE FUNCTION of a persisted RDD: partitions
    lost to a preemption recompute from lineage, and the aggregate is a sum over
    partitions, so re-running the action recomputes exactly the same number. A
    retried action is therefore idempotent — not "probably fine", but identical
    by construction. (The one thing a retry must NOT reuse is a broadcast whose
    blocks may have died with the executors, which is why the call sites below
    build their broadcast INSIDE the retried closure.)

    **Why a retry actually fixes the failure it is aimed at** (exp 0104,
    2026-08-28): spot-VM preemption waves accumulate Spark `excludeOnFailure`
    state until a task retry has no schedulable executor left and Spark aborts
    the TaskSet — "task 0 (partition 0) cannot run anywhere due to node and
    executor excludeOnFailure". Two independent clocks then work in our favour
    during the sleep: YARN replaces the killed containers within a couple of
    minutes, and — decisively — the per-taskset exclusion lists are per-JOB, so a
    NEW job starts with a clean slate regardless of how poisoned the aborted
    one's were. (App-level node exclusions age out on
    `spark.excludeOnFailure.timeout`, which is why this run's doc pins it to
    10m instead of the 1h default.) The sleeps are sized for that: 60s, 120s,
    240s.

    Retries on ANY Spark job failure rather than pattern-matching the abort
    message on purpose. The phrasings that matter here (excludeOnFailure aborts,
    `FetchFailed`, `ExecutorLostFailure`, "Killed by external signal") vary by
    Spark version and by which layer noticed first, and a missed pattern costs
    hours of solve. The cost of the opposite error — retrying a deterministic
    bug — is bounded at ~7 minutes of sleeping before the same exception is
    re-raised with its traceback intact, which is a trade this path should always
    take. Non-Spark exceptions (our own `ValueError`s, `KeyboardInterrupt`)
    propagate on the first raise.

    `label` is prefixed to the log lines so a failure in a multi-hour solve says
    WHICH pass died.
    """
    import time

    attempts = max(1, int(attempts))
    retryable = _spark_job_failure_types()
    tag = f"{label}: " if label else ""
    for k in range(attempts):
        try:
            return fn()
        except retryable as exc:
            if k + 1 >= attempts:
                print(f"[driver]   {tag}spark action FAILED on attempt "
                      f"{k + 1}/{attempts}, giving up: {_error_first_line(exc)}",
                      flush=True)
                raise
            sleep_s = float(base_sleep_s) * (2 ** k)
            print(f"[driver]   {tag}spark action FAILED on attempt "
                  f"{k + 1}/{attempts}: {_error_first_line(exc)}", flush=True)
            print(f"[driver]   {tag}retrying in {sleep_s:.0f}s (a fresh job resets "
                  "per-taskset excludeOnFailure state; YARN replaces preempted "
                  "containers meanwhile)", flush=True)
            time.sleep(sleep_s)
    raise AssertionError("unreachable")          # pragma: no cover


# --------------------------------------------------------------------------- #
# Spark wiring (cluster-covered; thin shells over the kernels above).         #
# --------------------------------------------------------------------------- #
def theta_topm_coverage(scored_df, K, *, ms=(64, 128, 256, 512),
                        topic_col="topicDistribution", n_bins=_COVERAGE_BINS,
                        depth=2, q=0.10):
    """One distributed pass measuring how much θ mass a top-m truncation keeps.

    Returns `{m: (mean_coverage, p10_coverage)}`. This is the MEASUREMENT that has
    to precede any use of the sparse-θ path: the lever's entire premise is that a
    Dirichlet(α=0.5) posterior mean over thousands of topics is concentrated, and a
    premise about the data is not something a kernel can assume. The p10 is the
    number that matters — a mean of 0.98 with a p10 of 0.4 means the truncation is
    fine for the typical doc and destroys the tail, and the tail is what the rare
    nodes are fit on.

    Cheap by construction (see `_topm_coverage_kernel`): one O(K) partition per doc
    and a fixed `(len(ms), n_bins)` int64 accumulator per partition, so this costs a
    fraction of one L-BFGS pass and nothing D-sized reaches the driver.
    """
    ms = tuple(int(m) for m in ms)

    def _local(rows, _col=topic_col, _ms=ms, _nb=int(n_bins)):
        return [_topm_coverage_kernel((_to_array(r[_col]) for r in rows), _ms, _nb)]

    def _combine(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    zero = (np.zeros(len(ms)), np.zeros((len(ms), int(n_bins)), dtype=np.int64), 0)
    # `.rdd` is taken INSIDE the retried closure: on a DataFrame whose cached
    # transform has not materialized yet, the `javaToPython` conversion behind
    # `.rdd` can itself run Spark jobs — and this coverage pass is the FIRST
    # action of the whole readout, so it is exactly where that happens. Exp
    # 0104's 08-28 relaunch died right there, one line OUTSIDE the wrapper.
    sums, hist, n = _retry_spark_action(
        lambda: scored_df.select(topic_col).rdd.mapPartitions(_local).treeAggregate(
            zero, _combine, _combine, depth=int(depth)),
        label="theta top-m coverage")
    return coverage_from_accum(sums, hist, n, ms, q=q)


def masked_moments(scored_df, C, K, *, topic_col="topicDistribution",
                   label_col="label", mask_col="labelMask", eps=1e-12, depth=2,
                   topm=0):
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

    `topm > 0` computes the moments of the TRUNCATED features — mandatory whenever
    the fit runs truncated, since a standardization derived from full θ describes a
    design matrix the solver never sees (module docstring, consistency rule).
    """
    C, K = int(C), int(K)
    topm = int(topm)

    def _local(rows, _cols=(topic_col, label_col, mask_col), _C=C, _K=K, _m=topm):
        if _m > 0:
            return [_sparse_moments_kernel(_row_sparse(rows, *_cols, _m), _C, _K)]
        return [_moments_kernel(_row_triples(rows, *_cols), _C, _K)]

    def _combine(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])

    zero = (np.zeros((C, K)), np.zeros((C, K)), np.zeros(C), np.zeros(C))
    # Retried: the moments pass is a pure sum over a lineage-recomputable
    # projection, and losing it to a preemption wave costs the whole fit (it is
    # the standardization every later pass is expressed in). `.rdd` is taken
    # inside the closure — `javaToPython` can run jobs of its own on an
    # unmaterialized frame (see theta_topm_coverage).
    sum_theta, sum_theta_sq, n_obs, n_pos = _retry_spark_action(
        lambda: scored_df.select(topic_col, label_col, mask_col)
        .rdd.mapPartitions(_local).treeAggregate(
            zero, _combine, _combine, depth=int(depth)),
        label="masked moments")
    mu, sd = moments_to_mu_sd(sum_theta, sum_theta_sq, n_obs, eps=eps)
    return mu, sd, n_obs, n_pos


class SparkStatsFn:
    """Callable `stats_fn(W_std, b_std, node_mask=None)
    -> (loss_data (C,), gW_std (C,K), gb (C,))`.

    The plan's per-pass seam: ONE `treeAggregate` returns all C heads' gradients, so
    a batched L-BFGS iteration costs one data scan regardless of C — versus Spark
    ML's C sequential jobs, "the wrong granularity" (plan §Design).

    `node_mask` is the SECOND economy, and at whole-Mondo scale the bigger one. A
    batched L-BFGS iteration is one pass only when every node accepts its first
    trial step; with C≈3,800 independent Armijo tests sharing the pass, some node
    is nearly always still backtracking, and the exp 0104 smoke paid ~26 full
    passes per iteration for it. The solver therefore names the nodes whose stats
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
      * A **fresh broadcast per call**, unpersisted (blocking) before returning. The
        parameters change every L-BFGS pass, so a long-lived broadcast would only
        pile up ~87 MB (C=K=3,300) of stale blocks per executor; the plan sizes this
        broadcast as "same order as the existing λ broadcast" precisely once per pass.
        The broadcast/unpersist pair lives INSIDE the retried closure
        (`_retry_spark_action`), so a preemption-wave job abort costs one pass and a
        backoff sleep rather than the whole solve — see that helper for why the
        retry is idempotent rather than hopeful.
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
            return _pack_partition(rows, *_cols, topm=_m)

        self._rdd = scored_df.select(*cols).rdd.mapPartitions(_pack)
        self._rdd = self._rdd.persist(
            StorageLevel.MEMORY_AND_DISK if storage_level is None else storage_level)
        self._rdd.count()                       # materialize before the first pass
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

        def _combine(a, b):
            return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

        zero = (np.zeros(C), np.zeros((C, K)), np.zeros(C))

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
                        return [_sparse_stats_kernel(packed, V_, b_, _C, _K,
                                                     node_mask=keep_)]
                    return [_stats_kernel(_dense_triples(packed, _C), V_, b_, _C,
                                          _K, node_mask=keep_)]

                return self._rdd.mapPartitions(_local).treeAggregate(
                    zero, _combine, _combine, depth=self._depth)
            finally:
                # Unpersist per ATTEMPT — the stale-block accounting that the
                # fresh-per-call broadcast exists for applies just as much to an
                # attempt that died as to one that returned.
                bcast.unpersist(blocking=True)

        # THE hot seam: L-BFGS calls this once per line-search trial, so a
        # multi-hour solve is overwhelmingly likely to be inside this action when
        # a preemption wave lands — which is exactly how exp 0104 lost 9,112s of
        # solve on 2026-08-28.
        loss, g_raw, s = _retry_spark_action(
            _pass,
            label=f"stats pass ({'all' if keep is None else int(keep.sum())}/{C} "
                  "nodes)")
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


def score_cells_df(scored_df, V, b_raw, C, *, topic_col="topicDistribution",
                   label_col="label", mask_col="labelMask", topm=0):
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
    topm = int(topm)
    sc = scored_df.sparkSession.sparkContext
    bcast = sc.broadcast((np.ascontiguousarray(V, dtype=np.float64),
                          np.ascontiguousarray(b_raw, dtype=np.float64)))
    cols = (topic_col, label_col, mask_col)

    def _local(rows, _b=bcast, _C=C, _cols=cols, _m=topm):
        V_, b_ = _b.value
        if _m > 0:
            # `(V, b_raw)` were FIT on truncated θ, so the cells they score must be
            # truncated too — same feature map, fit and eval.
            return _sparse_score_cells_kernel(_row_sparse(rows, *_cols, _m),
                                              V_, b_, _C)
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
