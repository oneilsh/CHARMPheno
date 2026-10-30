"""Pure-numpy kernels of the per-node readout, shared by the Spark and local backends.

Every number the readout computes -- the per-node masked standardization moments,
the per-pass (loss, gradient) statistics of the batched multi-head logistic, the
lean (row, node) probability block -- is a pure function of numpy arrays here. The
Spark backend (`readout.spark`) wraps them in `mapPartitions` + `treeAggregate`;
the local backend (`readout.local`) calls them on rows held in memory. Same kernels,
so the two backends agree up to floating-point summation order (ADR 0051 §2).
Nothing here imports pyspark; the row adapters (`row_*`, `pack_rows`) read Spark
`Row`s by column name and work on any mapping.

The features are any frozen per-row vector (topic proportions θ, raw-code presence,
...), called θ below because that is the main use. Labels and masks are (C,) per
row: node c's head is fit on the rows whose `mask[c]` is set.

**Top-m sparse features (the width lever).** At C and K in the thousands the cost of a data
pass is memory traffic, not flops: 56.2M observed cells × a K-wide dense dot product
≈ 1.7 TB per pass (C=3,820, K=3,827). Per-row θ is a Dirichlet posterior
mean over thousands of topics and is concentrated, so keeping each row's largest `m`
entries cuts that traffic by K/m. `topm_coverage_kernel` MEASURES the concentration
(the mass a truncation drops is reported before any fit uses it), and the `topm=`
argument on the row adapters applies it. Two properties make this a well-defined
estimator rather than an approximation:

  * **Truncated, NOT renormalized.** Kept entries equal the original ones; the kept
    mass is left as it is. Renormalizing would rescale each row by its own constant,
    which is not an affine reparameterization of the feature space, so the
    standardization fold would describe a different design matrix. Plain truncation
    keeps the feature map a fixed coordinate projection, which the fold handles
    exactly.
  * **Truncation happens ONCE, at ingest.** Moments, stats, scoring and the lean
    eval must all see the SAME truncated features, or the fit solves one problem and
    is scored on another. That is why `topm` lives on the row adapters (`row_sparse`,
    `row_quads`, `pack_rows`) and not inside the kernels.
"""
from __future__ import annotations

import numpy as np

# Scores are clipped to this magnitude before the sigmoid/log-loss
# (`np.clip(z, -50, 50)`). σ(±50) is
# 1∓2e-22, so the clip is a no-op for any non-separating parameter vector; it
# exists to keep `exp` from overflowing on a run-away head (the |w_CK|=273
# blowup). NOTE the clip flattens the loss beyond ±50, so the returned gradient
# is the gradient of the UNCLIPPED loss evaluated at the clipped score — exact
# wherever it matters.
# The solver's in-memory reference (`solver.make_inmemory_stats_fn`) uses an
# unclipped branch-stable sigmoid instead: the two agree to ~1e-22 in `p` and in
# every gradient, and can disagree in the reported LOSS only for a node driven
# past |z| = 50 (where the clipped objective is the bounded log-loss). Metrics —
# the A/B equality checks — read `p`, so they are unaffected.
_SCORE_CLIP = 50.0


# Mask-density cutoff for the per-row score computation in `stats_kernel`.
# `V[obs] @ theta` costs |obs|·K flops PLUS a gather that
# copies |obs|·K float64s out of V; the full `V @ theta` costs C·K flops with no
# copy and runs in one BLAS call. Break-even is therefore around |obs| ≈ C/2 —
# below it the gather wins (closure-mask mode touches "tens" of cells per
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
def topm_coverage_kernel(thetas, ms, n_bins=_COVERAGE_BINS, excess=False):
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

    `excess=True` measures coverage of the mass ABOVE THE PRIOR FLOOR instead:
    `e_k = max(θ_k - floor, 0)` with `floor` = the doc's smallest POSITIVE θ
    entry. Why: θ_k = (α + n_k) / (Σ_allowed α + N), so every allowed topic
    carries a floor of α/(Σα + N) whether or not a single token went to it. On a
    wide gated layout (0135: α=0.5 over a 1,200-topic background, ~600 prior
    pseudo-counts against docs of tens to hundreds of tokens) that floor IS most
    of θ's mass, and raw coverage reads "diffuse" (p10 0.117 at m=256, launch 3)
    for documents whose token assignments n_k are concentrated in a handful of
    topics. The top-m SET is the same either way (θ and e rank identically); what
    truncation discards is the floor on the dropped topics, a function of the
    doc's length and allowed set only, which carries no topic information. So the
    excess is the right denominator for "how much of the doc's own signal does a
    width-m truncation keep". Gated-out topics are exactly 0 and never set the
    floor; a doc whose positive entries are all equal has zero excess and counts
    as fully covered.
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
        if excess:
            pos = t[t > 0.0]
            if pos.size:
                t = np.maximum(t - float(pos.min()), 0.0)
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
    """Reduce `topm_coverage_kernel` accumulators to `{m: (mean, p_q)}`.

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


def moments_kernel(rows, C, K):
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

    Split out because it is the single hot line of `stats_kernel`, and because the sparse-vs-dense choice (see
    `_DENSE_MASK_FRACTION`) is the difference between the cheap `closure`
    mask mode ("each doc touches only its observed cells (tens) → cheap") and its
    heavy `full` mode ("a (C,K)@(K,) matvec per doc"). Returns a length-`|obs|`
    array aligned with `obs`.
    """
    if obs.size >= _DENSE_MASK_FRACTION * C:
        return (V @ theta)[obs] + b_raw[obs]
    return V[obs] @ theta + b_raw[obs]


def stats_kernel(rows, V, b_raw, C, K, node_mask=None):
    """Per-node DATA-TERM stats of the batched multi-head logistic, in RAW θ space.

    `rows` as in `moments_kernel`; `V (C,K)`, `b_raw (C,)` are the raw-θ scoring
    parameters (`solver.fold_standardization` output). For every OBSERVED cell
    (d, c) — `mask[c]` truthy — with `z = V[c]·θ_d + b_raw[c]` and `p = σ(z)`:

        loss[c]  += log(1 + e^z) − y·z          (= −log p if y=1, −log(1−p) if y=0)
        g_raw[c] += (p − y)·θ_d
        s[c]     += (p − y)

    `loss` is SUMMED (not averaged) over the node's observed cells because the
    sklearn oracle this must replicate uses `C=1` scaling: summed log-loss + ½‖w‖²
    with an unpenalized intercept. Averaging here would silently rescale the L2 the solver adds.

    The loss is written as `logaddexp(0, z) − y·z` rather than `−y·log p −
    (1−y)·log(1−p)`: the latter evaluates `log(0)` as soon as σ saturates (which
    rare nodes do immediately — their observed set is nearly all-negative), while
    `logaddexp` is exact in both tails. `p` itself is only ever used inside `p − y`,
    where saturation is harmless.

    `s` is the intercept gradient and `g_raw` the raw-θ coefficient gradient; the
    caller folds them to standardized space via the solver's
    `standardized_grad_from_raw` (which needs exactly this `(g_raw, s)` pair,
    because d z/d W_std = (θ − μ)/σ = θ/σ − μ/σ separates into the two sums).

    `node_mask` (a (C,) bool array, or `None` for "every node") is the solver's
    seam for skipping nodes whose stats it already holds — a frozen head, or one
    that already accepted its step this iteration (`solver.solve_batched_lr`).
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
# tried (C=K=3,827 at m=256: 5.9x under a full mask, 2.5-2.6x under a
# closure mask; small-C fixtures: 3.5-9x). There is no measured regime where the
# per-row form wins, so it is not kept as a branch.
def topm_sparse(theta, topm):
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
    doc's run — so `_node_groups` can regroup them off the same buffers.
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


def sparse_moments_kernel(packed, C, K, chunk_rows=_TOPM_CHUNK_ROWS):
    """`moments_kernel` on truncated θ: moments of the SAME features the fit sees.

    Only KEPT entries accumulate; a coordinate node c never keeps has `sum_theta = 0`
    and `sum_theta_sq = 0`, so `moments_to_mu_sd` reads it as a zero-variance column,
    hands it `sd = 1.0`, and its standardized column is identically 0 on the fitting
    rows — the coordinate is inert for that node, which is exactly the truth about
    the truncated design matrix. That is the consistency rule doing its job: the
    standardization describes the truncated features, not the original ones.

    Cells are regrouped by node for the reason given above the sparse block, and the
    scatter-adds go through `bincount` for the reason given in `sparse_stats_kernel`.
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


def sparse_stats_kernel(packed, V, b_raw, C, K, chunk_rows=_TOPM_CHUNK_ROWS,
                         node_mask=None):
    """`stats_kernel` on truncated θ — the hot kernel the whole lever exists for.

    Per observed cell the score is `z = V[c, idx]·val + b_raw[c]`: m terms instead of
    K, and (the part that actually matters at large K) m float64s of V pulled
    through cache instead of K. Everything else — the clip, the `logaddexp` loss
    form, the SUMMED (not averaged) reduction, the `(g_raw, s)` split the fold
    consumes — is identical to the dense kernel, because it must be: this is the same
    estimator on a different design matrix, not a different estimator.

    One trip per (node, chunk): a single `(n_c, m)` gather out of the contiguous
    `V[c]` row plus one row-wise dot serves ALL of node c's docs in the chunk. See
    the note above this block for why that beats the per-doc form 2.5-9x.

    `node_mask` (see `stats_kernel`) is nearly free here BECAUSE the cells are
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


def lean_eval_kernel(rows, C, V=None, b_raw=None):
    """Pack one partition of scored rows into the LEAN eval block.

    `rows` is an iterable of `(row_id, score, label (C,), mask (C,))` (`row_quads`).
    `score` is the raw feature vector (K,) when `(V, b_raw)` are given -- then
    `p = σ(clip(V·x + b_raw))` -- or an ALREADY-computed per-row (C,) probability
    when `V is None`. Under top-m truncation `score` arrives as the `(idx, val)` pair
    `row_quads` produces and the matvec becomes a `(C, m)` column gather out of V:
    the eval must see the truncated features because the heads were fit on them.

    Returns ONE block per partition, not one record per row:

        `(ids (n,) int64, P (n,C) float32, y_idx int32, y_ptr (n+1,) int64,
          m_idx int32 | None, m_ptr (n+1,) int64 | None)`

    Three things about that shape keep the driver budget at ~6 bytes per cell:

      * **float32 p, index-list y/mask.** The eval needs the FULL (D_te,C)
        probability matrix (detection takes a per-row max over all nodes), so p
        cannot be sparsified -- but float32 halves it, and its ~1e-7 resolution is
        orders below any solver-vs-sklearn disagreement. `label`/`labelMask` are
        sparse (closure membership), so they travel as indices and densify into
        uint8 on the driver (`densify_lean_blocks`).
      * **One block per partition.** A per-row record adds ~100 bytes of Python
        object overhead per row; a stacked numpy block pickles as one buffer and
        lets the driver free each partition's memory as it densifies it.
      * **`m_idx is None` = "every cell observed"** (an all-ones mask), where an
        index list would be C int32s per row.

    Rows are emitted in partition order and carry their int64 id, so two
    independently collected paths can be aligned on row identity without a shuffle.
    """
    C = int(C)
    if V is not None:
        V = np.asarray(V, dtype=np.float64)
        b_raw = np.asarray(b_raw, dtype=np.float64)
    ids, P = [], []
    y_parts, y_ptr = [], [0]
    m_parts, m_ptr = [], [0]
    for row_id, score, label, mask in rows:
        if V is None:
            p = np.asarray(score, dtype=np.float32)
        else:
            if isinstance(score, tuple):        # (idx, val): top-m truncated features
                idx, val = score
                z = V[:, idx] @ val + b_raw
            else:
                z = V @ score + b_raw
            z = np.clip(z, -_SCORE_CLIP, _SCORE_CLIP)
            p = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        ids.append(row_id)
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
        m_idx, m_ptr_arr = None, None           # every row observes every node
    else:
        dense = np.arange(C, dtype=np.int32)
        m_idx = np.concatenate([dense if mi is None else mi for mi in m_parts])
        m_ptr_arr = np.asarray(m_ptr, dtype=np.int64)
    return (np.asarray(ids, dtype=np.int64).reshape(n),
            P, y_idx, np.asarray(y_ptr, dtype=np.int64), m_idx, m_ptr_arr)


def densify_lean_blocks(blocks, C):
    """Per-partition lean blocks -> `(proba f32 (D,C), y u8, mask u8, ids list)`.

    The driver-side half of `lean_eval_kernel`. The destination arrays are
    (4 + 1 + 1) = **6 bytes per (row, node) cell** (~1.6 GB at D_te=80k, C=3,300).
    Each block is dropped as soon as it is copied, so the transient overshoot is
    the collected blocks (float32 p + index lists) rather than a second full copy.
    `mask` runs may be `None`: every row observes every node.
    """
    C = int(C)
    D = sum(int(b[1].shape[0]) for b in blocks)
    proba = np.zeros((D, C), dtype=np.float32)
    y = np.zeros((D, C), dtype=np.uint8)
    mask = np.zeros((D, C), dtype=np.uint8)
    ids = np.zeros(D, dtype=np.int64)
    at = 0
    for j in range(len(blocks)):
        b_ids, P, y_idx, y_ptr, m_idx, m_ptr = blocks[j]
        n = int(P.shape[0])
        proba[at:at + n] = P
        ids[at:at + n] = b_ids
        y[np.repeat(np.arange(at, at + n), np.diff(y_ptr)), y_idx] = 1
        if m_idx is None:
            mask[at:at + n] = 1
        else:
            mask[np.repeat(np.arange(at, at + n), np.diff(m_ptr)), m_idx] = 1
        at += n
        blocks[j] = None                      # free the partition's block eagerly
    id_list = ids.tolist()
    assert_unique_ids(id_list, where="densify_lean_blocks")
    return proba, y, mask, id_list


def assert_unique_ids(ids, *, where):
    """Uniqueness tripwire on the materialized row ids, run where they first reach
    the driver as a full array (one `set` build, no extra Spark pass). A duplicate
    means two rows share an id, which would let an id-keyed alignment silently
    overwrite one with the other; better a named raise than a wrong number."""
    n = len(ids)
    distinct = len(set(int(i) for i in ids))
    if n != distinct:
        raise ValueError(
            f"row-id collision in {where}: {n} rows carry {distinct} distinct ids. "
            "The id column must be unique per row (for the clinical corpora, the "
            "synthesized doc key: check the per-person document cap).")


def moments_to_mu_sd(sum_theta, sum_theta_sq, n_obs, *, eps=1e-12):
    """Reduce accumulated moments to `(mu (C,K), sd (C,K))`, population (ddof=0).

    ddof=0 because the oracle is `sklearn.preprocessing.StandardScaler`, which
    divides by n — using the sample std would put the distributed fit on a slightly
    different objective than the driver readout it must equal at the
    cardiovascular A/B gate.

    This is the driver-side reduction of `moments_kernel`'s sums, and it is the
    twin of `solver.standardization_moments`
    — it must agree with it cell-for-cell, so its zero-variance semantics are copied
    deliberately (this module stays solver-free):

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
def to_array(v):
    """Spark ML Vector / array<double> column value -> float64 ndarray."""
    to_array = getattr(v, "toArray", None)
    if to_array is not None:                     # DenseVector / SparseVector
        return np.asarray(to_array(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def row_triples(rows, topic_col, label_col, mask_col):
    """Stream `(theta, label, mask)` float64 triples straight off Spark Rows.

    Used by the SINGLE-pass aggregations (`masked_moments`),
    where paying the dense (C,) conversion once per row is cheaper than building a
    compact form to throw away.
    """
    for row in rows:
        yield (to_array(row[topic_col]),
               to_array(row[label_col]),
               to_array(row[mask_col]))


def row_sparse(rows, topic_col, label_col, mask_col, topm):
    """Spark Rows -> `(idx (m,) int32, val (m,), obs (n,) int32, y_obs (n,))` tuples.

    The truncated twin of `row_triples` for the SINGLE-pass sparse consumers
    (`masked_moments`). It supersedes rather than extends
    `row_triples`: every sparse kernel wants the O(nnz) cell form (it is the shape
    `_sparse_chunks` stacks), so there is no sparse consumer left that wants dense
    (C,) label/mask vectors, and `row_triples` stays dense-only and unchanged.

    This — with `pack_rows(topm=)` and `row_quads(topm=)` — is the ONLY place
    truncation happens, which is the consistency rule from the module docstring: one
    feature map at ingest, every kernel downstream sees it.
    """
    for row in rows:
        mask = to_array(row[mask_col])
        obs = np.flatnonzero(mask).astype(np.int32)
        label = to_array(row[label_col])
        idx, val = topm_sparse(to_array(row[topic_col]), topm)
        yield idx, val, obs, label[obs]


def row_quads(rows, id_col, score_col, label_col, mask_col, topm=0):
    """Stream `(row_id, score, label, mask)` off Spark Rows for `lean_eval_kernel`.

    `id_col` names an int64 row-id column (the clinical drivers synthesize a doc
    key and pass its name); `lean_eval_kernel` packs it into an int64 array, so a
    string id raises there. `score_col` is θ or an already-computed probability vector
    depending on the caller; both arrive as Spark ML Vectors, so the same
    `to_array` serves.

    `topm > 0` truncates the score to its top-m `(idx, val)` pair — valid ONLY when
    `score_col` is θ and the kernel is scoring it with `(V, b_raw)`. An
    already-computed probability column is not a feature vector and must never be
    truncated; `collect_lean_proba` enforces that by passing `topm` only on the
    branch that fits.
    """
    for row in rows:
        score = to_array(row[score_col])
        if int(topm) > 0:
            score = topm_sparse(score, topm)
        yield (row[id_col], score,
               to_array(row[label_col]), to_array(row[mask_col]))


def pack_rows(rows, topic_col, label_col, mask_col, topm=0):
    """Spark Rows -> compact `(theta (K,), obs_idx (n,) int32, y_obs (n,))` tuples.

    The form the REPEATED-pass path caches. The wall is that dense
    per-doc `label`/`labelMask` are ~53 KB/row at C≈3,300 while the closure mask
    observes only tens of cells; caching the dense C-vectors would re-import that
    wall into the readout's own persisted RDD. Storing `(obs_idx, y_obs)` makes the
    cached row O(nnz) instead of O(C).

    `topm > 0` switches the cached row to the truncated sparse form
    `(idx (m,) int32, val (m,), obs_idx, y_obs)` — a 4-tuple, so a consumer cannot
    confuse the two shapes — and shrinks the CACHE as well as the traffic: at
    K=3,827 and m=256 the θ half of a persisted row drops from 30 KB to 3 KB, which
    is the difference between an L-BFGS whose passes re-read from disk and one that
    stays in memory. Truncation is paid ONCE here, not on every pass.
    """
    topm = int(topm)
    for row in rows:
        mask = to_array(row[mask_col])
        obs = np.flatnonzero(mask).astype(np.int32)
        label = to_array(row[label_col])
        theta = to_array(row[topic_col])
        if topm > 0:
            idx, val = topm_sparse(theta, topm)
            yield (idx, val, obs, label[obs])
        else:
            yield (theta, obs, label[obs])


def dense_triples(packed, C):
    """Rehydrate `(theta, label, mask)` dense triples from `pack_rows` output.

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


