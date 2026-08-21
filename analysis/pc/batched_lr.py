"""Batched multi-head logistic regression: C independent per-node fits, one solver.

This is Package A of the distributed readout
(`docs/superpowers/plans/2026-08-20-distributed-readout-plan.md`, step 1): the
pure-numpy, Spark-free half. It replaces the per-node sklearn loop in
:func:`analysis.pc.evaluate._lr_proba_per_label_masked` — which is the ORACLE
this module must reproduce numerically — with ONE optimization over all C heads
that shares every pass over the data.

**Why batch at all.** On frozen θ the C per-node readouts are independent convex
(K+1)-dim problems that differ only in which rows they read (node c uses the rows
its mask observes) and in their labels. Fitting them one at a time costs C data
passes per L-BFGS iteration; fitting them together costs ONE, because doc d's
contribution to node c's gradient, `(σ(w_c·θ_d + b_c) − y_dc)·θ_d`, is computed
from the same θ_d every head already needs. At C≈3,300 that is the difference
between 3,300 Spark jobs and one. The whole point of the `stats_fn` seam below is
that this file never learns where the rows live: the in-memory reference and the
Spark treeAggregate implement the same three-number contract
(loss, raw gradient sums) and the solver cannot tell them apart.

**Why the objective looks like this.** The oracle is
`LogisticRegression(max_iter=1000, fit_intercept=True)` at sklearn defaults,
i.e. L2 with `C=1.0`, which minimizes the SUMMED (not averaged) log-loss plus
`0.5*‖w‖²` with the intercept UNPENALIZED, fit on features standardized by a
`StandardScaler` fit on that node's own observed train rows. Every one of those
choices is load-bearing for A/B equality against the driver readout, so:

  - the ridge is `0.5*l2*‖w_c‖²` with `l2=1.0` ≡ sklearn `C=1.0`, and `b_c` is
    outside the norm;
  - loss and gradient are SUMS over the node's observed cells, never means;
  - standardization is a fixed affine reparameterization of the SAME problem
    (see :func:`fold_standardization`), not a preprocessing step — the optimizer
    runs in standardized coordinates because that is where the problem is
    well-conditioned enough for L-BFGS to converge in tens of iterations, while
    aggregates are collected in RAW θ coordinates because that is what a
    distributed pass over θ can cheaply produce.

**Why the solver is L-BFGS and not Newton.** The plan (§"Design (v2)") rules out
the blockwise Newton of the co-fit head: an O(C·K²) Fisher block is 36 TB at
K≈3,300, whereas L-BFGS keeps `m` history pairs of the SAME shape as the
parameters — O(m·C·K). The separability of the objective is what makes the
batched version exact rather than a heuristic: a step-size VECTOR with a per-node
Armijo test is identical to running C independent line searches, and freezing a
converged node is identical to having stopped its own solver.

No Spark imports here, by design (packaging invariant: this must stay importable
on a driver, in a unit test, and inside an executor closure).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = [
    "standardization_moments",
    "fold_standardization",
    "standardized_grad_from_raw",
    "make_inmemory_stats_fn",
    "solve_batched_lr",
]

# Armijo sufficient-decrease constant. 1e-4 is the textbook value: loose enough
# that the unit step is accepted almost always once the L-BFGS Hessian estimate
# warms up (so a typical iteration costs ONE stats_fn call, i.e. one data pass).
_ARMIJO_C1 = 1e-4
# Curvature guard for the (s, y) pair. s·y <= 0 means the pair carries no usable
# curvature (would make the implicit inverse-Hessian indefinite); a tiny positive
# s·y is worse than useless because rho = 1/(s·y) then amplifies noise. Skipping
# is per-node: one bad node must not poison the shared history array.
_CURV_TOL = 1e-10
# Backtracking budget per iteration. 30 halvings = step ~1e-9; a node that cannot
# make progress at that scale is not going to, so we clear its history (fall back
# to steepest descent) and, if that fails too, stop it.
_MAX_BACKTRACK = 30
# Relative function-decrease floor = NUMERICAL convergence, the second stopping
# rule beside `gtol`. It is not optional: the objective is a SUM over a node's
# rows, so |F| ~ n and its roundoff is ~1e-16*n, which puts a floor of roughly
# sqrt(2*l2*1e-16*n) on the attainable gradient inf-norm — around 3e-6 at n=1000,
# i.e. ABOVE a tight `gtol`. Without this rule such a node spends the rest of
# max_iter failing 30 backtracks per iteration (30 wasted distributed passes
# each) while its parameters no longer move. Same constant sklearn hands scipy
# (`ftol=64*eps`), so the two solvers stop for the same reason at the same place.
_FTOL_REL = 64.0 * np.finfo(np.float64).eps


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic. The two branches keep `exp` on its safe side."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def standardization_moments(
    Pi: np.ndarray,
    obs_DC: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node masked feature moments: ``(mu (C,K), sd (C,K), n_obs (C,))``.

    ``mu[c]`` / ``sd[c]`` are the mean and POPULATION standard deviation (ddof=0,
    matching :class:`sklearn.preprocessing.StandardScaler`) of the features over
    exactly the rows node ``c`` observes — the same rows the oracle
    :func:`analysis.pc.evaluate._lr_proba_per_label_masked` hands to its
    per-node scaler. Computed from masked sums and sums-of-squares
    (``M.T @ Pi`` and ``M.T @ Pi²``) rather than a two-pass mean/var because that
    is the shape a single distributed treeAggregate can produce; θ lives on the
    simplex so the `E[x²] − mu²` cancellation is harmless at that scale.

    **Zero-variance semantics (read this before writing the Spark twin).** A
    feature that is constant across node ``c``'s observed rows carries no signal
    for that node, and its standardized column must therefore be inert. We flag
    it with a RELATIVE test — ``sd <= eps * max(1, |mu|)``, so the threshold
    survives features whose scale is far from 1 — and then return ``sd = 1.0``
    for it, exactly as sklearn's `_handle_zeros_in_scale` does. That gives a
    standardized column of `x − mu`, which is 0 on the fitting rows (so the
    coordinate's gradient is ~0 and its weight stays ~0, hence
    :func:`fold_standardization` maps it to a ~0 raw coefficient) and equals the
    oracle's own value off them.

    We deliberately do NOT return the literal floor ``sd = eps``. It is the same
    thing in exact arithmetic, but the gradient reaches the solver through
    :func:`standardized_grad_from_raw` as ``(g_raw − s*mu)/sd``, a cancellation
    of two O(n·|mu|) sums whose rounding residual is ~1e-16·n·|mu|. Dividing that
    residual by 1e-12 manufactures a spurious gradient of order 1e-4·n — many
    orders above any sensible ``gtol``, so the node would never converge. Unit
    scale keeps the residual at its true, negligible size. ``eps`` is therefore
    the DETECTION threshold, not the replacement value.

    Nodes with no observed rows get ``mu = 0``, ``sd = 1``: harmless identity
    moments, since their stats are all-zero and the solver no-ops on them.
    """
    Pi = np.asarray(Pi, dtype=np.float64)
    M = np.asarray(obs_DC).astype(np.float64)
    if M.shape[0] != Pi.shape[0]:
        raise ValueError(f"obs_DC rows {M.shape[0]} != Pi rows {Pi.shape[0]}")

    n_obs = M.sum(axis=0)                       # (C,)
    denom = np.maximum(n_obs, 1.0)[:, None]     # empty node -> divide by 1, sums are 0
    mu = (M.T @ Pi) / denom                     # (C,K)
    ex2 = (M.T @ (Pi * Pi)) / denom             # (C,K)
    var = np.maximum(ex2 - mu * mu, 0.0)        # clip the cancellation's negative tail
    sd = np.sqrt(var)

    degenerate = sd <= eps * np.maximum(1.0, np.abs(mu))
    sd = np.where(degenerate, 1.0, sd)
    empty = n_obs <= 0
    if empty.any():
        mu[empty] = 0.0
        sd[empty] = 1.0
    return mu, sd, n_obs


def fold_standardization(
    W_std: np.ndarray,
    b_std: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardized-space params -> raw-θ scoring params ``(V (C,K), b_raw (C,))``.

    Standardization is an affine change of variables, `x_std = (θ − mu)/sd`, so
    the fitted model is exactly linear in raw θ too:

        w_std·x_std + b_std = (w_std/sd)·θ + (b_std − sum_k w_std[k]*mu[k]/sd[k])

    i.e. ``V[c] = W_std[c]/sd[c]`` and ``b_raw[c] = b_std[c] − sum(W_std[c]*mu[c]/sd[c])``.

    This identity is why the scaler never has to travel with the model: scoring
    broadcasts `(V, b_raw)` and does one matvec against raw θ (plan step 2, "No
    collect"), and it is also how the solver evaluates its own objective — every
    score in this module goes through this function, so the fit and the eventual
    executor-side scoring cannot drift apart.
    """
    W_std = np.asarray(W_std, dtype=np.float64)
    b_std = np.asarray(b_std, dtype=np.float64)
    V = W_std / sd
    b_raw = b_std - np.einsum("ck,ck->c", V, np.asarray(mu, dtype=np.float64))
    return V, b_raw


def standardized_grad_from_raw(
    g_raw: np.ndarray,
    s: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """Fold RAW per-node aggregates into standardized-space weight gradients.

    A data pass can only cheaply produce raw-θ sums:
    ``g_raw[c] = sum over node c's observed cells of (p − y)·θ_d`` (C,K) and
    ``s[c] = sum of (p − y)`` (C,). The chain rule through `x_std = (θ − mu)/sd`
    turns those into the standardized gradient the optimizer works in:

        dL/dw_std[c,k] = sum (p−y)·(θ[k] − mu[c,k])/sd[c,k]
                       = (g_raw[c,k] − s[c]*mu[c,k]) / sd[c,k]

    ``s`` needs no folding: the intercept's derivative is `sum (p − y)` in either
    coordinate system (standardization does not touch the constant column), so
    ``s`` IS the intercept gradient. Keeping this fold on the driver is what lets
    the distributed pass stay standardization-agnostic — executors never need the
    (C,K) moment tables, only the broadcast (V, b_raw).

    NOTE: data term only. The ridge is the solver's business
    (:func:`solve_batched_lr`), so that the same aggregates serve any `l2`.
    """
    g_raw = np.asarray(g_raw, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return (g_raw - s[:, None] * np.asarray(mu, dtype=np.float64)) / sd


def make_inmemory_stats_fn(
    Pi: np.ndarray,
    y_DC: np.ndarray,
    obs_DC: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build the reference (single-process) implementation of the stats seam.

    Returns ``stats_fn(W_std, b_std) -> (loss_data (C,), gW_std (C,K), gb (C,))``
    where ``loss_data[c]`` is node ``c``'s SUMMED log-loss over its observed rows
    and the gradients are the matching standardized-space derivatives — DATA TERM
    ONLY, no ridge (the solver owns the penalty so the aggregate is reusable).

    It deliberately takes the long way round: fold to raw space, score raw θ,
    aggregate `g_raw`/`s`, fold back. Scoring the standardized matrix directly
    would be shorter and slightly more accurate, and would test nothing — this
    function exists so the unit tests exercise EXACTLY the arithmetic path the
    Spark treeAggregate will run (plan step 1), making it a usable A/B reference
    for the distributed implementation rather than merely a correct one.

    Unobserved cells are zeroed in the residual before aggregation, so a masked
    cell contributes nothing to any loss or gradient — the invariant
    :mod:`analysis.pc.tests.test_multitask_masking` pins for the co-fit head, and
    the reason a node with zero observed rows produces an exactly-zero gradient
    and is a no-op for the solver.
    """
    Pi = np.asarray(Pi, dtype=np.float64)
    Y = np.asarray(y_DC, dtype=np.float64)
    M = np.asarray(obs_DC).astype(np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    YM = Y * M                                  # labels only where observed

    def stats_fn(W_std, b_std):
        V, b_raw = fold_standardization(W_std, b_std, mu, sd)
        Z = Pi @ V.T + b_raw[None, :]           # (D,C) raw-space scores
        # sum over observed cells of [log(1+e^z) - y*z]; logaddexp is the stable form.
        loss_data = (M * np.logaddexp(0.0, Z) - YM * Z).sum(axis=0)
        R = M * (_sigmoid(Z) - Y)               # (D,C) residual, 0 off-mask
        g_raw = R.T @ Pi                        # (C,K)
        s = R.sum(axis=0)                       # (C,) == intercept gradient
        return loss_data, standardized_grad_from_raw(g_raw, s, mu, sd), s

    return stats_fn


def _two_loop(
    G: np.ndarray,
    S_hist: list[np.ndarray],
    Y_hist: list[np.ndarray],
    rho_hist: list[np.ndarray],
    gamma: np.ndarray,
) -> np.ndarray:
    """Vectorized L-BFGS two-loop recursion: returns the search direction ``-H·G``.

    Each history slot is a full (C, K+1) array, so the recursion runs for all C
    nodes at once with only `sum(..., axis=-1)` reductions — no per-node Python
    loop. Nodes that skipped a pair (bad curvature, frozen, or line-search
    failure) carry ``rho = 0`` and zeroed `s`/`y` in that slot, which makes their
    alpha/beta identically 0: the shared array holds a per-node-ragged history
    without any masking logic in the inner loop.
    """
    q = G.copy()
    alphas: list[np.ndarray] = []
    for S, Yv, rho in zip(reversed(S_hist), reversed(Y_hist), reversed(rho_hist)):
        a = rho * np.einsum("ck,ck->c", S, q)
        q -= a[:, None] * Yv
        alphas.append(a)
    r = gamma[:, None] * q
    for S, Yv, rho, a in zip(S_hist, Y_hist, rho_hist, reversed(alphas)):
        beta = rho * np.einsum("ck,ck->c", Yv, r)
        r += S * (a - beta)[:, None]
    return -r


def solve_batched_lr(
    stats_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    C: int,
    K: int,
    *,
    l2: float = 1.0,
    max_iter: int = 200,
    history: int = 6,
    gtol: float = 1e-6,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve C independent penalized logistic regressions with one batched L-BFGS.

    Minimizes, per node ``c`` and independently of every other node,

        F_c(w, b) = loss_data[c](w, b) + 0.5*l2*‖w‖²        (b unpenalized)

    where `loss_data` comes from ``stats_fn`` — the ONLY thing this solver knows
    about the data. `l2=1.0` reproduces sklearn's `C=1.0` on summed log-loss,
    which is what the oracle
    :func:`analysis.pc.evaluate._lr_proba_per_label_masked` fits.

    Returns ``(W_std (C,K), b_std (C,), info)``; `info` carries per-node
    ``n_iter``, ``converged``, ``converged_gtol``, ``stalled`` and
    ``grad_inf_norm`` (plus ``n_stats_calls``, the real cost metric — each one is
    a full distributed pass — and ``line_search_failures``). Params are in
    STANDARDIZED coordinates; pass them through :func:`fold_standardization` to
    score raw θ.

    ``progress_fn`` (optional) is called once per completed outer iteration with
    a small dict (`iter`, `n_stats_calls`, `n_active`, `n_converged`,
    `n_stalled`, `max_grad_inf_norm`) so a driver can heartbeat a long
    distributed solve — each stats call is a full cluster pass, so iterations
    are minutes, not microseconds, at whole-Mondo scale.

    ``converged`` is ``converged_gtol | stalled``: a node stops either because
    its gradient inf-norm fell below `gtol` or because the objective stopped
    moving at double precision (see `_FTOL_REL`). Both are convergence; the split
    is reported so a caller can tell "clean gradient stop" from "as good as this
    arithmetic gets", which is the diagnosis you want when `gtol` is set below
    the summed loss's own roundoff floor.

    **Why the batched step control is exact, not an approximation.** F is
    separable across nodes, so a step-size VECTOR with a per-node Armijo test
    accepts exactly the step each independent line search would have accepted;
    halving only the failing entries costs extra `stats_fn` calls but never
    perturbs a node that already passed (its candidate is pinned at its accepted
    point for the remaining trials). Likewise FREEZING: once a node has stopped
    (full gradient inf-norm, data + ridge, <= `gtol`, or the numerical stall
    above) its direction is set to 0, so it contributes nothing further and is
    excluded from history updates — which is not just an optimization but a
    correctness guard, since a frozen node would otherwise contribute the
    degenerate pair `s = y = 0` and pollute `rho`. This is the plan's "freeze
    nodes as they converge so late passes touch only stragglers".

    Degenerate nodes (the caller has masked them out of `stats_fn`, per the
    oracle's constant-prediction fallback for empty/single-class observed sets)
    arrive with an exactly-zero gradient at the zero initialization, converge at
    iteration 0, and are returned as all-zero params — a clean no-op.
    """
    n = K + 1
    X = np.zeros((C, n), dtype=np.float64)      # [w | b] per node; 0 = sklearn's w0
    calls = 0

    def full_obj(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Objective + gradient WITH the ridge, from the data-only stats seam."""
        nonlocal calls
        W, b = P[:, :K], P[:, K]
        loss, gW, gb = stats_fn(W, b)
        calls += 1
        F = np.asarray(loss, dtype=np.float64) + 0.5 * l2 * np.einsum("ck,ck->c", W, W)
        G = np.empty_like(P)
        G[:, :K] = np.asarray(gW, dtype=np.float64) + l2 * W
        G[:, K] = np.asarray(gb, dtype=np.float64)
        return F, G

    F, G = full_obj(X)
    gnorm = np.abs(G).max(axis=1)
    converged_gtol = gnorm <= gtol
    stalled = np.zeros(C, dtype=bool)
    frozen = converged_gtol.copy()
    n_iter = np.zeros(C, dtype=np.int64)
    stall_streak = np.zeros(C, dtype=np.int64)
    ls_failures = 0

    S_hist: list[np.ndarray] = []
    Y_hist: list[np.ndarray] = []
    rho_hist: list[np.ndarray] = []
    gamma = np.ones(C, dtype=np.float64)
    has_hist = np.zeros(C, dtype=bool)

    for it in range(1, max_iter + 1):
        active = ~frozen
        if not active.any():
            break

        d = _two_loop(G, S_hist, Y_hist, rho_hist, gamma) if S_hist else -G
        d[frozen] = 0.0
        gd = np.einsum("ck,ck->c", G, d)
        # A non-descent direction can only come from a corrupted history; fall
        # back to steepest descent for those nodes rather than abandoning them.
        bad = active & (gd >= 0.0)
        if bad.any():
            d[bad] = -G[bad]
            gd[bad] = -np.einsum("ck,ck->c", G[bad], G[bad])

        # First step from a cold (or cleared) history must be scaled: the unit
        # step along -G is meaningless when ‖G‖ is large, and burning 20
        # backtracks to discover that costs 20 data passes.
        had_hist = has_hist.copy()
        t = np.where(had_hist, 1.0, 1.0 / np.maximum(1.0, np.linalg.norm(G, axis=1)))
        X_best, F_best, G_best = X.copy(), F.copy(), G.copy()
        accepted = ~active                      # frozen nodes are trivially "done"
        f_scale = np.maximum(np.abs(F), 1.0)
        hopeless = np.zeros(C, dtype=bool)
        for _ in range(_MAX_BACKTRACK):
            # A node whose SUFFICIENT decrease has shrunk below the objective's
            # own roundoff can never pass Armijo again, so halving it further
            # only burns whole distributed passes for the nodes still searching.
            # Cutting it loose here (it is at its numerical optimum; the stall
            # rule below picks it up) is what keeps a converging batch from
            # paying 30 passes an iteration for its last straggler.
            hopeless |= (~accepted) & (t * np.abs(gd) <= _FTOL_REL * f_scale)
            searching = ~(accepted | hopeless)
            if not searching.any():
                break
            X_cand = np.where(searching[:, None], X + t[:, None] * d, X_best)
            F_cand, G_cand = full_obj(X_cand)
            ok = searching & np.isfinite(F_cand) & (F_cand <= F + _ARMIJO_C1 * t * gd)
            if ok.any():
                X_best[ok], F_best[ok], G_best[ok] = X_cand[ok], F_cand[ok], G_cand[ok]
                accepted |= ok
            t = np.where(accepted, t, 0.5 * t)
        failed = ~accepted
        ls_failures += int(failed.sum())

        s_new = X_best - X
        y_new = G_best - G
        sy = np.einsum("ck,ck->c", s_new, y_new)
        # Per-node curvature guard: a pair is stored only for a node that moved,
        # is not frozen, and has positive curvature. Everyone else gets zeros in
        # this slot, which the two-loop treats as "no pair" (see _two_loop).
        keep = active & (~failed) & (sy > _CURV_TOL)
        S_hist.append(np.where(keep[:, None], s_new, 0.0))
        Y_hist.append(np.where(keep[:, None], y_new, 0.0))
        rho_hist.append(np.where(keep, 1.0 / np.where(keep, sy, 1.0), 0.0))
        if len(S_hist) > history:
            S_hist.pop(0)
            Y_hist.pop(0)
            rho_hist.pop(0)
        has_hist |= keep
        yy = np.einsum("ck,ck->c", y_new, y_new)
        upd = keep & (yy > 0.0)
        # H0 = gamma*I, the Barzilai-Borwein scaling: without it the very first
        # L-BFGS step is on the wrong scale entirely.
        gamma[upd] = sy[upd] / yy[upd]

        # Numerical convergence: the objective no longer moves at double
        # precision. A node that failed the line search WITH a history is given
        # one retry from a cleared history first (below) — the direction, not the
        # arithmetic, is the likelier culprit there — so it is excluded here.
        dF = F - F_best
        scale = np.maximum(np.maximum(np.abs(F), np.abs(F_best)), 1.0)
        retry = failed & had_hist
        tiny = active & (dF <= _FTOL_REL * scale) & ~retry
        # Two CONSECUTIVE non-moves, not one: a single L-BFGS iteration can make
        # negligible progress and still be one step from a much better point
        # (the curvature pair it just stored is what unlocks the next step), and
        # a premature stop leaves the node short of `gtol` for no reason. One
        # extra pass is cheap insurance against that.
        stall_streak[tiny] += 1
        stall_streak[active & ~tiny] = 0
        newly_stalled = tiny & (stall_streak >= 2)

        X, F, G = X_best, F_best, G_best
        n_iter[active] = it
        gnorm = np.abs(G).max(axis=1)
        newly_gtol = active & (gnorm <= gtol)
        converged_gtol |= newly_gtol
        stalled |= newly_stalled
        frozen |= newly_gtol | newly_stalled

        if failed.any():
            # No progress at ~1e-9 of the step. Clear the history for those nodes
            # so the next iteration tries scaled steepest descent; a node that
            # fails again with no history to blame is stalled by the arithmetic
            # and was frozen by the rule above.
            for arr in S_hist:
                arr[failed] = 0.0
            for arr in Y_hist:
                arr[failed] = 0.0
            for arr in rho_hist:
                arr[failed] = 0.0
            gamma[failed] = 1.0
            has_hist[failed] = False

        if progress_fn is not None:
            # End-of-iteration hook for driver-side progress logging: every entry
            # in `n_stats_calls` is a full distributed data pass, which is where
            # the wall-clock goes, so the caller can turn this into a meaningful
            # "passes so far / nodes still active" heartbeat. Reporting-only —
            # exceptions are the caller's problem, not swallowed here.
            progress_fn({
                "iter": it,
                "n_stats_calls": calls,
                "n_active": int((~frozen).sum()),
                "n_converged": int((converged_gtol | stalled).sum()),
                "n_stalled": int(stalled.sum()),
                "max_grad_inf_norm": float(gnorm[~frozen].max()) if (~frozen).any()
                                     else float(gnorm.max()) if C else 0.0,
            })

    info = {
        "n_iter": n_iter,
        "converged": converged_gtol | stalled,
        "converged_gtol": converged_gtol,
        "stalled": stalled,
        "grad_inf_norm": np.abs(G).max(axis=1),
        "n_stats_calls": calls,
        "line_search_failures": ls_failures,
        "loss": F,
    }
    return X[:, :K].copy(), X[:, K].copy(), info
