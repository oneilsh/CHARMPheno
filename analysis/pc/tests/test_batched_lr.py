"""Correctness gates for the batched multi-head readout solver
(:mod:`analysis.pc.batched_lr`, Package A of
``docs/superpowers/plans/2026-08-20-distributed-readout-plan.md``).

The solver exists to replace the per-node sklearn loop in
:func:`analysis.pc.evaluate._lr_proba_per_label_masked` with ONE optimization
over all C heads, so the only gate that really matters is EQUIVALENCE to that
oracle: same masked rows, same per-node standardization, same
``LogisticRegression(max_iter=1000, fit_intercept=True)`` objective (L2, C=1.0),
same constant-prediction fallback for degenerate nodes. The plan's correctness
gate ("per-node AUC equal to numerical tolerance before the driver path is
retired") is that comparison at cardiovascular scale; this file is its unit-scale
rehearsal, plus the algebraic identities the distributed layer will lean on.

Seven gates:

  1. **Oracle equivalence** on a synthetic readout with the mask shapes the real
     readout sees — a fully-observed node, sparse (~5%) nodes, a rare node with
     ~20 positives, a single-class node and a zero-observation node. Held-out
     probabilities within 1e-4, per-node AUC within 1e-6.
  2. **fold_standardization** is an exact reparameterization (1e-10), which is
     what lets scoring broadcast raw-θ coefficients and drop the scaler.
  3. **standardized_grad_from_raw** matches both a direct standardized-space
     gradient and central finite differences (1e-8) — the fold is where a
     distributed raw aggregate becomes an optimizer gradient, so an error here
     would be invisible until the Spark A/B.
  4. **Convergence freezing** is inert: every non-degenerate node stops, and a
     node that froze early holds the same answer a fresh solve gives it.
  5. **Zero-variance features** (constant across a node's observed rows) produce
     no NaNs, a ~0 coefficient, and no material disturbance to the other
     coefficients — matching sklearn's `_handle_zeros_in_scale`.
  6. **The progress hook** fires once per outer iteration and reports the run it
     is heartbeating.
  7. **Warm starts** (`x0`): the raw-space round trip is exact, a warm-started
     split fit converges to the cold answer, it beats cold at every small
     iteration budget (the regime the readout runs in — full convergence is
     NOT reliably cheaper, see the solver docstring), a restart at the optimum
     stops instead of crawling out `max_iter`, an explicit zero `x0` is the cold
     path bit for bit, and a masked node stays a no-op only because its `x0` row
     is zeroed.

**On the reference's tolerance.** The oracle is fit here with ``tol=1e-8``
instead of sklearn's default ``tol=1e-4``. This is a statement about the ORACLE,
not about us: sklearn stops when its gradient inf-norm drops below `tol`, which
at the default leaves it ~1e-4 from the optimum, worth ~5e-4 in predicted
probability — five times the tolerance this file asserts, and in the direction of
sklearn being the less-converged of the two. Tightening the reference makes the
assertion measure OUR error rather than sklearn's stopping rule; the
default-`tol` oracle is checked too, at the looser bound that its own stopping
rule permits.

numpy + sklearn only (eval/baseline layer); deterministic given the seeds.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from analysis.pc.batched_lr import (
    fold_standardization,
    make_inmemory_stats_fn,
    solve_batched_lr,
    standardization_moments,
    standardized_grad_from_raw,
    unfold_standardization,
)

# See the module docstring: the reference is converged past its default so the
# comparison measures the solver, not sklearn's stopping rule.
_SK_TOL = 1e-8
_SK_DEFAULT_TOL = 1e-4


# --------------------------------------------------------------------------- #
# synthetic readout problem                                                    #
# --------------------------------------------------------------------------- #
def _readout_problem(seed=0, D=2000, D_te=500, K=30, C=12):
    """A frozen-θ readout problem with the mask shapes the real readout hits.

    Features are Dirichlet rows (θ is a simplex point, so the K standardized
    columns carry an exact linear dependency — the ill-conditioned direction the
    ridge has to catch, and a more honest test bed than isotropic gaussians).
    Labels are drawn from a true per-node logistic so every node has real signal
    and the fits are non-degenerate in the interesting cases.

    Node roles: 0 fully observed; 1-4 sparse (~5% of rows); 5 rare (~20
    positives, the Q1-tail case `readout_sample_frac` destroys); 6 single-class;
    7 zero-observation; 8-11 moderately observed.
    """
    rng = np.random.default_rng(seed)
    Pi = rng.dirichlet(np.full(K, 0.3), size=D)
    Pi_te = rng.dirichlet(np.full(K, 0.3), size=D_te)
    W_true = rng.standard_normal((C, K)) * 2.5
    b_true = rng.standard_normal(C) * 0.4

    def draw(P):
        z = P @ W_true.T + b_true
        return (rng.random(z.shape) < 1.0 / (1.0 + np.exp(-z))).astype(np.float64)

    y = draw(Pi)
    y_te = draw(Pi_te)

    obs = np.zeros((D, C), dtype=bool)
    obs[:, 0] = True                                     # fully observed
    for c in range(1, 5):                                # sparse
        obs[rng.random(D) < 0.05, c] = True
    for c in range(8, C):                                # moderate
        obs[rng.random(D) < 0.40, c] = True

    # rare node: keep 20 positives and a few hundred negatives
    pos = np.where(y[:, 5] == 1.0)[0]
    neg = np.where(y[:, 5] == 0.0)[0]
    obs[rng.choice(pos, size=20, replace=False), 5] = True
    obs[rng.choice(neg, size=400, replace=False), 5] = True

    # single-class node: observe only rows whose label is 0
    zero_rows = np.where(y[:, 6] == 0.0)[0]
    obs[rng.choice(zero_rows, size=300, replace=False), 6] = True

    # node 7 stays all-False (zero observations)
    assert obs[:, 7].sum() == 0
    return dict(Pi=Pi, y=y, obs=obs, Pi_te=Pi_te, y_te=y_te, K=K, C=C)


def _degenerate_mask(y, obs):
    """Which nodes the ORACLE refuses to fit: empty or single-class observed set.

    :func:`analysis.pc.evaluate._lr_proba_per_label_masked` falls back to a
    constant prediction for these (the lone class value, or 0.0 when nothing is
    observed). The batched solver's contract puts that decision on the caller —
    degenerate nodes are masked OUT of the stats so their gradient is exactly
    zero and the solve no-ops on them — so the harness here plays the caller.
    """
    C = y.shape[1]
    degen = np.zeros(C, dtype=bool)
    const = np.zeros(C, dtype=np.float64)
    for c in range(C):
        classes = np.unique(y[obs[:, c], c])
        if classes.size < 2:
            degen[c] = True
            const[c] = float(classes[0]) if classes.size else 0.0
    return degen, const


def _oracle_proba(Pi, y, obs, Pi_te, tol=_SK_TOL):
    """Re-implementation of ``_lr_proba_per_label_masked`` with a tunable `tol`.

    Kept literal (per-node StandardScaler on that node's observed train rows, then
    ``LogisticRegression(max_iter=1000)``) so this file compares against the
    oracle's ARITHMETIC and not against a paraphrase of it.
    """
    C = y.shape[1]
    proba = np.zeros((Pi_te.shape[0], C), dtype=np.float64)
    for c in range(C):
        rows = np.where(obs[:, c])[0]
        yc = y[rows, c]
        classes = np.unique(yc)
        if classes.size < 2:
            proba[:, c] = float(classes[0]) if classes.size else 0.0
            continue
        scaler = StandardScaler().fit(Pi[rows])
        lr = LogisticRegression(max_iter=1000, tol=tol, fit_intercept=True)
        lr.fit(scaler.transform(Pi[rows]), yc)
        pos = int(np.where(lr.classes_ == 1)[0][0])
        proba[:, c] = lr.predict_proba(scaler.transform(Pi_te))[:, pos]
    return proba


def _batched_fit(Pi, y, obs, C, K, warm_start=None, **solver_kw):
    """Fit all heads at once; returns ``(V, b_raw, info, degen, const)``.

    Degenerate columns are zeroed out of the mask BEFORE the moments/stats are
    built — that is the contract's "caller masks degenerate nodes out of stats",
    and it is what turns an unbounded single-class fit into an exact no-op.

    ``warm_start=(V, b_raw)`` plays the same caller role the cloud driver's
    `_fit_readout_heads` plays: a warm start arrives in RAW θ coordinates, is
    unfolded through THIS fit's own moments, and has the rows of the nodes THIS
    fit masks zeroed (their objective is the bare ridge, stationary only at 0).
    """
    degen, const = _degenerate_mask(y, obs)
    obs_fit = obs.copy()
    obs_fit[:, degen] = False
    mu, sd, _ = standardization_moments(Pi, obs_fit)
    stats_fn = make_inmemory_stats_fn(Pi, y, obs_fit, mu, sd)
    if warm_start is not None:
        W0, b0 = unfold_standardization(warm_start[0], warm_start[1], mu, sd)
        solver_kw["x0"] = (np.where(degen[:, None], 0.0, W0),
                           np.where(degen, 0.0, b0))
    W, b, info = solve_batched_lr(stats_fn, C, K, **solver_kw)
    V, b_raw = fold_standardization(W, b, mu, sd)
    return V, b_raw, info, degen, const, (W, b, mu, sd)


def _batched_proba(Pi_te, V, b_raw, degen, const):
    """Score raw θ and apply the oracle's constant fallback where it applies."""
    p = 1.0 / (1.0 + np.exp(-(Pi_te @ V.T + b_raw[None, :])))
    p[:, degen] = const[degen]
    return p


# --------------------------------------------------------------------------- #
# 1. equivalence with the sklearn oracle                                       #
# --------------------------------------------------------------------------- #
def test_matches_sklearn_oracle_per_node():
    """Held-out probabilities and per-node AUC match the per-node sklearn fits.

    This is the gate the whole package exists to pass: one batched optimization
    reproduces C independently-fit `LogisticRegression`s, each on its own masked
    rows with its own scaler, across the full range of mask densities — including
    a node whose entire training set is 20 positives, where an unstable solver
    would show up first.
    """
    p = _readout_problem()
    Pi, y, obs, Pi_te, y_te = p["Pi"], p["y"], p["obs"], p["Pi_te"], p["y_te"]
    C, K = p["C"], p["K"]

    V, b_raw, info, degen, const, _ = _batched_fit(Pi, y, obs, C, K)
    ours = _batched_proba(Pi_te, V, b_raw, degen, const)
    ref = _oracle_proba(Pi, y, obs, Pi_te)

    assert degen.tolist() == [c in (6, 7) for c in range(C)], (
        "problem construction changed: expected exactly nodes 6 (single-class) "
        "and 7 (zero-observation) to be degenerate"
    )
    assert np.isfinite(ours).all()

    for c in range(C):
        if degen[c]:
            # the oracle's constant fallback, reproduced exactly
            assert np.allclose(ours[:, c], ref[:, c], atol=0.0, rtol=0.0)
            continue
        dp = np.abs(ours[:, c] - ref[:, c]).max()
        assert dp < 1e-4, f"node {c}: max |Δp| = {dp:.3e} on held-out rows"
        d_auc = abs(
            roc_auc_score(y_te[:, c], ours[:, c])
            - roc_auc_score(y_te[:, c], ref[:, c])
        )
        assert d_auc < 1e-6, f"node {c}: |ΔAUC| = {d_auc:.3e}"

    # ...and the same comparison against the oracle AT ITS DEFAULT tol, which can
    # only agree to the accuracy its own stopping rule leaves it (~5e-4): the gap
    # below is sklearn's distance from the optimum, not ours.
    ref_default = _oracle_proba(Pi, y, obs, Pi_te, tol=_SK_DEFAULT_TOL)
    gap = np.abs(ours[:, ~degen] - ref_default[:, ~degen]).max()
    assert gap < 5e-3, f"default-tol oracle disagreement {gap:.3e}"


def test_zero_observation_node_is_a_clean_noop():
    """A node with no observed rows converges at iteration 0 with zero params.

    Its stats are identically zero, so the gradient is exactly zero at the
    initialization and there is nothing for the solver to do — the property that
    lets the caller keep degenerate nodes in the batch (preserving node indexing
    all the way to the metric tables) instead of compacting them out.
    """
    p = _readout_problem()
    C, K = p["C"], p["K"]
    _, _, info, degen, _, (W, b, _, _) = _batched_fit(
        p["Pi"], p["y"], p["obs"], C, K
    )
    for c in np.where(degen)[0]:
        assert info["n_iter"][c] == 0
        assert info["converged"][c]
        assert np.all(W[c] == 0.0) and b[c] == 0.0


# --------------------------------------------------------------------------- #
# 2. fold_standardization is an exact reparameterization                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(3))
def test_fold_standardization_scores_identically(seed):
    """Raw-θ scores equal standardized-space scores for the folded params.

    Scoring (plan step 2) broadcasts only `(V, b_raw)` and never ships the
    per-node scaler, so any drift in this identity would silently change every
    readout probability relative to the oracle.
    """
    rng = np.random.default_rng(seed)
    C, K, D = 5, 8, 40
    theta = rng.random((D, K))
    mu = rng.standard_normal((C, K))
    sd = rng.random((C, K)) + 0.5
    W_std = rng.standard_normal((C, K))
    b_std = rng.standard_normal(C)

    V, b_raw = fold_standardization(W_std, b_std, mu, sd)
    raw = theta @ V.T + b_raw[None, :]
    std = np.stack(
        [((theta - mu[c]) / sd[c]) @ W_std[c] + b_std[c] for c in range(C)], axis=1
    )
    assert np.abs(raw - std).max() < 1e-10

    # ...and the inverse map is exact, which is what lets a warm start travel in
    # raw coordinates between fits that standardize against DIFFERENT rows.
    W_back, b_back = unfold_standardization(V, b_raw, mu, sd)
    assert np.abs(W_back - W_std).max() < 1e-12
    assert np.abs(b_back - b_std).max() < 1e-12
    V2, b_raw2 = fold_standardization(W_back, b_back, mu, sd)
    assert np.abs(V2 - V).max() < 1e-12 and np.abs(b_raw2 - b_raw).max() < 1e-12


# --------------------------------------------------------------------------- #
# 3. the raw -> standardized gradient fold                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(3))
def test_standardized_grad_from_raw_matches_direct_and_finite_differences(seed):
    """The folded gradient equals the standardized objective's own gradient.

    Two independent checks on the same numbers, because this fold is the seam
    where a distributed raw-θ aggregate becomes the optimizer's gradient and an
    error in it would look like a mildly-wrong fit rather than a crash:

      a. against the textbook standardized-space formula ``sum (p−y)·x_std``
         computed directly per node from an explicitly standardized matrix;
      b. against central finite differences of the data loss (`gb` too), which
         also pins the loss/gradient pair `make_inmemory_stats_fn` returns.
    """
    rng = np.random.default_rng(seed)
    D, K, C = 60, 5, 4
    Pi = rng.random((D, K)) * 2.0
    y = (rng.random((D, C)) < 0.5).astype(np.float64)
    obs = rng.random((D, C)) < 0.7
    for c in range(C):                       # keep every node non-empty
        obs[rng.integers(D), c] = True

    mu, sd, _ = standardization_moments(Pi, obs)
    stats_fn = make_inmemory_stats_fn(Pi, y, obs, mu, sd)
    W = rng.standard_normal((C, K)) * 0.5
    b = rng.standard_normal(C) * 0.5
    loss, gW, gb = stats_fn(W, b)

    # (a) direct standardized-space aggregation, node by node
    gW_direct = np.zeros_like(gW)
    gb_direct = np.zeros_like(gb)
    g_raw = np.zeros_like(gW)
    s = np.zeros_like(gb)
    for c in range(C):
        rows = np.where(obs[:, c])[0]
        Xs = (Pi[rows] - mu[c]) / sd[c]
        r = 1.0 / (1.0 + np.exp(-(Xs @ W[c] + b[c]))) - y[rows, c]
        gW_direct[c] = r @ Xs
        gb_direct[c] = r.sum()
        g_raw[c] = r @ Pi[rows]
        s[c] = r.sum()
    assert np.abs(gW - gW_direct).max() < 1e-8
    assert np.abs(gb - gb_direct).max() < 1e-8
    # the fold itself, fed hand-built raw aggregates
    assert np.abs(standardized_grad_from_raw(g_raw, s, mu, sd) - gW_direct).max() < 1e-8

    # (b) central finite differences of the data loss
    eps = 1e-6
    for c in range(C):
        for k in range(K):
            Wp, Wm = W.copy(), W.copy()
            Wp[c, k] += eps
            Wm[c, k] -= eps
            fd = (stats_fn(Wp, b)[0][c] - stats_fn(Wm, b)[0][c]) / (2 * eps)
            assert abs(fd - gW[c, k]) < 1e-8 * max(1.0, abs(gW[c, k])) + 1e-6
        bp, bm = b.copy(), b.copy()
        bp[c] += eps
        bm[c] -= eps
        fd_b = (stats_fn(W, bp)[0][c] - stats_fn(W, bm)[0][c]) / (2 * eps)
        assert abs(fd_b - gb[c]) < 1e-8 * max(1.0, abs(gb[c])) + 1e-6


# --------------------------------------------------------------------------- #
# 4. convergence freezing                                                      #
# --------------------------------------------------------------------------- #
def test_freezing_terminates_and_does_not_corrupt_other_nodes():
    """Every node stops, early stoppers keep their answer, neighbours are unaffected.

    Freezing is the plan's mechanism for "late passes touch only stragglers", so
    it has to be provably inert: a node that stopped at iteration 8 must hold the
    same parameters it would have had if it had been solved on its own, and the
    nodes that kept iterating must not notice it left. The second half is the one
    that would catch a shared-history bug — a frozen node contributing the
    degenerate pair ``s = y = 0`` would corrupt `rho` for everyone.
    """
    p = _readout_problem()
    Pi, y, obs, C, K = p["Pi"], p["y"], p["obs"], p["C"], p["K"]

    V, b_raw, info, degen, _, (W, b, mu, sd) = _batched_fit(
        Pi, y, obs, C, K, gtol=1e-6, max_iter=200
    )
    live = ~degen
    assert info["converged"][live].all(), (
        f"unconverged nodes {np.where(live & ~info['converged'])[0].tolist()}; "
        f"grad inf-norms {info['grad_inf_norm'][live]}"
    )
    assert (info["n_iter"][live] < 200).all()
    # freezing really happened: the batch did not stop all at once
    assert info["n_iter"][live].min() < info["n_iter"][live].max()

    # The earliest-freezing live node, re-solved ALONE (C=1): a fresh solve of the
    # same convex problem, with none of the batch's shared line search.
    early = int(np.where(live)[0][np.argmin(info["n_iter"][live])])
    obs1 = obs[:, [early]].copy()
    mu1, sd1, _ = standardization_moments(Pi, obs1)
    W1, b1, info1 = solve_batched_lr(
        make_inmemory_stats_fn(Pi, y[:, [early]], obs1, mu1, sd1), 1, K, gtol=1e-6
    )
    assert info1["converged"][0]
    V1, b_raw1 = fold_standardization(W1, b1, mu1, sd1)
    # Compare where it matters — the scores the readout actually emits. Observed
    # agreement is ~1e-15 (the batch runs node `early`'s arithmetic unchanged);
    # the bounds leave room for a BLAS whose blocking differs with the matrix
    # width, which could nudge the two trajectories to different points on the
    # objective's roundoff floor.
    z_batch = Pi @ V[early] + b_raw[early]
    z_solo = Pi @ V1[0] + b_raw1[0]
    assert np.abs(z_batch - z_solo).max() < 1e-6
    assert np.abs(V[early] - V1[0]).max() < 1e-5

    # ...and dropping the early node from the batch leaves the others put
    # (bit-for-bit, in practice: the heads share passes, never arithmetic).
    keep = np.array([c for c in range(C) if c != early])
    V2, b_raw2, _, _, _, _ = _batched_fit(Pi, y[:, keep], obs[:, keep], C - 1, K)
    for j, c in enumerate(keep):
        if degen[c]:
            continue
        dz = np.abs((Pi @ V[c] + b_raw[c]) - (Pi @ V2[j] + b_raw2[j])).max()
        assert dz < 1e-6, f"node {c} moved by {dz:.3e} when node {early} was dropped"


def test_loose_gtol_stops_on_the_gradient_criterion():
    """At a gtol above the loss's roundoff floor, nodes stop for the stated reason.

    `converged` is `converged_gtol | stalled`, and which one fires is a property
    of the problem, not of the solver: the objective is a SUM over a node's rows,
    so its roundoff floor (~1e-16·n) can sit ABOVE a very tight `gtol`. At
    sklearn's own 1e-4 the gradient criterion is reachable and must be what
    stops the run — the integrator's sanity check that `gtol` still means
    something.
    """
    p = _readout_problem()
    _, _, info, degen, _, _ = _batched_fit(
        p["Pi"], p["y"], p["obs"], p["C"], p["K"], gtol=1e-4
    )
    live = ~degen
    assert info["converged"][live].all()
    assert info["converged_gtol"][live].all()
    assert (info["grad_inf_norm"][live] <= 1e-4).all()


# --------------------------------------------------------------------------- #
# 5. zero-variance features                                                    #
# --------------------------------------------------------------------------- #
def test_constant_feature_column_is_inert_and_finite():
    """A feature constant on a node's observed rows is inert, not a NaN factory.

    Real θ columns go constant on a node's cohort all the time (a topic no
    patient in that cohort loads on). The oracle handles it via sklearn's
    `_handle_zeros_in_scale` (unit scale, ~0 coefficient); this module must match
    — hence the relative degeneracy test and unit replacement documented in
    :func:`standardization_moments`, NOT a literal 1e-12 floor, which would
    amplify the raw aggregate's own rounding into a spurious gradient of order
    1e-4·n and keep the node from ever converging.
    """
    rng = np.random.default_rng(7)
    D, D_te, K, C = 900, 200, 12, 4
    Pi = rng.dirichlet(np.full(K, 0.4), size=D)
    Pi_te = rng.dirichlet(np.full(K, 0.4), size=D_te)
    W_true = rng.standard_normal((C, K)) * 2.0
    y = (rng.random((D, C)) < 1.0 / (1.0 + np.exp(-(Pi @ W_true.T)))).astype(float)

    obs = rng.random((D, C)) < 0.5
    rows = np.where(obs[:, 1])[0]
    Pi[rows, 3] = 0.137                     # constant on node 1's rows only
    assert np.unique(Pi[:, 3]).size > 1, "feature must still vary off node 1's rows"

    mu, sd, n_obs = standardization_moments(Pi, obs)
    assert sd[1, 3] == 1.0, "constant column must get sklearn's unit scale"
    assert (sd[:, 3][np.arange(C) != 1] > 1e-6).all()
    assert np.isfinite(mu).all() and np.isfinite(sd).all()

    V, b_raw, info, degen, const, (W, b, _, _) = _batched_fit(Pi, y, obs, C, K)
    assert not degen.any()
    assert np.isfinite(V).all() and np.isfinite(b_raw).all()
    assert info["converged"].all()
    # the constant coordinate carries no signal, so its weight stays at ~0 and it
    # contributes ~nothing to the raw-space score
    assert abs(W[1, 3]) < 1e-6

    ours = _batched_proba(Pi_te, V, b_raw, degen, const)
    ref = _oracle_proba(Pi, y, obs, Pi_te)
    assert np.abs(ours - ref).max() < 1e-4

    # the OTHER coefficients of node 1 are undisturbed: they equal the oracle's
    # own coef_/scale_ to the same tolerance the well-behaved nodes get.
    scaler = StandardScaler().fit(Pi[rows])
    lr = LogisticRegression(max_iter=1000, tol=_SK_TOL).fit(
        scaler.transform(Pi[rows]), y[rows, 1]
    )
    V_ref = lr.coef_[0] / scaler.scale_
    other = np.arange(K) != 3
    assert np.abs(V[1, other] - V_ref[other]).max() < 1e-4


# --------------------------------------------------------------------------- #
# 6. progress reporting                                                        #
# --------------------------------------------------------------------------- #
def test_progress_fn_fires_once_per_iteration_and_tracks_the_run():
    """The heartbeat hook exists because a whole-Mondo stats call is a full
    cluster pass (minutes), so the driver needs a per-iteration signal to log.
    It must fire exactly once per completed outer iteration, count the same
    passes the final info reports, and show convergence progressing monotonically
    — otherwise the log it feeds would misrepresent the run."""
    p = _readout_problem()
    events = []
    _, _, info, degen, _, _ = _batched_fit(
        p["Pi"], p["y"], p["obs"], p["C"], p["K"], gtol=1e-4,
        progress_fn=events.append,
    )
    assert events, "progress_fn never fired"
    assert [e["iter"] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]["iter"] == int(info["n_iter"].max())
    assert events[-1]["n_stats_calls"] == int(info["n_stats_calls"])
    assert events[-1]["n_converged"] == int(info["converged"].sum())
    n_conv = [e["n_converged"] for e in events]
    assert n_conv == sorted(n_conv)
    # degenerate nodes are converged from iteration 0, so every event already
    # counts them — the driver-side display subtracts them; the hook must not.
    assert all(e["n_converged"] >= int(degen.sum()) for e in events)


# --------------------------------------------------------------------------- #
# 7. warm starts                                                               #
# --------------------------------------------------------------------------- #
def _row_subset_problem(p, keep_frac=0.75, seed=99):
    """The driver's SECOND fit, in miniature: the same nodes on 75% of the rows.

    This is the shape both warm-started solves in the cloud driver have — the
    isotonic calibrator's 75/25 train split and the A/B harness's row sample —
    a near-identical convex problem whose optimum sits a short walk from the
    full-data one.
    """
    rng = np.random.default_rng(seed)
    rows = rng.random(p["Pi"].shape[0]) < keep_frac
    return p["Pi"][rows], p["y"][rows], p["obs"][rows]


def test_warm_start_lands_on_the_same_optimum():
    """A warm-started split fit converges to the cold fit's answer.

    The init cannot move a convex optimum, so run to `gtol` the two must agree —
    this is the gate that a warm start is a WALL-CLOCK device and never a
    numerical one. The start is the FULL-data fit, mapped into the 75% fit's own
    standardized coordinates: the raw-space hand-off `unfold_standardization`
    exists for exactly this.

    Deliberately NOT asserted here: that warm reaches `gtol` in fewer passes. It
    does not, on a problem this small (23 vs 14) — see the solver docstring's
    "what it does NOT buy": the endgame belongs to the curvature history, not to
    the starting point. The cost claim is the capped-budget test below, which is
    the regime the readout actually runs in.
    """
    p = _readout_problem()
    C, K = p["C"], p["K"]
    V_full, b_full, _, _, _, _ = _batched_fit(p["Pi"], p["y"], p["obs"], C, K,
                                              gtol=1e-4)
    Pi_s, y_s, obs_s = _row_subset_problem(p)

    V_cold, b_cold, i_cold, degen, _, _ = _batched_fit(Pi_s, y_s, obs_s, C, K,
                                                       gtol=1e-4)
    V_warm, b_warm, i_warm, degen_w, _, _ = _batched_fit(
        Pi_s, y_s, obs_s, C, K, warm_start=(V_full, b_full), gtol=1e-4)

    assert i_cold["warm_started"] is False and i_warm["warm_started"] is True
    assert degen_w.tolist() == degen.tolist()
    live = ~degen
    assert i_warm["converged"][live].all()
    # same answer: compare the SCORES (what the readout emits), on the fit rows.
    z_cold = Pi_s @ V_cold[live].T + b_cold[live]
    z_warm = Pi_s @ V_warm[live].T + b_warm[live]
    assert np.abs(z_warm - z_cold).max() < 1e-3


def test_warm_start_beats_cold_under_an_iteration_cap():
    """Capped at a handful of iterations, the warm run is the better answer.

    This is the property the CHARM_DEV readout cap leans on: with `max_iter`
    reached before `gtol`, the returned point is just "wherever the budget got
    to", so a start nearer the optimum is not merely faster, it is strictly
    better. Measured against the converged cold solve as ground truth.
    """
    p = _readout_problem()
    C, K = p["C"], p["K"]
    V_full, b_full, _, _, _, _ = _batched_fit(p["Pi"], p["y"], p["obs"], C, K,
                                              gtol=1e-4)
    Pi_s, y_s, obs_s = _row_subset_problem(p)
    V_ref, b_ref, _, degen, _, _ = _batched_fit(Pi_s, y_s, obs_s, C, K, gtol=1e-4)
    live = ~degen
    z_ref = Pi_s @ V_ref[live].T + b_ref[live]

    def _err(V, b):
        return np.abs((Pi_s @ V[live].T + b[live]) - z_ref).max()

    # Every budget from one iteration up, not a single lucky cap: the claim is
    # that the warm iterate DOMINATES, not that it wins at one point.
    for cap in (1, 2, 3, 5, 10):
        V_c, b_c, _, _, _, _ = _batched_fit(Pi_s, y_s, obs_s, C, K, gtol=1e-4,
                                            max_iter=cap)
        V_w, b_w, _, _, _, _ = _batched_fit(Pi_s, y_s, obs_s, C, K, gtol=1e-4,
                                            max_iter=cap,
                                            warm_start=(V_full, b_full))
        assert _err(V_w, b_w) < _err(V_c, b_c), f"cap={cap}"


def test_warm_start_at_the_optimum_stops_instead_of_crawling():
    """Restarted AT its own answer under an unreachable `gtol`, the batch stops.

    The failure this pins cost 200 distributed passes before the guard existed:
    a warm start lands in the flat bottom of the objective, where `y = ΔG` is
    roundoff, `s·y` falls under the curvature guard so no pair is ever stored
    again, and the frozen L-BFGS model proposes the same 1e-7 step forever. The
    steps still DECREASE F (by ~1e-13 relative), so the `_FTOL_REL` stall rule
    never fires and the node runs out `max_iter` one pass at a time. Two
    consecutive curvature-free iterations now stop it as `stalled` — the
    existing name for "as good as this arithmetic gets".

    `gtol=1e-8` is below the summed loss's own roundoff floor here, so NEITHER
    run can stop on the gradient: this measures the stopping rules alone.
    """
    p = _readout_problem()
    C, K = p["C"], p["K"]
    V, b_raw, _, degen, _, _ = _batched_fit(p["Pi"], p["y"], p["obs"], C, K,
                                            gtol=1e-4)
    _, _, i_warm, _, _, _ = _batched_fit(p["Pi"], p["y"], p["obs"], C, K,
                                         gtol=1e-8, warm_start=(V, b_raw))
    _, _, i_cold, _, _, _ = _batched_fit(p["Pi"], p["y"], p["obs"], C, K,
                                         gtol=1e-8)
    live = ~degen
    assert i_warm["converged"][live].all(), "the guard must stop every node"
    assert i_warm["n_iter"].max() < 15, (
        f"warm restart crawled for {int(i_warm['n_iter'].max())} iterations; "
        "the curvature-free stall guard is not firing")
    assert i_warm["n_stats_calls"] < i_cold["n_stats_calls"]


def test_zero_x0_is_the_cold_start_bit_for_bit():
    """An explicit zero `x0` is the default path, unchanged — the compatibility gate.

    `x0=None` must remain byte-identical to every solve that ran before this
    argument existed, so the only difference an explicit zero start may make is
    the `warm_started` flag.
    """
    p = _readout_problem()
    C, K = p["C"], p["K"]
    V0, b0, i0, _, _, (W0, bs0, _, _) = _batched_fit(p["Pi"], p["y"], p["obs"],
                                                     C, K, gtol=1e-4)
    Vz, bz, iz, _, _, (Wz, bsz, _, _) = _batched_fit(
        p["Pi"], p["y"], p["obs"], C, K, gtol=1e-4,
        x0=(np.zeros((C, K)), np.zeros(C)))
    assert np.array_equal(W0, Wz) and np.array_equal(bs0, bsz)
    assert np.array_equal(V0, Vz) and np.array_equal(b0, bz)
    assert int(i0["n_stats_calls"]) == int(iz["n_stats_calls"])
    assert i0["warm_started"] is False and iz["warm_started"] is True


def test_masked_node_freezes_only_when_its_warm_start_row_is_zero():
    """The degenerate-node no-op is a property of the ZERO row, not of the mask.

    A node the caller masked out of `stats_fn` has no data term left, so its
    objective is the bare ridge `0.5*l2*‖w‖²` — gradient `l2*w`, which vanishes
    only at the origin. Zeroing its `x0` row (what `_batched_fit` and the cloud
    driver's `_fit_readout_heads` both do) keeps the clean iteration-0 freeze and
    the all-zero params the constant-prediction fallback expects. Leaving a
    stale row there instead makes the solver spend real distributed passes
    walking a node that will be OVERWRITTEN by a constant at scoring time — this
    test pins both halves so the zeroing is never quietly dropped.
    """
    p = _readout_problem()
    C, K = p["C"], p["K"]
    V_full, b_full, _, _, _, _ = _batched_fit(p["Pi"], p["y"], p["obs"], C, K,
                                              gtol=1e-4)
    # nodes 6 (single-class) and 7 (zero-observation) are the masked ones
    _, _, info, degen, _, (W, b, mu, sd) = _batched_fit(
        p["Pi"], p["y"], p["obs"], C, K, gtol=1e-4, warm_start=(V_full, b_full))
    assert degen.any()
    for c in np.where(degen)[0]:
        assert info["n_iter"][c] == 0
        assert info["converged"][c]
        assert np.all(W[c] == 0.0) and b[c] == 0.0

    # the contrapositive: a NONZERO row on a masked node is not stationary.
    obs_fit = p["obs"].copy()
    obs_fit[:, degen] = False
    mu2, sd2, _ = standardization_moments(p["Pi"], obs_fit)
    stats_fn = make_inmemory_stats_fn(p["Pi"], p["y"], obs_fit, mu2, sd2)
    W0, b0 = unfold_standardization(V_full, b_full, mu2, sd2)
    W0[degen] = 1.0                              # deliberately NOT zeroed
    _, _, info_bad = solve_batched_lr(stats_fn, C, K, gtol=1e-4, max_iter=3,
                                      x0=(W0, b0))
    assert (info_bad["n_iter"][degen] > 0).all(), (
        "a masked node with a stale warm-start row must be seen to iterate — "
        "if this ever passes trivially the ridge argument above has changed"
    )
