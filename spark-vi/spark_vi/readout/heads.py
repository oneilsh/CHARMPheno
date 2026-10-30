"""Fit the per-node readout heads on either backend (ADR 0051 §2).

`fit_heads(data, ...)` is the readout's whole fitting procedure -- the masked
standardization moments, the degenerate-node fallback, the batched L-BFGS solve
(`readout.solver`), resumable checkpoints and the fold back to raw coordinates --
written once against a small data interface that `readout.spark.SparkReadoutData`
and `readout.local.LocalReadoutData` both implement:

    C, K, topm, can_measure_coverage
    coverage()                          -> {m: (mean, p10)} θ-mass coverage
    moments()                           -> (mu, sd, n_obs, n_pos)
    stats_fn(mu, sd, fold_standardization=, standardized_grad_from_raw=)
                                        -> context manager, callable as
                                           stats_fn(W_std, b_std, node_mask=None)
    lean_proba(V, b_raw, degenerate=, const=) -> (proba, y, mask, ids)

The heads persistence (`write_heads` / `read_heads`) is the npz a later scoring-only
pass reads instead of re-fitting.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np

from spark_vi.readout.solver import (fold_standardization, solve_batched_lr,
                                     standardized_grad_from_raw,
                                     unfold_standardization)

log = logging.getLogger(__name__)

# sklearn's own `LogisticRegression` default tol, and the oracle this readout must
# reproduce stops there too. Below ~3e-6 the gradient inf-norm of a SUMMED loss is
# unreachable (roundoff floor ~1e-16*n), so a tighter gtol only buys stalled nodes
# and wasted distributed passes.
READOUT_GTOL = 1e-4
# Iteration budget for ONE batched readout solve. At C=437 a cold solve spends all
# 200 of them (~1,200 distributed passes, ~30 min), so this is a real cap, not a
# safety net: the returned point is "wherever 200 iterations got to" and every
# iteration is a full pass over the train split. `--readout-max-iter` exposes it
# because that makes a development loop affordable (60 iterations already give a
# stable macro-AUC ranking).
READOUT_MAX_ITER = 200

# Fit size (C*K float64 parameter bytes) above which `fit_heads` pays for
# the θ mass-coverage measurement. 64 MB is C=K≈2,900 — far above a few-hundred-node
# fit (C=444: 1.6 MB) and below a C=K≈3,827 one (117 MB), which is exactly where the
# top-m lever is a live question and one extra cheap pass is noise against a
# multi-hour solve.
_COVERAGE_MIN_FIT_BYTES = 64 * 1024 * 1024


_CKPT_VERSION = "readout-ckpt-v2"


def ckpt_fingerprint(C, K, n_obs, n_pos, theta_topm=0):
    """Digest identifying the PROBLEM a solver checkpoint belongs to.

    Hashes only the EXACTLY-REPRODUCIBLE identity of the problem: `(C, K,
    theta_topm)` plus the per-node observed/positive COUNTS. Counts are integer
    sums, and integer addition is exact in float64 in any order — so `n_obs`/
    `n_pos` come back bit-identical from every run over the same rows and mask,
    and they are also exactly what distinguishes a different arm (labels change
    `n_pos`), a different corpus/split (`n_obs`), or a different degenerate mask.

    What is deliberately NOT hashed: the standardization moments `(mu, sd)`.
    The moments pass is deterministic in VALUE but not in BITS: `treeAggregate` combines partials in
    task-completion order, float addition is not associative, and the low-order
    bits differ on every run, so a fingerprint over them rejects every cross-run
    checkpoint of the identical solve and silently disables resume. The residual risk of not pinning the
    basis is bounded by the solver's own contract: `x0` changes the PATH, never
    the answer — a warm start whose basis drifted by float noise costs at most a
    few extra iterations, while a byte-exact check costs the whole feature.
    """
    h = hashlib.sha256()
    h.update(f"{_CKPT_VERSION}|C={int(C)}|K={int(K)}|topm={int(theta_topm)}|"
             .encode())
    for arr in (n_obs, n_pos):
        a = np.rint(np.ascontiguousarray(arr, dtype=np.float64)).astype(np.int64)
        h.update(f"|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_ckpt(path, W_std, b_std, it, fingerprint):
    """Atomically write the solver checkpoint to `path` (tmp + replace).

    A process death mid-write must not leave a TORN file (on an object-store
    mount such as gcsfuse a write is an object mutation), which would be worse
    than no checkpoint at all, since the thing that reads it is a recovery run. Write to a sibling
    tmp then rename, so the visible file is always a complete one.

    Failures are swallowed and reported: a checkpoint is insurance, and a full
    disk or a storage hiccup must not kill a solve that is otherwise healthy.
    Returns True on success.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            # File OBJECT, not a name: np.savez appends `.npz` to a str path,
            # which would make the tmp name (and thus the rename) a guess.
            np.savez(fh, W_std=np.ascontiguousarray(W_std, dtype=np.float64),
                     b_std=np.ascontiguousarray(b_std, dtype=np.float64),
                     iter=np.int64(it), fingerprint=np.str_(fingerprint))
        tmp.replace(path)
        return True
    except Exception as exc:                     # pragma: no cover - I/O failure
        log.info(f"readout checkpoint write FAILED ({exc}); the solve "
              "continues uncheckpointed")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def read_ckpt(path, fingerprint):
    """Load `(W_std, b_std, iter)` from `path` if it matches `fingerprint`.

    Returns `None` when there is nothing usable — no file, an unreadable/torn
    file, or a fingerprint mismatch — and PRINTS why in the mismatch case, which
    is the one that means something (a checkpoint left by a different arm or a
    different corpus). A mismatch does not delete the file: the fresh solve's
    own first checkpoint overwrites it, and until then a stale-but-explainable
    file on disk is more useful for diagnosis than a silent deletion.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            got = str(z["fingerprint"].item())
            if got != fingerprint:
                log.info(f"readout checkpoint {path.name} IGNORED: "
                      f"fingerprint {got[:12]} != {fingerprint[:12]} — it belongs "
                      "to a different arm/corpus/basis. Starting cold.")
                return None
            return (np.asarray(z["W_std"], dtype=np.float64),
                    np.asarray(z["b_std"], dtype=np.float64),
                    int(z["iter"]))
    except Exception as exc:
        log.info(f"readout checkpoint {path.name} UNREADABLE ({exc}); "
              "starting cold")
        return None


# --------------------------------------------------------------------------- #
# Persisted readout HEADS (the fitted per-node scoring params, saved for reuse) #
# --------------------------------------------------------------------------- #
# Distinct from the SOLVER checkpoint above: the checkpoint is a mid-solve
# resume point that is DELETED once the solve lands, whereas this is the
# COMPLETED fit's raw-θ scoring params, kept beside the run's other record files.
# It exists so a later scoring-only consumer (`DagHeadsModel.load`, the framings'
# saved-heads engines) can turn each doc's θ into per-node proba with ONE
# mapPartitions, instead of re-running the whole batched readout solve.
_HEADS_VERSION = "readout-heads-v1"


def heads_path(run_dir, label):
    """Path of the persisted heads npz for arm `label` in `run_dir`.

    Keyed by `label` for the same reason the solver checkpoint is
    (`readout_heads_gated_pc.npz`, `..._unsup_gated.npz`, `..._dag_head.npz`): a
    run fits several arms into one dir and each is a different scoring model."""
    return Path(run_dir) / f"readout_heads_{label or 'arm'}.npz"


def heads_fingerprint(C, K, theta_topm, label, degenerate):
    """Identity of a fitted arm's heads — the arm shape plus its fittable mask.

    Mirrors `ckpt_fingerprint`'s doctrine: hash only the
    exactly-reproducible integers, never the standardization moments' float bits
    (which `treeAggregate` combines in nondeterministic order). `degenerate` is a
    bool mask derived from integer observed/positive COUNTS, so it is bit-stable
    across runs of the same arm/corpus and is a compact signature of them."""
    h = hashlib.sha256()
    h.update(f"{_HEADS_VERSION}|label={label or 'arm'}|C={int(C)}|K={int(K)}|"
             f"topm={int(theta_topm)}|".encode())
    d = np.ascontiguousarray(np.asarray(degenerate, dtype=bool))
    h.update(f"|{d.shape}|".encode())
    h.update(d.tobytes())
    return h.hexdigest()


def write_heads(run_dir, label, V, b_raw, const, degenerate, C, K,
                         theta_topm, W_std=None):
    """Persist a COMPLETED readout fit's raw-θ scoring params to the run dir.

    Everything a scoring pass needs to turn a doc's θ into per-node proba —
    `(V, b_raw)` (raw-θ, so no scaler travels), plus the `degenerate`/`const`
    oracle fallback and the truncation width `theta_topm` the coefficients belong
    to. Written as a small npz (C×K×8 bytes, ~60 MB at C=K≈2,700) beside the fit's
    other record files, once the solve has RETURNED, and only when the arm has a
    durable run dir (the driver's `checkpoint_dir`). Atomic (tmp + replace), as
    `write_ckpt`; failures are swallowed and reported, because the fit's own results
    are the record and a missing heads file only means a later scoring pass asks for
    a one-time re-readout. Returns True on success.

    `W_std` (optional, (C,K)): the solve's STANDARDIZED weights, stored beside V
    so the off-cluster loadings read (`inspect_topics`) has an honest importance
    scale once the solver checkpoint is gone — V is W_std/sd and explodes on
    low-variance (starved) topics. Additive; older sidecars simply lack it."""
    path = heads_path(run_dir, label)
    tmp = path.with_name(path.name + ".tmp")
    fp = heads_fingerprint(C, K, theta_topm, label, degenerate)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            # File OBJECT, not a name: np.savez appends `.npz` to a str path,
            # which would make the tmp name (and thus the rename) a guess.
            extra = ({"W_std": np.ascontiguousarray(W_std, dtype=np.float64)}
                     if W_std is not None else {})
            np.savez(fh,
                     version=np.str_(_HEADS_VERSION),
                     label=np.str_(label or "arm"),
                     V=np.ascontiguousarray(V, dtype=np.float64),
                     **extra,
                     b_raw=np.ascontiguousarray(b_raw, dtype=np.float64),
                     const=np.ascontiguousarray(const, dtype=np.float64),
                     degenerate=np.ascontiguousarray(degenerate, dtype=bool),
                     C=np.int64(int(C)), K=np.int64(int(K)),
                     theta_topm=np.int64(int(theta_topm)),
                     fingerprint=np.str_(fp))
        tmp.replace(path)
        log.info(f"wrote readout heads {path.name} "
              f"(C={int(C)} K={int(K)} topm={int(theta_topm)}) — "
              "later scoring passes read this, no re-fit")
        return True
    except Exception as exc:                         # pragma: no cover - I/O failure
        log.info(f"readout heads write FAILED ({exc}); the fit is "
              "unaffected, but a later scoring pass will need a "
              "re-readout to persist them")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def read_heads(run_dir, label, *, C=None, K=None, theta_topm=None):
    """Load persisted readout heads for `label`, or None with a printed reason.

    Returns `(V, b_raw, const, degenerate, C, K, theta_topm)` — the exact tuple a
    scoring pass (`collect_lean_proba`) consumes — when a heads npz is present,
    readable, of a known version, and (where the caller passes them) consistent
    with the manifest's `C`/`K`/`theta_topm`. Any miss returns None so the caller
    can degrade to the overall (non-decile) deliverable, NEVER fall back to a
    re-fit. `K=None`/`theta_topm=None` skip that particular consistency check (an
    old manifest may not record K); the V-shape check against the STORED C/K still
    runs regardless."""
    path = heads_path(run_dir, label)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            ver = str(z["version"].item())
            if ver != _HEADS_VERSION:
                log.info(f"readout heads {path.name} IGNORED: version "
                      f"{ver!r} != {_HEADS_VERSION!r}")
                return None
            gC, gK, gm = int(z["C"]), int(z["K"]), int(z["theta_topm"])
            V = np.asarray(z["V"], dtype=np.float64)
            b_raw = np.asarray(z["b_raw"], dtype=np.float64)
            const = np.asarray(z["const"], dtype=np.float64)
            degenerate = np.asarray(z["degenerate"], dtype=bool)
    except Exception as exc:
        log.info(f"readout heads {path.name} UNREADABLE ({exc})")
        return None
    for name, want, got in (("C", C, gC), ("K", K, gK),
                            ("theta_topm", theta_topm, gm)):
        if want is not None and int(want) != got:
            log.info(f"readout heads {path.name} IGNORED: {name}={got} "
                  f"!= manifest {name}={int(want)} — a different arm/run")
            return None
    if V.shape != (gC, gK):
        log.info(f"readout heads {path.name} IGNORED: V shape {V.shape} "
              f"!= (C={gC}, K={gK})")
        return None
    return V, b_raw, const, degenerate, gC, gK, gm


def fit_heads(data, *, l2=1.0, gtol=READOUT_GTOL, max_iter=READOUT_MAX_ITER,
              history=6, label="", warm_start=None, checkpoint_path=None,
              checkpoint_every=10):
    """Fit all C per-node readout heads with ONE batched L-BFGS.

    `data` is the TRAIN split on either backend -- `readout.spark.SparkReadoutData`
    or `readout.local.LocalReadoutData` -- which supplies `C`, `K`, the truncation
    width `topm`, `coverage`, `moments` and `stats_fn`. Everything below is the same
    for both; only where a data pass runs differs.

    Returns `(V (C,K), b_raw (C,), const (C,), degenerate (C,) bool, info)` — the
    raw-θ scoring parameters, so nothing but `(V, b_raw)` has to travel to score a
    split (the standardization is folded away; scoring needs no scaler).

    Three driver-side steps:

      1. `masked_moments` — one pass for the per-node masked mean/sd (the sklearn
         oracle standardizes on each node's OWN observed train rows) plus the
         `n_obs`/`n_pos` counts that identify degenerate nodes for free.
      2. DEGENERATE MASKING. A node whose observed train set is empty or
         single-class has no finite optimum (the intercept runs to ±inf), so the
         oracle refuses to fit it and predicts the lone class. We reproduce that by
         zeroing its loss/gradient rows before the solver sees them: its gradient is
         then exactly 0, `solve_batched_lr` freezes it at iteration 0, and the
         constant is applied at scoring time. Wrapping the stats_fn (rather than
         re-deriving the moments off a masked frame) keeps it to one data pass.
      3. `solve_batched_lr` at `l2=1.0` — sklearn's `C=1.0` on the SUMMED log-loss,
         intercept unpenalized — then fold back to raw-θ coordinates.

    `warm_start=(V, b_raw)` starts the solve from ANOTHER fit's raw-θ scoring
    params (`dag_gated_readout --readout-warm-from`): a re-readout whose problem is
    a perturbation of an earlier fit's (another split, another theta width). It travels in RAW coordinates by necessity: standardized params
    only mean something next to the `(mu, sd)` that produced them, and this fit's
    moments come from ITS rows, so the start is unfolded here through the moments
    computed in step 1. Two things it must not do, both handled below:

      - resurrect a node this fit masks. A degenerate node's data term is zeroed,
        leaving the bare ridge, whose gradient `l2*w` vanishes only at `w = 0` —
        a stale warm-start row would therefore make the solver walk it back to
        zero over real distributed passes, for a node whose probability is
        overwritten by the constant fallback anyway. Its `x0` row is zeroed, which
        restores the exact iteration-0 no-op the cold path has;
      - change the answer. It cannot: the per-node objective is convex, so the
        start moves the path and not the optimum. What it buys is the iterate at
        a CAPPED budget (`max_iter`), which is the budget these solves run under.

    `theta_topm > 0` fits on TOP-M TRUNCATED θ (see `readout.kernels`' module
    docstring). It changes the DESIGN MATRIX, not the procedure: the moments pass,
    every L-BFGS pass and — by the caller's obligation — the scoring pass all read
    the same truncated features, so what comes back is the exact readout of a
    narrower model rather than an approximation of the wide one. The mass that
    truncation drops is measured and logged BEFORE the fit (`theta_topm_coverage`),
    because the whole premise is an empirical claim about θ's concentration.

    `checkpoint_path` turns the solve into a RESUMABLE one, which at C and K in the
    thousands is the difference between losing 15 minutes and losing an afternoon
    (a solve can run for hours, and each pass is a cluster job). Every
    `checkpoint_every` iterations the progress hook writes the current standardized
    iterate — the solver's entire resumable state, since `x0` takes a POINT —
    atomically, next to a FINGERPRINT of `(C, K, theta_topm, n_obs, n_pos)` that a
    resume must match (`ckpt_fingerprint`: integer counts only, never the moments'
    float bits). On the next run a matching fingerprint is fed back
    as `x0` (with the degenerate rows re-zeroed, exactly as the `warm_start` path
    does, or the bare ridge would spend real distributed passes walking them back
    to zero); a mismatch warns and starts cold. What a resume does NOT carry is
    the L-BFGS curvature history, so its first iterations re-learn the scaling
    and are slower than the ones that preceded the crash — expected, logged, and
    still overwhelmingly cheaper than restarting at iteration 0 (a warm start
    supplies a point, not curvature). The file is deleted once the
    solve returns: the fit results are the record, and a checkpoint that outlives
    its solve is only a way to confuse a later run (which the fingerprint would
    catch, but a deleted file cannot even raise the question).

    When BOTH `warm_start` and a matching checkpoint are present the checkpoint
    wins: it is a point on THIS problem's own optimization path, strictly further
    along than another fit's answer mapped in.
    """
    C, K, theta_topm = int(data.C), int(data.K), int(data.topm)
    tag = f"{label}: " if label else ""
    if C * K * 8 > _COVERAGE_MIN_FIT_BYTES and data.can_measure_coverage:
        # Only measured where truncation is on the table. Below ~64 MB of (C,K)
        # parameters the dense fit is a couple of seconds a pass and the θ-width
        # lever is not a decision anyone is making, so the extra pass would be pure
        # cost. Above it, this is the number that says whether top-m is a cheap
        # reparameterization or a lobotomy — reported on the TRAIN frame, before the
        # moments pass, whether or not `theta_topm` is set, so the run of record
        # carries the evidence for (or against) the setting it used.
        cov = data.coverage()
        log.info(f"{tag}theta top-m mass: "
              + " ".join(f"m={m}:{mean:.3f}/{p10:.3f}"
                         for m, (mean, p10) in sorted(cov.items()))
              + " (mean/p10)")
    mu, sd, n_obs, n_pos = data.moments()
    # Same three cases as `_lr_proba_per_label_masked`'s `np.unique(yc).size < 2`:
    # nothing observed (-> 0.0), all-negative (-> 0.0), all-positive (-> 1.0).
    degenerate = (n_obs <= 0) | (n_pos <= 0) | (n_pos >= n_obs)
    const = np.where((n_obs > 0) & (n_pos >= n_obs), 1.0, 0.0)
    keep = ~degenerate
    x0 = None
    if warm_start is not None:
        W0, b0 = unfold_standardization(warm_start[0], warm_start[1], mu, sd)
        x0 = (np.where(keep[:, None], W0, 0.0), np.where(keep, b0, 0.0))
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = None
    if ckpt_path is not None:
        # Derived AFTER the moments pass because the counts come from it —
        # exact-integer sums, the reproducible identity of this arm's problem
        # (see ckpt_fingerprint for why mu/sd bytes are NOT hashed).
        fingerprint = ckpt_fingerprint(C, K, n_obs, n_pos, theta_topm)
        resumed = read_ckpt(ckpt_path, fingerprint)
        if resumed is not None:
            W_ck, b_ck, ck_iter = resumed
            # Same zeroing rule as the warm-start path above, for the same reason:
            # a degenerate node's data term is zeroed, so its objective is the bare
            # ridge and any nonzero row would be walked back to 0 over real
            # distributed passes. (A checkpoint written by THIS problem already has
            # them at zero — the fingerprint pins the degenerate mask — so this is
            # a belt-and-braces restatement of the contract, not a fixup.)
            x0 = (np.where(keep[:, None], W_ck, 0.0), np.where(keep, b_ck, 0.0))
            log.info(f"{tag}resuming batched solve from checkpoint "
                  f"(iter {ck_iter} recorded); curvature history is not carried, "
                  "early iterations re-learn it")
    log.info(f"{tag}readout fit: C={C} K={K}, "
          f"{int(keep.sum())} fittable nodes, {int(degenerate.sum())} degenerate "
          f"(constant fallback), observed train cells={int(n_obs.sum())}"
          f"{' (warm start)' if x0 is not None else ''}"
          f"{f' (theta top-m={theta_topm})' if theta_topm > 0 else ''}")

    with data.stats_fn(
            mu, sd, fold_standardization=fold_standardization,
            standardized_grad_from_raw=standardized_grad_from_raw) as stats_fn:

        def _fittable_stats(W_std, b_std, node_mask=None, _f=stats_fn, _keep=keep):
            # `node_mask` passes STRAIGHT THROUGH to the pass (the solver uses it
            # to skip nodes whose stats it already holds) and composes with the
            # degenerate zeroing without interacting with it: masking decides
            # which cells are READ, `keep` decides which rows are ZEROED, and a
            # zeroed row is zero either way. Dropping the argument here would
            # silently re-inflate every trial pass back to all C heads.
            loss, gW, gb = _f(W_std, b_std, node_mask=node_mask)
            return (np.where(_keep, loss, 0.0),
                    np.where(_keep[:, None], gW, 0.0),
                    np.where(_keep, gb, 0.0))

        # Heartbeat: each stats call is a full treeAggregate over the train
        # split, so a large-C solve is minutes/iteration and looks hung
        # without it. Log every iteration for the first few (the cold-start
        # steps are the slow, uncertain ones), then every 5th, then the finish
        # line from the summary print below.
        t0 = time.time()

        every = max(1, int(checkpoint_every))

        # Window state for the heartbeat: per-iteration RATES (s/iter, passes/
        # iter, newly converged) read the solve's health far better than the
        # cumulative totals alone — a line search going pathological shows up as
        # passes/iter climbing within two heartbeats instead of a slowly bending
        # cumulative curve. `_prev` holds the totals at the last PRINTED line.
        _prev = {"iter": 0, "passes": 0, "conv": 0, "t": t0}

        def _progress(p, _t0=t0, _tag=tag, _ndeg=int(degenerate.sum()),
                      _nfit=int(keep.sum()), _path=ckpt_path, _fp=fingerprint,
                      _every=every, _prev=_prev):
            if _path is not None and p["iter"] % _every == 0:
                # BEFORE the logging gate, not after it: the two cadences are
                # independent (log every 5th iteration, checkpoint every 10th by
                # default), and hanging the checkpoint off the log's early return
                # would silently tie them together. `W_std`/`b_std` are live views
                # into the solver's iterate — `write_ckpt` copies them
                # into the npz, which is the copy the contract requires.
                write_ckpt(_path, p["W_std"], p["b_std"], p["iter"], _fp)
            if p["iter"] > 3 and p["iter"] % 5:
                return
            # Degenerate nodes converge (grad exactly 0) at iteration 0; report
            # progress over the FITTABLE nodes so the fraction reads as work left.
            # `nodes/pass` is the cost that moves now that trial passes are masked
            # to the nodes still searching: a deep line search shows up as extra
            # passes over FEW nodes, which reads very differently from extra
            # passes over all C.
            now = time.time()
            conv = max(0, p["n_converged"] - _ndeg)
            d_iter = max(1, p["iter"] - _prev["iter"])
            d_pass = p["n_stats_calls"] - _prev["passes"]
            log.info(f"{_tag}batched L-BFGS iter {p['iter']}: "
                  f"{(now - _prev['t']) / d_iter:.1f}s/iter, "
                  f"{d_pass / d_iter:.1f} passes/iter "
                  f"(avg {p['n_node_evals'] / max(p['n_stats_calls'], 1):.0f} "
                  f"nodes/pass), "
                  f"{conv}/{_nfit} converged (+{conv - _prev['conv']}), "
                  f"{p['n_active']} active, max|grad|={p['max_grad_inf_norm']:.3g}, "
                  f"{p['n_stats_calls']} passes/{now - _t0:.0f}s total")
            _prev.update(iter=p["iter"], passes=p["n_stats_calls"], conv=conv,
                         t=now)

        W_std, b_std, info = solve_batched_lr(
            _fittable_stats, C, K, l2=l2, max_iter=max_iter, history=history,
            gtol=gtol, x0=x0, progress_fn=_progress)
    if ckpt_path is not None:
        # The solve landed, so `(V, b_raw)` — and the results computed from them —
        # are the record; the checkpoint's only remaining effect would be to warm
        # a LATER run from a point it did not earn. (The fingerprint would refuse a
        # genuinely different problem, but an identical re-run resumed from a
        # finished solve would silently report "resuming from iter N" for a fit
        # that is already done.) Best-effort: a failed unlink is not worth failing
        # a completed fit over.
        try:
            ckpt_path.unlink(missing_ok=True)
        except OSError as exc:                   # pragma: no cover - I/O failure
            log.info(f"{tag}could not remove readout checkpoint "
                  f"{ckpt_path.name} ({exc})")
    V, b_raw = fold_standardization(W_std, b_std, mu, sd)
    gmax = float(info["grad_inf_norm"][keep].max()) if keep.any() else 0.0
    # `converged` = gtol OR the principled numerical stall; at gtol=1e-4 (sklearn's
    # own tol) every node should stop on the gradient, so a nonzero stalled count is
    # the diagnostic that the summed-loss roundoff floor was hit first.
    log.info(f"{tag}batched L-BFGS: {int(info['n_stats_calls'])} data passes "
          f"({int(info['n_node_evals'])} node-passes, "
          f"{info['n_node_evals'] / max(int(info['n_stats_calls']), 1):.0f} avg), "
          f"{int(info['converged'][keep].sum())}/{int(keep.sum())} converged "
          f"({int(info['converged_gtol'][keep].sum())} gtol, "
          f"{int(info['stalled'][keep].sum())} stalled), max|grad|={gmax:.3g}, "
          f"max iters={int(info['n_iter'].max())}, "
          f"line-search failures={int(info['line_search_failures'])}")
    # The STANDARDIZED weights ride along for the heads sidecar: V = W_std/sd is
    # the scoring decoder but inflates on low-variance (starved) topics, so any
    # loadings/importance read off-cluster needs W_std (inspect_topics).
    info["W_std"] = np.asarray(W_std, dtype=np.float64)
    return V, b_raw, const, degenerate, info


