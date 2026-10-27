"""Resumable checkpoints and heads persistence of `spark_vi.readout.heads`.

`fit_heads` is driven for real (the solver, the degenerate masking, the progress
hook, the fold) over a fake data backend whose stats come from the solver's
in-memory reference (`solver.make_inmemory_stats_fn`) and which can be told to die
mid-solve, which is what a preemption-wave job abort looks like from `fit_heads`.
Moved from `analysis/cloud/tests/test_gated_pc_cloud.py` with ADR 0051.
"""
import logging
from contextlib import nullcontext

import numpy as np
import pytest

from spark_vi.readout import heads
from spark_vi.readout import solver as heads_solver


@pytest.fixture(autouse=True)
def _info_logs(caplog):
    caplog.set_level(logging.INFO, logger="spark_vi.readout")


class _FakeData:
    """The `fit_heads` data interface over numpy, optionally failing on the n-th
    stats call."""

    topm = 0
    can_measure_coverage = False

    def __init__(self, Pi, y, obs, fail_at=None):
        self.Pi, self.y, self.obs = Pi, y, obs
        self.C, self.K = y.shape[1], Pi.shape[1]
        self.fail_at = fail_at
        self.n_calls = 0

    def moments(self):
        from spark_vi.readout.solver import standardization_moments
        mu, sd, n_obs = standardization_moments(self.Pi, self.obs.astype(bool))
        return mu, sd, n_obs, (self.y * self.obs).sum(axis=0)

    def stats_fn(self, mu, sd, **kw):
        from spark_vi.readout.solver import make_inmemory_stats_fn
        inner = make_inmemory_stats_fn(self.Pi, self.y, self.obs.astype(bool), mu, sd)

        def _stats(W_std, b_std, node_mask=None):
            self.n_calls += 1
            if self.fail_at is not None and self.n_calls >= self.fail_at:
                raise RuntimeError("job aborted: excludeOnFailure")
            return inner(W_std, b_std, node_mask=node_mask)

        return nullcontext(_stats)


# --------------------------------------------------------------------------- #
# Readout solver checkpoint + fingerprinted warm resume (exp 0104, 08-28).     #
#                                                                             #
# The failure this answers: a whole-Mondo readout solve is hours of distributed #
# L-BFGS, and on 2026-08-28 a spot-preemption wave starved the scheduler until  #
# Spark aborted the job 9,112s in, taking every computed iterate with it. The   #
# checkpoint makes that cost `checkpoint_every` iterations instead. These tests #
# drive the REAL `fit_heads` with the Spark seam swapped for the       #
# in-memory reference (`batched_lr.make_inmemory_stats_fn`) — the solver, the   #
# degenerate masking, the progress hook and the fold are all the production     #
# ones, only the treeAggregate is replaced.                                    #
# --------------------------------------------------------------------------- #
def _ckpt_problem(seed=0, C=6, K=4, D=80):
    """A tiny readout problem with one DEGENERATE (all-negative) node."""
    rng = np.random.default_rng(seed)
    Pi = rng.dirichlet(np.full(K, 0.5), size=D)
    W = rng.normal(size=(C, K)) * 3.0
    p = 1.0 / (1.0 + np.exp(-(Pi @ W.T)))
    y = (rng.random((D, C)) < p).astype(float)
    obs = (rng.random((D, C)) < 0.8).astype(float)
    y[:, C - 1] = 0.0                            # never positive => degenerate
    return Pi, y, obs


def test_ckpt_fingerprint_pins_the_problem_and_survives_float_noise():
    """The fingerprint must refuse a checkpoint from a DIFFERENT arm or corpus
    (labels change n_pos; a different corpus/split changes n_obs; C/K/top-m are
    the shape of the design matrix) — and it must NOT depend on anything a
    parallel float reduction perturbs. v1 hashed mu/sd bytes and every cross-run
    resume mismatched on treeAggregate combine-order noise (exp 0104, 08-29:
    an identical solve's iter-50 checkpoint was rejected). Counts are integer
    sums — exact in float64 in any order — so they, and only they, are hashed."""
    C, K = 3, 4
    n_obs, n_pos = np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])
    base = heads.ckpt_fingerprint(C, K, n_obs, n_pos)
    assert base == heads.ckpt_fingerprint(C, K, n_obs.copy(), n_pos.copy())
    assert len(base) == 64                        # sha256 hex
    # Sub-integer float noise (a count can only differ by whole units) is
    # rounded away — the property v1 lacked.
    assert base == heads.ckpt_fingerprint(C, K, n_obs + 1e-9, n_pos - 1e-9)
    # Whole-unit changes ARE a different problem.
    assert heads.ckpt_fingerprint(C, K, n_obs + 1.0, n_pos) != base
    assert heads.ckpt_fingerprint(C, K, n_obs, n_pos + 1.0) != base
    assert heads.ckpt_fingerprint(C + 1, K, n_obs, n_pos) != base
    assert heads.ckpt_fingerprint(C, K + 1, n_obs, n_pos) != base
    assert heads.ckpt_fingerprint(C, K, n_obs, n_pos, theta_topm=256) != base


def test_readout_ckpt_write_read_round_trip_and_rejections(tmp_path, caplog):
    """Round trip, plus the three ways a read must decline: absent, torn, and
    fingerprint mismatch. A mismatch is LOUD (it means a stale file from another
    run is sitting in the run dir) and does NOT delete the file — the fresh
    solve's own first checkpoint overwrites it."""
    path = tmp_path / "readout_ckpt_gated_pc.npz"
    W = np.arange(12.0).reshape(3, 4)
    b = np.array([0.5, -1.5, 2.0])

    assert heads.read_ckpt(path, "fp") is None          # nothing there yet
    assert heads.write_ckpt(path, W, b, 40, "fp") is True
    assert not (tmp_path / (path.name + ".tmp")).exists()   # tmp renamed away
    got = heads.read_ckpt(path, "fp")
    assert got is not None
    W2, b2, it = got
    assert np.array_equal(W2, W) and np.array_equal(b2, b) and it == 40

    assert heads.read_ckpt(path, "other-fingerprint") is None
    out = caplog.text
    assert "IGNORED" in out and "Starting cold" in out
    assert path.exists()                                    # not deleted

    path.write_bytes(b"not an npz")                         # torn / truncated
    assert heads.read_ckpt(path, "fp") is None
    assert "UNREADABLE" in caplog.text


def test_readout_heads_write_read_round_trip_and_rejections(tmp_path, caplog):
    """The PERSISTED heads (Part 1): a completed fit's raw-θ scoring params,
    written beside the run's record so conversion_analysis --deciles on scores
    from them instead of re-fitting. Round trip returns the exact scoring tuple;
    a read declines on absent, version mismatch, and a manifest C/K/theta_topm
    that disagrees with the file — every miss is None so the consumer degrades to
    the overall table rather than re-fitting."""
    C, K = 3, 4
    V = np.arange(12.0).reshape(C, K)
    b_raw = np.array([0.5, -1.5, 2.0])
    const = np.array([0.0, 1.0, 0.0])
    degenerate = np.array([False, True, False])

    assert heads.read_heads(tmp_path, "gated_pc") is None        # absent
    assert heads.write_heads(tmp_path, "gated_pc", V, b_raw, const,
                                degenerate, C, K, theta_topm=0) is True
    assert (tmp_path / "readout_heads_gated_pc.npz").exists()
    assert not (tmp_path / "readout_heads_gated_pc.npz.tmp").exists()  # renamed away

    got = heads.read_heads(tmp_path, "gated_pc", C=C, K=K, theta_topm=0)
    assert got is not None
    gV, gb, gconst, gdeg, gC, gK, gm = got
    assert np.array_equal(gV, V) and np.array_equal(gb, b_raw)
    assert np.array_equal(gconst, const) and np.array_equal(gdeg, degenerate)
    assert (gC, gK, gm) == (C, K, 0)

    # A manifest whose C/K/theta_topm disagrees with the file is a different run.
    assert heads.read_heads(tmp_path, "gated_pc", C=C, K=K, theta_topm=64) is None
    assert heads.read_heads(tmp_path, "gated_pc", C=99, K=K, theta_topm=0) is None
    assert "IGNORED" in caplog.text
    # K unknown in an old manifest (K=None) skips only that check; still loads.
    assert heads.read_heads(tmp_path, "gated_pc", C=C, K=None,
                               theta_topm=0) is not None

    (tmp_path / "readout_heads_gated_pc.npz").write_bytes(b"not an npz")
    assert heads.read_heads(tmp_path, "gated_pc") is None
    assert "UNREADABLE" in caplog.text


def test_fit_heads_checkpoint_survives_a_mid_solve_death(tmp_path, caplog,
                                                                monkeypatch):
    """The end-to-end claim: a solve that dies mid-flight leaves a checkpoint the
    next run resumes from, and the resumed run finishes at the same answer an
    uninterrupted one does (the per-node objective is convex, so `x0` moves the
    path and not the optimum). And once the solve lands, the checkpoint is gone —
    the fit is the record, and a checkpoint outliving its solve can only mislead
    a later run."""
    Pi, y, obs = _ckpt_problem()
    C, K = 6, 4
    path = tmp_path / "readout_ckpt_gated_pc.npz"

    # 1. Uninterrupted reference run, no checkpointing at all.
    data = _FakeData(Pi, y, obs)
    V_ref, b_ref, _, degen, _ = heads.fit_heads(data, gtol=1e-6, max_iter=200, label="gated_pc")

    # 2. The same solve, killed partway through.
    data = _FakeData(Pi, y, obs, fail_at=12)
    with pytest.raises(RuntimeError, match="excludeOnFailure"):
        heads.fit_heads(data, gtol=1e-6, max_iter=200,
                               label="gated_pc", checkpoint_path=path,
                               checkpoint_every=1)
    assert path.exists(), "the death left nothing to resume from"
    with np.load(path) as z:
        assert z["W_std"].shape == (C, K) and z["b_std"].shape == (C,)
        assert int(z["iter"]) >= 1
        recorded_fp = str(z["fingerprint"].item())

    # The stored fingerprint is the one THIS problem's counts produce.
    from spark_vi.readout.solver import standardization_moments
    _mu, _sd, n_obs = standardization_moments(Pi, obs.astype(bool))
    assert recorded_fp == heads.ckpt_fingerprint(
        C, K, n_obs, (y * obs).sum(axis=0), 0)

    # 3. Resume: same answer, and it says so in the log (curvature is NOT carried,
    #    so the first iterations re-learn it — expected, not a regression).
    caplog.clear()
    data = _FakeData(Pi, y, obs)
    V, b_raw, _, _, info = heads.fit_heads(data, gtol=1e-6, max_iter=200, label="gated_pc",
        checkpoint_path=path, checkpoint_every=1)
    out = caplog.text
    assert "resuming batched solve from checkpoint" in out
    assert "curvature history is not carried" in out
    assert info["warm_started"] is True
    assert np.abs(V - V_ref).max() < 1e-4
    assert np.abs(b_raw - b_ref).max() < 1e-4
    assert not path.exists(), "a completed solve must not leave a checkpoint"
    assert np.all(V[degen] == 0.0)


def test_fit_heads_resume_zeroes_degenerate_rows(tmp_path, monkeypatch):
    """A degenerate node's data term is zeroed, so its objective is the bare ridge
    — stationary only at 0. A resumed `x0` row for such a node would therefore be
    walked back to zero over REAL distributed passes, for a node whose probability
    is overwritten by the constant fallback anyway. Same rule the `warm_start`
    path has, restated for the checkpoint path."""
    Pi, y, obs = _ckpt_problem(seed=3)
    C, K = 6, 4
    path = tmp_path / "readout_ckpt_gated_pc.npz"
    _mu, _sd, n_obs = heads_solver.standardization_moments(Pi, obs.astype(bool))
    fp = heads.ckpt_fingerprint(C, K, n_obs, (y * obs).sum(axis=0), 0)
    # A checkpoint whose rows are ALL nonzero, including the degenerate node's.
    heads.write_ckpt(path, np.ones((C, K)), np.ones(C), 17, fp)

    seen = {}

    def _spy(stats_fn, C_, K_, **kw):
        seen["x0"] = kw.get("x0")
        seen["existed_during_solve"] = path.exists()
        return (np.zeros((C_, K_)), np.zeros(C_), _fake_info(C_, K_))

    data = _FakeData(Pi, y, obs)
    monkeypatch.setattr(heads, "solve_batched_lr", _spy)
    _, _, _, degen, _ = heads.fit_heads(data, label="gated_pc", checkpoint_path=path, checkpoint_every=1)
    W0, b0 = seen["x0"]
    assert np.all(W0[degen] == 0.0) and np.all(b0[degen] == 0.0)
    assert np.all(W0[~degen] == 1.0) and np.all(b0[~degen] == 1.0)


def test_fit_heads_mismatched_fingerprint_starts_cold(tmp_path, caplog,
                                                              monkeypatch):
    """A checkpoint from another arm/corpus deserializes fine and means nothing.
    It must WARN and be ignored — never silently used, and never deleted on the
    spot: until the fresh solve writes its own, the stale file on disk is the
    evidence for what went wrong."""
    Pi, y, obs = _ckpt_problem(seed=5)
    C, K = 6, 4
    path = tmp_path / "readout_ckpt_gated_pc.npz"
    heads.write_ckpt(path, np.ones((C, K)), np.ones(C), 17, "a-different-run")

    seen = {}

    def _spy(stats_fn, C_, K_, **kw):
        seen["x0"] = kw.get("x0")
        seen["existed_during_solve"] = path.exists()
        return (np.zeros((C_, K_)), np.zeros(C_), _fake_info(C_, K_))

    data = _FakeData(Pi, y, obs)
    monkeypatch.setattr(heads, "solve_batched_lr", _spy)
    heads.fit_heads(data, label="gated_pc", checkpoint_path=path,
                           checkpoint_every=1)
    out = caplog.text
    assert "IGNORED" in out and "different arm/corpus/basis" in out
    assert seen["x0"] is None                     # cold start, not the stale point
    assert seen["existed_during_solve"] is True   # not deleted on mismatch


def _fake_info(C, K):
    """The `info` dict `fit_heads`'s summary print reads."""
    return {
        "n_iter": np.zeros(C, dtype=int),
        "converged": np.ones(C, dtype=bool),
        "converged_gtol": np.ones(C, dtype=bool),
        "stalled": np.zeros(C, dtype=bool),
        "grad_inf_norm": np.zeros(C),
        "n_stats_calls": 1,
        "n_node_evals": C,
        "line_search_failures": 0,
        "loss": np.zeros(C),
        "warm_started": False,
    }




def test_readout_heads_sidecar_carries_w_std_when_given(tmp_path):
    """The heads sidecar stores the solve's STANDARDIZED weights beside V when
    the caller passes them (inspect_topics reads them as the honest loadings
    scale once the solver checkpoint is gone), and stays readable without."""
    C, K = 3, 5
    V = np.arange(C * K, dtype=float).reshape(C, K)
    W = V / 7.0
    ok = heads.write_heads(tmp_path, "gated_pc", V, np.zeros(C),
                                  np.zeros((C, K), dtype=bool),
                                  np.zeros(C, dtype=bool), C, K, 0, W_std=W)
    assert ok
    z = np.load(tmp_path / "readout_heads_gated_pc.npz")
    assert "W_std" in z.files and np.array_equal(z["W_std"], W)
    assert np.array_equal(z["V"], V)
    ok = heads.write_heads(tmp_path, "other", V, np.zeros(C),
                                  np.zeros((C, K), dtype=bool),
                                  np.zeros(C, dtype=bool), C, K, 0)
    assert ok and "W_std" not in np.load(tmp_path / "readout_heads_other.npz").files


def test_feature_mask_solve_is_an_exact_reduced_solve():
    """`apply_feature_mask` on the batched solver: masked coordinates stay
    EXACTLY 0 through the whole L-BFGS path, and the kept coordinates land on
    the optimum of the reduced problem (a solve over the kept features only)."""
    from spark_vi.readout.solver import solve_batched_lr
    rng = np.random.default_rng(3)
    C, K, n = 2, 5, 400
    X = rng.normal(size=(n, K))
    w_true = np.array([[1.5, -1.0, 0.0, 0.8, 0.0], [0.0, 2.0, -1.2, 0.0, 0.5]])
    Y = np.stack([(rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ w_true[c] + 0.2))))
                  for c in range(C)], axis=1).astype(float)

    def stats_fn(W, b, node_mask=None):
        z = X @ W.T + b                                   # (n, C)
        p = 1 / (1 + np.exp(-z))
        loss = -(Y * np.log(p + 1e-300) + (1 - Y) * np.log(1 - p + 1e-300)).sum(0)
        gW = (p - Y).T @ X                                # (C, K)
        gb = (p - Y).sum(0)
        return loss, gW, gb

    mask = np.ones((C, K), dtype=bool)
    mask[0, [1, 4]] = False                               # node 0 loses 2 features
    mask[1, [0, 2, 3]] = False                            # node 1 keeps 2
    Wm, bm, im = solve_batched_lr(heads.apply_feature_mask(stats_fn, mask), C, K,
                                  l2=1.0, max_iter=300, gtol=1e-8)
    assert np.all(Wm[~mask] == 0.0)                       # exactly, not approximately
    # reduced problem per node: solve on the kept columns only, compare
    for c in range(C):
        keep = np.flatnonzero(mask[c])
        Xr = X[:, keep]

        def sf_r(W, b, node_mask=None, _Xr=Xr, _y=Y[:, c]):
            z = _Xr @ W[0] + b[0]
            p = 1 / (1 + np.exp(-z))
            loss = -(_y * np.log(p + 1e-300) + (1 - _y) * np.log(1 - p + 1e-300)).sum()
            return (np.array([loss]), ((p - _y) @ _Xr)[None, :],
                    np.array([(p - _y).sum()]))
        Wr, br, _ = solve_batched_lr(sf_r, 1, len(keep), l2=1.0, max_iter=300,
                                     gtol=1e-8)
        assert np.allclose(Wm[c, keep], Wr[0], atol=1e-4), c
        assert abs(bm[c] - br[0]) < 1e-4


