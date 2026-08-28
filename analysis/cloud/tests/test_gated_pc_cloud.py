"""Tests for the gated_pc cloud driver: the pure pc_topics_lr scorer, the
DAG-closure-parents densifier, the arg surface, and the distributed readout's
solver CHECKPOINT (which is driver-side numpy + a file, so it is unit-testable
with the Spark seam swapped for `batched_lr`'s in-memory reference). The
end-to-end BQ+fit run is the cluster smoke (main() reads the CDR via the
spark-bigquery connector)."""

import numpy as np
import pytest


def test_pc_topics_lr_bundle_separable_theta_scores_high():
    from gated_pc_cloud import pc_topics_lr_bundle
    # C=2 nodes; theta dim 0 predicts node 0, dim 1 predicts node 1 (separable).
    Pi_tr = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0],
                      [0.8, 0.2], [0.2, 0.8]])
    y_tr = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], float)
    Pi_te = np.array([[0.95, 0.05], [0.05, 0.95], [0.85, 0.15], [0.15, 0.85]])
    y_te = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    m_tr = np.ones_like(y_tr)
    m_te = np.ones_like(y_te)
    b = pc_topics_lr_bundle(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C=2)
    assert b["macro"]["auc"] == 1.0            # perfectly separable
    assert b["macro"]["n_labels_scored"] == 2


def test_pc_topics_lr_bundle_masks_unobserved_cells():
    from gated_pc_cloud import pc_topics_lr_bundle
    # Node 1 is entirely UNOBSERVED in test (mask all zero) -> skipped from macro.
    Pi_tr = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    y_tr = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    Pi_te = np.array([[0.9, 0.1], [0.1, 0.9]])
    y_te = np.array([[1, 0], [0, 1]], float)
    m_tr = np.ones_like(y_tr)
    m_te = np.array([[1, 0], [1, 0]], float)   # node 1 unobserved in test
    b = pc_topics_lr_bundle(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C=2)
    assert b["macro"]["n_labels_scored"] == 1  # only node 0 had observed test cells
    assert b["per_label"][1]["skipped"] is not None


def test_dag_closure_parents_densifies_parent_int_over_range_C():
    from gated_pc_cloud import dag_closure_parents
    # engine ids: 0 root; 1,2 -> 0; 3 -> {1,2} (diamond). C=4.
    parent_int = {1: [0], 2: [0], 3: [1, 2]}
    cp = dag_closure_parents(parent_int, C=4)
    assert cp == [[], [0], [0], [1, 2]]        # root (0) and gaps -> []
    assert len(cp) == 4


def test_dag_children_and_depth():
    from gated_pc_cloud import _dag_children_and_depth
    # 0 root; 1,2 -> 0; 3,4 -> 1. C=5.
    parent_int = {1: [0], 2: [0], 3: [1], 4: [1]}
    children, depth = _dag_children_and_depth(parent_int, C=5)
    assert children[0] == [1, 2] and sorted(children[1]) == [3, 4]
    assert children[3] == [] and children[4] == []
    assert depth == {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}


def test_conditional_readout_sharpens_within_parent_cohort():
    from gated_pc_cloud import conditional_readout
    # 0 root; 1,2 -> 0; 3,4 -> 1. Among node-1's cohort, node 3 vs 4 is perfectly
    # separable, so the conditional AUC and multiclass top-1 must be 1.0.
    parent_int = {1: [0], 2: [0], 3: [1], 4: [1]}
    C = 5
    rng = np.random.default_rng(0)
    rows = []
    # closure labels: a doc at leaf L is positive at L, its parent 1, and root 0.
    for _ in range(20):   # at node 3
        rows.append(([1, 1, 0, 1, 0], [0.1, 0.5, 0.1, 0.9, 0.1]))
    for _ in range(20):   # at node 4
        rows.append(([1, 1, 0, 0, 1], [0.1, 0.5, 0.1, 0.1, 0.9]))
    for _ in range(20):   # at node 2 (sibling of 1, outside 1's cohort)
        rows.append(([1, 0, 1, 0, 0], [0.1, 0.1, 0.5, 0.1, 0.1]))
    y = np.array([r[0] for r in rows], float)
    proba = np.array([r[1] for r in rows], float)
    mask = np.ones_like(y)
    cond = conditional_readout(proba, y, mask, parent_int, C, min_count=5)

    # parent-1 cohort has both children scored, each perfectly separable.
    e13 = [e for e in cond["edges"] if e["parent"] == 1 and e["child"] == 3]
    e14 = [e for e in cond["edges"] if e["parent"] == 1 and e["child"] == 4]
    assert e13 and e14
    assert e13[0]["cond_auc"] == 1.0 and e14[0]["cond_auc"] == 1.0
    assert e13[0]["depth"] == 1                       # parent 1 is at depth 1
    p1 = [p for p in cond["parents"] if p["parent"] == 1]
    assert p1 and p1[0]["top1"] == 1.0               # argmax child always correct
    # honesty fields: majority baseline (20/20 -> 0.5), balanced accuracy, ECE.
    assert p1[0]["majority"] == 0.5                   # 20 vs 20 children -> 0.5 baseline
    assert p1[0]["bal_acc"] == 1.0                    # perfect per-child recall
    assert cond["ece"] is not None and cond["ece"] >= 0.0
    # per-node reliability: each scored edge carries its OWN ECE, and the summary
    # reports mean/max/worst over them so pooling can't hide a miscalibrated node.
    assert e13[0]["ece"] is not None and e13[0]["ece"] >= 0.0
    ne = cond["node_ece"]
    assert ne is not None and ne["n_nodes"] == len(cond["edges"])
    assert ne["max"] >= ne["mean"] >= 0.0
    assert ne["max"] == max(e["ece"] for e in cond["edges"])


def test_ece_perfectly_calibrated_is_zero_and_miscalibrated_is_positive():
    from gated_pc_cloud import _ece
    import numpy as np
    rng = np.random.default_rng(0)
    # p == empirical frequency in each bin -> ECE ~ 0.
    p = np.repeat([0.05, 0.25, 0.55, 0.85], 400)
    y = np.concatenate([(rng.random(400) < q).astype(float)
                        for q in (0.05, 0.25, 0.55, 0.85)])
    assert _ece(y, p) < 0.05
    # confident-but-wrong -> large ECE.
    assert _ece(np.zeros(100), np.full(100, 0.9)) > 0.5


def test_calibrate_per_node_fixes_ece_and_preserves_ranking():
    from gated_pc_cloud import calibrate_per_node, _ece
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    C = 1
    n = 4000
    # true P = sigmoid of a signal; miscalibrate by squaring the score (monotone,
    # so ranking is preserved but reliability is wrong).
    signal = rng.standard_normal(n)
    true_p = 1.0 / (1.0 + np.exp(-signal))
    y = (rng.random(n) < true_p).astype(float)
    bad = true_p ** 2                                    # miscalibrated, same ranking
    # split train/test
    tr = np.zeros(n, bool); tr[: n // 2] = True; te = ~tr
    proba_tr = bad[:, None]; proba_te = bad[:, None]
    yv = y[:, None]; mask = np.ones((n, 1))
    cal = calibrate_per_node(proba_tr[tr], yv[tr], mask[tr], proba_te[te], C)
    # calibration lowers ECE...
    assert _ece(yv[te, 0], cal[:, 0]) < _ece(yv[te, 0], proba_te[te, 0])
    # ...and approximately preserves the ranking (isotonic is monotone; only the
    # flat regions it induces create ties that move AUC slightly).
    assert abs(roc_auc_score(yv[te, 0], cal[:, 0])
               - roc_auc_score(yv[te, 0], proba_te[te, 0])) < 0.01


def test_precision_at_recall_and_recall_at_fdr():
    from gated_pc_cloud import precision_at_recall, recall_at_fdr
    # perfectly separable: at any recall, precision 1.0; at any FDR, recall 1.0.
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert precision_at_recall(y, p, [0.5, 1.0]) == {0.5: 1.0, 1.0: 1.0}
    assert recall_at_fdr(y, p, [0.0, 0.5]) == {0.0: 1.0, 0.5: 1.0}
    # one hard positive ranked below a negative: recall 1.0 now costs some precision.
    p2 = [0.1, 0.85, 0.8, 0.9]        # neg[1]=0.85 outranks pos[2]=0.8
    par = precision_at_recall(y, p2, [1.0])
    assert par[1.0] < 1.0             # to catch both positives we admit the negative
    # degenerate single-class column -> NaN, not an exception.
    import math
    assert math.isnan(precision_at_recall([1, 1, 1], [0.2, 0.5, 0.9], [0.5])[0.5])


def test_score_arm_separable_and_detection_shape():
    import numpy as np
    from gated_pc_cloud import score_arm
    Pi_tr = np.array([[1., 0], [.9, .1], [.1, .9], [0, 1.], [.8, .2], [.2, .8]])
    y_tr = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], float)
    Pi_te = np.array([[.95, .05], [.05, .95], [.85, .15], [.15, .85]])
    y_te = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    arm = score_arm(Pi_tr, y_tr, np.ones_like(y_tr), Pi_te, y_te, np.ones_like(y_te),
                    2, recall_targets=[0.5, 0.9], fdr_targets=[0.1, 0.5])
    assert arm["ranking"]["auc"] == 1.0
    assert set(arm["pr"]["par"]) == {0.5, 0.9}
    assert set(arm["pr"]["raf"]) == {0.1, 0.5}
    assert arm["pr"]["n_scored"] == 2
    # detection present with an AP + a precision@recall map (root=node0 label).
    assert "ap" in arm["detection"] and set(arm["detection"]["par"]) == {0.5, 0.9}


def test_parse_args_surface_defaults_and_arms():
    from gated_pc_cloud import parse_args
    a = parse_args([
        "--cdr", "p.d", "--billing", "bp", "--disease", "rare6",
        "--n-bg", "2", "--tpn", "1", "--weight-y", "80", "--out-dir", "/tmp/g",
        "--with-dag-head",
    ])
    assert a.disease == "rare6" and a.weight_y == 80.0
    assert a.head_optimizer == "newton"        # settled convergent head default
    assert a.head_l2 == 1e-3                    # absolute ridge (ADR 0041)
    assert a.label_mask_mode == "full"         # full-observation default
    assert a.with_dag_head is True
    assert a.skip_unsup_gated is False
    assert a.weight_y_warmup_iters == 10
    assert a.eval_every == 0                   # per-iter eval off by default


def test_head_l2_ridge_round_trips_through_estimator_and_engine():
    """--head-l2 flow: parse_args -> _build_pc_estimator (shim Params) ->
    OnlinePCLDA engine (head_l2 attribute). The ridge is the sole head regularizer
    (ADR 0043: Firth + Path B removed)."""
    from gated_pc_cloud import parse_args, _build_pc_estimator
    from spark_vi.mllib.topic.pc import _build_model_and_config
    a = parse_args([
        "--cdr", "p.d", "--billing", "bp", "--out-dir", "/tmp/g",
        "--head-optimizer", "newton", "--head-l2", "0.01", "--k", "4",
    ])
    assert a.head_l2 == 0.01 and a.head_optimizer == "newton"
    a._C = 1
    est = _build_pc_estimator(a, weight_y=50.0, gated=False)
    assert est.getOrDefault("headL2") == 0.01
    model, _ = _build_model_and_config(est, vocab_size=8)
    assert model.head_l2 == 0.01               # the shim built a ridge-only engine
    assert model.head_optimizer == "newton"


# --------------------------------------------------------------------------- #
# Readout solver checkpoint + fingerprinted warm resume (exp 0104, 08-28).     #
#                                                                             #
# The failure this answers: a whole-Mondo readout solve is hours of distributed #
# L-BFGS, and on 2026-08-28 a spot-preemption wave starved the scheduler until  #
# Spark aborted the job 9,112s in, taking every computed iterate with it. The   #
# checkpoint makes that cost `checkpoint_every` iterations instead. These tests #
# drive the REAL `_fit_readout_heads` with the Spark seam swapped for the       #
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


class _FakeDR:
    """Stand-in for `distributed_readout`: the same call surface, numpy behind it.

    `fail_at` raises on the n-th stats call, which is what a preemption-wave job
    abort looks like from `_fit_readout_heads`'s point of view (a Py4JJavaError
    escaping the treeAggregate after the retries are exhausted).
    """

    def __init__(self, Pi, y, obs, fail_at=None):
        self.Pi, self.y, self.obs = Pi, y, obs
        self.fail_at = fail_at
        self.n_calls = 0

    def masked_moments(self, train_scored, C, K, **kw):
        from analysis.pc.batched_lr import standardization_moments
        mu, sd, n_obs = standardization_moments(self.Pi, self.obs.astype(bool))
        return mu, sd, n_obs, (self.y * self.obs).sum(axis=0)

    def make_spark_stats_fn(self, train_scored, C, K, mu, sd, **kw):
        from contextlib import nullcontext

        from analysis.pc.batched_lr import make_inmemory_stats_fn
        inner = make_inmemory_stats_fn(self.Pi, self.y, self.obs.astype(bool),
                                       mu, sd)

        def _stats(W_std, b_std, node_mask=None):
            self.n_calls += 1
            if self.fail_at is not None and self.n_calls >= self.fail_at:
                raise RuntimeError("job aborted: excludeOnFailure")
            return inner(W_std, b_std, node_mask=node_mask)

        return nullcontext(_stats)


def test_readout_ckpt_fingerprint_pins_the_problem_not_just_the_shapes():
    """The fingerprint's job is to refuse a checkpoint from a DIFFERENT arm or
    corpus, which would otherwise deserialize cleanly (right shapes, wrong
    meaning) and warm-start a solve from a meaningless point. Every input that
    changes the standardized basis or the degenerate mask must change it."""
    from gated_pc_cloud import _readout_ckpt_fingerprint
    C, K = 3, 4
    mu = np.arange(C * K, dtype=float).reshape(C, K)
    sd = np.ones((C, K))
    n_obs, n_pos = np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])
    base = _readout_ckpt_fingerprint(C, K, mu, sd, n_obs, n_pos)
    assert base == _readout_ckpt_fingerprint(C, K, mu.copy(), sd, n_obs, n_pos)
    assert len(base) == 64                        # sha256 hex
    for kw in ("mu", "sd", "n_obs", "n_pos"):
        arrs = {"mu": mu.copy(), "sd": sd.copy(),
                "n_obs": n_obs.copy(), "n_pos": n_pos.copy()}
        arrs[kw].flat[0] += 1e-9                  # a moments pass is deterministic
        assert _readout_ckpt_fingerprint(C, K, **arrs) != base, kw
    assert _readout_ckpt_fingerprint(C + 1, K, mu, sd, n_obs, n_pos) != base
    assert _readout_ckpt_fingerprint(C, K + 1, mu, sd, n_obs, n_pos) != base


def test_readout_ckpt_write_read_round_trip_and_rejections(tmp_path, capsys):
    """Round trip, plus the three ways a read must decline: absent, torn, and
    fingerprint mismatch. A mismatch is LOUD (it means a stale file from another
    run is sitting in the run dir) and does NOT delete the file — the fresh
    solve's own first checkpoint overwrites it."""
    from gated_pc_cloud import _read_readout_ckpt, _write_readout_ckpt
    path = tmp_path / "readout_ckpt_gated_pc.npz"
    W = np.arange(12.0).reshape(3, 4)
    b = np.array([0.5, -1.5, 2.0])

    assert _read_readout_ckpt(path, "fp") is None          # nothing there yet
    assert _write_readout_ckpt(path, W, b, 40, "fp") is True
    assert not (tmp_path / (path.name + ".tmp")).exists()   # tmp renamed away
    got = _read_readout_ckpt(path, "fp")
    assert got is not None
    W2, b2, it = got
    assert np.array_equal(W2, W) and np.array_equal(b2, b) and it == 40

    assert _read_readout_ckpt(path, "other-fingerprint") is None
    out = capsys.readouterr().out
    assert "IGNORED" in out and "Starting cold" in out
    assert path.exists()                                    # not deleted

    path.write_bytes(b"not an npz")                         # torn / truncated
    assert _read_readout_ckpt(path, "fp") is None
    assert "UNREADABLE" in capsys.readouterr().out


def test_fit_readout_heads_checkpoint_survives_a_mid_solve_death(tmp_path, capsys,
                                                                monkeypatch):
    """The end-to-end claim: a solve that dies mid-flight leaves a checkpoint the
    next run resumes from, and the resumed run finishes at the same answer an
    uninterrupted one does (the per-node objective is convex, so `x0` moves the
    path and not the optimum). And once the solve lands, the checkpoint is gone —
    the fit is the record, and a checkpoint outliving its solve can only mislead
    a later run."""
    import gated_pc_cloud as gpc
    Pi, y, obs = _ckpt_problem()
    C, K = 6, 4
    path = tmp_path / "readout_ckpt_gated_pc.npz"

    # 1. Uninterrupted reference run, no checkpointing at all.
    monkeypatch.setattr(gpc, "_dr", _FakeDR(Pi, y, obs))
    V_ref, b_ref, _, degen, _ = gpc._fit_readout_heads(
        None, C, K, gtol=1e-6, max_iter=200, label="gated_pc")

    # 2. The same solve, killed partway through.
    monkeypatch.setattr(gpc, "_dr", _FakeDR(Pi, y, obs, fail_at=12))
    with pytest.raises(RuntimeError, match="excludeOnFailure"):
        gpc._fit_readout_heads(None, C, K, gtol=1e-6, max_iter=200,
                               label="gated_pc", checkpoint_path=path,
                               checkpoint_every=1)
    assert path.exists(), "the death left nothing to resume from"
    with np.load(path) as z:
        assert z["W_std"].shape == (C, K) and z["b_std"].shape == (C,)
        assert int(z["iter"]) >= 1
        recorded_fp = str(z["fingerprint"].item())

    # The stored fingerprint is the one THIS problem's moments produce.
    from analysis.pc.batched_lr import standardization_moments
    mu, sd, n_obs = standardization_moments(Pi, obs.astype(bool))
    assert recorded_fp == gpc._readout_ckpt_fingerprint(
        C, K, mu, sd, n_obs, (y * obs).sum(axis=0))

    # 3. Resume: same answer, and it says so in the log (curvature is NOT carried,
    #    so the first iterations re-learn it — expected, not a regression).
    capsys.readouterr()
    monkeypatch.setattr(gpc, "_dr", _FakeDR(Pi, y, obs))
    V, b_raw, _, _, info = gpc._fit_readout_heads(
        None, C, K, gtol=1e-6, max_iter=200, label="gated_pc",
        checkpoint_path=path, checkpoint_every=1)
    out = capsys.readouterr().out
    assert "resuming batched solve from checkpoint" in out
    assert "curvature history is not carried" in out
    assert info["warm_started"] is True
    assert np.abs(V - V_ref).max() < 1e-4
    assert np.abs(b_raw - b_ref).max() < 1e-4
    assert not path.exists(), "a completed solve must not leave a checkpoint"
    assert np.all(V[degen] == 0.0)


def test_fit_readout_heads_resume_zeroes_degenerate_rows(tmp_path, monkeypatch):
    """A degenerate node's data term is zeroed, so its objective is the bare ridge
    — stationary only at 0. A resumed `x0` row for such a node would therefore be
    walked back to zero over REAL distributed passes, for a node whose probability
    is overwritten by the constant fallback anyway. Same rule the `warm_start`
    path has, restated for the checkpoint path."""
    import gated_pc_cloud as gpc
    from analysis.pc import batched_lr
    Pi, y, obs = _ckpt_problem(seed=3)
    C, K = 6, 4
    path = tmp_path / "readout_ckpt_gated_pc.npz"
    mu, sd, n_obs = batched_lr.standardization_moments(Pi, obs.astype(bool))
    fp = gpc._readout_ckpt_fingerprint(C, K, mu, sd, n_obs, (y * obs).sum(axis=0))
    # A checkpoint whose rows are ALL nonzero, including the degenerate node's.
    gpc._write_readout_ckpt(path, np.ones((C, K)), np.ones(C), 17, fp)

    seen = {}

    def _spy(stats_fn, C_, K_, **kw):
        seen["x0"] = kw.get("x0")
        seen["existed_during_solve"] = path.exists()
        return (np.zeros((C_, K_)), np.zeros(C_), _fake_info(C_, K_))

    monkeypatch.setattr(gpc, "_dr", _FakeDR(Pi, y, obs))
    monkeypatch.setattr(batched_lr, "solve_batched_lr", _spy)
    _, _, _, degen, _ = gpc._fit_readout_heads(
        None, C, K, label="gated_pc", checkpoint_path=path, checkpoint_every=1)
    W0, b0 = seen["x0"]
    assert np.all(W0[degen] == 0.0) and np.all(b0[degen] == 0.0)
    assert np.all(W0[~degen] == 1.0) and np.all(b0[~degen] == 1.0)


def test_fit_readout_heads_mismatched_fingerprint_starts_cold(tmp_path, capsys,
                                                              monkeypatch):
    """A checkpoint from another arm/corpus deserializes fine and means nothing.
    It must WARN and be ignored — never silently used, and never deleted on the
    spot: until the fresh solve writes its own, the stale file on disk is the
    evidence for what went wrong."""
    import gated_pc_cloud as gpc
    from analysis.pc import batched_lr
    Pi, y, obs = _ckpt_problem(seed=5)
    C, K = 6, 4
    path = tmp_path / "readout_ckpt_gated_pc.npz"
    gpc._write_readout_ckpt(path, np.ones((C, K)), np.ones(C), 17, "a-different-run")

    seen = {}

    def _spy(stats_fn, C_, K_, **kw):
        seen["x0"] = kw.get("x0")
        seen["existed_during_solve"] = path.exists()
        return (np.zeros((C_, K_)), np.zeros(C_), _fake_info(C_, K_))

    monkeypatch.setattr(gpc, "_dr", _FakeDR(Pi, y, obs))
    monkeypatch.setattr(batched_lr, "solve_batched_lr", _spy)
    gpc._fit_readout_heads(None, C, K, label="gated_pc", checkpoint_path=path,
                           checkpoint_every=1)
    out = capsys.readouterr().out
    assert "IGNORED" in out and "different arm/corpus/basis" in out
    assert seen["x0"] is None                     # cold start, not the stale point
    assert seen["existed_during_solve"] is True   # not deleted on mismatch


def _fake_info(C, K):
    """The `info` dict `_fit_readout_heads`'s summary print reads."""
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
