"""Unsupervised-warm-start (Hughes et al.) for the VI-native PC path.

The warm-start protocol seeds a supervised PC fit from the TOPICS of an
unsupervised fit. Our analogue of Hughes' Gibbs-LDA warm-start is a two-phase SVI
fit through the SAME ``OnlinePCLDA`` machinery:

  * PHASE 1 (``weight_y == 0``) — unsupervised LDA-MAP; learns topics (lambda),
    leaves the logistic head at its zero init.
  * PHASE 2 (``weight_y > 0``) — the real supervised fit, WARM-INITIALIZED from
    phase-1's topics but with a FRESH Robbins-Monro iteration counter so rho
    restarts near rho_0 and the head can actually move.

The fresh counter is the whole subtlety: it is DISTINCT from resume (which
continues the counter, so rho would already be decayed and the head would barely
train). These tests pin, at both the ``VIRunner`` mechanism level and the
``OnlinePCLDAEstimator`` shim level:

  1. Fresh counter + warm topics + fresh (zero) head + fresh rho (not decayed),
     with phase-1 topics matching a standalone unsupervised fit at the boundary,
     and a trained head (|w_CK|max clearly > 0) after phase 2.
  2. The cold-start path (no warm start) is byte-for-byte unchanged.
  3. resume_from / warm_start_from are mutually exclusive.
  4. (slow) An A/B smoke: warm reaches a trained head in fewer phase-2 iters than
     cold — the head trajectory the user wants to see.
"""
from __future__ import annotations

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors

# `spark` is the session-scoped fixture from tests/conftest.py.


# ---------------------------------------------------------------------------
# Synthetic labeled corpus: K disjoint vocab blocks, y == 1 iff the doc's
# dominant block is block 0. Enough signal that a supervised head trains off
# zero within a few dozen full-batch SVI iters.
# ---------------------------------------------------------------------------

def _labeled_block_docs(n_per_topic=40, K=3, block=4, seed=0):
    """(list[PCDocument], V) — one predictive label cell per doc, all observed."""
    from spark_vi.models.topic.types import PCDocument

    rng = np.random.default_rng(seed)
    V = K * block
    docs = []
    for t in range(K):
        favored = list(range(t * block, (t + 1) * block))
        y = 1.0 if t == 0 else 0.0
        for _ in range(n_per_topic):
            counts = np.zeros(V)
            for w in rng.choice(favored, size=12, replace=True):
                counts[w] += 1.0
            idx = np.nonzero(counts)[0].astype(np.int32)
            docs.append(PCDocument(
                indices=idx, counts=counts[idx].astype(np.float64),
                length=int(counts.sum()),
                y=np.array([y]), label_mask=np.array([1.0]),
            ))
    return docs, V


def _labeled_block_rows(n_per_topic=40, K=3, block=4, seed=0):
    """Same corpus as Spark rows (features SparseVector + scalar label)."""
    rng = np.random.default_rng(seed)
    V = K * block
    rows = []
    for t in range(K):
        favored = list(range(t * block, (t + 1) * block))
        y = 1.0 if t == 0 else 0.0
        for _ in range(n_per_topic):
            counts = np.zeros(V)
            for w in rng.choice(favored, size=12, replace=True):
                counts[w] += 1.0
            idx = sorted(np.nonzero(counts)[0].tolist())
            fv = Vectors.sparse(V, idx, [float(counts[i]) for i in idx])
            rows.append((fv, y))
    return rows, ["features", "label"], V


class _RecordingPCLDA:
    """OnlinePCLDA subclass factory that records each driver-side update_global.

    Records the learning_rate (rho) and a snapshot of the incoming lambda / w_CK
    the step reads, so a test can assert the FIRST phase-2 step saw warm topics, a
    fresh (zero) head, and an undecayed rho. update_global runs only on the driver,
    so the recording lists populate on the driver instance; the extra list attrs
    stay picklable for the executor-shipped local_update.
    """

    def __new__(cls, base_cls):
        from spark_vi.models.topic.pc import OnlinePCLDA

        class _Rec(OnlinePCLDA):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.rho_calls: list[float] = []
                self.lambda_seen: list[np.ndarray] = []
                self.wck_seen: list[np.ndarray] = []

            def update_global(self, global_params, target_stats, learning_rate):
                self.rho_calls.append(float(learning_rate))
                self.lambda_seen.append(np.array(global_params["lambda"], copy=True))
                self.wck_seen.append(np.array(global_params["w_CK"], copy=True))
                return super().update_global(global_params, target_stats, learning_rate)

        return _Rec


def _persist(spark, docs):
    rdd = spark.sparkContext.parallelize(docs, numSlices=4).persist()
    rdd.count()
    return rdd


# ---------------------------------------------------------------------------
# 1. Mechanism: fresh counter, warm topics, fresh head, fresh rho, trained head.
# ---------------------------------------------------------------------------

def test_warm_start_mechanism_fresh_counter_warm_topics_trained_head(spark, tmp_path):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.io.export import save_result
    from spark_vi.models.topic.pc import OnlinePCLDA

    N, M = 8, 40
    TAU0, KAPPA, WY = 5.0, 0.6, 50.0
    docs, V = _labeled_block_docs(seed=0)
    rdd = _persist(spark, docs)
    common = dict(K=3, vocab_size=V, C=1, alpha=1.1, grad_cavi_iters=10,
                  random_seed=0)
    cfg_kw = dict(learning_rate_tau0=TAU0, learning_rate_kappa=KAPPA,
                  random_seed=0, convergence_tol=1e-12)

    # --- PHASE 1: unsupervised (weight_y=0), N iters -> save as warm-start dir.
    m1 = OnlinePCLDA(weight_y=0.0, **common)
    r1 = VIRunner(m1, VIConfig(max_iterations=N, **cfg_kw)).fit(rdd)
    lambda_p1 = np.array(r1.global_params["lambda"], copy=True)
    # Unsupervised path leaves the head at its zero seed.
    assert np.allclose(r1.global_params["w_CK"], 0.0)
    assert r1.n_iterations == N
    warm_dir = tmp_path / "phase1"
    save_result(r1, warm_dir)

    # --- A standalone unsupervised fit (same seed/data) must match phase 1's
    #     topics at the boundary — phase 1 IS an unsupervised fit (same path).
    m1b = OnlinePCLDA(weight_y=0.0, **common)
    r1b = VIRunner(m1b, VIConfig(max_iterations=N, **cfg_kw)).fit(rdd)
    np.testing.assert_allclose(r1b.global_params["lambda"], lambda_p1)

    # --- PHASE 2: supervised warm-start from phase 1, M iters, FRESH counter.
    m2 = _RecordingPCLDA(OnlinePCLDA)(weight_y=WY, **common)
    r2 = VIRunner(m2, VIConfig(max_iterations=M, **cfg_kw)).fit(
        rdd, warm_start_from=warm_dir,
    )
    rdd.unpersist(blocking=False)

    # Fresh counter: a warm-started M-iter fit reports M iters, NOT N + M
    # (that is the resume behavior — the whole distinction under test).
    assert r2.n_iterations == M, (
        f"warm-start must reset the counter (expected {M}, got {r2.n_iterations}); "
        f"N+M={N + M} would mean it wrongly continued the resume counter"
    )

    # The FIRST phase-2 step read phase-1's topics (warm) and a zero head (fresh).
    np.testing.assert_allclose(m2.lambda_seen[0], lambda_p1)
    assert np.allclose(m2.wck_seen[0], 0.0), "phase 2 must start the head at zero"

    # Fresh rho: the first phase-2 rho equals the COLD first rho (tau0+0+1)^-kappa,
    # and is strictly LARGER than the decayed rho a resume-at-t=N would have used
    # ((tau0+N+1)^-kappa) — i.e. the schedule restarted, it did not continue.
    fresh_rho0 = (TAU0 + 0 + 1) ** -KAPPA
    decayed_rhoN = (TAU0 + N + 1) ** -KAPPA
    assert m2.rho_calls[0] == pytest.approx(fresh_rho0)
    assert m2.rho_calls[0] > decayed_rhoN

    # Trained head after phase 2: |w_CK|max clearly off its zero seed.
    w_absmax = float(np.abs(r2.global_params["w_CK"]).max())
    assert w_absmax > 0.05, f"warm-started head did not train (|w_CK|max={w_absmax:.4g})"


# ---------------------------------------------------------------------------
# 2. Cold start unchanged: warm_start_from=None is byte-for-byte the default path.
# ---------------------------------------------------------------------------

def test_cold_start_warm_start_none_is_byte_identical(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.pc import OnlinePCLDA

    docs, V = _labeled_block_docs(seed=1)
    rdd = _persist(spark, docs)
    common = dict(K=3, vocab_size=V, C=1, alpha=1.1, grad_cavi_iters=10,
                  random_seed=0)
    cfg = VIConfig(max_iterations=6, learning_rate_tau0=5.0,
                   learning_rate_kappa=0.6, random_seed=0, convergence_tol=1e-12)

    r_default = VIRunner(OnlinePCLDA(weight_y=3.0, **common), cfg).fit(rdd)
    r_none = VIRunner(OnlinePCLDA(weight_y=3.0, **common), cfg).fit(
        rdd, warm_start_from=None,
    )
    rdd.unpersist(blocking=False)

    assert r_none.n_iterations == r_default.n_iterations
    for name, arr in r_default.global_params.items():
        np.testing.assert_array_equal(r_none.global_params[name], arr)
    assert r_none.elbo_trace == r_default.elbo_trace


def test_resume_and_warm_start_are_mutually_exclusive(spark, tmp_path):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.io.export import save_result
    from spark_vi.models.topic.pc import OnlinePCLDA

    docs, V = _labeled_block_docs(seed=2)
    rdd = _persist(spark, docs)
    r = VIRunner(
        OnlinePCLDA(weight_y=0.0, K=3, vocab_size=V, C=1, random_seed=0),
        VIConfig(max_iterations=2, random_seed=0),
    ).fit(rdd)
    ckpt = tmp_path / "ckpt"
    save_result(r, ckpt)

    with pytest.raises(ValueError, match="mutually exclusive"):
        VIRunner(
            OnlinePCLDA(weight_y=1.0, K=3, vocab_size=V, C=1, random_seed=0),
            VIConfig(max_iterations=2, random_seed=0),
        ).fit(rdd, resume_from=ckpt, warm_start_from=ckpt)
    rdd.unpersist(blocking=False)


# ---------------------------------------------------------------------------
# Shim-level: OnlinePCLDAEstimator.warmStartFrom end-to-end + validation + cold-start.
# ---------------------------------------------------------------------------

def test_shim_warm_start_from_trains_head_with_fresh_counter(spark, tmp_path):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    rows, cols, V = _labeled_block_rows(seed=0)
    df = spark.createDataFrame(rows, cols)

    N, M = 8, 40
    common = dict(k=3, seed=0, subsamplingRate=1.0, docConcentration=[1.1],
                  learningOffset=5.0, learningDecay=0.6, numLabels=1,
                  featuresCol="features", labelCol="label")

    # Phase 1: unsupervised warm-up, saved to a dir.
    warm_dir = tmp_path / "p1"
    m1 = OnlinePCLDAEstimator(weightY=0.0, maxIter=N, **common).fit(df)
    m1.save(str(warm_dir))
    assert np.allclose(m1.headWeights(), 0.0)

    # Phase 2: supervised, warm-started from phase 1 (fresh counter).
    m2 = OnlinePCLDAEstimator(
        weightY=50.0, maxIter=M, warmStartFrom=str(warm_dir), **common,
    ).fit(df)

    # Fresh counter: n_iterations == M (not N + M).
    assert m2.result.n_iterations == M
    # Warm topics: phase 2's initial lambda came from phase 1 — assert the fit
    # trained a head off zero (the whole point of the warm start).
    assert float(np.abs(m2.headWeights()).max()) > 0.05


def test_shim_warm_start_empty_is_cold_start(spark):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    rows, cols, V = _labeled_block_rows(seed=3)
    df = spark.createDataFrame(rows, cols)
    common = dict(k=3, maxIter=5, seed=0, subsamplingRate=1.0,
                  docConcentration=[1.1], numLabels=1,
                  featuresCol="features", labelCol="label", weightY=2.0)

    r_unset = OnlinePCLDAEstimator(**common).fit(df).result
    r_empty = OnlinePCLDAEstimator(warmStartFrom="", **common).fit(df).result

    assert r_empty.n_iterations == r_unset.n_iterations
    for name, arr in r_unset.global_params.items():
        np.testing.assert_array_equal(r_empty.global_params[name], arr)


def test_shim_warm_start_rejects_missing_manifest(spark, tmp_path):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    rows, cols, V = _labeled_block_rows(seed=4)
    df = spark.createDataFrame(rows, cols)
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    est = OnlinePCLDAEstimator(k=3, maxIter=2, seed=0, subsamplingRate=1.0, numLabels=1,
                      labelCol="label", weightY=1.0, warmStartFrom=str(empty))
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        est.fit(df)


def test_shim_rejects_warm_start_and_resume_together(spark, tmp_path):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    rows, cols, V = _labeled_block_rows(seed=5)
    df = spark.createDataFrame(rows, cols)
    # A real checkpoint so both paths pass their manifest existence check and the
    # mutual-exclusion guard is what fires.
    ckpt = tmp_path / "ckpt"
    OnlinePCLDAEstimator(k=3, maxIter=2, seed=0, subsamplingRate=1.0).fit(df).save(str(ckpt))

    est = OnlinePCLDAEstimator(k=3, maxIter=2, seed=0, subsamplingRate=1.0, numLabels=1,
                      labelCol="label", weightY=1.0,
                      resumeFrom=str(ckpt), warmStartFrom=str(ckpt))
    with pytest.raises(ValueError, match="mutually exclusive"):
        est.fit(df)


# ---------------------------------------------------------------------------
# 4. (slow) A/B smoke: warm reaches a trained head in fewer phase-2 iters than
#    cold. Prints both |w_CK|max trajectories.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ab_warm_vs_cold_head_trajectory(spark, tmp_path, capsys):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.io.export import save_result
    from spark_vi.models.topic.pc import OnlinePCLDA

    # Hughes regime: K_fit < K_dom, so unsupervised warm topics give phase 2 a
    # head start the cold fit must first discover. Reuse the analysis/pc generator.
    from analysis.pc.tests.test_synthetic_signal import _make_corpus, SEED, D, V, K_FIT

    X, y = _make_corpus(SEED)
    from test_pc_lda import _labeled_pc_docs  # sibling test helper on sys.path

    docs = _labeled_pc_docs(X, y)
    rdd = _persist(spark, docs)
    common = dict(K=K_FIT, vocab_size=V, C=1, alpha=1.1, grad_cavi_iters=10,
                  random_seed=0)
    cfg_kw = dict(learning_rate_tau0=10.0, learning_rate_kappa=0.6,
                  random_seed=0, convergence_tol=1e-12)
    N, M, WY = 40, 60, 50.0
    THRESH = 0.05

    def _trajectory(warm_from):
        traj: list[float] = []
        model = OnlinePCLDA(weight_y=WY, **common)
        VIRunner(model, VIConfig(max_iterations=M, **cfg_kw)).fit(
            rdd, warm_start_from=warm_from,
            on_iteration=lambda it, gp, elbo: traj.append(
                float(np.abs(gp["w_CK"]).max())
            ),
        )
        return traj

    # Phase 1 warm-up (unsupervised), saved.
    warm_dir = tmp_path / "p1"
    r1 = VIRunner(
        OnlinePCLDA(weight_y=0.0, **common), VIConfig(max_iterations=N, **cfg_kw),
    ).fit(rdd)
    save_result(r1, warm_dir)

    warm_traj = _trajectory(warm_dir)
    cold_traj = _trajectory(None)
    rdd.unpersist(blocking=False)

    def _first_cross(traj):
        for i, v in enumerate(traj):
            if v > THRESH:
                return i + 1
        return None

    warm_cross = _first_cross(warm_traj)
    cold_cross = _first_cross(cold_traj)
    with capsys.disabled():
        print(f"\n[warm-vs-cold |w_CK|max] threshold={THRESH}")
        print(f"  warm first-cross iter: {warm_cross}  final={warm_traj[-1]:.4g}")
        print(f"  cold first-cross iter: {cold_cross}  final={cold_traj[-1]:.4g}")
        print(f"  warm traj[:12]: {[round(v, 4) for v in warm_traj[:12]]}")
        print(f"  cold traj[:12]: {[round(v, 4) for v in cold_traj[:12]]}")

    # Both eventually train a head; warm crosses the threshold no later than cold.
    assert warm_traj[-1] > THRESH and cold_traj[-1] > THRESH
    assert warm_cross is not None
    if cold_cross is not None:
        assert warm_cross <= cold_cross, (
            f"warm-start should reach a trained head no later than cold "
            f"(warm={warm_cross}, cold={cold_cross})"
        )
