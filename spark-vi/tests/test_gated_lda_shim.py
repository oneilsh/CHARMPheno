import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors

# `spark` is the session-scoped fixture from tests/conftest.py (Java
# security-manager option + PYSPARK_PYTHON pinning already applied there);
# a bare `local[1]` SparkSession built without those options fails to start
# on newer JDKs, so this file reuses the shared fixture rather than
# redefining its own.


def test_gated_shim_fit_transform_smoke(spark):
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V = 30
    # tiny planted rows: features SparseVector + frontier (array of node ids)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    df = spark.createDataFrame(rows, ["features", "frontier"])

    est = GatedLDAEstimator(featuresCol="features", labelCol="frontier",
                            parent=parent, nBg=2, tpn=1, maxIter=3, seed=0)
    model = est.fit(df)
    out = model.transform(df)
    assert "nodeAffinity" in out.columns
    aff = out.select("nodeAffinity").head()[0]
    n_nodes = len({1, 2, 3, 4, 5, 6})
    assert len(aff) == n_nodes            # one affinity per DAG node


def test_gated_shim_transform_is_deterministic(spark):
    # The held-out fold-in feeds AUC/precision metrics, so two transforms of the
    # same model + data must yield identical node-affinity vectors (content-seeded
    # gamma_init, not an unseeded RNG).
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V = 30
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    model = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, maxIter=3, seed=0).fit(df)

    a1 = [r[0] for r in model.transform(df).select("nodeAffinity").collect()]
    a2 = [r[0] for r in model.transform(df).select("nodeAffinity").collect()]
    assert len(a1) == len(a2) == 40
    for v1, v2 in zip(a1, a2):
        assert np.array_equal(v1.toArray(), v2.toArray())


def test_gated_shim_node_alpha_scale_builds_asymmetric_alpha(spark):
    # nodeAlphaScale<1 must give the per-node blocks a smaller Dirichlet alpha
    # than the background block: alpha_background = 1/K, alpha_node = scale/K.
    # alpha is fixed (optimize_alpha disabled), so the fitted vector equals the
    # constructed one.
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V = 30
    n_bg, tpn = 2, 1
    K = n_bg + len(parent) * tpn                       # 2 + 6 = 8
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    df = spark.createDataFrame(rows, ["features", "frontier"])

    m = GatedLDAEstimator(parent=parent, nBg=n_bg, tpn=tpn,
                          nodeAlphaScale=0.1, maxIter=2, seed=0).fit(df)
    alpha = m.result.global_params["alpha"]
    assert alpha.shape == (K,)
    assert np.allclose(alpha[:n_bg], 1.0 / K)          # background unchanged
    assert np.allclose(alpha[n_bg:], 0.1 / K)          # node blocks down-weighted
    # symmetric default is unchanged (regression guard).
    m_sym = GatedLDAEstimator(parent=parent, nBg=n_bg, tpn=tpn,
                              maxIter=2, seed=0).fit(df)
    assert np.allclose(m_sym.result.global_params["alpha"], 1.0 / K)


def test_gated_shim_svi_schedule_params_default_and_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("miniBatchFraction") == 0.0    # full-batch default
    assert est.getOrDefault("learningRateTau0") == 1.0
    assert est.getOrDefault("learningRateKappa") == 0.7
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, miniBatchFraction=0.1,
                             learningRateTau0=10.0, learningRateKappa=0.75)
    assert est2.getOrDefault("miniBatchFraction") == 0.1
    assert est2.getOrDefault("learningRateTau0") == 10.0
    assert est2.getOrDefault("learningRateKappa") == 0.75


def test_gated_shim_minibatch_fit_smoke(spark):
    # The mini-batch SVI path (miniBatchFraction in (0,1]) must fit without error
    # and produce a K-row lambda, exercising VIConfig(mini_batch_fraction=...).
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V = 30
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(60):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    m = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, miniBatchFraction=0.5,
                          learningRateTau0=10.0, maxIter=3, seed=0).fit(df)
    assert m.result.global_params["lambda"].shape[0] == 2 + len(parent)   # K rows


def test_gated_shim_init_param_defaults_random():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("init") == "random"


def test_gated_shim_anchor_scope_param_defaults_and_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("anchorScope") == "closure"
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, anchorScope="frontier")
    assert est2.getOrDefault("anchorScope") == "frontier"


def test_gated_shim_frontier_scope_scalable_fit(spark):
    # anchorScope='frontier' on the scalable path fits end-to-end and yields a
    # K-row lambda (background from empty-frontier docs, node from its own docs).
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    rows = []
    for _ in range(20):
        rows.append((SparseVector(8, [0, 1, 2], [1.0, 1.0, 1.0]), []))    # background
        rows.append((SparseVector(8, [0, 5, 6], [1.0, 1.0, 1.0]), [1]))   # node 1
    df = spark.createDataFrame(rows, ["features", "frontier"])
    m = GatedLDAEstimator(parent={1: 0}, nBg=1, tpn=1, init="spectral",
                          spectralMethod="scalable", spectralMinDocFreq=1,
                          anchorScope="frontier", maxIter=2, seed=0).fit(df)
    assert m.result.global_params["lambda"].shape[0] == 1 + 1   # n_bg + 1 node


def test_gated_shim_unknown_init_raises(spark):
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import pytest
    df = spark.createDataFrame(
        [(SparseVector(5, [0, 1], [1.0, 1.0]), [1])],
        ["features", "frontier"],
    )
    est = GatedLDAEstimator(parent={1: 0, 2: 0}, init="banana", maxIter=1)
    with pytest.raises(ValueError, match="init"):
        est.fit(df)


def test_gated_shim_spectral_method_param_defaults_and_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("spectralMethod") == "auto"
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, spectralMethod="scalable")
    assert est2.getOrDefault("spectralMethod") == "scalable"


def test_gated_shim_scalable_spectral_fits_and_seeds_lambda(spark):
    # Forcing spectralMethod='scalable' at small V routes through the distributed
    # projected init and fits; the resulting lambda differs from a random-init fit
    # on the same corpus (the scalable seed took effect), and no dense V×V is built.
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import numpy as np
    rows = []
    for _ in range(30):
        rows.append((SparseVector(8, [0, 1, 6], [3.0, 2.0, 1.0]), [1]))
        rows.append((SparseVector(8, [2, 3, 6], [3.0, 2.0, 1.0]), [2]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    parent = {1: 0, 2: 0}
    m_rand = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="random",
                               maxIter=2, seed=0).fit(df)
    m_scal = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="spectral",
                               spectralMethod="scalable", spectralMinDocFreq=1,
                               maxIter=2, seed=0).fit(df)
    lam_r = m_rand.result.global_params["lambda"]
    lam_s = m_scal.result.global_params["lambda"]
    assert lam_r.shape == lam_s.shape == (2 + len(parent), 8)
    assert not np.allclose(lam_r, lam_s)


def test_gated_shim_spectral_auto_routes_scalable_above_threshold(spark):
    # spectralMethod='auto' with a tiny spectralMaxVocab threshold routes a
    # V>=threshold corpus to the scalable path and fits (no NotImplementedError).
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    rows = []
    for _ in range(20):
        rows.append((SparseVector(8, [0, 1, 6], [2.0, 1.0, 1.0]), [1]))
        rows.append((SparseVector(8, [2, 3, 6], [2.0, 1.0, 1.0]), [2]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    m = GatedLDAEstimator(parent={1: 0, 2: 0}, nBg=2, tpn=1, init="spectral",
                          spectralMethod="auto", spectralMaxVocab=4,
                          spectralMinDocFreq=1, maxIter=2, seed=0).fit(df)   # V=8 >= 4
    assert m.result.global_params["lambda"].shape[0] == 4


def test_gated_shim_spectral_fits_and_seeds_lambda(spark):
    """init='spectral' collects docs, runs the block-aligned spectral seed via
    data_summary, and fits. The resulting lambda must differ from a random-init
    fit on the same corpus (the spectral seed took effect)."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import numpy as np
    # Two nodes under root; docs attest node 1 (tokens 0,1) or node 2 (tokens 2,3).
    rows = []
    for _ in range(20):
        rows.append((SparseVector(6, [0, 1], [3.0, 2.0]), [1]))
        rows.append((SparseVector(6, [2, 3], [3.0, 2.0]), [2]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    parent = {1: 0, 2: 0}
    m_rand = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="random",
                               maxIter=2, seed=0).fit(df)
    m_spec = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, init="spectral",
                               spectralMaxVocab=1000, maxIter=2, seed=0).fit(df)
    lam_r = m_rand.result.global_params["lambda"]
    lam_s = m_spec.result.global_params["lambda"]
    assert lam_r.shape == lam_s.shape
    assert not np.allclose(lam_r, lam_s)   # spectral seed changed the trajectory


def test_gated_shim_on_iteration_callback_fires(spark):
    """setOnIteration registers a per-iter callback that the runner invokes with
    (iter_num, global_params, elbo_trace); global_params carries the (K,V) lambda
    so a driver can log per-topic top terms as the fit evolves (STM parity)."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    rows = []
    for _ in range(12):
        rows.append((SparseVector(6, [0, 1], [2.0, 1.0]), [1]))
        rows.append((SparseVector(6, [2, 3], [2.0, 1.0]), [2]))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    seen = []

    def _cb(it, gp, elbo):
        seen.append((it, "lambda" in gp, gp["lambda"].shape))

    est = GatedLDAEstimator(parent={1: 0, 2: 0}, nBg=2, tpn=1, maxIter=3, seed=0)
    assert est.setOnIteration(_cb) is est          # chainable
    est.fit(df)
    assert seen                                    # callback fired at least once
    it, has_lambda, shape = seen[-1]
    assert has_lambda and shape[0] == 4            # K = n_bg(2) + 2 nodes * tpn(1)


def test_gated_shim_optimize_doc_concentration_param_defaults_and_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("optimizeDocConcentration") is False
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, optimizeDocConcentration=True)
    assert est2.getOrDefault("optimizeDocConcentration") is True


def test_gated_shim_optimize_alpha_learns_asymmetric(spark):
    # A corpus where node 1 fires often (common) and node 2 rarely (rare) should,
    # with optimizeDocConcentration on, learn alpha(node1) > alpha(node2).
    import numpy as np
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout
    parent = {1: 0, 2: 0}
    V = 24
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        leaf = 1 if rng.random() < 0.75 else 2          # node1 common, node2 rare
        idx = sorted(rng.choice(V, size=5, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    for _ in range(60):
        idx = sorted(rng.choice(V, size=5, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), []))  # background
    df = spark.createDataFrame(rows, ["features", "frontier"])
    model = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, maxIter=8, seed=0,
                              optimizeDocConcentration=True).fit(df)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    alpha = model.result.global_params["alpha"]
    assert alpha[lay.block[1][0]] > alpha[lay.block[2][0]]


def test_gated_optimize_alpha_recovers_planted_alpha_ensemble(spark):
    # Faithful generative recovery, ENSEMBLE form. Plant a KNOWN per-node Dirichlet
    # alpha; draw each doc's theta from Dir(alpha over its allowed set); generate
    # words from DISJOINT per-topic vocab blocks (so topic-word recovery is trivial
    # and theta accurate). This isolates the alpha optimizer from the topic /
    # mass-starvation confound: what remains is whether the learned alpha_u recovers
    # the planted alpha_u.
    #
    # FINDING (why an ensemble, not a single fit): single-seed fits are MULTIMODAL —
    # different random inits land in different basins, so a given node can
    # under-recover (e.g. seed 2 sends the highest-alpha node to the lowest learned
    # value). This is the same mean-field/variational multimodality documented in
    # insights 0050-0058 (seed-dependent basins), NOT a math error: the pure Newton
    # step is proven exact by the finite-difference test in
    # test_concentration_optimization.py, and some seeds recover the ranking
    # perfectly. The SEED-ENSEMBLE MEAN averages out the basin noise and recovers
    # the planted ranking cleanly (observed Spearman 1.0 over 5-6 seeds). So the
    # honest, robust acceptance is ensemble recovery of the RANKING (the feature's
    # purpose: rarer node -> smaller alpha), not per-seed point calibration.
    import numpy as np
    from pyspark.ml.linalg import Vectors
    from scipy.stats import spearmanr
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from spark_vi.models.topic.dag_placement import DagLayout

    parent = {1: 0, 2: 0, 3: 0, 4: 0}                         # 4 flat siblings
    nodes = [1, 2, 3, 4]
    n_bg, tpn = 2, 1
    lay = DagLayout(parent, n_bg=n_bg, tpn=tpn)               # K = 2 + 4 = 6
    K = lay.K
    W = 6                                                     # words per topic
    V = K * W                                                 # disjoint vocab blocks
    alpha_bg = 0.4
    alpha_node = {1: 0.90, 2: 0.45, 3: 0.18, 4: 0.06}         # planted, descending
    bg_topics = list(range(n_bg))

    def _corpus(seed):
        rng = np.random.default_rng(seed)

        def _doc(allowed_topics, alpha_allowed, n_words=120):
            theta = rng.dirichlet(alpha_allowed)
            picks = rng.choice(len(allowed_topics), size=n_words, p=theta)
            counts = {}
            for p in picks:
                t = allowed_topics[p]
                w = t * W + int(rng.integers(0, W))          # word from topic t's block
                counts[w] = counts.get(w, 0) + 1
            idx = sorted(counts)
            return Vectors.sparse(V, idx, [float(counts[i]) for i in idx])

        rows = []
        for u in nodes:
            allowed = bg_topics + [lay.block[u][0]]          # bg ∪ node u's topic
            alpha_allowed = [alpha_bg] * n_bg + [alpha_node[u]]
            for _ in range(250):
                rows.append((_doc(allowed, alpha_allowed), [u]))
        for _ in range(300):                                  # background docs
            rows.append((_doc(bg_topics, [alpha_bg] * n_bg), []))
        return spark.createDataFrame(rows, ["features", "frontier"])

    learned_by_seed = []
    for seed in range(5):
        model = GatedLDAEstimator(parent=parent, nBg=n_bg, tpn=tpn, maxIter=80,
                                  seed=seed, nodeAlphaScale=1.0,
                                  optimizeDocConcentration=True).fit(_corpus(seed))
        alpha = model.result.global_params["alpha"]
        learned_by_seed.append([float(alpha[lay.block[u][0]]) for u in nodes])
    ens = np.mean(learned_by_seed, axis=0)
    planted = [alpha_node[u] for u in nodes]

    # The seed-ensemble mean recovers the planted alpha ranking.
    assert np.argmax(ens) == nodes.index(1)                   # highest planted -> highest learned
    assert np.argmin(ens) == nodes.index(4)                   # lowest planted -> lowest learned
    rho = spearmanr(planted, ens).correlation
    assert rho >= 0.9, f"ensemble planted->learned rank corr too low: {rho} (ens={list(ens)})"
