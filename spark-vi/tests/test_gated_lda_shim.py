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


def test_deployment_alpha_modes():
    # _deployment_alpha builds the fold-in prior: fitted unchanged; symmetric flat;
    # block_balanced keeps nodes equal but sets the bg-vs-node collective split.
    import numpy as np
    from spark_vi.models.topic.dag_placement import DagLayout
    from spark_vi.mllib.topic.gated_lda import _deployment_alpha
    lay = DagLayout({1: 0, 2: 0, 3: 0}, n_bg=2, tpn=1)          # K=5, 2 bg + 3 nodes
    fitted = np.arange(5, dtype=float) + 1.0

    assert np.array_equal(_deployment_alpha(fitted, lay, "fitted", 0.0, 0.5), fitted)

    sym = _deployment_alpha(fitted, lay, "symmetric", 0.0, 0.5)  # <=0 -> 1/K
    assert np.allclose(sym, 1.0 / 5)
    assert np.allclose(_deployment_alpha(fitted, lay, "symmetric", 0.02, 0.5), 0.02)

    bb = _deployment_alpha(fitted, lay, "block_balanced", 0.0, 0.6)
    assert np.allclose(bb[:2].sum(), 0.6)                        # background collective = w
    node_vals = [bb[lay.block[u][0]] for u in lay.nodes]
    assert np.allclose(node_vals, node_vals[0])                  # all nodes equal (unbiased)
    assert np.allclose(bb.sum(), 1.0)                            # total concentration 1

    import pytest
    with pytest.raises(ValueError, match="transformBgWeight"):
        _deployment_alpha(fitted, lay, "block_balanced", 0.0, 1.5)
    with pytest.raises(ValueError, match="transformAlphaMode"):
        _deployment_alpha(fitted, lay, "banana", 0.0, 0.5)


def test_gated_shim_transform_alpha_mode_param_defaults_and_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert est.getOrDefault("transformAlphaMode") == "fitted"   # byte-identical default
    assert est.getOrDefault("transformBgWeight") == 0.5
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, transformAlphaMode="block_balanced",
                             transformBgWeight=0.9)
    assert est2.getOrDefault("transformAlphaMode") == "block_balanced"
    assert est2.getOrDefault("transformBgWeight") == 0.9


def test_gated_shim_transform_symmetric_mode_runs(spark):
    # transform with a symmetric deployment alpha runs end-to-end and yields the
    # per-node affinity vector (decoupled from whatever alpha was fitted).
    import numpy as np
    from pyspark.ml.linalg import Vectors
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    parent = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    V = 30
    rng = np.random.default_rng(0)
    rows = [(Vectors.sparse(V, sorted(rng.choice(V, 6, replace=False).tolist()),
                            [1.0] * 6), [int(rng.choice([3, 4, 5, 6]))]) for _ in range(40)]
    df = spark.createDataFrame(rows, ["features", "frontier"])
    model = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, maxIter=3, seed=0).fit(df)
    model.set(model.transformAlphaMode, "symmetric")
    out = model.transform(df).select("nodeAffinity").head()[0]
    assert len(out) == 6                                         # one affinity per node


def test_shim_spectral_topo_order_param_default_and_set():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator()
    assert est.getOrDefault("spectralTopoOrder") == "forward"
    est2 = GatedLDAEstimator(spectralTopoOrder="reverse")
    assert est2.getOrDefault("spectralTopoOrder") == "reverse"


def test_concat_domain_features_offsets_and_sorts():
    """Per-domain vectors concatenate into the engine's single id space: domain m
    ids shift by bounds[m], and the result stays globally sorted."""
    import numpy as np
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import _concat_domain_features
    v0 = SparseVector(4, {0: 2.0, 3: 1.0})
    v1 = SparseVector(3, {1: 5.0})
    idx, cnt = _concat_domain_features([v0, v1], [4, 3])
    np.testing.assert_array_equal(idx, np.array([0, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(cnt, np.array([2.0, 1.0, 5.0]))
    assert np.all(np.diff(idx) > 0)


def test_concat_domain_features_rejects_a_mis_sized_vector():
    """A vector whose size disagrees with the established layout must raise, naming
    the domain and both sizes -- silently re-laying-out the vocabulary would
    corrupt the fit with no symptom."""
    import pytest
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import _concat_domain_features
    with pytest.raises(ValueError, match=r"domain 1.*size 9.*expected 3"):
        _concat_domain_features([SparseVector(4, {0: 1.0}), SparseVector(9, {1: 1.0})],
                                [4, 3])


def test_multidomain_shim_fit_derives_domain_sizes(spark):
    """A fit with featuresCols derives the per-domain widths from the first row and
    produces a per-domain dict lambda of those widths."""
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [
        (SparseVector(6, {0: 2.0, 1: 1.0}), SparseVector(4, {0: 3.0}), [2]),
        (SparseVector(6, {1: 1.0, 2: 2.0}), SparseVector(4, {1: 1.0}), [2]),
        (SparseVector(6, {3: 3.0, 4: 1.0}), SparseVector(4, {2: 2.0}), [3]),
        (SparseVector(6, {4: 1.0, 5: 1.0}), SparseVector(4, {3: 4.0}), [3]),
    ]
    df = spark.createDataFrame(rows, schema=schema).repartition(2)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=1, seed=0)
    model = est.fit(df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and sorted(lam) == [0, 1]
    assert lam[0].shape[1] == 6 and lam[1].shape[1] == 4
    assert model.result.metadata["domains"] == [6, 4]


def test_multidomain_shim_fit_rejects_a_mis_sized_row(spark):
    """A row disagreeing with the derived layout fails the fit rather than
    silently re-laying-out the vocabulary."""
    import pytest
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [
        (SparseVector(6, {0: 1.0}), SparseVector(4, {0: 1.0}), [2]),
        (SparseVector(6, {1: 1.0}), SparseVector(9, {1: 1.0}), [3]),   # wrong width
    ]
    df = spark.createDataFrame(rows, schema=schema)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=1, seed=0)
    with pytest.raises(Exception, match="domain 1"):
        est.fit(df)


def test_explicit_domain_bounds_override_an_unrepresentative_first_row(spark):
    """domainBounds is the escape hatch for a dataset whose first row does not
    carry the true per-domain widths (a narrower vector, e.g. a producer that
    sized it to the max nonzero id). When set it is AUTHORITATIVE: the fit uses
    those widths and every row -- the first included -- is validated against them,
    so a first row that disagrees fails instead of silently defining the layout."""
    import pytest
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    # Every row is width 6/4; bounds declaring 6/4 must fit fine...
    ok = spark.createDataFrame(
        [(SparseVector(6, {0: 1.0}), SparseVector(4, {0: 1.0}), [2]),
         (SparseVector(6, {3: 1.0}), SparseVector(4, {2: 1.0}), [3])], schema=schema)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            domainBounds=[0, 6, 10], parent={1: 0, 2: 1, 3: 1},
                            nBg=2, tpn=1, maxIter=1, seed=0)
    model = est.fit(ok)
    assert model.result.metadata["domains"] == [6, 4]
    # ...and bounds that disagree with the actual vectors must fail per-row.
    bad = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            domainBounds=[0, 5, 9], parent={1: 0, 2: 1, 3: 1},
                            nBg=2, tpn=1, maxIter=1, seed=0)
    with pytest.raises(Exception, match="domain 0"):
        bad.fit(ok)
    # Malformed bounds are rejected on the driver, before any Spark work.
    with pytest.raises(ValueError, match="strictly increasing"):
        GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                          domainBounds=[0, 6, 6], parent={1: 0, 2: 1, 3: 1},
                          nBg=2, tpn=1, maxIter=1, seed=0).fit(ok)


def test_concat_domain_features_rejects_a_column_count_mismatch():
    """len(vectors) != len(sizes) must raise, not zip-truncate. A short `sizes`
    would silently drop the trailing domain -- the same invisible re-layout the
    per-vector size check exists to stop -- and the columns and the widths do not
    always arrive together (a model carries them as separate Params)."""
    import pytest
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import _concat_domain_features
    v0, v1 = SparseVector(4, {0: 1.0}), SparseVector(3, {1: 1.0})
    with pytest.raises(ValueError, match=r"2 vector\(s\) for 1 domain size\(s\)"):
        _concat_domain_features([v0, v1], [4])
    with pytest.raises(ValueError, match=r"1 vector\(s\) for 2 domain size\(s\)"):
        _concat_domain_features([v0], [4, 3])


def _multidomain_spectral_rows():
    """Two domains with per-node-distinct tokens in BOTH domains, so the
    block-aligned anchor recipe has a candidate per block in each domain."""
    from pyspark.ml.linalg import SparseVector
    rows = []
    for _ in range(20):
        rows.append((SparseVector(6, {0: 3.0, 1: 2.0}), SparseVector(4, {0: 2.0}), [1]))
        rows.append((SparseVector(6, {2: 3.0, 3: 2.0}), SparseVector(4, {1: 2.0}), [2]))
        rows.append((SparseVector(6, {4: 3.0, 5: 2.0}), SparseVector(4, {2: 2.0,
                                                                        3: 1.0}), []))
    return rows


def _multidomain_schema():
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    return StructType([
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])


def test_multidomain_scalable_spectral_init_yields_per_domain_dict_lambda(spark):
    """featuresCols + init='spectral' routed SCALABLE must fit and seed a per-domain
    dict lambda. scalable_block_aligned_lambda returns the JOINT (K, V) array while
    the engine's multi-domain arm calls .items() on the handed-over seed, so an
    unconverted joint raises a bare AttributeError naming neither domains nor init.
    Reachable on shipped defaults: spectralMethod='auto' routes scalable at
    V >= spectralMaxVocab, and a CONCATENATED multi-domain V crosses that easily.

    The row-mass assertion pins the NORMALIZATION convention, which is the half a
    shape-only check would miss: split_domains returns row-stochastic blocks, so a
    conversion that forgets to re-apply the lambda scale hands the engine a seed of
    row mass ~1 instead of ~200 -- a fit that runs, and simply never concentrates.
    """
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = spark.createDataFrame(_multidomain_spectral_rows(), schema=_multidomain_schema())
    m = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                          parent={1: 0, 2: 0}, nBg=1, tpn=1,
                          init="spectral", spectralMethod="scalable",
                          spectralMinDocFreq=1, maxIter=1, seed=0).fit(df)
    lam = m.result.global_params["lambda"]
    assert isinstance(lam, dict) and sorted(lam) == [0, 1]
    assert lam[0].shape[1] == 6 and lam[1].shape[1] == 4
    assert m.result.metadata["domains"] == [6, 4]
    # Seed magnitude survived the per-domain split (scale ~200, not ~1). One SVI
    # step keeps a large fraction of the seed, so ~1 and ~200 are far apart here.
    for md in (0, 1):
        assert lam[md].sum(axis=1).min() > 50.0, lam[md].sum(axis=1)


def test_multidomain_dense_spectral_init_builds_docs_through_the_shared_helper(spark):
    """featuresCols + init='spectral' routed DENSE collects to the driver and builds
    train_docs through _concat_domain_features, so the seed sees exactly the
    concatenated ids the fit will see. Covers the dense multi-domain branch (the
    other fit tests all use init='random', which never reaches it)."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = spark.createDataFrame(_multidomain_spectral_rows(), schema=_multidomain_schema())
    m = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                          parent={1: 0, 2: 0}, nBg=1, tpn=1,
                          init="spectral", spectralMethod="dense",
                          maxIter=1, seed=0).fit(df)
    lam = m.result.global_params["lambda"]
    assert isinstance(lam, dict) and sorted(lam) == [0, 1]
    assert lam[0].shape[1] == 6 and lam[1].shape[1] == 4


def test_single_domain_shim_path_unchanged(spark):
    """featuresCols unset keeps the existing single-featuresCol behavior: a single
    (K, V) array lambda and no domains in metadata."""
    import numpy as np
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    schema = StructType([
        StructField("features", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [(SparseVector(8, {0: 1.0, 1: 2.0}), [2]),
            (SparseVector(8, {2: 1.0, 3: 1.0}), [3])]
    df = spark.createDataFrame(rows, schema=schema)
    est = GatedLDAEstimator(labelCol="frontier", parent={1: 0, 2: 1, 3: 1},
                            nBg=2, tpn=1, maxIter=1, seed=0)
    model = est.fit(df)
    assert isinstance(model.result.global_params["lambda"], np.ndarray)
    assert "domains" not in model.result.metadata
