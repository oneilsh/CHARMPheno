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


def test_gated_shim_explicit_none_omega_eta_is_unset_not_set_to_none():
    # A driver forwarding an unset CLI arg passes omega=None / etaPerDomain=None
    # explicitly. @keyword_only funnels those into _input_kwargs, so a naive
    # _set would mark the Param SET-with-value-None -- tripping the isSet() guard
    # in _fit (`list(None)` -> TypeError) and _transform. explicit-None MUST be
    # treated as not-passed for these no-default, isSet-gated params.
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0}, omega=None, etaPerDomain=None,
                            domainBounds=None)
    assert not est.isSet("omega")
    assert not est.isSet("etaPerDomain")
    assert not est.isSet("domainBounds")
    # a real value still sets the Param (explicit-None is the only special case).
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, omega=[1.0, 0.5])
    assert est2.isSet("omega")
    assert est2.getOrDefault("omega") == [1.0, 0.5]


def test_gated_shim_explicit_none_omega_fits_single_domain(spark):
    # End-to-end regression for the exp-0070 driver crash: a single-domain fit
    # constructed with explicit omega=None / etaPerDomain=None must run (it used
    # to raise `TypeError: 'NoneType' object is not iterable` at _fit line 390).
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
    model = GatedLDAEstimator(parent=parent, nBg=2, tpn=1, maxIter=3, seed=0,
                              omega=None, etaPerDomain=None).fit(df)
    out = model.transform(df)                       # transform must not crash either
    assert "nodeAffinity" in out.columns


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


@pytest.mark.slow
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
    # The seed's DIRECTION survived the per-domain split, asserted via per-topic
    # CONCENTRATION rather than row mass.
    #
    # Row mass can no longer carry this check. Full batch takes an undamped rho=1
    # step (see tests/test_runner.py::test_vi_runner_full_batch_uses_undamped_step),
    # which REPLACES lambda with eta + the first E-step's expected counts, so after
    # one iteration lambda's row mass reflects the CORPUS, not the seed -- a seed
    # wrongly scaled to row mass ~1 would land at the same row mass as a correct
    # ~200 one, and the old `> 50.0` floor became unsatisfiable (measured
    # [62.1, 41.3, 40.5]) without indicating any defect.
    #
    # What a correctly-scaled seed still buys is a DECISIVE first E-step, which
    # shows up as BALANCED topic mass. A seed flattened toward row mass ~1 is
    # dominated by the eta prior, the first E-step is mushy, and mass collapses
    # onto fewer topics leaving one starved -- the insight 0066 topic-death
    # signature, and the real content of "a fit that runs and never concentrates".
    #
    # Threshold validated by deliberately patching SPECTRAL_LAMBDA_SCALE to 1.0 and
    # re-running this fit. Domain 1 row masses, min/max ratio:
    #     correct (scale 200): [62.1, 41.3, 40.5] -> 0.652
    #     broken  (scale 1.0): [85.6, 41.3, 17.1] -> 0.199
    # so 0.4 sits with ~1.6x margin either side. Two rejected alternatives, both
    # measured on the same pair: per-topic peak fraction (0.649 vs 0.471) discriminates
    # only on domain 1 and only just; distinct argmax per topic is 3/3 in BOTH cases
    # and is therefore vacuous. Domain 0 does not discriminate either way (0.987 vs
    # 0.927) -- it is asserted for symmetry, and domain 1 carries the check.
    for md in (0, 1):
        mass = lam[md].sum(axis=1)
        assert mass.min() / mass.max() > 0.4, (md, mass)


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


def _two_domain_df(spark, *, empty_domain1=False):
    """Six two-domain rows: domain 0 width 6, domain 1 width 4, frontiers over
    nodes 2 and 3 of the DAG {1:0, 2:1, 3:1}, two partitions so the reduce
    combines more than one, and a `rid` row key. Deliberately tiny -- these are
    shim-contract tests, not recovery tests.

    `rid` exists because collect() order is NOT a contract here: a repartition(2)
    round-robin can hand two runs the same rows in different orders, so comparing
    two transforms POSITIONALLY silently compares different documents (observed:
    max|diff| 1.7e-2 raw, 1.2e-15 after sorting). Compare via
    `_affinities_by_rid`.

    empty_domain1=True keeps every row's domain-0 vector, frontier and rid, and
    empties ONLY the domain-1 vector (same width 4, no nonzeros). That frame is
    the omega=0 counterfactual: a weight of 0 removes a domain's tokens from the
    gamma sum exactly, so it must fold in identically to a frame where those
    tokens are simply absent (see the direction test)."""
    from pyspark.ml.linalg import SparseVector, VectorUDT
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    schema = StructType([
        StructField("rid", IntegerType()),
        StructField("c", VectorUDT()), StructField("d", VectorUDT()),
        StructField("frontier", ArrayType(IntegerType())),
    ])
    rows = [
        (0, SparseVector(6, {0: 2.0, 1: 1.0}), SparseVector(4, {0: 3.0}), [2]),
        (1, SparseVector(6, {1: 1.0, 2: 2.0}), SparseVector(4, {1: 1.0}), [2]),
        (2, SparseVector(6, {0: 1.0, 2: 1.0}), SparseVector(4, {0: 2.0, 1: 1.0}), [2]),
        (3, SparseVector(6, {3: 3.0, 4: 1.0}), SparseVector(4, {2: 2.0}), [3]),
        (4, SparseVector(6, {4: 1.0, 5: 1.0}), SparseVector(4, {3: 4.0}), [3]),
        (5, SparseVector(6, {3: 1.0, 5: 2.0}), SparseVector(4, {2: 1.0, 3: 1.0}), [3]),
    ]
    if empty_domain1:
        rows = [(rid, c, SparseVector(4, {}), fr) for rid, c, d, fr in rows]
    return spark.createDataFrame(rows, schema=schema).repartition(2)


def _affinities_by_rid(model, df):
    """Transform df and return the nodeAffinity matrix ordered by the `rid` key.

    Never compare two transforms by collect() order: it is not stable across runs
    (see _two_domain_df), so a positional comparison can pair up different
    documents and either invent a difference or hide one."""
    import numpy as np
    rows = model.transform(df).select("rid", "nodeAffinity").collect()
    return np.array([list(r[1]) for r in sorted(rows, key=lambda r: r[0])])


def test_multidomain_shim_transform_produces_node_affinity(spark):
    """A multi-domain fitted model must transform: _transform currently assumes an
    ndarray lambda and raises on the per-domain dict."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)                     # helper added in this task
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=2, seed=0, omega=[1.0, 0.4])
    model = est.fit(df)
    out = model.transform(df).select("nodeAffinity").collect()
    assert len(out) == df.count()
    for r in out:
        vals = list(r[0])
        assert len(vals) == 3                      # one per DAG node
        assert all(v >= 0.0 for v in vals)


def test_transform_applies_omega_so_deployed_theta_matches_fitted(spark):
    """omega weights theta, and theta IS the deployed read-out, so transform must
    apply the same per-token weight the fit used. Directional: down-weighting
    domain 1 must change nodeAffinity relative to omega=1 on the SAME fitted
    lambda."""
    import numpy as np
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=2, seed=0, omega=[1.0, 1.0])
    model = est.fit(df)
    base = _affinities_by_rid(model, df)
    model._set(omega=[1.0, 0.05])                  # same lambda, different dial
    down = _affinities_by_rid(model, df)
    assert not np.allclose(base, down), "transform ignored omega"


def test_transform_reads_omega_from_metadata_on_a_rebuilt_model(spark):
    """A model rebuilt from a LOADED VIResult carries no omega Param, so transform
    falls back to the fitted omega recorded in result.metadata -- otherwise a
    served theta would silently differ from the fitted one (the train/serve skew
    this task exists to prevent). Param, when set, still wins."""
    import numpy as np
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator, GatedLDAModel
    df = _two_domain_df(spark)
    parent = {1: 0, 2: 1, 3: 1}
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent=parent, nBg=2, tpn=1,
                            maxIter=2, seed=0, omega=[1.0, 0.05])
    fitted = est.fit(df)
    assert fitted.result.metadata["omega"] == [1.0, 0.05]

    # Rebuild as a loader would: only the result + DAG shape, no fit-time Params.
    rebuilt = GatedLDAModel(fitted.result, parent=parent, nBg=2, tpn=1)
    rebuilt._set(featuresCols=["c", "d"])          # a loader knows its input columns
    assert not rebuilt.isSet("omega")
    a_fit = _affinities_by_rid(fitted, df)
    a_reb = _affinities_by_rid(rebuilt, df)
    np.testing.assert_allclose(a_reb, a_fit)

    # ...and an explicit Param overrides the recorded value.
    rebuilt._set(omega=[1.0, 1.0])
    a_one = _affinities_by_rid(rebuilt, df)
    assert not np.allclose(a_one, a_fit)


def test_transform_omega_zero_equals_dropping_that_domains_tokens(spark):
    """Pins the omega->domain DIRECTION, which "changing omega changes the output"
    cannot: reversing the gather (omega[::-1]) or using side="left" leaves every
    other omega assertion in this file green, and the engine's own exactness
    instrument does not cover the shim because the gather is duplicated there.

    Exact, not directional: omega_m = 0 multiplies domain m's tokens by 0 in the
    gamma recurrence's count vector, and phi_norm is per-token, so the gamma map
    becomes term-for-term the map of a document that simply lacks those tokens.
    So omega=[1, 0] on the real frame must reproduce omega=[1, 1] on the frame
    whose domain-1 vectors are empty. A reversed gather would zero domain 0
    instead and the two would disagree grossly; side="left" misassigns every id
    that sits exactly on a bound (global id 6 = domain 1's first id) plus id 0
    (searchsorted -> -1 -> the LAST omega), so it is caught too.

    The two frames hold different token content, so the content-seeded gamma_init
    differs; caviTol is tightened so both runs sit ON the (identical) fixed point,
    which is what the equality is about. Comparison is keyed by rid, never by
    collect() order (see _two_domain_df)."""
    import numpy as np
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)
    df_no_d1 = _two_domain_df(spark, empty_domain1=True)
    model = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                              parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                              maxIter=2, seed=0, omega=[1.0, 1.0]).fit(df)
    model._set(caviTol=1e-12, caviMaxIter=2000)     # converge to the fixed point

    model._set(omega=[1.0, 0.0])                    # domain 1 weighted out
    weighted_out = _affinities_by_rid(model, df)
    model._set(omega=[1.0, 1.0])                    # domain 1 absent instead
    absent = _affinities_by_rid(model, df_no_d1)
    np.testing.assert_allclose(weighted_out, absent, atol=1e-9)
    # Guard against the assertion being vacuous: domain 1 does move the read-out,
    # so the agreement above is omega=0 acting on the right domain, not a no-op.
    unweighted = _affinities_by_rid(model, df)
    assert not np.allclose(unweighted, absent, atol=1e-9)


def test_transform_rejects_a_malformed_omega_through_the_shared_validator(spark):
    """omega goes through domains.resolve_per_domain -- the ONE per-domain
    validator -- so the shim and the engine reject the same caller errors with the
    same message. The NEGATIVE case is the one that matters: transform is the only
    place a bad omega can still enter (the engine validates at fit time, and a
    Param can be _set on a model afterwards), and a negative weight drives gamma
    negative, whereupon digamma returns NaN and nodeAffinity comes out NaN with no
    error raised at all."""
    import numpy as np
    import pytest
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                            maxIter=1, seed=0)
    model = est.fit(df)
    model._set(omega=[1.0, 0.5, 0.25])             # 3 weights, 2 domains
    with pytest.raises(ValueError, match=r"omega must be a scalar or a length-2"):
        model.transform(df)
    model._set(omega=[1.0, -5.0])                  # would silently emit NaN
    with pytest.raises(ValueError, match=r"omega components must be finite and >= 0"):
        model.transform(df)
    # 0.0 IS legal (drop the domain), and the read-out is finite.
    model._set(omega=[1.0, 0.0])
    assert np.all(np.isfinite(_affinities_by_rid(model, df)))


def test_transform_rejects_a_features_cols_domain_count_mismatch(spark):
    """featuresCols disagreeing with the fitted domain count is knowable on the
    driver, so it raises there rather than surfacing per-row from inside a Spark
    task."""
    import pytest
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)
    model = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                              parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1,
                              maxIter=1, seed=0).fit(df)
    model._set(featuresCols=["c"])                 # 1 column, 2 fitted domains
    with pytest.raises(ValueError, match=r"1 column\(s\).*2 domain\(s\)"):
        model.transform(df)


def test_directly_constructed_single_domain_model_transforms(spark):
    """A loader rebuilds a GatedLDAModel directly from a saved VIResult, so NO
    Params are copied from a fit -- and a single-domain loader has no reason to
    set featuresCols. _transform reads that Param unconditionally, so without a
    model-level default it raises KeyError on a path with nothing to do with
    multi-domain. No _set call here on purpose."""
    import numpy as np
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator, GatedLDAModel
    parent = {1: 0, 2: 0}
    df = spark.createDataFrame(
        [(0, SparseVector(6, {0: 1.0, 1: 2.0}), [1]),
         (1, SparseVector(6, {2: 1.0, 3: 1.0}), [2])], ["rid", "features", "frontier"])
    fitted = GatedLDAEstimator(labelCol="frontier", parent=parent, nBg=1, tpn=1,
                               maxIter=1, seed=0).fit(df)
    rebuilt = GatedLDAModel(fitted.result, parent=parent, nBg=1, tpn=1)
    assert rebuilt.getOrDefault("featuresCols") == []      # unset = single-domain
    np.testing.assert_array_equal(_affinities_by_rid(rebuilt, df),
                                  _affinities_by_rid(fitted, df))


def test_single_domain_transform_rejects_an_omega_set_after_the_fit(spark):
    """A single-domain model has no domains to weight, so an omega set on it is
    rejected at transform instead of being silently dropped (the fit-time engine
    check cannot see a Param set afterwards)."""
    import pytest
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = spark.createDataFrame(
        [(SparseVector(6, {0: 1.0, 1: 2.0}), [1]),
         (SparseVector(6, {2: 1.0}), [2])], ["features", "frontier"])
    model = GatedLDAEstimator(labelCol="frontier", parent={1: 0, 2: 0}, nBg=1,
                              tpn=1, maxIter=1, seed=0).fit(df)
    model.transform(df).select("nodeAffinity").collect()      # fine without omega
    model._set(omega=[1.0, 0.5])
    with pytest.raises(ValueError, match="omega requires multi-domain mode"):
        model.transform(df)


def test_shim_omega_and_eta_per_domain_params_settable():
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    est = GatedLDAEstimator(parent={1: 0, 2: 0})
    assert not est.isSet("omega")                  # unset = unweighted MixEHR default
    assert not est.isSet("etaPerDomain")
    est2 = GatedLDAEstimator(parent={1: 0, 2: 0}, featuresCols=["c", "d"],
                             omega=[1.0, 0.25], etaPerDomain=[0.02, 0.1])
    assert est2.getOrDefault("omega") == [1.0, 0.25]
    assert est2.getOrDefault("etaPerDomain") == [0.02, 0.1]


def test_omega_without_features_cols_raises_rather_than_being_ignored(spark):
    """omega is forwarded whenever SET, not only when featuresCols is, so a
    per-domain weight on a single-domain fit hits the engine's named error. Gating
    it on featuresCols instead would discard the caller's omega silently and serve
    an unweighted theta they never asked for."""
    import pytest
    from pyspark.ml.linalg import SparseVector
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = spark.createDataFrame(
        [(SparseVector(6, {0: 1.0, 1: 2.0}), [1]),
         (SparseVector(6, {2: 1.0}), [2])], ["features", "frontier"])
    est = GatedLDAEstimator(labelCol="frontier", parent={1: 0, 2: 0}, nBg=1, tpn=1,
                            maxIter=1, seed=0, omega=[1.0, 0.5])
    with pytest.raises(ValueError, match="omega requires multi-domain mode"):
        est.fit(df)


def test_multidomain_fit_forwards_eta_per_domain(spark):
    """etaPerDomain reaches the engine's per-domain Dirichlet prior; unset keeps
    the scalar 1/K the single-domain path uses."""
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    df = _two_domain_df(spark)
    kw = dict(featuresCols=["c", "d"], labelCol="frontier",
              parent={1: 0, 2: 1, 3: 1}, nBg=2, tpn=1, maxIter=1, seed=0)
    m = GatedLDAEstimator(etaPerDomain=[0.02, 0.5], **kw).fit(df)
    assert m.result.metadata["eta_m"] == [0.02, 0.5]
    K = 2 + 3                                       # n_bg + nodes * tpn
    m_default = GatedLDAEstimator(**kw).fit(df)
    assert m_default.result.metadata["eta_m"] == [1.0 / K, 1.0 / K]


def test_multidomain_save_load_reproduces_node_affinity(spark, tmp_path):
    """The whole persistence story end to end: fit multi-domain through the shim,
    score, save, load, rebuild the Model from the loaded VIResult, score again --
    and get the SAME nodeAffinity. A dict lambda that round-trips with string keys
    or reordered blocks would load cleanly and score differently.

    Comparison is rid-keyed via `_affinities_by_rid`, not positional: collect()
    order is not stable across two equivalently-built repartition(2) frames (see
    `_two_domain_df`). omega is deliberately left UNSET on the rebuilt model --
    `_transform_omega` falls back to `result.metadata["omega"]`, so this test
    exercises that metadata path rather than papering over it with an explicit
    `_set`. featuresCols IS set on the rebuilt model: a loader knows its own input
    column names, but they are not recorded in the metadata (only the widths are)."""
    from spark_vi.io.export import load_result, save_result
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator, GatedLDAModel
    df = _two_domain_df(spark)
    parent = {1: 0, 2: 1, 3: 1}
    est = GatedLDAEstimator(featuresCols=["c", "d"], labelCol="frontier",
                            parent=parent, nBg=2, tpn=1,
                            maxIter=2, seed=0, omega=[1.0, 0.4])
    model = est.fit(df)
    before = _affinities_by_rid(model, df)

    save_result(model.result, tmp_path / "fit")
    loaded = load_result(tmp_path / "fit")
    assert isinstance(loaded.global_params["lambda"], dict)
    assert sorted(loaded.global_params["lambda"]) == [0, 1]
    assert all(isinstance(k, int) for k in loaded.global_params["lambda"])
    assert loaded.metadata["domains"] == [6, 4]
    assert loaded.metadata["omega"] == [1.0, 0.4]

    rebuilt = GatedLDAModel(loaded, parent=parent, nBg=2, tpn=1)
    rebuilt._set(featuresCols=["c", "d"])          # a loader knows its input columns
    assert not rebuilt.isSet("omega")              # metadata fallback does the work
    after = _affinities_by_rid(rebuilt, df)
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=0.0)


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
