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


def test_gated_shim_spectral_vocab_guard_raises(spark):
    from spark_vi.mllib.topic.gated_lda import GatedLDAEstimator
    from pyspark.ml.linalg import SparseVector
    import pytest
    # V = 6 features but spectralMaxVocab = 4 -> dense V x V guard trips.
    df = spark.createDataFrame(
        [(SparseVector(6, [0, 1], [1.0, 1.0]), [1])],
        ["features", "frontier"],
    )
    est = GatedLDAEstimator(parent={1: 0, 2: 0}, init="spectral",
                            spectralMaxVocab=4, maxIter=1)
    with pytest.raises(NotImplementedError, match="scalable"):
        est.fit(df)


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
