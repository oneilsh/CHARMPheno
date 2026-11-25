"""OnlineSTMModel.transform and the covariate-formula path end to end (ADR 0056).

transform must give, per document, exactly the gated mode theta of the per-document
Laplace E-step (``_stm_doc_inference`` -> ``_gated_mode_theta``) that the corpus
helpers compute; a formula model must encode covariates identically before and
after a save / load (the spec is rebuilt from JSON, not unpickled)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyspark")
from pyspark.ml.linalg import DenseVector, SparseVector


def _oracle_theta(model, docs):
    """theta per (indices, counts, x, groups) by the engine's own primitives."""
    from scipy.special import digamma

    from spark_vi.mllib.topic.stm import _gated_mode_theta
    from spark_vi.models.topic._linalg import safe_inverse
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.stm import _stm_doc_inference

    gp = model.result.global_params
    lam = gp["lambda"]
    K = lam.shape[0]
    eb = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    part = model.topicBlockPartition() or TopicBlockPartition("", background_k=K, foreground=())
    ref = 0 if model.result.metadata["stm_hardening"]["reference_topic"] else None
    out = []
    for idx, cnt, x, groups in docs:
        allowed = part.allowed_indices(groups)
        prec = safe_inverse(gp["Sigma"][np.ix_(allowed, allowed)])
        eta, _, _ = _stm_doc_inference(
            indices=np.asarray(idx), counts=np.asarray(cnt, dtype=float),
            expElogbeta=eb, Gamma=gp["Gamma"], Sigma_inv_allowed=prec,
            x=np.asarray(x, dtype=float), max_iter=model.getOrDefault("lbfgsMaxIter"),
            tol=model.getOrDefault("lbfgsTol"), allowed=allowed, reference=ref)
        out.append(_gated_mode_theta(eta, allowed, K))
    return np.asarray(out)


def _cov_rows():
    return [
        (SparseVector(8, [0, 2], [3.0, 1.0]), DenseVector([1.0, 0.0])),
        (SparseVector(8, [1, 3], [2.0, 2.0]), DenseVector([0.0, 1.0])),
        (SparseVector(8, [0, 4], [1.0, 2.0]), DenseVector([1.0, 0.5])),
        (SparseVector(8, [5, 6], [1.0, 1.0]), DenseVector([0.0, 1.0])),
        (SparseVector(8, [2, 7], [2.0, 1.0]), DenseVector([1.0, 0.2])),
        (SparseVector(8, [1, 6], [1.0, 3.0]), DenseVector([0.0, 0.8])),
    ]


def test_transform_matches_the_e_step_oracle(spark):
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    df = spark.createDataFrame(_cov_rows(), ["features", "covariates"])
    model = OnlineSTMEstimator(k=3, covariateNames=["a", "b"], seed=0,
                               spectralInit=False, maxIter=3,
                               subsamplingRate=1.0).fit(df)
    out = model.transform(df).collect()
    got = np.asarray([r["topicDistribution"].toArray() for r in out])
    want = _oracle_theta(model, [
        (r["features"].indices, r["features"].values, r["covariates"].toArray(),
         frozenset()) for r in out])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.sum(axis=1), 1.0)
    assert set(out[0].asDict()) == {"features", "covariates", "topicDistribution"}


def test_gated_transform_zeroes_closed_blocks_and_matches_oracle(spark):
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition("grp", background_k=2, foreground=(("rare", 1),))
    rows = []
    for i in range(20):
        rows.append((SparseVector(5, {i % 5: 2.0, (i + 1) % 5: 1.0}),
                     DenseVector([1.0, float(i % 2)]),
                     "rare" if i % 4 == 0 else "common"))
    df = spark.createDataFrame(rows, ["features", "covariates", "grp"])
    model = OnlineSTMEstimator(k=3, covariateNames=["Intercept", "age"],
                               topicBlocks=part, docGroupCol="grp", seed=0,
                               spectralInit=False, maxIter=3,
                               subsamplingRate=1.0).fit(df)
    out = model.transform(df).collect()
    got = np.asarray([r["topicDistribution"].toArray() for r in out])
    common = np.asarray([r["grp"] == "common" for r in out])
    assert np.all(got[common, 2] == 0.0)          # the rare block is closed
    assert np.all(got[~common, 2] > 0.0)
    want = _oracle_theta(model, [
        (r["features"].indices, r["features"].values, r["covariates"].toArray(),
         frozenset({r["grp"]})) for r in out])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def _pandas(df):
    """A small DataFrame as pandas without toPandas (the suite's conftest stubs
    distutils, which toPandas' version check trips on)."""
    import pandas as pd
    return pd.DataFrame([r.asDict() for r in df.collect()])


def _formula_df(spark):
    rows = []
    sexes = ["F", "M", "X"]
    for i in range(24):
        rows.append((SparseVector(6, {i % 6: 2.0, (i + 2) % 6: 1.0}),
                     sexes[i % 3], float(20 + i), i % 2))
    return spark.createDataFrame(rows, ["features", "sex", "age", "site"])


def test_formula_fit_resolves_names_and_persists_json_spec(spark, tmp_path):
    from spark_vi.mllib.topic.stm import OnlineSTMModel, OnlineSTMEstimator
    df = _formula_df(spark)
    est = OnlineSTMEstimator(k=3, covariateFormula="~ C(sex) + age + C(site)", seed=0,
                             spectralInit=False, maxIter=2, subsamplingRate=1.0)
    model = est.fit(df)
    names = model.getOrDefault("covariateNames")
    assert names == ["Intercept", "C(sex)[T.M]", "C(sex)[T.X]", "age",
                     "C(site)[T.1]"]
    assert model.result.global_params["Gamma"].shape == (5, 3)
    assert model.covariateSpec["formula"] == "~ C(sex) + age + C(site)"
    assert model.covariateSpec["categorical_cols"] == ["sex", "site"]

    before = model.transform(df).collect()
    out = str(tmp_path / "stm_formula")
    model.save(out)
    assert not list((tmp_path / "stm_formula").rglob("*.pkl"))
    loaded = OnlineSTMModel.load(out)
    assert loaded.covariateSpec == model.covariateSpec
    pdf = _pandas(df.select("sex", "age", "site"))
    assert loaded.encodeCovariates(pdf).tobytes() == model.encodeCovariates(pdf).tobytes()
    after = loaded.transform(df).collect()
    for a, b in zip(before, after):
        assert a["topicDistribution"].toArray().tobytes() == \
            b["topicDistribution"].toArray().tobytes()


def test_formula_matches_the_prebuilt_covariate_column(spark):
    """The formula path is an encoding step only: a fit on the formula equals a fit
    on the same design matrix given as a covariate column (same seed)."""
    from pyspark.ml.linalg import DenseVector as DV
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    df = _formula_df(spark)
    kw = dict(k=3, seed=0, spectralInit=False, maxIter=2, subsamplingRate=1.0)
    m_formula = OnlineSTMEstimator(covariateFormula="~ C(sex) + age", **kw).fit(
        df.coalesce(1))
    X = m_formula.encodeCovariates(_pandas(df.select("sex", "age")))
    rows = [(r["features"], DV(x)) for r, x in zip(df.collect(), X)]
    df_a = spark.createDataFrame(rows, ["features", "covariates"]).coalesce(1)
    m_col = OnlineSTMEstimator(covariateNames=m_formula.getOrDefault("covariateNames"),
                               **kw).fit(df_a)
    for key in ("lambda", "Gamma", "Sigma"):
        np.testing.assert_allclose(m_formula.result.global_params[key],
                                   m_col.result.global_params[key], rtol=1e-10)


def test_formula_nulls_dropped_at_fit_and_null_at_transform(spark):
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    df = _formula_df(spark)
    rows = df.collect()
    nulls = [(rows[0]["features"], None, 30.0, 0),
             (rows[1]["features"], "F", None, 1)]
    df_n = spark.createDataFrame(list(map(tuple, rows)) + nulls, df.schema)
    model = OnlineSTMEstimator(k=3, covariateFormula="~ C(sex) + age", seed=0,
                               spectralInit=False, maxIter=1,
                               subsamplingRate=1.0).fit(df_n)
    out = model.transform(df_n).collect()
    thetas = [r["topicDistribution"] for r in out]
    assert sum(t is None for t in thetas) == 2
    assert all(t is not None for t in thetas[:len(rows)])


def test_formula_transform_fails_on_an_unseen_level(spark):
    from spark_vi.mllib.topic.stm import OnlineSTMEstimator
    df = _formula_df(spark)
    model = OnlineSTMEstimator(k=3, covariateFormula="~ C(sex) + age", seed=0,
                               spectralInit=False, maxIter=1,
                               subsamplingRate=1.0).fit(df)
    unseen = spark.createDataFrame(
        [(SparseVector(6, {0: 1.0}), "Q", 40.0, 0)], df.schema)
    with pytest.raises(Exception, match="not seen"):
        model.transform(unseen).collect()
