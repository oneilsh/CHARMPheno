"""End-to-end wrapper tests for CharmPhenoHDP.

The wrapper is a thin clinical layer around `spark_vi.models.OnlineHDP`.
Construction validation runs in unit tier; fit/transform exercise the
full Spark-local path in slow tier.
"""
import numpy as np
import pytest


def _tiny_corpus():
    from spark_vi.core import BOWDocument
    rng = np.random.default_rng(0)
    docs = []
    for d in range(20):
        n = int(rng.integers(2, 6))
        idx = np.sort(rng.choice(20, size=n, replace=False)).astype(np.int32)
        cnt = rng.gamma(2.0, 1.0, n).astype(np.float64)
        docs.append(BOWDocument(
            indices=idx,
            counts=cnt,
            length=int(cnt.sum()),
        ))
    return docs


def test_charm_pheno_hdp_constructor_validates():
    from charmpheno.phenotype import CharmPhenoHDP

    with pytest.raises(ValueError):
        CharmPhenoHDP(vocab_size=0)

    m = CharmPhenoHDP(vocab_size=20, max_topics=5, max_doc_topics=3)
    assert m.vocab_size == 20
    assert m.max_topics == 5
    assert m.max_doc_topics == 3
    # Translation to inner OnlineHDP names.
    assert m.model.T == 5
    assert m.model.K == 3


def test_charm_pheno_hdp_exposes_underlying_online_hdp():
    """The wrapper's .model attribute is the spark_vi OnlineHDP instance."""
    from spark_vi.models import OnlineHDP
    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=10)
    assert isinstance(m.model, OnlineHDP)


@pytest.mark.slow
def test_charm_pheno_hdp_fit_smoke_tiny(spark):
    """Tiny end-to-end fit completes; ELBO trace is finite."""
    from charmpheno.phenotype import CharmPhenoHDP
    from spark_vi.core import VIConfig

    docs = _tiny_corpus()
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()

    np.random.seed(0)
    m = CharmPhenoHDP(vocab_size=20, max_topics=5, max_doc_topics=3)
    result = m.fit(rdd, config=VIConfig(max_iterations=5))

    assert result.elbo_trace is not None
    assert all(np.isfinite(v) for v in result.elbo_trace)


@pytest.mark.slow
def test_charm_pheno_hdp_transform_returns_simplex_thetas(spark):
    """fit then transform; per-doc θ is a length-T simplex vector."""
    from charmpheno.phenotype import CharmPhenoHDP
    from spark_vi.core import VIConfig

    docs = _tiny_corpus()
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()

    np.random.seed(0)
    m = CharmPhenoHDP(vocab_size=20, max_topics=5, max_doc_topics=3)
    m.fit(rdd, config=VIConfig(max_iterations=5))

    out = m.transform(rdd).collect()
    assert len(out) == len(docs)
    for _, theta in out:
        assert theta.shape == (5,)
        assert np.isclose(theta.sum(), 1.0, atol=1e-6)
        assert np.all(theta >= 0)
