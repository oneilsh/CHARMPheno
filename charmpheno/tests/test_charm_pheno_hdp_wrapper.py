"""CharmPhenoHDP is a thin wrapper around spark_vi.models.OnlineHDP.

Its public surface is real (construction validated, API in place); its
.fit() propagates the underlying OnlineHDP stub's NotImplementedError
until the real OnlineHDP lands.
"""
import pytest


def test_charm_pheno_hdp_constructs_with_vocab_size():
    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=500, max_topics=50)
    assert m.vocab_size == 500
    assert m.max_topics == 50


def test_charm_pheno_hdp_rejects_invalid_vocab_size():
    from charmpheno.phenotype import CharmPhenoHDP

    with pytest.raises(ValueError):
        CharmPhenoHDP(vocab_size=0)


def test_charm_pheno_hdp_fit_raises_not_implemented(spark):
    """Until the real OnlineHDP lands, fit raises NotImplementedError."""
    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=10)
    # We don't need real data — the stub raises before touching the rdd.
    empty_rdd = spark.sparkContext.parallelize([], numSlices=1)
    with pytest.raises(NotImplementedError):
        m.fit(empty_rdd)


def test_charm_pheno_hdp_exposes_underlying_online_hdp():
    """The wrapper's .model attribute is the spark_vi OnlineHDP instance."""
    from spark_vi.models import OnlineHDP

    from charmpheno.phenotype import CharmPhenoHDP

    m = CharmPhenoHDP(vocab_size=10)
    assert isinstance(m.model, OnlineHDP)
