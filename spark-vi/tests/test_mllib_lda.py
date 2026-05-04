"""Tests for spark_vi.mllib.lda — fast unit tests for the MLlib shim."""
from __future__ import annotations

import pytest


def test_default_params_match_mllib_lda(spark):
    """Each shared Param defaults to the same value pyspark.ml.clustering.LDA uses."""
    from pyspark.ml.clustering import LDA as MLlibLDA
    from spark_vi.mllib.lda import VanillaLDAEstimator

    ours = VanillaLDAEstimator()
    theirs = MLlibLDA()

    for name in [
        "k", "maxIter", "featuresCol", "topicDistributionCol",
        "optimizer", "learningOffset", "learningDecay",
        "subsamplingRate", "optimizeDocConcentration",
    ]:
        assert ours.getOrDefault(name) == theirs.getOrDefault(name), (
            f"Param {name!r} default mismatch: ours={ours.getOrDefault(name)!r} "
            f"theirs={theirs.getOrDefault(name)!r}"
        )


def test_our_extras_have_adr_0008_defaults():
    from spark_vi.mllib.lda import VanillaLDAEstimator

    e = VanillaLDAEstimator()
    assert e.getOrDefault("gammaShape") == 100.0
    assert e.getOrDefault("caviMaxIter") == 100
    assert e.getOrDefault("caviTol") == 1e-3
