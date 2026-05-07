"""MLlib Estimator/Transformer shims for spark_vi models.

Opt-in import path — `import spark_vi` does not transitively load this
subpackage, so users who don't need MLlib integration don't pay the
pyspark.ml import cost. Per ADR 0009.
"""
from spark_vi.mllib.hdp import OnlineHDPEstimator, OnlineHDPModel
from spark_vi.mllib.lda import VanillaLDAEstimator, VanillaLDAModel

__all__ = [
    "OnlineHDPEstimator",
    "OnlineHDPModel",
    "VanillaLDAEstimator",
    "VanillaLDAModel",
]
