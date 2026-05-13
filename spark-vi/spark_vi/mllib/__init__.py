"""MLlib Estimator/Transformer shims for spark_vi models.

Topic-model shims live under spark_vi.mllib.topic; this file re-exports
their public surface for the convenience top-level
`from spark_vi.mllib import OnlineLDAEstimator` style imports.

Opt-in import path — `import spark_vi` does not transitively load this
subpackage, so users who don't need MLlib integration don't pay the
pyspark.ml import cost. Per ADR 0009.
"""
from spark_vi.mllib.topic import (
    OnlineHDPEstimator,
    OnlineHDPModel,
    OnlineLDAEstimator,
    OnlineLDAModel,
)

__all__ = [
    "OnlineHDPEstimator",
    "OnlineHDPModel",
    "OnlineLDAEstimator",
    "OnlineLDAModel",
]
