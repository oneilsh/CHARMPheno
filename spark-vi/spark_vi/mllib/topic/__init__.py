"""MLlib Estimator/Model shims for spark_vi topic models.

OnlineLDAEstimator / OnlineLDAModel: streaming variational Bayes for LDA.
OnlineHDPEstimator / OnlineHDPModel: streaming variational Bayes for HDP.
DagGatedLDAEstimator / DagGatedLDAModel: LDA gated by a label DAG (ADR 0050).

Lives under topic/ to mirror the spark_vi.models.topic and
spark_vi.eval.topic namespacing — anything topic-specific gets a topic/
subpackage. The parent spark_vi.mllib namespace stays available for any
future non-topic shim (e.g. Gaussian-mixture, factor analysis).
"""
from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator, DagGatedLDAModel
from spark_vi.mllib.topic.hdp import OnlineHDPEstimator, OnlineHDPModel
from spark_vi.mllib.topic.lda import OnlineLDAEstimator, OnlineLDAModel

__all__ = [
    "DagGatedLDAEstimator",
    "DagGatedLDAModel",
    "OnlineHDPEstimator",
    "OnlineHDPModel",
    "OnlineLDAEstimator",
    "OnlineLDAModel",
]
