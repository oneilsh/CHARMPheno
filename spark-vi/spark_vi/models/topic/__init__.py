"""Topic-model implementations of the spark_vi.core.VIModel contract.

OnlineLDA: streaming variational Bayes for Latent Dirichlet Allocation.
OnlineHDP: streaming variational Bayes for the Hierarchical Dirichlet Process.
DagGatedOnlineLDA: LDA whose training E-step is gated by a label DAG (ADR 0050).
CountingModel: trivial coin-flip-posterior reference model used to exercise
    the framework contract end-to-end. Not a topic model in the LDA/HDP sense;
    lives under topic/ because it shares the bag-of-words input shape and is
    used as a contract-conformance fixture for topic-model infrastructure.
"""
from spark_vi.models.topic.counting import CountingModel
from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
from spark_vi.models.topic.lda import OnlineLDA
from spark_vi.models.topic.online_hdp import OnlineHDP
from spark_vi.models.topic.types import BOWDocument, GatedBOWDocument

__all__ = [
    "BOWDocument",
    "CountingModel",
    "DagGatedOnlineLDA",
    "GatedBOWDocument",
    "OnlineHDP",
    "OnlineLDA",
]
