"""spark-vi: a PySpark-native framework for distributed variational inference.

Public API:

    from spark_vi.core import VIModel, VIRunner, LocalRunner, VIConfig, VIResult

Two engines run one model contract: VIRunner on Spark, LocalRunner in memory
(ADR 0051). Pre-built models live in spark_vi.models.topic (LDA, HDP, STM, the
DAG-gated LDA) with MLlib shims in spark_vi.mllib; the per-node readout over any
frozen feature column is spark_vi.readout.
"""
__version__ = "0.1.0"
