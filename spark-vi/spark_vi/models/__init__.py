"""Pre-built models for spark-vi.

The topic-model implementations live under spark_vi.models.topic. This file
re-exports their public surface for backward-compatibility-by-default of the
top-level `from spark_vi.models import OnlineLDA` style imports.
"""
from spark_vi.models.topic import CountingModel, OnlineHDP, OnlineLDA

__all__ = ["CountingModel", "OnlineHDP", "OnlineLDA"]
