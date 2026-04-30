"""Pre-built models for spark-vi."""
from spark_vi.models.counting import CountingModel
from spark_vi.models.lda import VanillaLDA
from spark_vi.models.online_hdp import OnlineHDP

__all__ = ["CountingModel", "OnlineHDP", "VanillaLDA"]
