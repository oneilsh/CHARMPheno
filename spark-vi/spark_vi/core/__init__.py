"""Public API for spark_vi.core.

BOWDocument used to live here; it moved to spark_vi.models.topic.types
because it's a topic-model row type, not a generic framework primitive.
Import it from spark_vi.models.topic.
"""
from spark_vi.core.config import VIConfig
from spark_vi.core.model import VIModel
from spark_vi.core.result import VIResult
from spark_vi.core.runner import VIRunner

__all__ = ["VIConfig", "VIModel", "VIResult", "VIRunner"]
