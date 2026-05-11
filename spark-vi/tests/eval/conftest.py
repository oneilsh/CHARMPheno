"""Eval-tier pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def sc(spark):
    """SparkContext derived from the shared session-scoped spark fixture."""
    return spark.sparkContext
