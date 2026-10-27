"""Pytest fixtures for charmpheno.

Session-scoped local Spark (shared with integration tests). Same config as
spark-vi's conftest to keep behavior identical across packages.
"""
import os
import sys
import warnings

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    # Pin worker Python to the driver's interpreter (mirrors spark-vi's
    # conftest). Without this, workers default to `python3` on PATH — which
    # may be a different venv whose distutils/numpy don't match the driver,
    # breaking broadcast unpickling and pyspark.ml imports. sys.executable =
    # the venv we're running in.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[2]")
        .appName("charmpheno-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Every test that starts local Spark is `slow` (AGENTS.md, Testing).

    The session's JVM alone costs seconds and each Spark action ~1 s, so a Spark
    test cannot live in the unit suite's ~10 s budget. Marking by fixture keeps the
    rule mechanical: a new Spark test is slow without anyone remembering to say so.
    Pure-numpy tests that are slow for other reasons carry `@pytest.mark.slow`
    explicitly. `tryfirst` so the mark exists before `-m` deselects."""
    for item in items:
        if "spark" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)
