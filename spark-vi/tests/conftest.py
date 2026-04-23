"""Pytest fixtures for spark-vi.

The session-scoped local Spark session is the only fixture all tests share.
"""
import os
import sys
import warnings

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Session-scoped local Spark (2 cores, small shuffle, UI disabled).

    Why 2 cores: large enough to exercise parallel map/reduce, small enough
    to keep startup cost negligible in CI and local iteration.
    """
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    # Pin worker Python to the driver's interpreter. Dataproc presets
    # PYSPARK_PYTHON to the cluster's system conda Python, which has a
    # different numpy than our poetry venv; broadcast unpickling then
    # fails across the boundary. sys.executable = the venv we're in.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[2]")
        .appName("spark-vi-tests")
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
