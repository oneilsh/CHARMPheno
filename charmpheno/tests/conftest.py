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
