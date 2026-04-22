"""Pytest fixtures for charmpheno.

Session-scoped local Spark (shared with integration tests). Same config as
spark-vi's conftest to keep behavior identical across packages.
"""
import os
import warnings

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
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
