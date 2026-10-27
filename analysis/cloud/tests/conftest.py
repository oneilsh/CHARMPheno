"""Shared fixtures for analysis/cloud driver tests: import path + local Spark."""
import os
import sys
import warnings
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

_CLOUD = str(Path(__file__).resolve().parent.parent)   # analysis/cloud
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)


@pytest.fixture(scope="session")
def spark():
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[2]")
        .appName("cloud-tests")
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
    """Every test that starts local Spark is `slow` (AGENTS.md, Testing); see
    spark-vi/tests/conftest.py for the rule. `tryfirst` so the mark exists before
    `-m` deselects."""
    for item in items:
        if "spark" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)
