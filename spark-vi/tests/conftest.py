"""Pytest fixtures for spark-vi.

The session-scoped local Spark session is the only fixture all tests share.
"""
import os
import pathlib
import sys
import warnings

# Put the tests/ directory on sys.path so that helper modules starting with _
# (e.g. _stm_synth) can be imported with a plain `from _stm_synth import ...`
# alongside regular `from spark_vi...` imports.
_TESTS_DIR = str(pathlib.Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Python 3.12 compat: distutils was removed, but PySpark 3.5 still tries to import it
# from pyspark.ml.image. Create a mock module before PySpark imports.
if sys.version_info >= (3, 12):
    from unittest.mock import MagicMock
    sys.modules['distutils'] = MagicMock()
    sys.modules['distutils.version'] = MagicMock()

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
