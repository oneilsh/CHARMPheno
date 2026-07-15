"""Fixtures for script-level tests."""
import os
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make scripts/ importable in tests so we can call functions directly.
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Python 3.12 compat: pin the Spark worker to the venv interpreter so it finds
# the same packages as the driver (e.g. numpy, formulaic).  Without this the
# executor falls back to the system Python, which lacks distutils (removed in
# 3.12) and fails to deserialize PySpark's own ML UDFs.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    """Run the test from inside a fresh tmp dir so relative-path defaults work."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def spark():
    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    warnings.filterwarnings("ignore")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    session = (
        SparkSession.builder.master("local[2]")
        .appName("charmpheno-scripts-tests")
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
