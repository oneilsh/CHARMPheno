"""Zip artifact is importable when placed on sys.path.

This simulates what `sc.addPyFile('spark_vi.zip')` does on a Spark worker.
"""
import os
import site
import sys
import subprocess
from pathlib import Path


def test_spark_vi_zip_imports_in_fresh_subprocess():
    zip_path = Path(__file__).resolve().parents[1] / "dist" / "spark_vi.zip"
    if not zip_path.is_file():
        import pytest
        pytest.skip(f"dist/spark_vi.zip not built; run `make zip` first")

    # The fresh subprocess only sees what we put on PYTHONPATH. Include:
    #   - the zip itself (simulating sc.addPyFile), and
    #   - the current venv's site-packages (so numpy/scipy are importable).
    # The second entry stands in for whatever dependency resolution a real
    # Spark worker gets via its base image / --archives.
    venv_site = site.getsitepackages()[0]
    pythonpath = os.pathsep.join([str(zip_path), venv_site])

    proc = subprocess.run(
        [sys.executable, "-c",
         "import spark_vi; "
         "from spark_vi.core import VIModel, VIRunner, VIConfig, VIResult; "
         "from spark_vi.models import CountingModel, OnlineHDP; "
         "print(spark_vi.__version__)"],
        env={"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"zip import failed:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert proc.stdout.strip() == "0.1.0"
