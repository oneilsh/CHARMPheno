"""Artifact-level tests for dist/spark_vi.zip.

Phase 3 promises that `dist/spark_vi.zip` is a flat, pure-Python archive that
can be shipped to Spark executors via `sc.addPyFile(...)` or `--py-files`.
Two tests cover two things:
  1. Structural: the zip contains the expected package layout and nothing
     spurious (no __pycache__, no .egg-info).
  2. Behavioral: the zip alone is sufficient to import spark_vi's top-level
     module in a fresh, isolated Python subprocess. We use `python -S` to
     disable site-packages .pth processing so that the editable install
     present in a dev environment cannot accidentally satisfy the import.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ZIP_PATH = Path(__file__).resolve().parents[1] / "dist" / "spark_vi.zip"

EXPECTED_CONTENTS = {
    "spark_vi/__init__.py",
    "spark_vi/core/__init__.py",
    "spark_vi/core/config.py",
    "spark_vi/core/model.py",
    "spark_vi/core/result.py",
    "spark_vi/core/runner.py",
    "spark_vi/models/__init__.py",
    "spark_vi/models/counting.py",
    "spark_vi/models/online_hdp.py",
    "spark_vi/io/__init__.py",
    "spark_vi/io/export.py",
    "spark_vi/diagnostics/__init__.py",
    "spark_vi/diagnostics/checkpoint.py",
}


def test_spark_vi_zip_contains_expected_structure():
    """The zip has the flat spark_vi package and no build artifacts."""
    if not ZIP_PATH.is_file():
        pytest.skip("dist/spark_vi.zip not built; run `make zip` first")
    import zipfile
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = set(z.namelist())
    missing = EXPECTED_CONTENTS - names
    assert not missing, f"zip missing expected files: {missing}"
    assert not any("__pycache__" in n for n in names), \
        "zip contains __pycache__ entries"
    assert not any(".egg-info" in n for n in names), \
        "zip contains egg-info entries"
    assert not any(n.endswith(".pyc") for n in names), \
        "zip contains .pyc bytecode"


def test_spark_vi_zip_top_level_imports_in_isolation(tmp_path):
    """`python -S` prevents the subprocess from picking up site-package
    .pth files (which the dev environment's editable install creates).
    With the zip as the only PYTHONPATH entry, importing spark_vi proves
    the zip itself contains the top-level module.

    We only import `spark_vi` (not its submodules) because submodules pull
    in numpy/scipy/pyspark which aren't available under -S. That's fine:
    this test covers the zip packaging; other tests cover module contents.
    """
    if not ZIP_PATH.is_file():
        pytest.skip("dist/spark_vi.zip not built; run `make zip` first")
    # Copy to an isolated dir so no parent-directory .pth files interfere.
    staged = tmp_path / "spark_vi.zip"
    shutil.copy(ZIP_PATH, staged)
    proc = subprocess.run(
        [sys.executable, "-S", "-c",
         "import spark_vi; print(spark_vi.__version__)"],
        env={"PYTHONPATH": str(staged), "PATH": "/usr/bin:/bin"},
        cwd=str(tmp_path),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"zip-only import failed:\n"
        f"stdout={proc.stdout!r}\n"
        f"stderr={proc.stderr!r}"
    )
    assert proc.stdout.strip() == "0.1.0"
