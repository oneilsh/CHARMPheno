"""Regression: spark_vi.io.export must be importable BEFORE spark_vi.core.

spark_vi.io.export imports spark_vi.core.result, which triggers
spark_vi.core.__init__ -> spark_vi.core.runner. runner used to import
load_result/save_result from export at MODULE level, so importing export first
(as a post-hoc readout does with `from spark_vi.io.export import load_result`)
deadlocked with a partially-initialized-module ImportError. runner now imports
those names lazily inside fit(), breaking the cycle.

Run in a FRESH subprocess: an in-process test is masked once any other test has
already imported spark_vi.core (the module cache hides the ordering bug).
"""
import subprocess
import sys


def test_export_importable_before_core():
    # If the cycle regresses, this subprocess exits non-zero with the
    # "cannot import name 'load_result' ... partially initialized module" error.
    code = "from spark_vi.io.export import load_result, save_result; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"export-first import failed (circular import regressed):\n{r.stderr}")
    assert "ok" in r.stdout
