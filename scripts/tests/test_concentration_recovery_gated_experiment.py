"""Light smoke test for scripts/concentration_recovery_gated_experiment.py.

The math (gated planting, gated held-out sweep) is already unit-validated by
spark-vi/tests/test_heldout_scale_sweep.py (HS-1, including the KEY
argmax-recovers-planted-scale test this script's planting/global_params
construction mirrors); this driver only orchestrates those calls for a small
grid of planted scales x {gated, non-gated} partitions, so we just check it
runs end-to-end on a tiny config and returns the expected shape.
"""
from __future__ import annotations

import numpy as np

# scripts/tests/conftest.py already inserts scripts/ into sys.path.
import concentration_recovery_gated_experiment as crge


class TestRunSmallSmoke:
    def test_run_small_smoke(self):
        results = crge.run(
            groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=30, doc_len=25,
            holdout_frac=0.3, seed=0, scales=[5.0], c_grid=[1, 5, 20],
        )

        assert set(results.keys()) == {"config", "cells"}
        assert results["config"]["K"] == 4  # bg_k=2 + 2 groups * fg_per_group=1
        assert len(results["cells"]) == 1

        cell = results["cells"][0]
        assert cell["s"] == 5.0
        assert np.isfinite(cell["planted"]["top_mass"])
        assert np.isfinite(cell["planted"]["eff_topics"])

        for regime in ("gated", "nongated"):
            block = cell[regime]
            assert block["argmax_c"] in (1, 5, 20)
            assert np.isfinite(block["recovered_top_mass"])
            assert np.isfinite(block["recovered_eff_topics"])
            assert np.isfinite(block["abs_err"])
            assert np.isfinite(block["leaked_mass"])
            assert block["n_docs"] > 0

        # Gated recovery never assigns mass to a topic outside the doc's true
        # allowed set (the gated softmax is exactly 0 there by construction).
        assert cell["gated"]["leaked_mass"] == 0.0

        table = crge.render_markdown_table(results)
        assert "GATED_argmax_c" in table

        summary = crge.build_summary(results)
        assert isinstance(summary, str) and len(summary) > 0
