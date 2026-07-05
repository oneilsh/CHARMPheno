"""Light smoke test for scripts/concentration_recovery_experiment.py.

The math (planting, recovery, held-out gold standard) is already unit-tested
in spark-vi/tests/eval/test_concentration_recovery.py (CR-1/CR-2); this
driver only orchestrates those calls, so we just check it runs end-to-end on
a tiny config and returns the expected shape.
"""
from __future__ import annotations

import numpy as np

# scripts/tests/conftest.py already inserts scripts/ into sys.path.
import concentration_recovery_experiment as cre


class TestRunSmallSmoke:
    def test_run_small_smoke(self):
        results = cre.run(
            K=4, V=80, D=40, doc_len=30, holdout_frac=0.3, seed=0,
            mechanism_levels={"logistic_normal": [4]},
            c_grid=[1, 4], alpha_grid=[0.1, 1.0],
        )

        assert set(results.keys()) == {"config", "cells"}
        assert len(results["cells"]) == 1

        cell = results["cells"][0]
        assert cell["mechanism"] == "logistic_normal"
        assert cell["level"] == 4

        assert np.isfinite(cell["planted"]["top_mass_p50"])
        assert np.isfinite(cell["planted"]["eff_topics_p50"])

        assert cell["stm_heldout"]["argmax_c"] in (1, 4)
        assert np.isfinite(cell["stm_heldout"]["recovered_top_mass_p50"])
        assert np.isfinite(cell["stm_heldout"]["abs_err"])

        assert cell["lda_heldout"]["argmax_alpha"] in (0.1, 1.0)
        assert np.isfinite(cell["lda_heldout"]["recovered_top_mass_p50"])
        assert np.isfinite(cell["lda_heldout"]["abs_err"])

        assert np.isfinite(cell["lda_opt_alpha"]["recovered_top_mass_p50"])
        assert np.isfinite(cell["lda_opt_alpha"]["abs_err"])

        table = cre.render_markdown_table(results)
        assert "logistic_normal" in table

        summary = cre.build_summary(results)
        assert isinstance(summary, str) and len(summary) > 0
