"""Fast smoke test for scripts/marginalized_scale_decomposition_experiment.py.

The math (planting, MAP held-out sweep, marginalized held-out sweep, the
log-quadratic c* reducer) is already unit-tested in spark-vi (CR-1/CR-2 and
the Task-2 marginalized sweep). This driver only orchestrates those calls, so
the smoke test just checks it runs end-to-end on a tiny config and returns the
expected shape. It deliberately does NOT assert the scientific direction (drift
sign/magnitude) -- at this size the estimate is far too underpowered for that.
"""
from __future__ import annotations

import numbers

import numpy as np

# scripts/tests/conftest.py already inserts scripts/ into sys.path.
import marginalized_scale_decomposition_experiment as msd


class TestRunSmallSmoke:
    def test_run_small_smoke(self):
        holdouts = [0.5, 0.9]
        grid = np.round(np.geomspace(0.5, 20.0, 6), 4).tolist()
        results = msd.run(
            regimes={
                "smoke": dict(K=6, V=120, doc_len=44, D=40),
            },
            level=5.0,
            c_grid=grid,
            holdouts=holdouts,
            n_samples=16,
            seed=0,
        )

        assert set(results.keys()) == {"config", "regimes"}
        assert set(results["regimes"].keys()) == {"smoke"}

        cell = results["regimes"]["smoke"]

        # Per regime: map_cstar and marg_cstar keyed by holdout fraction.
        assert set(cell["map_cstar"].keys()) == set(holdouts)
        assert set(cell["marg_cstar"].keys()) == set(holdouts)
        for h in holdouts:
            assert np.isfinite(cell["map_cstar"][h])
            assert np.isfinite(cell["marg_cstar"][h])

        # A numeric marginalized residual drift (max-min across holdouts).
        assert isinstance(cell["marg_residual_drift"], numbers.Real)
        assert np.isfinite(cell["marg_residual_drift"])
        assert isinstance(cell["map_residual_drift"], numbers.Real)
        assert np.isfinite(cell["map_residual_drift"])
        assert np.isfinite(cell["marg_scale_error"])
        assert np.isfinite(cell["planted_top_mass_p50"])

        # Renderers must not blow up on the result.
        table = msd.render_markdown_table("smoke", cell)
        assert "smoke" in table
        summary = msd.build_summary(results)
        assert isinstance(summary, str) and len(summary) > 0
