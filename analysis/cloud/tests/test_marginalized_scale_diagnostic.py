"""Unit tests for the flagged MAP-vs-marginalized eta_scale diagnostic.

build_marginalized_scale_diagnostic is a PURE helper (no Spark): given the
smoothed c* per holdout fraction for each estimator (MAP plug-in vs
Laplace-MC marginalized), it assembles a comparison dict recording both
curves plus each estimator's residual drift (max - min c* across holdout
fractions -- the quantity that should be ~0 for a well-behaved,
prefix-independent scale). This is the pure-Python half of Task 6'; the
Spark wiring that calls corpus_heldout_scale_sweep_gated_rdd on a sampled
real corpus is verified on the cluster, not here.
"""
import json
import sys
from pathlib import Path

import pytest

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

import build_dashboard_cloud as bdc  # noqa: E402


def test_diagnostic_shape_and_drift():
    holdouts = [0.5, 0.8, 0.95]
    map_by = {"0.5": 3.4, "0.8": 3.45, "0.95": 3.39}
    marg_by = {"0.5": 2.6, "0.8": 2.8, "0.95": 3.6}
    out = bdc.build_marginalized_scale_diagnostic(
        map_cstar_by_holdout=map_by,
        marg_cstar_by_holdout=marg_by,
        n_samples="64",
        n_docs_sampled="123",
        c_grid=[0.5, 1.0, 2.0],
        holdouts=holdouts,
    )
    assert out["map_residual_drift"] == pytest.approx(0.06)
    assert out["marg_residual_drift"] == pytest.approx(1.0)
    assert out["map_cstar_by_holdout"] == {
        "0.5": pytest.approx(3.4),
        "0.8": pytest.approx(3.45),
        "0.95": pytest.approx(3.39),
    }
    assert out["marg_cstar_by_holdout"] == {
        "0.5": pytest.approx(2.6),
        "0.8": pytest.approx(2.8),
        "0.95": pytest.approx(3.6),
    }
    assert out["holdouts"] == [0.5, 0.8, 0.95]
    assert out["c_grid"] == [0.5, 1.0, 2.0]
    assert out["n_samples"] == 64 and isinstance(out["n_samples"], int)
    assert out["n_docs_sampled"] == 123 and isinstance(out["n_docs_sampled"], int)


def test_diagnostic_is_pure_json_safe():
    holdouts = [0.5, 0.8, 0.95]
    map_by = {"0.5": 3.4, "0.8": 3.45, "0.95": 3.39}
    marg_by = {"0.5": 2.6, "0.8": 2.8, "0.95": 3.6}
    out = bdc.build_marginalized_scale_diagnostic(
        map_cstar_by_holdout=map_by,
        marg_cstar_by_holdout=marg_by,
        n_samples=64,
        n_docs_sampled=123,
        c_grid=[0.5, 1.0, 2.0],
        holdouts=holdouts,
    )
    # should not raise -- every value is a plain float/int/list/dict/str
    json.dumps(out)
