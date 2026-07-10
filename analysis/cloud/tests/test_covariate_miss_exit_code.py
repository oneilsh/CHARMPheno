"""COVARIATE_CACHE_MISS_EXIT sentinel: the dashboard-only build must exit with
a DISTINCT code when a gated STM build has no covariate sidecar, so
scripts/run_experiment.py (`_build_only_with_auto_covariates`) can catch it,
rebuild the covariate cache in-cluster, and retry the build once — instead of
the operator having to notice the failure and re-run `make build-covariates`
by hand.
"""
import sys
from pathlib import Path

import pytest

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

import build_dashboard_cloud as bdc  # noqa: E402


def test_miss_exit_code_is_42():
    assert bdc.COVARIATE_CACHE_MISS_EXIT == 42


def test_sidecar_missing_gated_raises_sentinel_exit_code():
    with pytest.raises(SystemExit) as ei:
        bdc.assert_covariate_sidecar_present(
            is_stm=True, gated=True, sidecar_present=False,
            allow_incomplete=False, exp_hint="0028")
    assert ei.value.code == bdc.COVARIATE_CACHE_MISS_EXIT
    assert ei.value.code == 42


def test_sidecar_missing_gated_allow_incomplete_returns_none():
    result = bdc.assert_covariate_sidecar_present(
        is_stm=True, gated=True, sidecar_present=False, allow_incomplete=True)
    assert result is None


def test_sidecar_present_returns_none():
    result = bdc.assert_covariate_sidecar_present(
        is_stm=True, gated=True, sidecar_present=True, allow_incomplete=False)
    assert result is None


def test_non_stm_returns_none():
    result = bdc.assert_covariate_sidecar_present(
        is_stm=False, gated=False, sidecar_present=False, allow_incomplete=False)
    assert result is None
