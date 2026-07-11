"""Unit tests for the flagged BUILD_T_PRIOR_SCALE driver wiring.

Two pure, unit-testable pieces of the block added around
build_dashboard_cloud.py:1028 (mirroring BUILD_CONCENTRATION_HETEROGENEITY_
DIAGNOSTIC): the (c, nu) grid-spec parser, and the zip-optional-files helper
that closes the omission that broke two prior cluster runs (a diagnostic's
JSON written to out_dir but left out of the downloadable zip). The Spark
wiring that calls corpus_tprior_scale_sweep_gated_rdd on a sampled real
corpus is exercised on the cluster, not here.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

_CLOUD = str(Path(__file__).resolve().parent.parent)
if _CLOUD not in sys.path:
    sys.path.insert(0, _CLOUD)

import build_dashboard_cloud as bdc  # noqa: E402


def test_nu_grid_parser_handles_inf():
    assert bdc._parse_scale_grid("2.5,5,10,20,inf") == [2.5, 5.0, 10.0, 20.0, float("inf")]
    assert bdc._parse_scale_grid("2,4,8") == [2.0, 4.0, 8.0]


def test_zip_includes_t_prior_scale_when_present(tmp_path: Path):
    out_dir = tmp_path
    (out_dir / "topics.json").write_text("{}")
    (out_dir / "t_prior_scale.json").write_text("{}")
    zip_path = tmp_path / "bundle.zip"
    bdc._zip_optional_files(out_dir, zip_path, required=("topics.json",))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "t_prior_scale.json" in names
    assert "topics.json" in names


def test_zip_omits_t_prior_scale_when_absent(tmp_path: Path):
    out_dir = tmp_path
    (out_dir / "topics.json").write_text("{}")
    zip_path = tmp_path / "bundle.zip"
    bdc._zip_optional_files(out_dir, zip_path, required=("topics.json",))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "t_prior_scale.json" not in names
    assert "topics.json" in names
