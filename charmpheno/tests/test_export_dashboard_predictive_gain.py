"""Tests for the predictive-gain aggregates in write_phenotypes_bundle
(Phase-2 Task 5: threading spark_vi.mllib.topic.predictive_gain's per-topic
corpus aggregates into the dashboard's phenotypes.json).

Schema (PROVISIONAL — see write_phenotypes_bundle's docstring): a single
top-level "predictive_gain" object nesting the per-topic arrays
(presence, mean_gain, depth, prominence_hist, length_corr, dedup_gain) and
the bundle-level diagnostics (prominence_bin_edges, null_band,
observed_delta_range, downdate_audit, scale, n_docs). Backward compat:
when every predictive-gain param is None (LDA/HDP bundles, or an STM build
where the enhancement-only phase failed), the "predictive_gain" key must be
entirely absent so existing bundles are byte-unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from charmpheno.export.dashboard import write_phenotypes_bundle


def test_write_phenotypes_bundle_omits_predictive_gain_when_none(tmp_path: Path):
    """Backward compat: no predictive-gain params -> no predictive_gain key
    anywhere in the payload (byte-unchanged bundle for LDA/HDP/legacy STM)."""
    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[0.1, 0.2],
        pair_coverage=[0.9, 0.8],
        corpus_prevalence=[0.5, 0.5],
    )
    payload = json.loads(out.read_text())
    assert "predictive_gain" not in payload


def test_write_phenotypes_bundle_predictive_gain_well_formed(tmp_path: Path):
    out = tmp_path / "phenotypes.json"
    K = 3
    n_bins = 5
    presence = [0.1, 0.4, float("nan")]
    mean_gain = [0.02, 0.5, -0.1]
    depth = [0.05, 0.6, float("nan")]
    prominence_hist = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [float("nan"), 1.0, 1.0, 1.0, 1.0],
    ]
    length_corr = [0.3, float("nan"), -0.2]
    dedup_gain = [0.01, 0.4, -0.05]
    prominence_bin_edges = [-1.0, 1.4, 3.8, 6.2, 8.6, 11.0]
    null_band = {"mean": 0.01, "std": 0.02, "n": 400, "hist": [1, 2, 3, 4, 5], "p95": 0.1}
    observed_delta_range = [-0.8, 9.5]
    downdate_audit = {"max_abs_overall": 0.0021, "n_docs_audited": 50}

    write_phenotypes_bundle(
        out,
        npmi=[0.1, 0.2, 0.3],
        pair_coverage=[0.9, 0.8, 0.7],
        corpus_prevalence=[0.5, 0.3, 0.2],
        presence=presence,
        mean_gain=mean_gain,
        depth=depth,
        prominence_hist=prominence_hist,
        length_corr=length_corr,
        dedup_gain=dedup_gain,
        prominence_bin_edges=prominence_bin_edges,
        null_band=null_band,
        observed_delta_range=observed_delta_range,
        predictive_gain_downdate_audit=downdate_audit,
        predictive_gain_scale=4.6,
        predictive_gain_n_docs=950,
        n_bins=n_bins,
    )
    payload = json.loads(out.read_text())
    assert "NaN" not in out.read_text()

    pg = payload["predictive_gain"]
    assert len(pg["presence"]) == K
    assert len(pg["mean_gain"]) == K
    assert len(pg["depth"]) == K
    assert len(pg["prominence_hist"]) == K
    assert all(len(row) == n_bins for row in pg["prominence_hist"])
    assert len(pg["length_corr"]) == K
    assert len(pg["dedup_gain"]) == K

    # NaN -> None (json null), not the literal NaN token.
    assert pg["presence"][2] is None
    assert pg["depth"][2] is None
    assert pg["length_corr"][1] is None
    assert pg["prominence_hist"][2][0] is None
    assert pg["prominence_hist"][2][1] == pytest.approx(1.0)

    # Finite entries preserved.
    assert pg["presence"][0] == pytest.approx(0.1)
    assert pg["mean_gain"][1] == pytest.approx(0.5)

    # Bundle-level diagnostics present and pass through untouched.
    assert pg["prominence_bin_edges"] == pytest.approx(prominence_bin_edges)
    assert pg["null_band"] == null_band
    assert pg["observed_delta_range"] == pytest.approx(observed_delta_range)
    assert pg["downdate_audit"]["max_abs_overall"] == pytest.approx(0.0021)
    assert pg["downdate_audit"]["n_docs_audited"] == 50
    assert pg["scale"] == pytest.approx(4.6)
    assert pg["n_docs"] == 950


def test_write_phenotypes_bundle_predictive_gain_length_mismatch_raises(tmp_path: Path):
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="mean_gain length"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9, 0.8],
            corpus_prevalence=[0.5, 0.5],
            mean_gain=[0.1],  # wrong length: 1 vs K=2
        )


def test_write_phenotypes_bundle_predictive_gain_prominence_hist_wrong_bins(tmp_path: Path):
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="prominence_hist row"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9, 0.8],
            corpus_prevalence=[0.5, 0.5],
            prominence_hist=[[0.0] * 5, [0.0] * 4],  # row 1 wrong length
            n_bins=5,
        )
