"""Tests for the labeled effective-rank post-fit readout (pure, no Spark)."""
import math

from analysis.cloud.effrank_readout import (
    build_rows,
    node_depths,
    node_names,
    pa_volume_correlation,
    pearson,
    pr_volume_correlation,
    render,
)


def test_node_names_maps_engine_id_to_concept_name():
    manifest = {"corpus_manifest": {
        "int2cid": {"1": "100", "2": "200"},
        "name_by_id": {"100": "Disorder of X", "200": "Disorder of Y"},
    }}
    assert node_names(manifest) == {1: "Disorder of X", 2: "Disorder of Y"}


def test_node_names_falls_back_to_concept_id():
    manifest = {"corpus_manifest": {
        "int2cid": {"3": "999"}, "name_by_id": {},
    }}
    assert node_names(manifest) == {3: "999"}


def test_node_depths_longest_path():
    # 0(root) -> 1 -> 2 -> 3 ; 1 -> 3 (multi-parent, longest wins)
    parent = {1: [0], 2: [1], 3: [2, 1]}
    d = node_depths(parent)
    assert d[1] == 1 and d[2] == 2 and d[3] == 3


def test_pearson_perfect_and_degenerate():
    assert pearson([1, 2, 3], [2, 4, 6]) == 1.0
    assert pearson([1, 1, 1], [2, 4, 6]) == 0.0   # zero variance -> 0
    assert pearson([5], [5]) == 0.0               # too short


def test_build_rows_sorted_by_participation():
    sidecar = {
        "7": {"participation": 2.0, "threshold": 2, "eigengap": 2,
              "n_probed": 2, "n_docs": 50},
        "3": {"participation": 20.0, "threshold": 40, "eigengap": 3,
              "n_probed": 40, "n_docs": 40000},
    }
    names = {7: "leaf", 3: "big class"}
    depths = {7: 5, 3: 1}
    rows = build_rows(sidecar, names, depths)
    assert [r["node"] for r in rows] == [3, 7]     # 20.0 before 2.0
    assert rows[0]["name"] == "big class"
    assert rows[0]["n_docs"] == 40000


def test_pr_volume_correlation_detects_volume_tracking():
    # participation rises with n_docs -> strong positive correlation
    sidecar = {
        str(i): {"participation": float(i), "threshold": i, "eigengap": 1,
                 "n_probed": i, "n_docs": 10 ** i}
        for i in range(1, 6)
    }
    rows = build_rows(sidecar, {}, {})
    assert pr_volume_correlation(rows) > 0.99


def test_render_includes_labels_counts_and_correlation():
    sidecar = {
        "3": {"participation": 20.0, "threshold": 40, "eigengap": 3,
              "n_probed": 40, "n_docs": 40000},
        "7": {"participation": 2.0, "threshold": 2, "eigengap": 2,
              "n_probed": 2, "n_docs": 50},
    }
    rows = build_rows(sidecar, {3: "big class", 7: "leaf"}, {3: 1, 7: 5})
    out = render(rows, k_uniform=4)
    assert "big class" in out and "leaf" in out
    assert "corr(PR, log10 n_docs)" in out
    assert "current foreground K: 4" in out
    # diversity-driven K = round(20)+round(2) = 22
    assert "Σround(PR) [diversity-driven K]: 22" in out


# --- parallel-analysis (pa_k) columns ---------------------------------------

def _pa_sidecar():
    # pa_k is DECORRELATED from n_docs: the biggest-volume node (40k docs) has a
    # middling pa_k, a mid-volume node has the largest, and the 26-doc node has a
    # tiny pa_k -- even though its raw PR stays ~flat-high (the closed-negative
    # effective-rank behavior pa_k is meant to fix).
    return {
        "3": {"participation": 90.0, "pa_k": 5, "pa_k_all": 12, "pa_pr_raw": 90.0,
              "threshold": 0, "eigengap": 0, "n_probed": 60, "n_docs": 40000},
        "7": {"participation": 88.0, "pa_k": 2, "pa_k_all": 90, "pa_pr_raw": 88.0,
              "threshold": 0, "eigengap": 0, "n_probed": 60, "n_docs": 26},
        "5": {"participation": 85.0, "pa_k": 9, "pa_k_all": 14, "pa_pr_raw": 85.0,
              "threshold": 0, "eigengap": 0, "n_probed": 60, "n_docs": 5000},
    }


def test_build_rows_sorts_by_pa_k_when_present():
    rows = build_rows(_pa_sidecar(), {}, {})
    assert [r["node"] for r in rows] == [5, 3, 7]     # pa_k 9,5,2
    assert rows[0]["pa_k"] == 9 and "pa_pr_raw" in rows[0]


def test_pa_volume_correlation_is_low_while_pr_is_high():
    rows = build_rows(_pa_sidecar(), {}, {})
    # raw PR is ~flat-high regardless of n_docs; pa_k does not rise with n_docs.
    assert abs(pa_volume_correlation(rows)) < 0.9
    # the 26-doc node keeps a big raw PR (the closed-negative behavior) ...
    small = next(r for r in rows if r["node"] == 7)
    assert small["participation"] > 50 and small["pa_k"] <= 3   # ... but small pa_k


def test_render_leads_with_pa_k_when_present():
    rows = build_rows(_pa_sidecar(), {3: "big", 7: "tiny", 5: "mid"}, {})
    out = render(rows, k_uniform=100)
    assert "Σpa_k [parallel-analysis K, leading-run]: 16" in out   # 5+2+9
    assert "corr(pa_k, log10 n_docs)" in out
    assert "current foreground K: 100" in out
    # the count-all diagnostic + the impossible-flag (node 7: pa_k_all 90 > 26 docs)
    assert "count-all diagnostic" in out
    assert "1 nodes had pa_k_all > n_docs" in out
    assert "by support:" in out
    header = next(ln for ln in out.splitlines() if ln.strip().startswith("pa_k"))
    assert "pa_k" in header and "PR" in header
