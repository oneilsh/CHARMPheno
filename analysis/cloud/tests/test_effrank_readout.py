"""Tests for the labeled effective-rank post-fit readout (pure, no Spark)."""
import math

from analysis.cloud.effrank_readout import (
    build_rows,
    node_depths,
    node_names,
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
