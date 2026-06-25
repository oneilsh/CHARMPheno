from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from charmpheno.export.dashboard import (
    select_top_n_by_marginal,
    write_model_and_vocab_bundles,
)


def _row_stochastic(arr: np.ndarray) -> np.ndarray:
    """Test helper: normalize rows to sum to 1."""
    arr = np.asarray(arr, dtype=float)
    return arr / arr.sum(axis=1, keepdims=True)


def test_top_n_trim_reorders_and_renormalizes(tmp_path: Path):
    K, V = 2, 5
    # β: row 0 mostly on col 4; row 1 mostly on col 1. Other cols low.
    beta = _row_stochastic(np.array([
        [1, 1, 1, 1, 100],
        [1, 100, 1, 1, 1],
    ], dtype=float))
    alpha = np.array([0.1, 0.1])
    marginals = [0.10, 0.30, 0.01, 0.01, 0.58]   # col 4, then col 1 are top-2
    vocab_ids = [101, 202, 303, 404, 505]
    descriptions = {101: "A", 202: "B", 505: "E"}
    domains = {101: "condition", 202: "drug", 505: "procedure"}

    write_model_and_vocab_bundles(
        out_dir=tmp_path,
        beta=beta, alpha=alpha,
        vocab_ids=vocab_ids, descriptions=descriptions, domains=domains,
        code_marginals=marginals,
        top_n=2,
    )

    model = json.loads((tmp_path / "model.json").read_text())
    vocab = json.loads((tmp_path / "vocab.json").read_text())

    # vocab kept top-2 by marginal: col 4 (505), then col 1 (202)
    assert [c["code"] for c in vocab["codes"]] == ["505", "202"]
    assert [c["description"] for c in vocab["codes"]] == ["E", "B"]
    assert [c["corpus_freq"] for c in vocab["codes"]] == pytest.approx([0.58, 0.30])
    # ids are 0..top_N-1
    assert [c["id"] for c in vocab["codes"]] == [0, 1]

    # model V matches trimmed vocab; β rows renormalize
    assert model["K"] == 2
    assert model["V"] == 2
    beta = np.array(model["beta"])
    np.testing.assert_allclose(beta.sum(axis=1), np.ones(2), atol=1e-6)
    # row 0 was concentrated on the kept col 4 → still concentrated there (column 0 of trimmed)
    assert beta[0, 0] > 0.9
    # row 1 was concentrated on col 1 → trimmed-column 1
    assert beta[1, 1] > 0.9


def test_returns_v_displayed(tmp_path: Path):
    K, V = 2, 4
    beta = _row_stochastic(np.ones((K, V)))
    alpha = np.array([0.1, 0.1])
    marginals = [0.4, 0.3, 0.2, 0.1]
    vocab_ids = [10, 20, 30, 40]
    v_disp = write_model_and_vocab_bundles(
        out_dir=tmp_path,
        beta=beta, alpha=alpha,
        vocab_ids=vocab_ids, descriptions={}, domains={},
        code_marginals=marginals,
        top_n=10,
    )
    assert v_disp == V


def test_model_json_includes_sigma_when_provided(tmp_path: Path):
    # STM exports the per-topic prior variance (diagonal Sigma) into model.json
    # so the dashboard can sample the faithful logistic-normal prior (ADR 0028
    # Alternative B). Length K, topic-aligned with alpha/beta rows; the vocab
    # top-N trim (which only touches V) leaves it untouched.
    K, V = 2, 4
    write_model_and_vocab_bundles(
        out_dir=tmp_path,
        beta=_row_stochastic(np.ones((K, V))), alpha=np.array([0.1, 0.1]),
        vocab_ids=[10, 20, 30, 40], descriptions={}, domains={},
        code_marginals=[0.4, 0.3, 0.2, 0.1], top_n=10,
        sigma=np.array([0.5, 0.9]),
    )
    model = json.loads((tmp_path / "model.json").read_text())
    assert model["sigma"] == pytest.approx([0.5, 0.9])


def test_model_json_omits_sigma_by_default(tmp_path: Path):
    # Non-STM bundles (LDA/HDP) pass sigma=None; the key must be ABSENT so the
    # frontend can treat its presence as the "use the STM sampler" signal.
    K, V = 2, 4
    write_model_and_vocab_bundles(
        out_dir=tmp_path,
        beta=_row_stochastic(np.ones((K, V))), alpha=np.array([0.1, 0.1]),
        vocab_ids=[10, 20, 30, 40], descriptions={}, domains={},
        code_marginals=[0.4, 0.3, 0.2, 0.1], top_n=10,
    )
    model = json.loads((tmp_path / "model.json").read_text())
    assert "sigma" not in model


def test_select_top_n_pure_marginal_ranking():
    """select_top_n_by_marginal returns top-N by marginal, no doc-count filter."""
    code_marginals = [0.05, 0.30, 0.10, 0.40, 0.15]
    selected = select_top_n_by_marginal(code_marginals, top_n=3)
    # Expect descending marginal: idx 3 (0.40), idx 1 (0.30), idx 4 (0.15)
    assert selected == [3, 1, 4]


def test_select_top_n_respects_top_n_cap():
    code_marginals = [0.5, 0.3, 0.2]
    assert select_top_n_by_marginal(code_marginals, top_n=2) == [0, 1]
    assert select_top_n_by_marginal(code_marginals, top_n=10) == [0, 1, 2]


def test_write_phenotypes_bundle(tmp_path: Path):
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    K = 3
    n_bins = 50
    # Build K × 50 histograms; include some None entries to exercise null round-trip.
    theta_hist = [[float(i) for i in range(n_bins)] for _ in range(K)]
    theta_hist[0][5] = None   # explicit null in row 0
    theta_hist[1][0] = None   # explicit null in row 1
    theta_pcts = [
        {"p5": 0.01, "p25": 0.05, "p50": 0.10, "p75": 0.20, "p95": 0.40},
        {"p5": 0.02, "p25": 0.06, "p50": 0.12, "p75": 0.22, "p95": 0.42},
        {"p5": 0.03, "p25": 0.07, "p50": 0.14, "p75": 0.24, "p95": 0.44},
    ]
    write_phenotypes_bundle(
        out,
        npmi=[0.18, 0.05, -0.10],
        pair_coverage=[0.90, 0.50, 0.0],
        corpus_prevalence=[0.30, 0.40, 0.30],
        labels=["Cardiac", "", ""],
        topic_indices=[0, 1, 2],
        theta_histogram=theta_hist,
        theta_percentiles=theta_pcts,
        n_bins=n_bins,
        min_count=20,
    )
    payload = json.loads(out.read_text())
    assert "npmi_threshold" not in payload
    # Core per-phenotype fields still present.
    assert payload["phenotypes"][0]["id"] == 0
    assert payload["phenotypes"][0]["label"] == "Cardiac"
    assert payload["phenotypes"][0]["description"] == ""
    assert payload["phenotypes"][0]["quality"] is None
    assert payload["phenotypes"][0]["npmi"] == pytest.approx(0.18)
    assert payload["phenotypes"][0]["pair_coverage"] == pytest.approx(0.90)
    assert payload["phenotypes"][0]["corpus_prevalence"] == pytest.approx(0.30)
    assert payload["phenotypes"][0]["original_topic_id"] == 0
    # New per-phenotype theta fields present.
    assert payload["phenotypes"][0]["theta_histogram"] == theta_hist[0]
    assert payload["phenotypes"][0]["theta_percentiles"] == theta_pcts[0]
    assert payload["phenotypes"][1]["theta_histogram"] == theta_hist[1]
    assert payload["phenotypes"][2]["theta_histogram"] == theta_hist[2]
    # pair_coverage=0 means "unrated".
    assert payload["phenotypes"][2]["pair_coverage"] == 0.0
    # Top-level histogram metadata.
    bin_edges = payload["theta_histogram_bin_edges"]
    assert isinstance(bin_edges, list)
    assert len(bin_edges) == n_bins + 1
    assert bin_edges[0] == pytest.approx(0.0)
    assert bin_edges[-1] == pytest.approx(1.0)
    assert payload["theta_histogram_min_count"] == 20
    assert set(payload["phenotypes"][0].keys()) == {
        "id", "label", "description", "quality",
        "npmi", "pair_coverage", "corpus_prevalence",
        "original_topic_id", "theta_histogram", "theta_percentiles",
    }


def test_write_phenotypes_bundle_nan_npmi_serializes_as_null(tmp_path: Path):
    """NaN npmi/pair_coverage must serialize as JSON null, not the literal NaN."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[float("nan"), 0.5],
        pair_coverage=[0.0, 0.5],
        corpus_prevalence=[0.0, 0.4],
    )
    # File must contain no literal NaN token.
    assert "NaN" not in out.read_text()
    # Must parse as strict JSON (json.loads rejects literal NaN).
    payload = json.loads(out.read_text())
    assert payload["phenotypes"][0]["npmi"] is None
    assert payload["phenotypes"][0]["pair_coverage"] == pytest.approx(0.0)
    assert payload["phenotypes"][1]["npmi"] == pytest.approx(0.5)
    assert payload["phenotypes"][1]["pair_coverage"] == pytest.approx(0.5)


def test_write_phenotypes_bundle_preserves_hdp_original_indices(tmp_path: Path):
    """For HDP, the displayed phenotype ids are 0..K_display-1 but
    original_topic_id carries the source truncation index."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[0.2, 0.15],
        pair_coverage=[0.8, 0.7],
        corpus_prevalence=[0.4, 0.3],
        labels=None,
        topic_indices=[42, 7],
    )
    payload = json.loads(out.read_text())
    assert [p["id"] for p in payload["phenotypes"]] == [0, 1]
    assert [p["original_topic_id"] for p in payload["phenotypes"]] == [42, 7]


def test_write_phenotypes_bundle_length_mismatch_raises(tmp_path: Path):
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="pair_coverage length"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9],  # wrong length
            corpus_prevalence=[0.5, 0.5],
        )


def test_write_phenotypes_bundle_omits_distribution_when_none(tmp_path: Path):
    """When theta_histogram and theta_percentiles are both None, the output
    must not include any histogram-related top-level keys or per-phenotype
    theta fields."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[0.1, 0.2],
        pair_coverage=[0.9, 0.8],
        corpus_prevalence=[0.5, 0.5],
    )
    payload = json.loads(out.read_text())
    assert "theta_histogram_bin_edges" not in payload
    assert "theta_histogram_min_count" not in payload
    for p in payload["phenotypes"]:
        assert "theta_histogram" not in p
        assert "theta_percentiles" not in p


def test_write_phenotypes_bundle_preserves_null_in_histogram(tmp_path: Path):
    """None entries in histogram rows must survive the JSON round-trip as None."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    n_bins = 50
    row = [float(i) for i in range(n_bins)]
    row[3] = None
    row[49] = None
    write_phenotypes_bundle(
        out,
        npmi=[0.1],
        pair_coverage=[0.9],
        corpus_prevalence=[0.5],
        theta_histogram=[row],
        n_bins=n_bins,
    )
    payload = json.loads(out.read_text())
    loaded_row = payload["phenotypes"][0]["theta_histogram"]
    assert loaded_row[3] is None
    assert loaded_row[49] is None
    # Non-None entries are preserved.
    assert loaded_row[0] == pytest.approx(0.0)
    assert loaded_row[10] == pytest.approx(10.0)


def test_write_phenotypes_bundle_length_mismatch_histogram(tmp_path: Path):
    """theta_histogram of wrong outer length raises ValueError."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="theta_histogram length"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9, 0.8],
            corpus_prevalence=[0.5, 0.5],
            theta_histogram=[[0.0] * 50],  # length 1, K=2
        )


def test_write_phenotypes_bundle_length_mismatch_percentiles(tmp_path: Path):
    """theta_percentiles of wrong length raises ValueError."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="theta_percentiles length"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9, 0.8],
            corpus_prevalence=[0.5, 0.5],
            theta_histogram=[[0.0] * 50, [0.0] * 50],  # valid K=2 histogram required
            theta_percentiles=[{"p5": 0.01, "p25": 0.05, "p50": 0.10, "p75": 0.20, "p95": 0.40}],  # length 1, K=2
        )


def test_write_phenotypes_bundle_wrong_n_bins(tmp_path: Path):
    """Histogram row with wrong length raises ValueError naming the row index."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="row 1"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9, 0.8],
            corpus_prevalence=[0.5, 0.5],
            theta_histogram=[
                [0.0] * 50,   # row 0 correct
                [0.0] * 49,   # row 1 wrong (49 instead of 50)
            ],
            n_bins=50,
        )


def test_write_phenotypes_bundle_length_mismatch_corpus_prevalence(tmp_path: Path):
    """corpus_prevalence of wrong length raises ValueError."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="corpus_prevalence length"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9, 0.8],
            corpus_prevalence=[0.5],  # length 1, K=2
        )


def test_write_phenotypes_bundle_percentiles_without_histogram_errors(tmp_path: Path):
    """theta_percentiles without theta_histogram raises ValueError."""
    from charmpheno.export.dashboard import write_phenotypes_bundle
    out = tmp_path / "phenotypes.json"
    with pytest.raises(ValueError, match="theta_percentiles requires theta_histogram"):
        write_phenotypes_bundle(
            out,
            npmi=[0.1, 0.2],
            pair_coverage=[0.9, 0.8],
            corpus_prevalence=[0.5, 0.5],
            theta_percentiles=[
                {"p5": 0.01, "p25": 0.05, "p50": 0.10, "p75": 0.20, "p95": 0.40},
                {"p5": 0.02, "p25": 0.06, "p50": 0.12, "p75": 0.22, "p95": 0.42},
            ],
            theta_histogram=None,
        )
