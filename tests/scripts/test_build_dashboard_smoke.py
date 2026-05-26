from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_build_dashboard_smoke(tmp_path: Path):
    from spark_vi.core.result import VIResult
    from spark_vi.io import save_result

    from charmpheno.export.theta_aggregates import compute_theta_aggregates

    K, V = 4, 12
    lambda_ = np.random.RandomState(0).rand(K, V) + 0.5
    alpha = np.full(K, 0.1)
    vocab = [1000 + i for i in range(V)]
    # Aggregates landed in metadata at fit time (ADR / patient-prevalence
    # work). Adapter hard-errors without them, so the smoke fixture mirrors
    # the production training-end shape.
    gamma = np.random.RandomState(1).gamma(shape=0.1, scale=1.0, size=(50, K)) + 0.01
    aggregates = compute_theta_aggregates(gamma)
    result = VIResult(
        global_params={"lambda": lambda_, "alpha": alpha},
        elbo_trace=[1.0, 2.0, 3.0], n_iterations=3, converged=True,
        metadata={
            "vocab": vocab,
            "concept_names": {1000: "Atrial fibrillation"},
            "concept_domains": {1000: "condition"},
            "corpus_manifest": {"doc_spec": {"name": "patient"}},
            "theta_histogram": aggregates["theta_histogram"],
            "theta_percentiles": aggregates["theta_percentiles"],
            "corpus_prevalence": aggregates["corpus_prevalence"],
            "n_patients": aggregates["n_patients"],
        },
    )
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)
    parquet = Path("data/simulated/omop_N10000_seed42.parquet")
    assert parquet.exists(), "fixture parquet missing; run `make data`"
    out = tmp_path / "data"
    subprocess.check_call([
        sys.executable, "analysis/local/build_dashboard.py",
        "--checkpoint", str(ckpt),
        "--input", str(parquet),
        "--out-dir", str(out),
        "--vocab-top-n", "8",
        # Tiny synthetic vocab can have per-code doc-counts below the
        # production AoU-style guard; disable it for this smoke test.
        "--min-doc-count", "0",
    ])
    assert {p.name for p in out.iterdir()} == {
        "model.json", "vocab.json", "phenotypes.json", "corpus_stats.json"
    }
    stats = json.loads((out / "corpus_stats.json").read_text())
    assert stats["k"] == 4 and stats["v"] == 8 and stats["v_full"] == 12
    model = json.loads((out / "model.json").read_text())
    assert model["V"] == 8
    np.testing.assert_allclose(np.array(model["beta"]).sum(axis=1), np.ones(4), atol=1e-5)
    # Adapter populated original_topic_id (identity for LDA)
    phenos = json.loads((out / "phenotypes.json").read_text())["phenotypes"]
    assert [p["original_topic_id"] for p in phenos] == [0, 1, 2, 3]
