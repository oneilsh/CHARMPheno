"""End-to-end smoke: simulator parquet → VIRunner → exported artifact.

Bootstrap-scope smoke: uses CountingModel through VIRunner (not real HDP,
which is stubbed). Proves the plumbing — data loading, Spark distribution,
training loop, export — works.

Marked @slow because it invokes the data simulator with a small N, which
takes a few seconds more than unit tests.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis" / "local"))


@pytest.mark.slow
def test_fit_charmpheno_local_smoke_runs(tmp_path):
    # Arrange: generate a tiny synthetic parquet on the fly so the test is
    # hermetic (doesn't depend on an earlier `make data` invocation).
    import pandas as pd
    fixture = tmp_path / "sim.parquet"
    pd.DataFrame({
        "person_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "visit_occurrence_id": [10, 10, 20, 20, 30, 30, 40, 40],
        "concept_id": [1, 0, 1, 0, 1, 1, 0, 0],
        "concept_name": ["head", "tail", "head", "tail",
                         "head", "head", "tail", "tail"],
        "true_topic_id": [0, 0, 0, 0, 0, 0, 0, 0],
    }).to_parquet(fixture, index=False)

    from fit_charmpheno_local import main

    out_dir = tmp_path / "result"
    rc = main([
        "--input", str(fixture),
        "--output", str(out_dir),
        "--max-iterations", "3",
    ])
    assert rc == 0

    # Artifact exists and can be loaded back.
    from spark_vi.io import load_result
    result = load_result(out_dir)
    assert result.n_iterations == 3
    assert len(result.elbo_trace) == 3
    # CountingModel's posterior counts should have moved past the prior.
    assert float(result.global_params["alpha"]) > 2.0
