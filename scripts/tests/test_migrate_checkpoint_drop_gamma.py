"""Tests for scripts/migrate_checkpoint_drop_gamma.py.

Pytest discovers this via `poetry run pytest scripts/tests/ -v`.
The conftest.py in this directory inserts scripts/ onto sys.path so
`import migrate_checkpoint_drop_gamma` resolves.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from spark_vi.core.result import VIResult
from spark_vi.io import load_result, save_result
from charmpheno.export.theta_aggregates import compute_theta_aggregates

import migrate_checkpoint_drop_gamma as mig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lda_result(K: int, V: int, N: int, *, seed: int = 0) -> VIResult:
    """Minimal synthetic LDA VIResult with gamma in global_params."""
    rng = np.random.RandomState(seed)
    gamma = rng.gamma(shape=1.0, scale=1.0, size=(N, K)).astype(np.float64)
    lambda_ = rng.gamma(shape=1.0, scale=1.0, size=(K, V)).astype(np.float64)
    alpha = np.full(K, 1.0 / K, dtype=np.float64)
    return VIResult(
        global_params={"gamma": gamma, "lambda": lambda_, "alpha": alpha},
        elbo_trace=[-1000.0, -900.0],
        n_iterations=2,
        converged=False,
        metadata={"model_class": "lda"},
        diagnostic_traces={},
    )


def _make_hdp_result(T: int, V: int, *, seed: int = 42) -> VIResult:
    """Minimal synthetic HDP VIResult with u, v, lambda but no corpus_prevalence."""
    rng = np.random.RandomState(seed)
    # u and v are Beta concentration params for the GEM sticks
    u = rng.gamma(shape=2.0, scale=1.0, size=T).astype(np.float64)
    v = rng.gamma(shape=1.0, scale=1.0, size=T).astype(np.float64)
    lambda_ = rng.gamma(shape=1.0, scale=1.0, size=(T, V)).astype(np.float64)
    return VIResult(
        global_params={"u": u, "v": v, "lambda": lambda_},
        elbo_trace=[-500.0],
        n_iterations=1,
        converged=False,
        metadata={"model_class": "hdp"},
        diagnostic_traces={},
    )


# ---------------------------------------------------------------------------
# Test 1: LDA migration writes aggregates and drops gamma
# ---------------------------------------------------------------------------

def test_migrate_lda_writes_aggregates_and_drops_gamma(tmp_path):
    K, V, N = 4, 20, 150
    result = _make_lda_result(K, V, N)
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    status = mig.migrate(ckpt)

    # Check return value
    assert status["kind"] == "lda"
    assert status["action"] == "migrated"
    assert status["n_patients"] == N

    # Reload and verify structure
    loaded = load_result(ckpt)
    assert "gamma" not in loaded.global_params, "gamma should have been dropped"
    assert "theta_histogram" in loaded.metadata
    assert "theta_percentiles" in loaded.metadata
    assert "corpus_prevalence" in loaded.metadata
    assert "n_patients" in loaded.metadata

    assert len(loaded.metadata["theta_histogram"]) == K
    assert loaded.metadata["n_patients"] == N

    # Verify orphan file is gone
    orphan = ckpt / "params" / "gamma.npy"
    assert not orphan.exists(), "orphan params/gamma.npy should have been removed"


# ---------------------------------------------------------------------------
# Test 2: LDA idempotency
# ---------------------------------------------------------------------------

def test_migrate_lda_idempotent(tmp_path):
    K, V, N = 3, 15, 80
    result = _make_lda_result(K, V, N)
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    status1 = mig.migrate(ckpt)
    assert status1["action"] == "migrated"

    # Capture metadata after first migration
    loaded1 = load_result(ckpt)
    cp1 = loaded1.metadata["corpus_prevalence"]
    n1 = loaded1.metadata["n_patients"]

    # Second migration should be a no-op
    status2 = mig.migrate(ckpt)
    assert status2["action"] == "noop"

    # Values must be unchanged
    loaded2 = load_result(ckpt)
    assert loaded2.metadata["corpus_prevalence"] == cp1
    assert loaded2.metadata["n_patients"] == n1


# ---------------------------------------------------------------------------
# Test 3: Migrated aggregates match independent compute_theta_aggregates
# ---------------------------------------------------------------------------

def test_migrate_lda_aggregates_match_compute_theta_aggregates(tmp_path):
    K, V, N = 5, 30, 200
    rng = np.random.RandomState(7)
    gamma = rng.gamma(shape=1.0, scale=1.0, size=(N, K)).astype(np.float64)
    lambda_ = rng.gamma(shape=1.0, scale=1.0, size=(K, V)).astype(np.float64)
    result = VIResult(
        global_params={"gamma": gamma, "lambda": lambda_},
        elbo_trace=[-100.0],
        n_iterations=1,
        converged=False,
        metadata={"model_class": "lda"},
        diagnostic_traces={},
    )
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)
    mig.migrate(ckpt, n_bins=30, min_count=5)

    # Independent reference computation
    ref = compute_theta_aggregates(gamma, n_bins=30, min_count=5)

    loaded = load_result(ckpt)
    meta = loaded.metadata

    # n_patients
    assert meta["n_patients"] == ref["n_patients"]

    # corpus_prevalence — float64 values, should match closely
    for a, b in zip(meta["corpus_prevalence"], ref["corpus_prevalence"]):
        assert math.isclose(a, b, rel_tol=1e-9), f"corpus_prevalence mismatch: {a} vs {b}"

    # theta_percentiles
    for topic_idx, (migrated_pct, ref_pct) in enumerate(
        zip(meta["theta_percentiles"], ref["theta_percentiles"])
    ):
        for key in ("p5", "p25", "p50", "p75", "p95"):
            assert math.isclose(migrated_pct[key], ref_pct[key], rel_tol=1e-9), (
                f"topic {topic_idx} {key} mismatch: {migrated_pct[key]} vs {ref_pct[key]}"
            )

    # theta_histogram — may contain None (suppressed cells)
    for topic_idx, (migrated_row, ref_row) in enumerate(
        zip(meta["theta_histogram"], ref["theta_histogram"])
    ):
        assert len(migrated_row) == len(ref_row)
        for bin_idx, (mv, rv) in enumerate(zip(migrated_row, ref_row)):
            if rv is None:
                assert mv is None, f"topic {topic_idx} bin {bin_idx}: expected None"
            elif mv is None:
                assert rv is None, f"topic {topic_idx} bin {bin_idx}: unexpected None"
            else:
                assert math.isclose(mv, rv, rel_tol=1e-9), (
                    f"topic {topic_idx} bin {bin_idx}: {mv} vs {rv}"
                )


# ---------------------------------------------------------------------------
# Test 4: HDP migration writes corpus_prevalence from sticks
# ---------------------------------------------------------------------------

def test_migrate_hdp_writes_corpus_prevalence_from_sticks(tmp_path):
    T, V = 10, 25
    result = _make_hdp_result(T, V)
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    status = mig.migrate(ckpt)
    assert status["kind"] == "hdp"
    assert status["action"] == "migrated"
    assert status["n_patients"] is None

    loaded = load_result(ckpt)
    assert "corpus_prevalence" in loaded.metadata
    cp = loaded.metadata["corpus_prevalence"]
    assert len(cp) == T

    # Check values match the expected stick-breaking formula
    u = result.global_params["u"]
    v = result.global_params["v"]
    expected = mig._hdp_corpus_prevalence(u, v)
    for i, (got, exp) in enumerate(zip(cp, expected)):
        assert math.isclose(got, exp, rel_tol=1e-9), (
            f"corpus_prevalence[{i}]: {got} vs {exp}"
        )


# ---------------------------------------------------------------------------
# Test 5: HDP no-op when corpus_prevalence already present
# ---------------------------------------------------------------------------

def test_migrate_hdp_noop_when_corpus_prevalence_present(tmp_path):
    T, V = 8, 20
    rng = np.random.RandomState(13)
    u = rng.gamma(2.0, 1.0, T).astype(np.float64)
    v = rng.gamma(1.0, 1.0, T).astype(np.float64)
    precomputed_cp = [float(x) for x in rng.dirichlet(np.ones(T))]
    result = VIResult(
        global_params={"u": u, "v": v, "lambda": rng.gamma(1.0, 1.0, (T, V)).astype(np.float64)},
        elbo_trace=[-300.0],
        n_iterations=1,
        converged=False,
        metadata={"model_class": "hdp", "corpus_prevalence": precomputed_cp},
        diagnostic_traces={},
    )
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    status = mig.migrate(ckpt)
    assert status["kind"] == "hdp"
    assert status["action"] == "noop"

    # Values must be unchanged
    loaded = load_result(ckpt)
    assert loaded.metadata["corpus_prevalence"] == precomputed_cp


# ---------------------------------------------------------------------------
# Test 6: LDA raises ValueError when no gamma and no aggregates
# ---------------------------------------------------------------------------

def test_migrate_raises_when_no_gamma_and_no_aggregates(tmp_path):
    K, V = 3, 15
    rng = np.random.RandomState(99)
    # LDA without gamma — as if it was already dropped but aggregates are missing
    result = VIResult(
        global_params={"lambda": rng.gamma(1.0, 1.0, (K, V)).astype(np.float64)},
        elbo_trace=[-200.0],
        n_iterations=1,
        converged=False,
        metadata={"model_class": "lda"},
        diagnostic_traces={},
    )
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    with pytest.raises(ValueError, match="re-fit"):
        mig.migrate(ckpt)


# ---------------------------------------------------------------------------
# Test 7: LDA warns and recomputes when gamma AND aggregates both present
# ---------------------------------------------------------------------------

def test_migrate_warns_when_gamma_and_aggregates_both_present(tmp_path, capsys):
    K, V, N = 4, 20, 100
    rng = np.random.RandomState(55)
    gamma = rng.gamma(1.0, 1.0, (N, K)).astype(np.float64)
    lambda_ = rng.gamma(1.0, 1.0, (K, V)).astype(np.float64)

    # Stale aggregates: deliberately wrong n_patients to show they get overwritten
    stale_metadata = {
        "model_class": "lda",
        "theta_histogram": [list([0.0] * 50)] * K,
        "theta_percentiles": [{"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}] * K,
        "corpus_prevalence": [0.25] * K,
        "n_patients": 9999,  # stale / wrong
    }
    result = VIResult(
        global_params={"gamma": gamma, "lambda": lambda_},
        elbo_trace=[-100.0],
        n_iterations=1,
        converged=False,
        metadata=stale_metadata,
        diagnostic_traces={},
    )
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    status = mig.migrate(ckpt)
    assert status["action"] == "recomputed"
    assert status["kind"] == "lda"

    # Warning logged to stdout
    captured = capsys.readouterr()
    assert "WARNING" in captured.out

    # Final state: gamma dropped, aggregates recomputed from actual gamma
    loaded = load_result(ckpt)
    assert "gamma" not in loaded.global_params
    assert loaded.metadata["n_patients"] == N  # correct value, not 9999

    orphan = ckpt / "params" / "gamma.npy"
    assert not orphan.exists()


# ---------------------------------------------------------------------------
# Test 8: migrate with explicit --out path leaves source unchanged
# ---------------------------------------------------------------------------

def test_migrate_with_explicit_out_path(tmp_path):
    K, V, N = 3, 12, 60
    result = _make_lda_result(K, V, N, seed=77)
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    save_result(result, src)

    status = mig.migrate(src, out_path=dst)
    assert status["action"] == "migrated"

    # Source must be unchanged
    src_loaded = load_result(src)
    assert "gamma" in src_loaded.global_params, "source gamma should still be present"
    assert "theta_histogram" not in src_loaded.metadata, "source should have no aggregates"

    # Destination must have migrated state
    dst_loaded = load_result(dst)
    assert "gamma" not in dst_loaded.global_params
    assert "theta_histogram" in dst_loaded.metadata
    assert "corpus_prevalence" in dst_loaded.metadata
    assert dst_loaded.metadata["n_patients"] == N

    # No orphan cleanup applies to dst (it was a fresh save without gamma)
    # but we double-check dst/params/gamma.npy doesn't exist either way
    assert not (dst / "params" / "gamma.npy").exists()
