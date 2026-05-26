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

def _make_lda_result_no_aggregates(K: int, V: int) -> VIResult:
    """Minimal LDA VIResult with no gamma and no theta aggregates in metadata.

    Represents a checkpoint produced by an old driver that dropped gamma but
    forgot to write aggregates — or a corrupt/incomplete state.
    """
    rng = np.random.RandomState(0)
    lambda_ = rng.gamma(shape=1.0, scale=1.0, size=(K, V)).astype(np.float64)
    alpha = np.full(K, 1.0 / K, dtype=np.float64)
    return VIResult(
        global_params={"lambda": lambda_, "alpha": alpha},
        elbo_trace=[-1000.0, -900.0],
        n_iterations=2,
        converged=False,
        metadata={"model_class": "lda"},
        diagnostic_traces={},
    )


def _make_lda_result_with_aggregates(K: int, V: int, N: int, *, seed: int = 0) -> VIResult:
    """LDA VIResult that already has theta aggregates in metadata (post-fit state).

    Represents a checkpoint produced by the current fit drivers that call
    model.transform(fit_df) and embed aggregates at fit time.
    """
    rng = np.random.RandomState(seed)
    lambda_ = rng.gamma(shape=1.0, scale=1.0, size=(K, V)).astype(np.float64)
    alpha = np.full(K, 1.0 / K, dtype=np.float64)
    # Generate synthetic theta for aggregates
    theta = rng.dirichlet(np.ones(K), size=N).astype(np.float64)
    aggregates = compute_theta_aggregates(theta)
    return VIResult(
        global_params={"lambda": lambda_, "alpha": alpha},
        elbo_trace=[-1000.0, -900.0],
        n_iterations=2,
        converged=False,
        metadata={
            "model_class": "lda",
            "theta_histogram": aggregates["theta_histogram"],
            "theta_percentiles": aggregates["theta_percentiles"],
            "corpus_prevalence": aggregates["corpus_prevalence"],
            "n_patients": aggregates["n_patients"],
        },
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
# Test 1: LDA no-op when aggregates already present (the normal post-fit case)
# ---------------------------------------------------------------------------

def test_migrate_lda_noop_when_already_has_aggregates(tmp_path):
    K, V, N = 4, 20, 150
    result = _make_lda_result_with_aggregates(K, V, N)
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    status = mig.migrate(ckpt)

    assert status["kind"] == "lda"
    assert status["action"] == "noop"
    assert status["n_patients"] == N

    # Checkpoint is unchanged
    loaded = load_result(ckpt)
    assert "gamma" not in loaded.global_params
    assert "theta_histogram" in loaded.metadata
    assert loaded.metadata["n_patients"] == N


# ---------------------------------------------------------------------------
# Test 2: LDA no-op is idempotent across multiple calls
# ---------------------------------------------------------------------------

def test_migrate_lda_noop_is_idempotent(tmp_path):
    K, V, N = 3, 15, 80
    result = _make_lda_result_with_aggregates(K, V, N, seed=7)
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    status1 = mig.migrate(ckpt)
    assert status1["action"] == "noop"

    # Capture metadata after first call
    loaded1 = load_result(ckpt)
    cp1 = loaded1.metadata["corpus_prevalence"]
    n1 = loaded1.metadata["n_patients"]

    # Second call must also be a no-op with unchanged values
    status2 = mig.migrate(ckpt)
    assert status2["action"] == "noop"

    loaded2 = load_result(ckpt)
    assert loaded2.metadata["corpus_prevalence"] == cp1
    assert loaded2.metadata["n_patients"] == n1


# ---------------------------------------------------------------------------
# Test 3: HDP migration writes corpus_prevalence from sticks
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
# Test 4: HDP no-op when corpus_prevalence already present
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
# Test 5: LDA raises ValueError when no aggregates and no gamma
# ---------------------------------------------------------------------------

def test_migrate_raises_when_no_gamma_and_no_aggregates(tmp_path):
    K, V = 3, 15
    result = _make_lda_result_no_aggregates(K, V)
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    with pytest.raises(ValueError, match="(?i)re-fit"):
        mig.migrate(ckpt)


# ---------------------------------------------------------------------------
# Test 6: LDA raises ValueError specifically mentions gamma not being persistable
# ---------------------------------------------------------------------------

def test_migrate_lda_raises_mentions_gamma_not_persisted(tmp_path):
    K, V = 3, 15
    result = _make_lda_result_no_aggregates(K, V)
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    with pytest.raises(ValueError, match="never"):
        mig.migrate(ckpt)


# ---------------------------------------------------------------------------
# Test 7: migrate raises ValueError for unknown model_class
# ---------------------------------------------------------------------------

def test_migrate_raises_on_unknown_model_class(tmp_path):
    K, V = 3, 15
    rng = np.random.RandomState(11)
    result = VIResult(
        global_params={"lambda": rng.gamma(1.0, 1.0, (K, V)).astype(np.float64)},
        elbo_trace=[-100.0],
        n_iterations=1,
        converged=False,
        metadata={"model_class": "ctm"},
        diagnostic_traces={},
    )
    ckpt = tmp_path / "ckpt"
    save_result(result, ckpt)

    with pytest.raises(ValueError, match="unsupported model class"):
        mig.migrate(ckpt)
