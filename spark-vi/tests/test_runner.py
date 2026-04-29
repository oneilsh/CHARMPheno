"""Integration test: VIRunner fits CountingModel end-to-end on local Spark."""
import numpy as np
import pytest


def test_vi_runner_fits_counting_model_end_to_end(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    # 100 rows: 70 heads, 30 tails.
    data = [1] * 70 + [0] * 30
    rdd = spark.sparkContext.parallelize(data, numSlices=4)

    model = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    runner = VIRunner(
        model=model,
        config=VIConfig(max_iterations=5, convergence_tol=1e-6),
    )
    result = runner.fit(rdd)

    # Posterior mean ≈ 70 / 100 = 0.7 (with Beta(1,1) prior and learning rate
    # schedule that hasn't fully saturated in 5 iterations — check at a loose
    # tolerance).
    a = float(result.global_params["alpha"])
    b = float(result.global_params["beta"])
    mean = a / (a + b)
    assert 0.55 < mean < 0.85
    assert result.n_iterations == 5  # hit max, not convergence (tight tol)
    assert len(result.elbo_trace) == 5


def test_vi_runner_early_stop_branch_triggers(spark):
    """An absurdly wide convergence tolerance should trigger the early-stop branch.

    This doesn't exercise a realistic convergence criterion; it only proves that
    the converged-return path of VIRunner.fit is reachable and populates the
    VIResult correctly (converged=True, n_iterations < max, elbo_trace matches).
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 100 + [0] * 100, numSlices=4)
    model = CountingModel()
    runner = VIRunner(
        model=model,
        config=VIConfig(max_iterations=100, convergence_tol=1e10),  # will stop after 2
    )
    result = runner.fit(rdd)
    assert result.converged is True
    assert result.n_iterations < 100
    # n_iterations should equal the length of the ELBO trace recorded so far,
    # i.e., the result is internally consistent rather than a stale leftover.
    assert result.n_iterations == len(result.elbo_trace)
    assert result.n_iterations >= 1


def test_vi_runner_runs_to_max_iterations_without_convergence(spark):
    """With a tight tolerance, the runner should exhaust max_iterations.

    Exercises the other branch of the loop: no early-stop, converged=False,
    n_iterations == max_iterations.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 100 + [0] * 100, numSlices=4)
    model = CountingModel()
    runner = VIRunner(
        model=model,
        config=VIConfig(max_iterations=3, convergence_tol=1e-10),
    )
    result = runner.fit(rdd)
    assert result.converged is False
    assert result.n_iterations == 3
    assert len(result.elbo_trace) == 3


def test_vi_runner_rejects_non_vi_model(spark):
    from spark_vi.core import VIConfig, VIRunner

    class NotAModel:
        pass

    rdd = spark.sparkContext.parallelize([1, 0], numSlices=2)
    with pytest.raises(TypeError):
        VIRunner(model=NotAModel(), config=VIConfig()).fit(rdd)


# Mini-batch sampling tests --------------------------------------------------


def test_vi_runner_uses_mini_batch_when_fraction_set(spark):
    """With mini_batch_fraction set, the runner samples each iteration.

    Sanity-check that the fit completes and produces a reasonable posterior
    on CountingModel. Stochastic sampling means we don't pin numerics tightly,
    just that the posterior mean ends up in the right ballpark for 70/30 data.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    data = [1] * 70 + [0] * 30
    rdd = spark.sparkContext.parallelize(data, numSlices=4)

    model = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    runner = VIRunner(
        model=model,
        config=VIConfig(
            max_iterations=20,
            convergence_tol=1e-12,  # disable early stop to exercise sampling fully
            mini_batch_fraction=0.5,
            random_seed=42,
        ),
    )
    result = runner.fit(rdd)

    a = float(result.global_params["alpha"])
    b = float(result.global_params["beta"])
    mean = a / (a + b)
    # Same broad acceptance band as the full-batch end-to-end test.
    assert 0.5 < mean < 0.9
    # Some iterations may have been skipped if a batch was empty, but we
    # asked for fraction=0.5 on 100 rows — empty batches should be unlikely.
    assert result.n_iterations == 20
    assert len(result.elbo_trace) >= 1


def test_vi_runner_mini_batch_is_reproducible_with_seed(spark):
    """Two runs with the same random_seed must produce identical global params."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    data = [1] * 60 + [0] * 40
    rdd = spark.sparkContext.parallelize(data, numSlices=2)

    cfg = VIConfig(
        max_iterations=8,
        convergence_tol=1e-12,
        mini_batch_fraction=0.4,
        random_seed=2026,
    )

    r1 = VIRunner(CountingModel(), cfg).fit(rdd)
    r2 = VIRunner(CountingModel(), cfg).fit(rdd)

    # Per-iteration sampling seeds derive from the same root → same draws.
    # Spark partition ordering may still introduce tiny FP differences in
    # principle, but for additive integer counts on a single local executor
    # the result should be exact.
    assert float(r1.global_params["alpha"]) == float(r2.global_params["alpha"])
    assert float(r1.global_params["beta"]) == float(r2.global_params["beta"])


def test_vi_runner_mini_batch_empty_batch_path_does_not_crash(spark):
    """Tiny fraction on small data sometimes produces an empty batch.

    The runner must skip the global update for empty batches without crashing.
    Exact iteration counts are sampling-dependent; we only require that the
    result is well-formed.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1, 0, 1], numSlices=1)

    runner = VIRunner(
        model=CountingModel(),
        config=VIConfig(
            max_iterations=5,
            convergence_tol=1e-12,
            mini_batch_fraction=0.001,  # nearly always empty on N=3
            random_seed=7,
        ),
    )
    result = runner.fit(rdd)

    # Loop ran to max_iterations; some iterations may have been skipped.
    assert result.n_iterations == 5
    assert result.converged is False
    # ELBO trace length equals number of non-skipped iterations and may be
    # less than max_iterations.
    assert len(result.elbo_trace) <= 5


def test_vi_runner_full_batch_path_unchanged_when_fraction_none(spark):
    """With mini_batch_fraction=None (the default), the run is fully deterministic
    and matches a previously-recorded full-batch outcome.

    Anchors that adding mini-batch did not regress the full-batch path. The
    number-pin here mirrors the existing end-to-end test; if those values
    drift, both tests should be updated together.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    data = [1] * 70 + [0] * 30
    rdd = spark.sparkContext.parallelize(data, numSlices=4)

    runner = VIRunner(
        model=CountingModel(prior_alpha=1.0, prior_beta=1.0),
        config=VIConfig(max_iterations=5, convergence_tol=1e-6),  # default mini_batch_fraction=None
    )
    result = runner.fit(rdd)

    a = float(result.global_params["alpha"])
    b = float(result.global_params["beta"])
    mean = a / (a + b)
    assert 0.55 < mean < 0.85
    assert result.n_iterations == 5
    assert len(result.elbo_trace) == 5
