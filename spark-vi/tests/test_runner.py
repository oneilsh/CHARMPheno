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
