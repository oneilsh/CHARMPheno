"""Ensure VIRunner unpersists prior broadcasts (prevents OOM in long runs).

See docs/architecture/RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
"""
from unittest.mock import patch


def _run_with_broadcast_tracking(spark, cfg):
    """Run VIRunner.fit with the cfg, returning (result, unpersist_calls)."""
    from spark_vi.core import VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1, 0, 1, 0], numSlices=2)
    model = CountingModel()
    runner = VIRunner(model=model, config=cfg)

    real_broadcast = spark.sparkContext.broadcast
    unpersist_calls = []

    class _WrappedBcast:
        def __init__(self, inner):
            self._inner = inner
        @property
        def value(self):
            return self._inner.value
        def unpersist(self, blocking=False):
            unpersist_calls.append(self._inner)
            return self._inner.unpersist(blocking=blocking)

    def _wrapping_broadcast(value):
        inner = real_broadcast(value)
        return _WrappedBcast(inner)

    with patch.object(spark.sparkContext, "broadcast", side_effect=_wrapping_broadcast):
        result = runner.fit(rdd)

    return result, unpersist_calls


def test_vi_runner_unpersists_prior_broadcasts_max_iterations_path(spark):
    """Full-loop path: max_iterations=4 with tight tol produces exactly 4 unpersist calls.

    Semantics per runner.py: each iteration after the first unpersists the
    previous iteration's broadcast (3 swaps across 4 iterations), and the
    final return-site cleanup unpersists the last broadcast (+1). Total = 4.
    """
    from spark_vi.core import VIConfig

    cfg = VIConfig(max_iterations=4, convergence_tol=1e-10)
    result, unpersist_calls = _run_with_broadcast_tracking(spark, cfg)

    assert result.converged is False
    assert result.n_iterations == 4
    assert len(unpersist_calls) == 4, (
        "Expected 3 mid-loop swaps + 1 final cleanup = 4 unpersists on the "
        f"max-iterations path; got {len(unpersist_calls)}"
    )


def test_vi_runner_unpersists_prior_broadcasts_convergence_path(spark):
    """Early-stop path: wide tol converges on iteration 2 with exactly 2 unpersist calls.

    The convergence-return branch must also clean up: 1 mid-loop swap from
    iteration 2 + 1 final cleanup before the converged return = 2.
    """
    from spark_vi.core import VIConfig

    cfg = VIConfig(max_iterations=100, convergence_tol=1e10)
    result, unpersist_calls = _run_with_broadcast_tracking(spark, cfg)

    assert result.converged is True
    assert result.n_iterations == 2
    assert len(unpersist_calls) == 2, (
        "Expected 1 mid-loop swap + 1 final cleanup = 2 unpersists on the "
        f"convergence path; got {len(unpersist_calls)}"
    )
