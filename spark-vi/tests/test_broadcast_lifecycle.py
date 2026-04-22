"""Ensure VIRunner unpersists prior broadcasts (prevents OOM in long runs).

See docs/architecture/RISKS_AND_MITIGATIONS.md §Broadcast lifecycle.
"""
from unittest.mock import patch


def test_vi_runner_unpersists_prior_broadcasts(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1, 0, 1, 0], numSlices=2)
    model = CountingModel()
    cfg = VIConfig(max_iterations=4, convergence_tol=1e-10)
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
        runner.fit(rdd)

    # With 4 iterations, we expect at least 3 unpersist calls for the
    # previous broadcasts plus one for the final broadcast on return = 4.
    # The regression we guard against is zero calls.
    assert len(unpersist_calls) >= 3, \
        f"Expected VIRunner to unpersist prior broadcasts; got {len(unpersist_calls)} calls"
