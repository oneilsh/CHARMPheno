"""Ensure VIRunner unpersists prior broadcasts (prevents OOM in long runs).

Strategy: wrap each broadcast in a transparent proxy that records its own
unpersist() calls. The runner sees and uses real Spark broadcasts; the wrapper
only adds an observation side-channel. We then assert the exact unpersist
count for each terminal branch (max-iterations vs convergence) of fit().

See docs/architecture/RISKS_AND_MITIGATIONS.md §Broadcast lifecycle for the
failure mode this guards against.
"""
from unittest.mock import patch


def _run_with_broadcast_tracking(spark, cfg):
    """Run VIRunner.fit with the cfg, returning (result, unpersist_calls)."""
    from spark_vi.core import VIRunner
    from spark_vi.models.topic.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1, 0, 1, 0], numSlices=2).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition
    model = CountingModel()
    runner = VIRunner(model=model, config=cfg)

    # Capture the real broadcast method *before* patching, so the wrapper can
    # delegate to the original. Without this, _wrapping_broadcast would call
    # the patched method recursively.
    real_broadcast = spark.sparkContext.broadcast
    unpersist_calls = []

    class _WrappedBcast:
        """Transparent proxy: forwards .value to the real broadcast and only
        adds an observation hook to unpersist(). The runner cannot tell it
        apart from a real Broadcast object — important so we exercise the
        real broadcast lifecycle, not a mock substitute.
        """

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

    # Scoped patch: reverted on context exit even if the test fails.
    with patch.object(spark.sparkContext, "broadcast", side_effect=_wrapping_broadcast):
        result = runner.fit(rdd)

    return result, unpersist_calls


def test_vi_runner_unpersists_prior_broadcasts_max_iterations_path(spark):
    """Full-loop path: max_iterations=4 with tight tol produces exactly 4 unpersist calls.

    Counting math: each iteration creates one broadcast (4 total). The runner's
    "unpersist the *previous* one at the start of cleanup" pattern produces 3
    mid-loop unpersists across 4 iterations (iter 2 frees iter 1's, iter 3
    frees iter 2's, iter 4 frees iter 3's). The max-iter terminal branch in
    runner.py then frees the final broadcast (+1). Total = 4.

    If a future change removes either the mid-loop unpersist or the terminal
    one, this count will break — surfacing the leak before it manifests as
    OOM in production runs of 100+ iterations.
    """
    from spark_vi.core import VIConfig

    # mini_batch_fraction=1.0 (not None) to guarantee 4 iterations: this test is
    # about broadcast lifecycle across the full loop, and a full-batch fit can no
    # longer supply them. Full batch takes an undamped rho=1 step (see
    # test_vi_runner_full_batch_uses_undamped_step), so the conjugate model behind
    # _run_with_broadcast_tracking reaches its exact posterior in ONE iteration and
    # then converges at iteration 2. A set fraction is always treated as mini-batch
    # and never early-stops (VIRunner.fit docstring); 1.0 keeps essentially all the
    # data, and sampling creates no broadcasts, so the counting math is unchanged.
    cfg = VIConfig(max_iterations=4, convergence_tol=1e-10,
                   mini_batch_fraction=1.0, random_seed=0)
    result, unpersist_calls = _run_with_broadcast_tracking(spark, cfg)

    assert result.converged is False
    assert result.n_iterations == 4
    assert len(unpersist_calls) == 4, (
        "Expected 3 mid-loop swaps + 1 final cleanup = 4 unpersists on the "
        f"max-iterations path; got {len(unpersist_calls)}"
    )


def test_vi_runner_unpersists_prior_broadcasts_convergence_path(spark):
    """Early-stop path: wide tol converges on iteration 2 with exactly 2 unpersist calls.

    Counterpart to the max-iterations test above: the convergence-return
    branch is a *separate* return site in runner.fit() and must independently
    perform the terminal unpersist. Loose tol forces convergence after iter 2:
    1 mid-loop unpersist (iter 2 frees iter 1's) + 1 terminal cleanup before
    the converged-return = 2.

    Together, the two tests pin down that *both* terminal branches clean up.
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


def test_vi_runner_transform_does_not_eagerly_unpersist_its_broadcast(spark):
    """transform() must NOT eagerly unpersist its broadcast.

    The returned RDD captures the broadcast in its closure, so the broadcast has
    to live as long as the RDD is used. Unlike fit() — whose per-iteration
    broadcasts are unpersisted AFTER that iteration's action has read them — a
    lazy transform has no action of its own, so an eager unpersist frees nothing
    and forces a re-broadcast on every downstream action. Lifetime is delegated
    to Spark's ContextCleaner (GC of the returned RDD), matching MLlib's own
    models. We assert (1) the broadcast is not eagerly unpersisted, and (2) the
    result is correct across MULTIPLE actions on the same returned RDD.
    """
    from unittest.mock import patch
    from spark_vi.core import VIRunner
    from spark_vi.core.model import VIModel
    import numpy as np

    class _ToyModel(VIModel):
        def initialize_global(self, data_summary=None):
            return {"k": np.array(1.0)}
        def local_update(self, rows, global_params):
            return {"x": np.array(0.0)}
        def update_global(self, global_params, target_stats, learning_rate):
            return global_params
        def infer_local(self, row, global_params):
            return float(row)

    rdd = spark.sparkContext.parallelize([1.0, 2.0], numSlices=2)
    runner = VIRunner(_ToyModel())

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
        return _WrappedBcast(real_broadcast(value))

    with patch.object(spark.sparkContext, "broadcast", side_effect=_wrapping_broadcast):
        out = runner.transform(rdd, global_params={"k": np.array(1.0)})
        first = sorted(out.collect())
        second = sorted(out.collect())   # a second action must still work

    assert first == second == [1.0, 2.0], (
        "transform result must be correct and reusable across multiple actions"
    )
    assert unpersist_calls == [], (
        "transform must NOT eagerly unpersist its broadcast — lifetime is the "
        f"returned RDD's (ContextCleaner reclaims it on GC); got {len(unpersist_calls)} "
        "eager unpersist call(s)"
    )
