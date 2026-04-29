"""Auto-checkpoint and resume_from semantics for VIRunner.fit.

Three properties pinned down here:
  1. Checkpoint-then-resume produces a final state indistinguishable from a
     continuous run of equal total length (the "resume continuity" invariant).
  2. With checkpoint_interval/checkpoint_dir set, the runner auto-saves a
     VIResult every N iterations.
  3. The auto-checkpoint can be fed straight back into resume_from for a
     monkey-patch-free resume.
"""
import numpy as np


def test_checkpoint_then_resume_matches_continuous_run(spark, tmp_path):
    """Manual save_result + resume_from preserves the Robbins-Monro schedule
    across the checkpoint boundary."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.io.export import save_result
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 60 + [0] * 40, numSlices=4)

    cfg6 = VIConfig(max_iterations=6, convergence_tol=1e-12)
    continuous = VIRunner(CountingModel(), cfg6).fit(rdd)

    cfg3 = VIConfig(max_iterations=3, convergence_tol=1e-12)
    r3 = VIRunner(CountingModel(), cfg3).fit(rdd)
    ckpt = tmp_path / "ckpt"
    save_result(r3, ckpt)

    resumed = VIRunner(CountingModel(), cfg3).fit(rdd, resume_from=ckpt)

    np.testing.assert_allclose(
        resumed.global_params["alpha"],
        continuous.global_params["alpha"],
        rtol=1e-6,
    )


def test_auto_checkpoint_writes_per_interval(spark, tmp_path):
    """With checkpoint_interval set, runner writes a VIResult every N iterations.

    The checkpoint reflects the most recent loop state and matches the runner's
    own returned result (identical global_params).
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.io.export import load_result
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1, 0, 1, 0], numSlices=2)
    ckpt = tmp_path / "auto_ckpt"
    cfg = VIConfig(
        max_iterations=4,
        convergence_tol=1e-12,
        checkpoint_interval=2,
        checkpoint_dir=ckpt,
    )
    runner = VIRunner(CountingModel(), cfg)
    result = runner.fit(rdd)

    # Last on-disk checkpoint reflects the final loop state (iter 4; 4 % 2 == 0).
    loaded = load_result(ckpt)
    assert loaded.n_iterations == 4
    assert loaded.converged is False
    assert loaded.metadata.get("checkpoint") is True
    np.testing.assert_array_equal(
        loaded.global_params["alpha"], result.global_params["alpha"],
    )


def test_auto_checkpoint_then_resume_via_kwarg(spark, tmp_path):
    """End-to-end: auto-checkpoint during a 3-iteration run, then resume via
    resume_from for 3 more, and verify equivalence with a 6-iteration continuous
    run. No monkey-patching anywhere — the API is the user-facing contract.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 60 + [0] * 40, numSlices=2)
    ckpt = tmp_path / "ckpt"

    cfg6 = VIConfig(max_iterations=6, convergence_tol=1e-12)
    continuous = VIRunner(CountingModel(), cfg6).fit(rdd)

    # Session 1: fit with auto-checkpoint at the final iteration.
    cfg3_with_ckpt = VIConfig(
        max_iterations=3,
        convergence_tol=1e-12,
        checkpoint_interval=3,
        checkpoint_dir=ckpt,
    )
    VIRunner(CountingModel(), cfg3_with_ckpt).fit(rdd)

    # Session 2: a fresh runner resumes from the checkpoint and runs 3 more.
    cfg3 = VIConfig(max_iterations=3, convergence_tol=1e-12)
    resumed = VIRunner(CountingModel(), cfg3).fit(rdd, resume_from=ckpt)

    np.testing.assert_allclose(
        resumed.global_params["alpha"],
        continuous.global_params["alpha"],
        rtol=1e-6,
    )
    # The resumed run reports its total iteration count (session 1 + session 2).
    assert resumed.n_iterations == 6
