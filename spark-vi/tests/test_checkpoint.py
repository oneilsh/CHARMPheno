"""Checkpoint-then-resume produces a VIResult indistinguishable from a
continuous run on the same data."""
import numpy as np


def test_checkpoint_then_resume_matches_continuous_run(spark, tmp_path):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.diagnostics.checkpoint import load_checkpoint, save_checkpoint
    from spark_vi.models.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 60 + [0] * 40, numSlices=4)

    # Run continuously for 6 iterations.
    cfg6 = VIConfig(max_iterations=6, convergence_tol=1e-12)
    continuous = VIRunner(CountingModel(), cfg6).fit(rdd)

    # Run for 3 iterations, checkpoint, then resume for 3 more.
    cfg3 = VIConfig(max_iterations=3, convergence_tol=1e-12)
    r3 = VIRunner(CountingModel(), cfg3).fit(rdd)
    ckpt = tmp_path / "ckpt"
    save_checkpoint(r3.global_params, r3.elbo_trace, iteration=3, path=ckpt)

    global_params, elbo_trace, completed = load_checkpoint(ckpt)
    assert completed == 3

    # Resume: run the remaining 3 iterations, starting from the checkpoint state.
    runner2 = VIRunner(CountingModel(), VIConfig(max_iterations=3, convergence_tol=1e-12))
    # Use the internal attribute to seed _resume_ — we patch global_params
    # into the runner via a dedicated helper: for this test we inject via
    # fit(... data_summary=) since CountingModel ignores it; the restart is
    # equivalent to constructing a runner whose first broadcast is the
    # checkpointed params. We emulate that by starting a fresh runner but
    # skipping initialize_global.
    model = CountingModel()
    # Monkey-patch initialize_global to return the checkpointed params.
    orig_init = model.initialize_global
    model.initialize_global = lambda _ds: global_params  # type: ignore
    # Offset the Robbins-Monro step counter so rho matches a continuous run.
    resumed = VIRunner(model, VIConfig(max_iterations=3, convergence_tol=1e-12)).fit(
        rdd, start_iteration=completed
    )
    model.initialize_global = orig_init  # restore

    # Final alpha should match (posterior counts are additive and deterministic).
    np.testing.assert_allclose(
        resumed.global_params["alpha"],
        continuous.global_params["alpha"],
        rtol=1e-6,
    )
