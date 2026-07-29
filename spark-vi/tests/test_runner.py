"""Integration test: VIRunner fits CountingModel end-to-end on local Spark."""
import numpy as np
import pytest


def test_vi_runner_fits_counting_model_end_to_end(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.counting import CountingModel

    # 100 rows: 70 heads, 30 tails.
    data = [1] * 70 + [0] * 30
    rdd = spark.sparkContext.parallelize(data, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

    model = CountingModel(prior_alpha=1.0, prior_beta=1.0)
    runner = VIRunner(
        model=model,
        config=VIConfig(max_iterations=5, convergence_tol=1e-6),
    )
    result = runner.fit(rdd)

    # Full batch takes an undamped rho = 1 step, so this CONJUGATE model lands on
    # its EXACT posterior in a single iteration: Beta(1 + 70, 1 + 30). Iteration 2
    # then sees zero ELBO change and the fit converges.
    #
    # This used to assert only a loose 0.55 < mean < 0.85 over 5 iterations,
    # because the runner applied a decaying rho to full batches and the fit merely
    # crept toward the posterior (see test_vi_runner_full_batch_uses_undamped_step).
    # The exact pin below is the point: batch VB on a conjugate model is a
    # one-step, closed-form answer, and anything less means the step is damped.
    a = float(result.global_params["alpha"])
    b = float(result.global_params["beta"])
    assert (a, b) == (71.0, 31.0)
    assert a / (a + b) == pytest.approx(70 / 100, abs=0.005)
    assert result.n_iterations == 2 and result.converged is True
    assert len(result.elbo_trace) == 2


def test_vi_runner_early_stop_branch_triggers(spark):
    """An absurdly wide convergence tolerance should trigger the early-stop branch.

    This doesn't exercise a realistic convergence criterion; it only proves that
    the converged-return path of VIRunner.fit is reachable and populates the
    VIResult correctly (converged=True, n_iterations < max, elbo_trace matches).
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 100 + [0] * 100, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition
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
    """With a tight tolerance and an ELBO that keeps moving, the runner exhausts
    max_iterations: no early-stop, converged=False, n_iterations == max.

    Uses the strictly-changing-ELBO stub (with the REAL has_converged) rather than
    CountingModel, because a conjugate model can no longer reach this branch: a
    full-batch fit takes an undamped rho=1 step, so CountingModel lands on its
    exact posterior in ONE iteration, the ELBO stops changing, and any positive
    tolerance then converges. That is a property of correct batch VB, not a
    regression — see test_vi_runner_full_batch_uses_undamped_step. Testing the
    tolerance branch against a model whose ELBO genuinely keeps changing is also
    what this test always meant to do; previously it leaned on a damped step size
    to keep a conjugate model from converging.
    """
    from spark_vi.core import VIConfig, VIRunner

    rdd = spark.sparkContext.parallelize([1] * 100 + [0] * 100, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition
    runner = VIRunner(
        model=_make_stub(real_has_converged=True),
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


def test_vi_runner_mini_batch_never_early_stops(spark):
    """Mini-batch fits ignore the (noisy) per-minibatch ELBO convergence check
    and run the full max_iterations, matching Spark MLlib's online LDA (which has
    no convergence tolerance). Even an absurdly wide tolerance that WOULD trip
    has_converged after 2 iters must not early-stop a mini-batch fit — whereas
    the same wide tolerance DOES stop a full-batch fit (see
    test_vi_runner_early_stop_branch_triggers)."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1] * 100 + [0] * 100, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition
    model = CountingModel()
    runner = VIRunner(
        model=model,
        config=VIConfig(max_iterations=6, convergence_tol=1e10,   # would stop @2 if gated off
                        mini_batch_fraction=0.5, random_seed=42),
    )
    result = runner.fit(rdd)
    assert result.converged is False
    assert result.n_iterations == 6
    assert len(result.elbo_trace) == 6


def _rho_spy_model():
    """CountingModel that records every learning_rate the runner hands it."""
    from spark_vi.models.topic.counting import CountingModel

    class _RhoSpy(CountingModel):
        def __init__(self):
            super().__init__(prior_alpha=1.0, prior_beta=1.0)
            self.rhos = []

        def update_global(self, global_params, target_stats, learning_rate):
            self.rhos.append(float(learning_rate))
            return super().update_global(global_params, target_stats,
                                        learning_rate)

    return _RhoSpy()


def test_vi_runner_full_batch_uses_undamped_step(spark):
    """A full-batch fit must use rho = 1 (batch variational EM), NOT the decaying
    Robbins-Monro schedule.

    Regression for a real defect: the runner applied rho_t = (tau0+t+1)^-kappa
    unconditionally, so full-batch fits took damped steps toward a DETERMINISTIC
    target. Since the step magnitude is rho·‖T(params)-params‖, a vanishing rho
    freezes the parameters while the fixed-point residual is still large -- the fit
    lands short of the batch-VB fixed point and the relative-ELBO early stop fires
    on the vanishing step rather than on convergence (insight 0074; it silently
    under-fit the exp 0070/0071 full-batch baselines).

    Also pins that learning_rate_tau0/kappa are INERT on the full-batch path: the
    values below would give rho_1 = 0.19 if the schedule were (wrongly) applied.
    """
    from spark_vi.core import VIConfig, VIRunner

    rdd = spark.sparkContext.parallelize([1] * 70 + [0] * 30, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition
    model = _rho_spy_model()
    result = VIRunner(
        model=model,
        config=VIConfig(max_iterations=4, convergence_tol=1e-12,
                        mini_batch_fraction=None,                 # full batch
                        learning_rate_tau0=10.0, learning_rate_kappa=0.7),
    ).fit(rdd)

    # Every step is undamped, whatever tau0/kappa say.
    assert model.rhos == [1.0] * result.n_iterations, model.rhos
    # The CONSEQUENCE, which is what actually matters: an undamped step reaches
    # this conjugate model's EXACT posterior — Beta(1 + 70, 1 + 30) — in one
    # iteration, so iteration 2 sees zero ELBO change and the fit converges before
    # max_iterations. Under the damped schedule the first step moved only
    # rho_1 = (10+1)^-0.7 = 0.19 of the way and no iteration was ever exact.
    assert (float(result.global_params["alpha"]),
            float(result.global_params["beta"])) == (71.0, 31.0)
    assert result.n_iterations == 2 and result.converged is True


def test_vi_runner_mini_batch_still_decays(spark):
    """The counterpart to test_vi_runner_full_batch_uses_undamped_step: with a
    fraction set, the Robbins-Monro schedule is still applied exactly as
    Hoffman et al. 2013 specify, rho_t = (tau0 + t + 1)^-kappa with t from 0."""
    from spark_vi.core import VIConfig, VIRunner

    rdd = spark.sparkContext.parallelize([1] * 70 + [0] * 30, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition
    model = _rho_spy_model()
    VIRunner(
        model=model,
        config=VIConfig(max_iterations=4, convergence_tol=1e-12,
                        mini_batch_fraction=0.5, random_seed=42,
                        learning_rate_tau0=10.0, learning_rate_kappa=0.7),
    ).fit(rdd)

    expected = [(10.0 + t + 1) ** -0.7 for t in range(4)]
    assert np.allclose(model.rhos, expected), (model.rhos, expected)
    assert model.rhos[0] < 1.0 and model.rhos == sorted(model.rhos, reverse=True)


def test_vi_runner_uses_mini_batch_when_fraction_set(spark):
    """With mini_batch_fraction set, the runner samples each iteration.

    Sanity-check that the fit completes and produces a reasonable posterior
    on CountingModel. Stochastic sampling means we don't pin numerics tightly,
    just that the posterior mean ends up in the right ballpark for 70/30 data.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.counting import CountingModel

    data = [1] * 70 + [0] * 30
    rdd = spark.sparkContext.parallelize(data, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

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
    from spark_vi.models.topic.counting import CountingModel

    data = [1] * 60 + [0] * 40
    rdd = spark.sparkContext.parallelize(data, numSlices=2).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

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
    from spark_vi.models.topic.counting import CountingModel

    rdd = spark.sparkContext.parallelize([1, 0, 1], numSlices=1).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

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

    The pin is now the EXACT conjugate posterior rather than a loose band: the
    full-batch step is rho = 1 (batch variational EM), which reaches Beta(1+70,
    1+30) in one iteration. See test_vi_runner_full_batch_uses_undamped_step.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.topic.counting import CountingModel

    data = [1] * 70 + [0] * 30
    rdd = spark.sparkContext.parallelize(data, numSlices=4).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

    runner = VIRunner(
        model=CountingModel(prior_alpha=1.0, prior_beta=1.0),
        config=VIConfig(max_iterations=5, convergence_tol=1e-6),  # default mini_batch_fraction=None
    )
    result = runner.fit(rdd)

    a = float(result.global_params["alpha"])
    b = float(result.global_params["beta"])
    assert (a, b) == (71.0, 31.0)
    assert 0.55 < a / (a + b) < 0.85
    assert result.n_iterations == 2
    assert len(result.elbo_trace) == 2


def test_runner_transform_calls_infer_local_on_each_row(spark):
    """VIRunner.transform applies infer_local across the RDD, returning per-row results."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.core.model import VIModel
    import numpy as np

    class _ToyModel(VIModel):
        def initialize_global(self, data_summary=None):
            return {"scale": np.array(2.0)}
        def local_update(self, rows, global_params):
            return {"x": np.array(0.0)}
        def update_global(self, global_params, target_stats, learning_rate):
            return global_params
        def infer_local(self, row, global_params):
            return {"y": float(row) * float(global_params["scale"])}

    rdd = spark.sparkContext.parallelize([1, 2, 3, 4], numSlices=2)
    runner = VIRunner(_ToyModel(), config=VIConfig())
    out = runner.transform(rdd, global_params={"scale": np.array(2.0)})
    collected = sorted([r["y"] for r in out.collect()])
    assert collected == [2.0, 4.0, 6.0, 8.0]


# Metadata merge / diagnostics / final-save tests --------------------------


def _make_stub(get_metadata=None, iteration_diagnostics=None, *,
               real_has_converged=False):
    """Construct a deterministic VIModel suitable for fit() integration tests.

    Single-key float global param; local_update is a no-op stat so update_global
    is a pure-rho mix. ELBO is a strictly-decreasing finite sequence so
    has_converged with the default relative tolerance only fires once values
    plateau — by default we keep ELBO changing each step so fits run to
    max_iterations.

    `real_has_converged=True` keeps VIModel's DEFAULT relative-change
    has_converged instead of the always-False stub, for tests that need to
    exercise the tolerance branch itself on a model whose ELBO keeps moving.
    """
    from spark_vi.core.model import VIModel
    import numpy as np

    class _Stub(VIModel):
        def initialize_global(self, data_summary=None):
            return {"theta": np.array(0.0)}

        def local_update(self, rows, global_params):
            return {"x": np.array(0.0)}

        def update_global(self, global_params, target_stats, learning_rate):
            return {"theta": global_params["theta"] + np.array(1.0)}

        def compute_elbo(self, global_params, aggregated_stats):
            # Strictly-changing finite ELBO so default has_converged stays False
            # at any reasonable tol — the global step counter we encode into
            # theta is the cleanest source of monotone change.
            return float(global_params["theta"])

    if not real_has_converged:
        _Stub.has_converged = lambda self, elbo_trace, convergence_tol: False

    if get_metadata is not None:
        _Stub.get_metadata = lambda self, _gm=get_metadata: _gm()
    if iteration_diagnostics is not None:
        _Stub.iteration_diagnostics = (
            lambda self, gp, _fn=iteration_diagnostics: _fn(gp)
        )
    return _Stub()


def test_runner_merges_get_metadata_into_result_metadata(spark):
    from spark_vi.core import VIConfig, VIRunner

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=2).persist()
    rdd.count()

    model = _make_stub(get_metadata=lambda: {"K": 5, "V": 100})
    result = VIRunner(model, VIConfig(max_iterations=3, convergence_tol=1e-12)).fit(rdd)

    assert result.metadata["K"] == 5
    assert result.metadata["V"] == 100
    assert result.metadata["model_class"] == type(model).__name__


def test_runner_get_metadata_does_not_override_model_class(spark):
    from spark_vi.core import VIConfig, VIRunner

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=2).persist()
    rdd.count()

    model = _make_stub(get_metadata=lambda: {"model_class": "Imposter"})
    result = VIRunner(model, VIConfig(max_iterations=2, convergence_tol=1e-12)).fit(rdd)

    assert result.metadata["model_class"] == type(model).__name__
    assert result.metadata["model_class"] != "Imposter"


def test_runner_accumulates_iteration_diagnostics(spark):
    from spark_vi.core import VIConfig, VIRunner
    import numpy as np

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=2).persist()
    rdd.count()

    # iteration_diagnostics is called *after* update_global, so the stub
    # observes theta = step + 1 (initial 0.0 + step+1 increments). We encode
    # alpha = theta - 1 to get the predictable 0,1,2,... sequence. eta tracks
    # 2*alpha as np.float64 to exercise the numpy-scalar path through the
    # save/load round-trip for diagnostic_traces.
    def _diag(gp):
        alpha = float(gp["theta"]) - 1.0
        return {"alpha": alpha, "eta": np.float64(alpha * 2.0)}

    model = _make_stub(iteration_diagnostics=_diag)
    result = VIRunner(model, VIConfig(max_iterations=5, convergence_tol=1e-12)).fit(rdd)

    assert result.diagnostic_traces["alpha"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [float(x) for x in result.diagnostic_traces["eta"]] == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_runner_default_iteration_diagnostics_yields_no_traces(spark):
    from spark_vi.core import VIConfig, VIRunner

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=2).persist()
    rdd.count()

    model = _make_stub()  # no iteration_diagnostics override
    result = VIRunner(model, VIConfig(max_iterations=3, convergence_tol=1e-12)).fit(rdd)

    assert result.diagnostic_traces == {}


def test_runner_final_save_writes_to_checkpoint_dir_on_convergence(spark, tmp_path):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.core.model import VIModel
    from spark_vi.io.export import load_result
    import numpy as np

    class _ConvergesAtIter2(VIModel):
        def initialize_global(self, data_summary=None):
            return {"theta": np.array(0.0)}
        def local_update(self, rows, global_params):
            return {"x": np.array(0.0)}
        def update_global(self, global_params, target_stats, learning_rate):
            return {"theta": global_params["theta"] + np.array(1.0)}
        def has_converged(self, elbo_trace, convergence_tol):
            # Trip on iter 2 (i.e., once we have >=2 elbo entries).
            return len(elbo_trace) >= 2

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=2).persist()
    rdd.count()

    ckpt = tmp_path / "ckpt_conv"
    cfg = VIConfig(
        max_iterations=10,
        convergence_tol=1e-12,
        # Set checkpoint_interval high enough that no interim save fires before
        # convergence at iter 2 — this isolates the final-save guarantee path.
        checkpoint_interval=99,
        checkpoint_dir=ckpt,
    )
    result = VIRunner(_ConvergesAtIter2(), cfg).fit(rdd)
    assert result.converged is True

    assert ckpt.exists()
    loaded = load_result(ckpt)
    assert loaded.converged is True
    assert loaded.n_iterations == result.n_iterations


def test_runner_final_save_writes_to_checkpoint_dir_on_max_iter(spark, tmp_path):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.io.export import load_result

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=2).persist()
    rdd.count()

    ckpt = tmp_path / "ckpt_max"
    cfg = VIConfig(
        max_iterations=3,
        convergence_tol=1e-12,
        checkpoint_interval=99,  # no interim save in 3 iters → only final-save runs
        checkpoint_dir=ckpt,
    )
    model = _make_stub()
    result = VIRunner(model, cfg).fit(rdd)
    assert result.converged is False
    assert result.n_iterations == 3

    assert ckpt.exists()
    loaded = load_result(ckpt)
    assert loaded.converged is False
    assert loaded.n_iterations == 3


def test_runner_no_save_when_checkpoint_dir_unset(spark, tmp_path):
    from spark_vi.core import VIConfig, VIRunner

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=2).persist()
    rdd.count()

    # tmp_path is empty before the run; assert it remains empty after.
    assert list(tmp_path.iterdir()) == []
    model = _make_stub()
    VIRunner(model, VIConfig(max_iterations=3, convergence_tol=1e-12)).fit(rdd)
    assert list(tmp_path.iterdir()) == []


def test_runner_transform_propagates_not_implemented(spark):
    """Calling transform on a model without infer_local raises NotImplementedError."""
    import pytest
    from spark_vi.core import VIRunner
    from spark_vi.models.topic.counting import CountingModel
    import numpy as np

    rdd = spark.sparkContext.parallelize([0, 1], numSlices=1)
    runner = VIRunner(CountingModel())
    out = runner.transform(rdd, global_params={"alpha": np.array(1.0), "beta": np.array(1.0)})
    with pytest.raises(Exception) as exc:
        out.collect()
    # The Spark task wraps the original error; the message survives.
    assert "CountingModel" in str(exc.value)
