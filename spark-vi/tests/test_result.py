"""VIResult captures the outcome of a completed training run."""
import numpy as np


def test_vi_result_holds_global_params_and_metrics():
    from spark_vi.core import VIResult

    result = VIResult(
        global_params={"lambda": np.array([1.0, 2.0])},
        elbo_trace=[-10.0, -9.5, -9.1],
        n_iterations=3,
        converged=False,
        metadata={"model_class": "TestModel"},
    )
    assert "lambda" in result.global_params
    assert result.elbo_trace[-1] == -9.1
    assert result.n_iterations == 3
    assert result.converged is False
    assert result.metadata["model_class"] == "TestModel"


def test_vi_result_final_elbo_accessor():
    from spark_vi.core import VIResult

    r = VIResult(global_params={}, elbo_trace=[-10.0, -9.0], n_iterations=2,
                 converged=True, metadata={})
    assert r.final_elbo == -9.0


def test_vi_result_empty_trace_has_none_final_elbo():
    from spark_vi.core import VIResult

    r = VIResult(global_params={}, elbo_trace=[], n_iterations=0,
                 converged=False, metadata={})
    assert r.final_elbo is None


def test_vi_result_diagnostic_traces_default_empty():
    from spark_vi.core import VIResult

    r = VIResult(global_params={}, elbo_trace=[-1.0], n_iterations=1,
                 converged=False, metadata={})
    assert r.diagnostic_traces == {}


def test_vi_result_diagnostic_traces_round_trip_constructor():
    from spark_vi.core import VIResult

    traces = {"alpha": [0.1, 0.2, 0.3], "eta": [1.0, 1.1, 1.2]}
    r = VIResult(
        global_params={},
        elbo_trace=[-3.0, -2.5, -2.1],
        n_iterations=3,
        converged=True,
        metadata={},
        diagnostic_traces=traces,
    )
    assert r.diagnostic_traces == {"alpha": [0.1, 0.2, 0.3], "eta": [1.0, 1.1, 1.2]}
