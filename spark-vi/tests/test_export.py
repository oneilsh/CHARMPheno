"""VIResult.save + load roundtrip via JSON + .npy sidecar files."""
import numpy as np


def test_export_roundtrip_preserves_params_exactly(tmp_path):
    from spark_vi.core import VIResult
    from spark_vi.io.export import save_result, load_result

    r = VIResult(
        global_params={
            "alpha": np.array(3.5),
            "lambda": np.array([[1.0, 2.0], [3.0, 4.0]]),
        },
        elbo_trace=[-100.0, -50.0, -10.0],
        n_iterations=3,
        converged=True,
        metadata={"model_class": "CountingModel", "git_sha": "abc123"},
    )
    out_dir = tmp_path / "result"
    save_result(r, out_dir)

    loaded = load_result(out_dir)
    np.testing.assert_array_equal(loaded.global_params["alpha"], r.global_params["alpha"])
    np.testing.assert_array_equal(loaded.global_params["lambda"], r.global_params["lambda"])
    assert loaded.elbo_trace == r.elbo_trace
    assert loaded.n_iterations == r.n_iterations
    assert loaded.converged is True
    assert loaded.metadata == r.metadata


def test_export_produces_inspectable_files(tmp_path):
    """Files on disk are plain JSON + .npy so a human can inspect them."""
    import json

    from spark_vi.core import VIResult
    from spark_vi.io.export import save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[-1.0],
        n_iterations=1,
        converged=False,
        metadata={},
    )
    out = tmp_path / "x"
    save_result(r, out)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_iterations"] == 1
    assert manifest["converged"] is False
    assert (out / "params" / "alpha.npy").is_file()
