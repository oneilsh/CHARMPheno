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


def test_export_writes_format_version_in_manifest(tmp_path):
    """Manifest carries a format_version field; load enforces it matches."""
    import json

    from spark_vi.core import VIResult
    from spark_vi.io.export import save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[],
        n_iterations=0,
        converged=False,
        metadata={},
    )
    out = tmp_path / "v"
    save_result(r, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["format_version"] == 1


def test_load_result_rejects_unknown_format_version(tmp_path):
    """A future format version that this build doesn't know yields a clear ValueError."""
    import json

    from spark_vi.core import VIResult
    from spark_vi.io.export import load_result, save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[],
        n_iterations=0,
        converged=False,
        metadata={},
    )
    out = tmp_path / "future"
    save_result(r, out)

    # Tamper with the manifest to simulate a future format.
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    import pytest as _pytest

    with _pytest.raises(ValueError, match="format_version"):
        load_result(out)


def test_export_roundtrip_preserves_in_progress_checkpoint_state(tmp_path):
    """A VIResult with converged=False (an interim checkpoint) round-trips
    identically. Confirms the dual-purpose semantics: same dataclass, same
    save/load, for both completed runs and in-progress checkpoints.
    """
    from spark_vi.core import VIResult
    from spark_vi.io.export import load_result, save_result

    r = VIResult(
        global_params={"alpha": np.array(7.0), "beta": np.array(3.0)},
        elbo_trace=[-50.0, -42.0, -39.0],
        n_iterations=3,
        converged=False,
        metadata={"checkpoint": True, "model_class": "CountingModel"},
    )
    out = tmp_path / "interim"
    save_result(r, out)
    loaded = load_result(out)

    assert loaded.converged is False
    assert loaded.n_iterations == 3
    assert loaded.metadata["checkpoint"] is True
    np.testing.assert_array_equal(loaded.global_params["alpha"], r.global_params["alpha"])


def test_save_result_round_trips_scalar_diagnostic_traces(tmp_path):
    """Scalar-valued diagnostic traces (lists of floats) live inline in the
    manifest and round-trip exactly."""
    from spark_vi.core import VIResult
    from spark_vi.io.export import load_result, save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[-10.0, -5.0, -1.0],
        n_iterations=3,
        converged=True,
        metadata={},
        diagnostic_traces={"gamma": [1.0, 2.0, 3.0]},
    )
    out = tmp_path / "scalar_trace"
    save_result(r, out)
    loaded = load_result(out)

    assert loaded.diagnostic_traces == {"gamma": [1.0, 2.0, 3.0]}
    # No traces/ directory should be created for purely scalar traces.
    assert not (out / "traces").exists()


def test_save_result_round_trips_vector_diagnostic_traces(tmp_path):
    """Vector-valued diagnostic traces emit traces/<name>.npy as a 2D array
    and round-trip element-wise."""
    from spark_vi.core import VIResult
    from spark_vi.io.export import load_result, save_result

    alpha_trace = [
        np.array([0.1, 0.2]),
        np.array([0.15, 0.25]),
        np.array([0.18, 0.28]),
    ]
    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[-10.0, -5.0, -1.0],
        n_iterations=3,
        converged=True,
        metadata={},
        diagnostic_traces={"alpha": alpha_trace},
    )
    out = tmp_path / "vector_trace"
    save_result(r, out)

    # Sidecar file must exist and be 2D (n_iterations x dim).
    sidecar = out / "traces" / "alpha.npy"
    assert sidecar.is_file()
    on_disk = np.load(sidecar)
    assert on_disk.shape == (3, 2)

    loaded = load_result(out)
    assert list(loaded.diagnostic_traces.keys()) == ["alpha"]
    loaded_alpha = loaded.diagnostic_traces["alpha"]
    assert len(loaded_alpha) == len(alpha_trace)
    for got, want in zip(loaded_alpha, alpha_trace):
        np.testing.assert_array_equal(got, want)


def test_save_result_round_trips_mixed_scalar_and_vector_traces(tmp_path):
    """Both scalar and vector traces in the same VIResult round-trip via
    their respective storage strategies."""
    from spark_vi.core import VIResult
    from spark_vi.io.export import load_result, save_result

    alpha_trace = [np.array([0.1, 0.2]), np.array([0.15, 0.25])]
    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[-10.0, -1.0],
        n_iterations=2,
        converged=True,
        metadata={},
        diagnostic_traces={"gamma": [1.0, 2.0], "alpha": alpha_trace},
    )
    out = tmp_path / "mixed_trace"
    save_result(r, out)
    loaded = load_result(out)

    assert loaded.diagnostic_traces["gamma"] == [1.0, 2.0]
    loaded_alpha = loaded.diagnostic_traces["alpha"]
    assert len(loaded_alpha) == 2
    for got, want in zip(loaded_alpha, alpha_trace):
        np.testing.assert_array_equal(got, want)


def test_load_result_legacy_manifest_without_diagnostic_traces(tmp_path):
    """Older checkpoints written before the diagnostic_traces field load as
    diagnostic_traces={} (forward-compatibility, no format_version bump)."""
    import json

    from spark_vi.io.export import load_result

    out = tmp_path / "legacy"
    out.mkdir()
    params_dir = out / "params"
    params_dir.mkdir()
    np.save(params_dir / "alpha.npy", np.array(1.0))

    legacy_manifest = {
        "format_version": 1,
        "elbo_trace": [-1.0],
        "n_iterations": 1,
        "converged": True,
        "metadata": {},
        "param_names": ["alpha"],
        # Note: no "diagnostic_traces" key.
    }
    (out / "manifest.json").write_text(json.dumps(legacy_manifest))

    loaded = load_result(out)
    assert loaded.diagnostic_traces == {}


def test_save_result_round_trips_zero_d_ndarray_as_scalar(tmp_path):
    """0-d ndarrays are semantically scalar and must be classified as such;
    treating them as "vector" would corrupt the round-trip via np.stack."""
    from spark_vi.core import VIResult
    from spark_vi.io.export import load_result, save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[-1.0, -0.5],
        n_iterations=2,
        converged=True,
        metadata={},
        diagnostic_traces={"foo": [np.array(1.5), np.array(2.5)]},
    )
    out = tmp_path / "zero_d"
    save_result(r, out)
    loaded = load_result(out)

    assert [float(x) for x in loaded.diagnostic_traces["foo"]] == [1.5, 2.5]
    # 0-d arrays are scalar-classified, so no traces/ sidecar is written.
    assert not (out / "traces").exists()


def test_save_result_rejects_high_rank_arrays(tmp_path):
    """Per-iteration arrays with ndim > 1 are rejected at save time; only
    scalar or 1-D-array values are supported."""
    import pytest as _pytest

    from spark_vi.core import VIResult
    from spark_vi.io.export import save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[-1.0, -0.5],
        n_iterations=2,
        converged=True,
        metadata={},
        diagnostic_traces={"foo": [np.zeros((2, 3)), np.ones((2, 3))]},
    )
    out = tmp_path / "high_rank"
    with _pytest.raises(ValueError, match="ndim=2"):
        save_result(r, out)


def test_save_result_round_trips_empty_diagnostic_trace(tmp_path):
    """An empty trace list round-trips as an empty list (no sidecar file)."""
    from spark_vi.core import VIResult
    from spark_vi.io.export import load_result, save_result

    r = VIResult(
        global_params={"alpha": np.array(1.0)},
        elbo_trace=[],
        n_iterations=0,
        converged=False,
        metadata={},
        diagnostic_traces={"foo": []},
    )
    out = tmp_path / "empty_trace"
    save_result(r, out)
    loaded = load_result(out)

    assert loaded.diagnostic_traces == {"foo": []}
    assert not (out / "traces").exists()
