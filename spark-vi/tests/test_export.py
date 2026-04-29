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
