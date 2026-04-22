"""Save and load VIResults in a human-inspectable format.

Layout:
    <dir>/
      manifest.json           # everything except the np.ndarrays
      params/
        <name>.npy            # one file per entry in global_params

Rationale: JSON + .npy is the simplest format that is inspectable from the
command line, survives long-term storage without opaque binary blobs, and
doesn't require any non-standard library to read. See
docs/architecture/SPARK_VI_FRAMEWORK.md#viresult-and-model-export.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spark_vi.core.result import VIResult


def save_result(result: VIResult, out_dir: Path | str) -> None:
    """Write `result` to `out_dir`. Creates the dir if needed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params_dir = out / "params"
    params_dir.mkdir(exist_ok=True)

    for name, arr in result.global_params.items():
        np.save(params_dir / f"{name}.npy", np.asarray(arr))

    manifest = {
        "elbo_trace": list(result.elbo_trace),
        "n_iterations": int(result.n_iterations),
        "converged": bool(result.converged),
        "metadata": dict(result.metadata),
        "param_names": list(result.global_params.keys()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_result(in_dir: Path | str) -> VIResult:
    """Load a VIResult previously written by `save_result`."""
    in_path = Path(in_dir)
    manifest = json.loads((in_path / "manifest.json").read_text())
    params_dir = in_path / "params"
    global_params = {
        name: np.load(params_dir / f"{name}.npy") for name in manifest["param_names"]
    }
    return VIResult(
        global_params=global_params,
        elbo_trace=manifest["elbo_trace"],
        n_iterations=manifest["n_iterations"],
        converged=manifest["converged"],
        metadata=manifest["metadata"],
    )
