"""Checkpoint / resume support for long training runs.

The design mirrors the VIResult export format: params go to .npy files,
everything else goes to a JSON manifest. This keeps checkpoints inspectable
and platform-agnostic (any filesystem path works).

See docs/architecture/RISKS_AND_MITIGATIONS.md §No built-in checkpointing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_checkpoint(
    global_params: dict[str, np.ndarray],
    elbo_trace: list[float],
    iteration: int,
    path: Path | str,
) -> None:
    """Write an in-progress training state to `path/` (a directory)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    params_dir = out / "params"
    params_dir.mkdir(exist_ok=True)
    for name, arr in global_params.items():
        np.save(params_dir / f"{name}.npy", np.asarray(arr))
    (out / "manifest.json").write_text(json.dumps({
        "elbo_trace": list(elbo_trace),
        "iteration": int(iteration),
        "param_names": list(global_params.keys()),
    }, indent=2))


def load_checkpoint(
    path: Path | str,
) -> tuple[dict[str, np.ndarray], list[float], int]:
    """Read a checkpoint. Returns (global_params, elbo_trace, iteration_completed)."""
    in_path = Path(path)
    manifest = json.loads((in_path / "manifest.json").read_text())
    params_dir = in_path / "params"
    global_params = {
        name: np.load(params_dir / f"{name}.npy") for name in manifest["param_names"]
    }
    return global_params, manifest["elbo_trace"], int(manifest["iteration"])
