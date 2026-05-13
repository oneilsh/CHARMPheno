"""Save and load VIResults in a human-inspectable format.

Layout:
    <dir>/
      manifest.json           # everything except the np.ndarrays
      params/
        <name>.npy            # one file per entry in global_params
      traces/                 # only created if any vector-valued
        <name>.npy            # diagnostic_traces entry exists; one
                              # 2D array per trace, shape (n_iter, dim)

Rationale: JSON + .npy is the simplest format that is inspectable from the
command line, survives long-term storage without opaque binary blobs, and
doesn't require any non-standard library to read. The same format serves
both "final fit outcome" exports and "interim checkpoint" auto-saves
written during a fit; see ADR 0006 for the unification rationale.

See docs/architecture/SPARK_VI_FRAMEWORK.md#viresult-and-model-export and
docs/decisions/0006-unified-persistence-format.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spark_vi.core.result import VIResult

# Manifest schema version. Bump when changing the on-disk shape; load_result
# rejects unknown versions with a clear error to provide a migration handle.
_FORMAT_VERSION = 1


def _classify_trace(name: str, trace: list) -> str:
    """Decide on-disk strategy for a single diagnostic_traces entry.

    Returns "empty", "scalar", "vector", or "json". Raises ValueError if a
    trace mixes kinds (we don't silently coerce), or if any per-iteration
    array has rank > 1 (only scalar or 1-D-array values are supported).

    Storage strategy by kind:
        scalar  — inline list of floats in manifest.json.
        vector  — sidecar traces/<name>.npy of shape (n_iter, dim).
        json    — wrapped object {"json": [...]} in manifest.json, values
                  stored as-is (must be JSON-serializable).
        empty   — bare empty list in manifest.json.
    """
    if len(trace) == 0:
        return "empty"

    def _kind(x: object) -> str:
        # 0-d ndarrays are semantically scalar; treating them as "vector"
        # would corrupt the round-trip via np.stack (which would then yield
        # a 1-D array whose rows are not arrays at all).
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return "scalar"
        # bool is a subclass of int — keep it in scalar bucket; bool fidelity
        # through float() cast is preserved (0.0/1.0 round-trip exactly).
        if isinstance(x, (int, float, np.floating, np.integer)):
            return "scalar"
        if isinstance(x, np.ndarray):
            return "vector"
        return "json"

    kinds = {_kind(x) for x in trace}
    if len(kinds) > 1:
        raise ValueError(
            f"trace {name!r} has mixed value kinds {sorted(kinds)}; each "
            f"trace must be homogeneous across iterations."
        )
    kind = kinds.pop()
    if kind == "vector":
        for x in trace:
            if isinstance(x, np.ndarray) and x.ndim > 1:
                raise ValueError(
                    f"trace {name!r} has elements with ndim={x.ndim}; only "
                    f"scalar or 1-D-array per-iteration values are supported."
                )
    return kind


def save_result(result: VIResult, out_dir: Path | str) -> None:
    """Write `result` to `out_dir`. Creates the dir if needed.

    `diagnostic_traces` is split by value kind:
      * scalar-valued traces (lists of floats) are stored inline in
        manifest.json under the top-level "diagnostic_traces" key as plain
        JSON lists.
      * vector-valued traces (1-D per-iter arrays only) are persisted as a
        2-D array of shape (n_iterations, dim) and written to
        traces/<name>.npy. The manifest records a small marker dict
        ``{"file": "traces/<name>.npy"}`` for that key — explicit and self-
        documenting compared to a sentinel string. Per-iter arrays with
        ndim > 1 are rejected at save time (YAGNI).
      * json-valued traces (anything else JSON-serializable: strings,
        lists, dicts) are stored inline in manifest.json as a wrapped
        marker dict ``{"json": [...]}`` to distinguish them from scalar
        traces (which appear as bare JSON lists). Values are written
        as-is and round-trip via the same JSON path.

    An empty trace list round-trips inline as ``[]``; an empty
    diagnostic_traces dict produces no traces/ directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params_dir = out / "params"
    params_dir.mkdir(exist_ok=True)

    for name, arr in result.global_params.items():
        np.save(params_dir / f"{name}.npy", np.asarray(arr))

    # Split diagnostic_traces by storage strategy. Vector traces go to
    # traces/<name>.npy; scalar (and empty) traces stay inline in JSON.
    diagnostic_traces_manifest: dict = {}
    traces_dir: Path | None = None
    for name, trace in result.diagnostic_traces.items():
        kind = _classify_trace(name, list(trace))
        if kind == "scalar":
            diagnostic_traces_manifest[name] = [float(x) for x in trace]
        elif kind == "empty":
            diagnostic_traces_manifest[name] = []
        elif kind == "json":
            # Validate JSON-serializability eagerly so the failure points at
            # the offending trace, not at the final manifest.json write.
            try:
                json.dumps(trace)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"trace {name!r} is not JSON-serializable: {exc}. Only "
                    f"plain JSON types (str, list, dict, bool, int/float, "
                    f"None) are supported for non-numeric diagnostics."
                ) from exc
            diagnostic_traces_manifest[name] = {"json": list(trace)}
        else:  # vector
            if traces_dir is None:
                traces_dir = out / "traces"
                traces_dir.mkdir(exist_ok=True)
            stacked = np.stack([np.asarray(x) for x in trace], axis=0)
            np.save(traces_dir / f"{name}.npy", stacked)
            diagnostic_traces_manifest[name] = {"file": f"traces/{name}.npy"}

    manifest = {
        "format_version": _FORMAT_VERSION,
        "elbo_trace": list(result.elbo_trace),
        "n_iterations": int(result.n_iterations),
        "converged": bool(result.converged),
        "metadata": dict(result.metadata),
        "param_names": list(result.global_params.keys()),
        "diagnostic_traces": diagnostic_traces_manifest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_result(in_dir: Path | str) -> VIResult:
    """Load a VIResult previously written by `save_result`.

    Restores `diagnostic_traces` per the storage strategy used at save:
    a list value in the manifest is a scalar trace (returned as
    ``[float(x), ...]``); a dict value with a "file" key points to a
    sidecar traces/<name>.npy that is loaded and split row-wise back into
    a list of np.ndarray; a dict value with a "json" key holds an inline
    list of arbitrary JSON-serializable values (returned as-is).

    Manifests written before the diagnostic_traces field existed simply
    omit the key; those load with ``diagnostic_traces={}`` for forward-
    compatibility (no format_version bump was required).

    Raises ValueError if the manifest's format_version is not understood by
    this build. Manifests written before format_version was introduced are
    treated as version 1 (no production checkpoints predate this field).
    """
    in_path = Path(in_dir)
    manifest = json.loads((in_path / "manifest.json").read_text())
    version = manifest.get("format_version", 1)
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported persistence format_version {version}; this build "
            f"reads format_version {_FORMAT_VERSION}."
        )
    params_dir = in_path / "params"
    global_params = {
        name: np.load(params_dir / f"{name}.npy") for name in manifest["param_names"]
    }

    diagnostic_traces: dict[str, list] = {}
    for name, entry in manifest.get("diagnostic_traces", {}).items():
        if isinstance(entry, dict) and "file" in entry:
            arr = np.load(in_path / entry["file"])
            # Split rows back into a list of arrays. arr is 2D
            # (n_iterations, dim); we copy each row so a caller mutating
            # one row can't silently corrupt the others through the shared
            # backing buffer.
            diagnostic_traces[name] = [arr[i].copy() for i in range(arr.shape[0])]
        elif isinstance(entry, dict) and "json" in entry:
            # Inline json-mode trace: arbitrary JSON-serializable values
            # round-tripped as-is (strings, lists, dicts, ...).
            diagnostic_traces[name] = list(entry["json"])
        else:
            # Inline scalar trace (possibly empty).
            diagnostic_traces[name] = [float(x) for x in entry]

    return VIResult(
        global_params=global_params,
        elbo_trace=manifest["elbo_trace"],
        n_iterations=manifest["n_iterations"],
        converged=manifest["converged"],
        metadata=manifest["metadata"],
        diagnostic_traces=diagnostic_traces,
    )
