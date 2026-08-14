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
#   v2 (2026-07-25): global_params values may be a dict of arrays, written as
#   one params/<name>_<key>.npy per key with the keys listed in the manifest's
#   "dict_param_keys". Motivated by the multi-domain gated model's per-domain
#   lambda {m: (K, V_m)} (MixEHR-style storage; Li, Nair, Lu et al. 2020,
#   Nat. Commun.). v1 archives have no "dict_param_keys" and still load.
_FORMAT_VERSION = 2
_READABLE_FORMAT_VERSIONS = (1, 2)


class UnsupportedGlobalParamError(TypeError):
    """A global_params value this format cannot write and read back.

    `save_result` stores one params/<name>.npy per global_params entry and
    `load_result` reads them with np.load's default allow_pickle=False (they
    are model parameters, not our own trusted trace sidecars). Anything that does
    not convert to a NUMERIC array therefore becomes an OBJECT array, which
    np.save happily pickles -- the write SUCCEEDS and the later load fails with
    "Object arrays cannot be loaded when allow_pickle=False". Raising here
    converts that silent, delayed corruption into an immediate named failure.
    """


def _check_saveable_param(name: str, arr: object) -> np.ndarray:
    """Return `arr` as the ndarray to write, or raise UnsupportedGlobalParamError.

    The test is on the CONVERTED dtype, not on ``isinstance(arr, np.ndarray)``:
    0-d array arithmetic in numpy returns an np.float64 SCALAR (``np.array(0.0) +
    np.array(1.0)`` is not an ndarray), so models that keep a scalar parameter in
    global_params legitimately hand this function numpy scalars, and those write
    and load back as ordinary 0-d numeric .npy files. Object dtype is exactly the
    boundary that matters: it is the only thing np.save can store solely as a
    pickle, and therefore the only thing `load_result`'s allow_pickle=False
    cannot read.

    The one object-dtype shape this function is handed directly for is the
    multi-domain gated topic model's per-domain dict lambda ({m: (K, V_m)},
    MixEHR-style storage; Li, Nair, Lu et al. 2020, Nat. Commun.): np.asarray on
    a dict yields a 0-d object array that writes as a few hundred pickled bytes
    and never loads back. `save_result` no longer hands a dict to this function
    directly -- it unpacks the dict one key at a time and calls this on each
    block (one params/<name>_<key>.npy per domain, with the domain keys recorded
    in the manifest's "dict_param_keys") -- so what reaches here per call is
    always a single block. This guard therefore still fires for a NON-numeric
    block inside a dict param (e.g. a stray nested dict or object array under one
    domain key), and the caller passes a ``name[key]``-shaped label so the error
    names the offending key, not just the param.
    """
    try:
        out = np.asarray(arr)
    except Exception as exc:                      # ragged input, etc.
        raise UnsupportedGlobalParamError(
            f"global_params[{name!r}] ({type(arr).__name__}) is not convertible "
            f"to a numeric array: {exc}"
        ) from exc
    if out.dtype.hasobject:
        raise UnsupportedGlobalParamError(
            f"global_params[{name!r}] is a {type(arr).__name__}, which converts "
            f"to an OBJECT-dtype array; np.save can only store that as a pickle "
            f"and load_result reads params with allow_pickle=False, so the write "
            f"would succeed and the read would fail. Only numeric arrays (and "
            f"dicts of numeric arrays, one .npy per key) are supported."
        )
    return out


def _classify_trace(name: str, trace: list) -> str:
    """Decide on-disk strategy for a single diagnostic_traces entry.

    Returns "empty", "scalar", "array", or "json". Raises ValueError if a
    trace mixes kinds (we don't silently coerce).

    Storage strategy by kind:
        scalar  — inline list of floats in manifest.json.
        array   — sidecar traces/<name>.npy of shape (n_iter, *dims); any
                  per-iteration ndarray rank is supported (1-D, 2-D, ...).
        json    — wrapped object {"json": [...]} in manifest.json, values
                  stored as-is (must be JSON-serializable).
        empty   — bare empty list in manifest.json.

    No size cap is imposed: emitting a trace is an explicit opt-in (the base
    iteration_diagnostics returns {}), so heavy per-iter state is a deliberate
    model choice the framework persists faithfully. A model that should not
    carry such state suppresses it by returning {} / omitting the key.
    """
    if len(trace) == 0:
        return "empty"

    def _kind(x: object) -> str:
        # 0-d ndarrays are semantically scalar; treating them as "array"
        # would corrupt the round-trip via np.stack (which would then yield
        # a 1-D array whose rows are not arrays at all).
        if isinstance(x, np.ndarray) and x.ndim == 0:
            return "scalar"
        # bool is a subclass of int — keep it in scalar bucket; bool fidelity
        # through float() cast is preserved (0.0/1.0 round-trip exactly).
        if isinstance(x, (int, float, np.floating, np.integer)):
            return "scalar"
        if isinstance(x, np.ndarray):
            return "array"
        return "json"

    kinds = {_kind(x) for x in trace}
    if len(kinds) > 1:
        raise ValueError(
            f"trace {name!r} has mixed value kinds {sorted(kinds)}; each "
            f"trace must be homogeneous across iterations."
        )
    return kinds.pop()


def save_result(result: VIResult, out_dir: Path | str) -> None:
    """Write `result` to `out_dir`. Creates the dir if needed.

    `diagnostic_traces` is split by value kind:
      * scalar-valued traces (lists of floats) are stored inline in
        manifest.json under the top-level "diagnostic_traces" key as plain
        JSON lists.
      * array-valued traces (np.ndarray per iter, any rank) are stacked to
        shape (n_iterations, *dims) and written to traces/<name>.npy. The
        manifest records a small marker dict ``{"file": "traces/<name>.npy"}``
        for that key — explicit and self-documenting compared to a sentinel
        string. Higher-rank per-iter arrays (e.g. STM's (P, K) Gamma) are
        supported; the framework persists whatever a model emits, so keeping
        per-iter state small is the model's responsibility.
      * json-valued traces (anything else JSON-serializable: strings,
        lists, dicts) are stored inline in manifest.json as a wrapped
        marker dict ``{"json": [...]}`` to distinguish them from scalar
        traces (which appear as bare JSON lists). Values are written
        as-is and round-trip via the same JSON path.

    An empty trace list round-trips inline as ``[]``; an empty
    diagnostic_traces dict produces no traces/ directory.

    A `global_params` value may itself be a dict of arrays (the multi-domain
    gated model's per-domain lambda, {m: (K, V_m)}, MixEHR-style storage; Li,
    Nair, Lu et al. 2020, Nat. Commun.): each block is written as its own
    params/<name>_<key>.npy, since the blocks have different widths (V_m
    differs per domain) and there is no single array to write. The domain
    keys are recorded in the manifest's "dict_param_keys" so `load_result`
    knows which sidecar files to read back, in domain order.

    Raises UnsupportedGlobalParamError if a `global_params` value (or, for a
    dict-valued param, one of its per-key blocks) does not convert to a
    NUMERIC array (see `_check_saveable_param`). `diagnostic_traces` are
    unaffected: they have their own sidecar path and are loaded with
    allow_pickle=True.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params_dir = out / "params"
    params_dir.mkdir(exist_ok=True)

    # A dict-valued param is stored per key, NOT as one array: the blocks have
    # different widths (V_m differs per domain), so there is no single array to
    # write, and np.asarray on the dict would yield a 0-d object array that
    # np.save can only pickle and load_result could never read back.
    dict_param_keys: dict[str, list[int]] = {}
    for name, arr in result.global_params.items():
        if isinstance(arr, dict):
            keys = sorted(arr)
            for k in keys:
                np.save(params_dir / f"{name}_{k}.npy",
                        _check_saveable_param(f"{name}[{k}]", arr[k]))
            dict_param_keys[name] = [int(k) for k in keys]
        else:
            np.save(params_dir / f"{name}.npy", _check_saveable_param(name, arr))

    # Split diagnostic_traces by storage strategy. Array traces go to
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
        else:  # array
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
        "dict_param_keys": dict_param_keys,
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

    A `global_params` entry named in the manifest's "dict_param_keys" (the
    multi-domain gated model's per-domain lambda) is loaded back as a dict,
    one params/<name>_<key>.npy per domain, in domain order. JSON object
    keys are strings, so the keys are converted back to int -- every
    consumer indexes this dict with an int domain id. A v1 archive has no
    "dict_param_keys" key at all (``manifest.get(..., {})`` below is empty),
    so every param loads as a plain array exactly as before.

    Raises ValueError if the manifest's format_version is not understood by
    this build (see `_READABLE_FORMAT_VERSIONS`). Manifests written before
    format_version was introduced are treated as version 1 (no production
    checkpoints predate this field).
    """
    in_path = Path(in_dir)
    manifest = json.loads((in_path / "manifest.json").read_text())
    version = manifest.get("format_version", 1)
    if version not in _READABLE_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported persistence format_version {version}; this build "
            f"reads format_version(s) {_READABLE_FORMAT_VERSIONS}."
        )
    params_dir = in_path / "params"
    # JSON object keys are strings; the per-domain lambda is keyed by INT domain
    # index and every consumer indexes it with an int, so convert back.
    dict_param_keys = {n: [int(k) for k in ks]
                       for n, ks in manifest.get("dict_param_keys", {}).items()}
    global_params: dict[str, object] = {}
    for name in manifest["param_names"]:
        if name in dict_param_keys:
            global_params[name] = {
                k: np.load(params_dir / f"{name}_{k}.npy")
                for k in dict_param_keys[name]
            }
        else:
            global_params[name] = np.load(params_dir / f"{name}.npy")

    diagnostic_traces: dict[str, list] = {}
    for name, entry in manifest.get("diagnostic_traces", {}).items():
        if isinstance(entry, dict) and "file" in entry:
            # allow_pickle=True: a model may emit object-dtype array traces
            # (e.g. STM's per-iter topic_block_labels, a length-K array of
            # label strings), which np.save pickles. These are our own trusted
            # checkpoint sidecars. Numeric traces load identically either way;
            # global_params (line above) stay strict (numeric only).
            arr = np.load(in_path / entry["file"], allow_pickle=True)
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
