"""Shared generic helpers for the MLlib shims.

Persistence Params, the apply_persistence_params splice and the MLlib persistence
every shim stage uses (_StagePersistence: Params metadata + our data files, spec R13
Level 2; _ResultStagePersistence for the topic models, whose data is one VIResult;
our own reader and writer of MLlib's metadata JSON, ADR 0055) live here because they apply to any shim (topic-model or otherwise). Topic-specific helpers like the
SparseVector → BOWDocument converter live under
spark_vi.mllib.topic._common so this module stays free of topic content.
"""
from __future__ import annotations

import dataclasses

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import (DefaultParamsReader, MLReadable, MLReader, MLWritable,
                             MLWriter)


class _PersistenceParams(Params):
    """Shared MLlib Params for shim persistence (saveInterval, saveDir, resumeFrom).

    Every shim estimator and model (OnlineLDA, OnlineHDP, DAG-gated LDA, ...)
    inherits this so the Param surface is declared once. Defaults are seeded via :meth:`_set_persistence_defaults`
    which the concrete Estimator/Model ``__init__`` must call (Spark's
    ``_setDefault`` is per-instance, not per-class).

    Two coupled invariants that a future maintainer adding a Param here must
    preserve — neither is enforced by the type system, both have regression
    tests that will fail if forgotten:

    1. **Enumerate the new Param in the concrete ``__init__`` signature.**
       MLlib's ``@keyword_only``-style kwarg construction
       (``OnlineLDAEstimator(saveInterval=10, ...)``) requires the kwarg to
       appear *literally* in the ``__init__`` signature; declaring it on
       this mixin makes it reachable via ``setX`` but not via the
       constructor. The cloud drivers construct via kwargs, so the
       mismatch crashes at startup with a ``TypeError``. See
       ``test_*_persistence.py::test_*_constructor_accepts_persistence_kwargs``.
    2. **Seed the default in ``_set_persistence_defaults``** to match the
       default in the ``__init__`` signature. Two sources of truth for the
       same number is a known wart; until MLlib gives us a per-class
       default mechanism, sync them by hand.
    """

    saveInterval = Param(
        Params._dummy(), "saveInterval",
        "Save every N iters during fit. -1 (default) = off. When > 0 and "
        "saveDir is set, the runner writes a VIResult checkpoint every N "
        "iterations. The directory is also written once at end-of-fit "
        "regardless of where the iteration count falls.",
        typeConverter=TypeConverters.toInt,
    )
    saveDir = Param(
        Params._dummy(), "saveDir",
        "Directory for auto-saves. Empty (default) = no auto-save. When "
        "set, fit writes a VIResult on completion (and at every "
        "saveInterval iters if that is also set). The directory is the "
        "authoritative post-fit artifact — load via the Model class's classmethod load(...).",
        typeConverter=TypeConverters.toString,
    )
    resumeFrom = Param(
        Params._dummy(), "resumeFrom",
        "Path to a previously-written save dir. Empty (default) = fresh "
        "start. When set, fit loads the saved VIResult and continues from "
        "that iteration count, preserving Robbins-Monro continuity and "
        "ELBO trace.",
        typeConverter=TypeConverters.toString,
    )

    def setSaveInterval(self, value: int):
        """Set saveInterval (iterations between auto-saves; -1 disables)."""
        return self._set(saveInterval=value)

    def getSaveInterval(self) -> int:
        return int(self.getOrDefault(self.saveInterval))

    def setSaveDir(self, value: str):
        """Set saveDir (auto-save directory; empty disables)."""
        return self._set(saveDir=value)

    def getSaveDir(self) -> str:
        return str(self.getOrDefault(self.saveDir))

    def setResumeFrom(self, value: str):
        """Set resumeFrom (path to a previously-written save dir; empty = fresh start)."""
        return self._set(resumeFrom=value)

    def getResumeFrom(self) -> str:
        return str(self.getOrDefault(self.resumeFrom))

    def _set_persistence_defaults(self) -> None:
        """Seed the three persistence Params with their defaults.

        Call from the concrete Estimator/Model ``__init__``. Spark's
        ``_setDefault`` operates per-instance, so the mixin can't seed
        these at class-construction time.
        """
        self._setDefault(saveInterval=-1, saveDir="", resumeFrom="")


def apply_persistence_params(estimator, base_config):
    """Validate save Params and return updated VIConfig + resume path.

    Encapsulates the validation + VIConfig splice that both shim ``_fit``
    methods need. Raises ``ValueError`` on invalid Param combinations and
    ``FileNotFoundError`` if ``resumeFrom`` is set but the path lacks a
    ``manifest.json``. Returns ``(updated_config, resume_path)`` where
    ``resume_path`` is a path or URI string, or ``None``.

    The ``interval = max_iterations + 1`` trick handles the case where
    ``saveDir`` is set but ``saveInterval`` is -1: VIConfig requires both
    ``checkpoint_dir`` and ``checkpoint_interval`` to be set or unset
    (validated in ``core/config.py``), so we make the in-loop modulo
    ``(step+1) % interval == 0`` never fire while the runner's
    final-save guarantee still writes the dir at end-of-fit.
    """
    save_interval = estimator.getSaveInterval()
    save_dir = estimator.getSaveDir()
    resume_from = estimator.getResumeFrom()

    if save_interval == 0:
        raise ValueError(
            "saveInterval=0 is not meaningful; use -1 to disable saves"
        )
    if save_interval > 0 and save_dir == "":
        raise ValueError(
            "saveInterval > 0 requires saveDir to be set"
        )
    from spark_vi.io import fs
    if resume_from != "" and not fs.exists(fs.join(resume_from, "manifest.json")):
        raise FileNotFoundError(
            f"No manifest.json at resumeFrom path: {resume_from}"
        )

    if save_dir != "":
        interval = save_interval if save_interval > 0 else base_config.max_iterations + 1
        config = dataclasses.replace(
            base_config,
            checkpoint_dir=str(save_dir),        # a URI must not go through Path
            checkpoint_interval=interval,
        )
    else:
        config = base_config

    resume_path = str(resume_from) if resume_from else None
    return config, resume_path


# --------------------------------------------------------------------------- #
# MLlib persistence (spec R13 Level 2): Params metadata + our own data files.  #
# --------------------------------------------------------------------------- #
def params_metadata_json(instance, spark_version: str, extra_metadata=None,
                         timestamp_ms=None) -> str:
    """MLlib's Params metadata for ``instance`` as one JSON line: ``class``,
    ``timestamp`` (ms), ``sparkVersion``, ``uid``, ``paramMap`` (explicitly set
    values), ``defaultParamMap`` (default values), then ``extra_metadata``'s keys.

    The same document, keys in the same order, that MLlib's
    ``DefaultParamsWriter.saveMetadata`` writes, so MLlib's readers (and ours) load
    it. Built here from the public ``Params`` API rather than through pyspark's
    private ``DefaultParamsWriter._get_metadata_to_save`` (ADR 0055): private API can
    change in any pyspark release, and spark-vi declares no pyspark upper bound.

    ``Params`` exposes a set value (``isSet`` / ``getOrDefault``) but no getter for
    the default of a Param that is ALSO set; that one is read by clearing the Param
    and restoring it (``clear`` / ``set``), which leaves the instance as it was."""
    import json
    import time
    param_map, default_map = {}, {}
    for p in instance.params:
        if instance.isSet(p):
            value = instance.getOrDefault(p)
            param_map[p.name] = value
            if instance.hasDefault(p):
                instance.clear(p)
                try:
                    default_map[p.name] = instance.getOrDefault(p)
                finally:
                    instance.set(p, value)
        elif instance.hasDefault(p):
            default_map[p.name] = instance.getOrDefault(p)
    meta = {
        "class": _class_name(type(instance)),
        "timestamp": (int(round(time.time() * 1000)) if timestamp_ms is None
                      else int(timestamp_ms)),
        "sparkVersion": str(spark_version),
        "uid": instance.uid,
        "paramMap": param_map,
        "defaultParamMap": default_map,
    }
    if extra_metadata is not None:
        meta.update(extra_metadata)
    return json.dumps(meta, separators=(",", ":"))


def parse_params_metadata(line: str, expected_class: str = "") -> dict:
    """Parse one metadata JSON line (``params_metadata_json``'s, or MLlib's own);
    with ``expected_class`` non-empty, refuse a different saved class."""
    import json
    meta = json.loads(line)
    for key in ("class", "uid", "sparkVersion", "paramMap"):
        if key not in meta:
            raise ValueError(f"Params metadata has no {key!r}: {line[:200]}")
    if expected_class and meta["class"] != expected_class:
        raise ValueError(f"Error loading metadata: expected class name "
                         f"{expected_class} but found class name {meta['class']}")
    return meta


def write_params_metadata(instance, uri: str, sc, extra_metadata=None) -> None:
    """Write MLlib's standard Params metadata for ``instance`` under ``<uri>/metadata``
    FROM THE DRIVER (``spark_vi.io.fs``), in the file layout
    ``DefaultParamsWriter.saveMetadata`` produces (one JSON line in ``part-00000``,
    plus ``_SUCCESS``), so MLlib's own readers still read it.

    Why not ``DefaultParamsWriter.saveMetadata``: it writes through a one-task Spark
    job (``sc.parallelize([json], 1).saveAsTextFile``), i.e. on an EXECUTOR. For a
    ``file://`` path that is a directory on the driver's machine only (a run dir on
    the master's workspace mount), so every task attempt fails on the workers and a
    finished fit cannot be saved. Writing from the driver works for local paths and
    any Hadoop URI alike, and runs no job."""
    from spark_vi.io import fs
    md = fs.join(uri, "metadata")
    text = params_metadata_json(instance, sc.version, extra_metadata)
    fs.write_text(fs.join(md, "part-00000"), text + "\n")
    fs.write_bytes(fs.join(md, "_SUCCESS"), b"")


def read_params_metadata(uri: str, expected_class: str = "") -> dict:
    """Read the Params metadata under ``<uri>/metadata`` from the driver: the
    counterpart of ``write_params_metadata``, and of ``DefaultParamsReader.loadMetadata``
    without its executor-side ``sc.textFile`` (same failure for a driver-local path).
    Reads either writer's files (the first non-empty line of the first ``part-*``)."""
    from spark_vi.io import fs
    md = fs.join(uri, "metadata")
    parts = sorted(n for n in fs.listdir(md) if n.startswith("part-"))
    if not parts:
        raise FileNotFoundError(f"{uri}: no saved stage here (no metadata/part-*)")
    line = next((ln for ln in fs.read_text(fs.join(md, parts[0])).splitlines()
                 if ln.strip()), None)
    if line is None:
        raise ValueError(f"{fs.join(md, parts[0])} is empty")
    return parse_params_metadata(line, expected_class)


def refuse_existing_path(path: str, uri: str) -> None:
    """MLlib's rule for ``write().save(path)``: refuse a path that exists. Called from
    ``saveImpl``, which ``MLWriter.save`` reaches only after ``write().overwrite()``
    has removed the path (``FileSystemOverwrite``), so an overwrite never trips it."""
    from spark_vi.io import fs
    if fs.exists(uri):
        raise FileExistsError(f"Path {path} already exists. To overwrite it, use "
                              f"write().overwrite().save(path)")


class DriverParamsWriter(MLWriter):
    """MLlib's ``DefaultParamsWriter`` behaviour with the metadata written from the
    driver (``write_params_metadata``), for stages whose whole state is their Params.
    Refuses an existing path unless ``write().overwrite()`` (MLlib's rule)."""

    def __init__(self, instance):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        from spark_vi.io import fs
        uri = fs.qualify(path)
        refuse_existing_path(path, uri)
        write_params_metadata(self.instance, uri, self.sc)


class DriverParamsReader(MLReader):
    """Loads a Params-only stage saved by ``DriverParamsWriter`` (or MLlib's own
    writer), reading its metadata from the driver."""

    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def load(self, path: str):
        from spark_vi.io import fs
        metadata = read_params_metadata(fs.qualify(path), _class_name(self.cls))
        instance = self.cls()
        instance._resetUid(metadata["uid"])
        DefaultParamsReader.getAndSetParams(instance, metadata)
        return instance


class DriverParamsWritable(MLWritable, MLReadable):
    """``DefaultParamsWritable`` / ``DefaultParamsReadable`` with driver-side metadata
    I/O (see ``write_params_metadata`` for why)."""

    def write(self):
        return DriverParamsWriter(self)

    @classmethod
    def read(cls):
        return DriverParamsReader(cls)


class _StagePersistence(MLWritable, MLReadable):
    """``MLWritable`` / ``MLReadable`` for a stage whose state is Params plus data.

    On-disk layout under the stage's directory (any local path or Hadoop URI):

    - ``metadata/``: MLlib's standard Params metadata (``DefaultParamsWriter``: class,
      uid, Spark version, ``paramMap``, ``defaultParamMap``, plus the stage's
      ``_extra_metadata()``), written LAST: on an object store a reader that finds it
      finds every data file it names already complete;
    - the stage's own data files (``_save_data``; npy / npz / json through
      ``spark_vi.io.fs``).

    The path is qualified once (``fs.qualify``: MLlib's rule, a path with no scheme
    resolves against ``fs.defaultFS``) and the same URI goes to Spark's metadata
    writer and to our data writer, so the two never land on different filesystems.
    ``save`` refuses an existing path unless ``write().overwrite()`` (MLlib's rule;
    the overwrite itself is MLlib's ``FileSystemOverwrite``). ``load`` checks the
    saved class name.

    A subclass defines ``_save_data(path)``, ``_load_data(cls, path, metadata)``
    (returns the instance, before its uid and Params are restored) and optionally
    ``_extra_metadata()``. ``save(path)`` / ``load(path)`` are MLlib's
    ``write().save`` / ``read().load``, so a ``Pipeline`` / ``PipelineModel`` holding
    the stage saves and loads like one of built-in stages."""

    def _save_data(self, path: str) -> None:
        raise NotImplementedError

    def _extra_metadata(self):
        return None

    @classmethod
    def _load_data(cls, path: str, metadata: dict):
        raise NotImplementedError

    @classmethod
    def _load_without_metadata(cls, path: str):
        """A directory with no ``metadata/``: a subclass that can read an older layout
        returns the instance; ``None`` (the default) lets the load fail as usual."""
        return None

    def write(self):
        return _StageWriter(self)

    def save(self, path: str) -> None:
        """Save this stage to ``path`` (``write().save(path)``; refuses an existing
        path: use ``write().overwrite().save(path)`` to replace it)."""
        self.write().save(path)

    @classmethod
    def read(cls):
        return _StageReader(cls)

    @classmethod
    def load(cls, path: str):
        """Load a stage saved by ``save`` / ``write().save`` (``read().load(path)``)."""
        return cls.read().load(path)


class _ResultStagePersistence(_StagePersistence):
    """``_StagePersistence`` for a Model whose data is one ``VIResult``: the topic
    models' (OnlineLDA, OnlineHDP, DAG-gated LDA). Layout::

        <path>/metadata/   Params metadata (MLlib's layout), written last
        <path>/result/     the VIResult in the fit checkpoints' format
                           (spark_vi.io.export.save_result)

    A subclass sets ``_expected_model_class`` (the ``model_class`` the runner stamps
    into the result's metadata) and takes the result as its one ``__init__``
    argument; ``load`` refuses a result of another model class.

    ``_legacy_result_dir = True`` also loads a bare VIResult directory (``manifest.json``
    at the top, no ``metadata/``): the layout ``OnlineLDAModel`` / ``OnlineHDPModel``
    saved before ADR 0055, and the one every fit checkpoint (``saveDir``) and the
    drivers' ``save_result`` output still use. Such a model comes back with its Params
    at their defaults, as the old ``load`` returned it."""

    _expected_model_class: str = ""
    _legacy_result_dir: bool = False

    def _save_data(self, path: str) -> None:
        from spark_vi.io import fs
        from spark_vi.io.export import save_result
        save_result(self._result, fs.join(path, "result"))

    @classmethod
    def _result_instance(cls, result, where: str):
        saved = (result.metadata or {}).get("model_class")
        if saved != cls._expected_model_class:
            raise ValueError(f"{where}: the saved result is a {saved!r}, not a "
                             f"{cls._expected_model_class!r}")
        return cls(result)

    @classmethod
    def _load_data(cls, path: str, metadata: dict):
        from spark_vi.io import fs
        from spark_vi.io.export import load_result
        return cls._result_instance(load_result(fs.join(path, "result")), path)

    @classmethod
    def _load_without_metadata(cls, path: str):
        from spark_vi.io import fs
        from spark_vi.io.export import load_result
        if not (cls._legacy_result_dir and fs.exists(fs.join(path, "manifest.json"))):
            return None
        return cls._result_instance(load_result(path), path)


def _class_name(cls) -> str:
    return f"{cls.__module__}.{cls.__name__}"


class _StageWriter(MLWriter):
    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self._in_place = False

    def saveImpl(self, path: str) -> None:
        from spark_vi.io import fs
        uri = fs.qualify(path)
        if self._in_place:
            # Replace only this stage's files in a directory that may hold others:
            # the old metadata goes first, so a reader never pairs it with new data.
            fs.remove_tree(fs.join(uri, "metadata"))
        else:
            refuse_existing_path(path, uri)
        self.instance._save_data(uri)
        # Metadata last: it is what a reader looks for (object-store atomicity).
        write_params_metadata(self.instance, uri, self.sc,
                              extra_metadata=self.instance._extra_metadata())


def save_stage_in_place(stage, path: str) -> None:
    """Save ``stage`` (a ``_StagePersistence``) into ``path`` even when that directory
    exists and holds other files (a fit's ``saveDir``): the stage's own files are
    replaced, nothing else is touched (``write().overwrite()`` would delete the whole
    directory)."""
    writer = stage.write()
    writer._in_place = True
    writer.save(path)


class _StageReader(MLReader):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def load(self, path: str):
        from spark_vi.io import fs
        uri = fs.qualify(path)
        if not fs.exists(fs.join(uri, "metadata")):
            legacy = self.cls._load_without_metadata(uri)
            if legacy is not None:
                return legacy
        metadata = read_params_metadata(uri)
        if metadata["class"] != _class_name(self.cls):
            raise ValueError(f"{path} holds a {metadata['class']}, not a "
                             f"{_class_name(self.cls)}")
        instance = self.cls._load_data(uri, metadata)
        instance._resetUid(metadata["uid"])
        DefaultParamsReader.getAndSetParams(instance, metadata)
        return instance
