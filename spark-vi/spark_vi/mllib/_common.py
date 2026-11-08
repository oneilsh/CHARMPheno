"""Shared generic helpers for the MLlib shims.

Persistence Params, the apply_persistence_params splice, the
_PersistableModel mixin (the custom VIResult save/load OnlineLDA / OnlineHDP still
use) and _StagePersistence (MLlib persistence: Params metadata + our data files, spec
R13 Level 2) live here because they apply to any shim (topic-model or otherwise). Topic-specific helpers like the
SparseVector → BOWDocument converter live under
spark_vi.mllib.topic._common so this module stays free of topic content.
"""
from __future__ import annotations

import dataclasses

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import (DefaultParamsReader, DefaultParamsWriter, MLReadable,
                             MLReader, MLWritable, MLWriter)


class _PersistenceParams(Params):
    """Shared MLlib Params for shim persistence (saveInterval, saveDir, resumeFrom).

    Both OnlineLDA and OnlineHDP shims inherit this so the Param surface is
    declared once. Defaults are seeded via :meth:`_set_persistence_defaults`
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


class _PersistableModel:
    """Mixin providing ``save(path)`` / ``load(path)`` to shim Models.

    Subclass MUST define the class attribute ``_expected_model_class: str``
    matching the class name the runner stamps into
    ``VIResult.metadata["model_class"]`` (e.g. ``"OnlineLDA"``,
    ``"OnlineHDP"``). :meth:`load` validates this and raises on mismatch
    so a user can't silently load an HDP checkpoint into an
    ``OnlineLDAModel``.

    Subclass ``__init__`` must accept a single ``result: VIResult``
    positional arg (per-shim shape constants like T/K/V come from
    ``result.metadata`` via the model's ``get_metadata`` contribution).
    """

    _expected_model_class: str  # subclass sets this

    def save(self, path: str) -> None:
        """Persist this trained model to ``path``.

        Wraps :func:`spark_vi.io.export.save_result`. The directory contents
        round-trip through this class's ``load(path)``.
        """
        from spark_vi.io.export import save_result
        save_result(self._result, path)

    @classmethod
    def load(cls, path: str):
        """Load a previously-saved model from ``path``.

        Validates that the saved metadata identifies the expected model
        class; raises ``ValueError`` on type mismatch (e.g. trying to
        load an OnlineHDP checkpoint into an OnlineLDAModel).
        """
        from spark_vi.io.export import load_result
        result = load_result(path)
        saved_class = result.metadata.get("model_class")
        if saved_class is None:
            raise ValueError(
                f"Checkpoint at {path} has no 'model_class' in its metadata; "
                f"cannot determine model type. Was this saved by a recent "
                f"version of spark_vi?"
            )
        if saved_class != cls._expected_model_class:
            raise ValueError(
                f"Expected '{cls._expected_model_class}' checkpoint at "
                f"{path}, got {saved_class!r}. Did you mean a different "
                f"Model class (e.g. OnlineHDPModel.load)?"
            )
        # Reconstruct from VIResult; shape info comes from metadata.
        # (No Param round-trip — Pipeline.save persistence is deferred per
        # ADRs 0009 / 0012.)
        return cls(result)


# --------------------------------------------------------------------------- #
# MLlib persistence (spec R13 Level 2): Params metadata + our own data files.  #
# --------------------------------------------------------------------------- #
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
    text = DefaultParamsWriter._get_metadata_to_save(instance, sc, extra_metadata)
    fs.write_text(fs.join(md, "part-00000"), text + "\n")
    fs.write_bytes(fs.join(md, "_SUCCESS"), b"")


def read_params_metadata(uri: str, expected_class: str = "") -> dict:
    """Read the Params metadata under ``<uri>/metadata`` from the driver: the
    counterpart of ``write_params_metadata``, and of ``DefaultParamsReader.loadMetadata``
    without its executor-side ``sc.textFile`` (same failure for a driver-local path).
    Reads either writer's files (the first non-empty line of the first ``part-*``)."""
    from spark_vi.io import fs
    md = fs.join(uri, "metadata")
    parts = [n for n in fs.listdir(md) if n.startswith("part-")]
    if not parts:
        raise FileNotFoundError(f"{uri}: no saved stage here (no metadata/part-*)")
    line = next((ln for ln in fs.read_text(fs.join(md, parts[0])).splitlines()
                 if ln.strip()), None)
    if line is None:
        raise ValueError(f"{fs.join(md, parts[0])} is empty")
    return DefaultParamsReader._parseMetaData(line, expected_class)


class DriverParamsWriter(DefaultParamsWriter):
    """``DefaultParamsWriter`` whose metadata is written from the driver
    (``write_params_metadata``), for stages whose whole state is their Params."""

    def saveImpl(self, path: str) -> None:
        from spark_vi.io import fs
        write_params_metadata(self.instance, fs.qualify(path), self.sc)


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
        elif fs.exists(uri):
            raise FileExistsError(f"Path {path} already exists. To overwrite it, use "
                                  f"write().overwrite().save(path)")
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
        metadata = read_params_metadata(uri)
        if metadata["class"] != _class_name(self.cls):
            raise ValueError(f"{path} holds a {metadata['class']}, not a "
                             f"{_class_name(self.cls)}")
        instance = self.cls._load_data(uri, metadata)
        instance._resetUid(metadata["uid"])
        DefaultParamsReader.getAndSetParams(instance, metadata)
        return instance
