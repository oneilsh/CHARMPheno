"""Shared generic helpers for the MLlib shims.

Persistence Params, the apply_persistence_params splice, and the
_PersistableModel mixin live here because they apply to any shim
(topic-model or otherwise). Topic-specific helpers like the
SparseVector → BOWDocument converter live under
spark_vi.mllib.topic._common so this module stays free of topic content.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from pyspark.ml.param import Param, Params, TypeConverters


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
    ``resume_path`` is a :class:`Path` or ``None``.

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
    if resume_from != "" and not (Path(resume_from) / "manifest.json").exists():
        raise FileNotFoundError(
            f"No manifest.json at resumeFrom path: {resume_from}"
        )

    if save_dir != "":
        interval = save_interval if save_interval > 0 else base_config.max_iterations + 1
        config = dataclasses.replace(
            base_config,
            checkpoint_dir=Path(save_dir),
            checkpoint_interval=interval,
        )
    else:
        config = base_config

    resume_path = Path(resume_from) if resume_from else None
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
