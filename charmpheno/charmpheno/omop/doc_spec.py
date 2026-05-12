"""DocSpec: how event-level OMOP rows become topic-model documents.

A DocSpec adds a `doc_id` column to an event-level OMOP DataFrame; each
distinct doc_id becomes one bag-of-words document downstream. The doc_id
shape and the event-to-doc mapping is the entire substance of the
abstraction: same OMOP frame, different DocSpec ⇒ different topic-model
input.

Two specs ship in v1 (ADR 0018):

- PatientDocSpec: doc_id = person_id. One doc per patient over their
  full event history. The pre-ADR default; reproduces existing behavior.
- PatientYearDocSpec: doc_id = "{person_id}:{year}". With era replication
  (default), each condition_era contributes one event-row to every
  calendar year it spans, so chronic conditions populate multiple
  patient-year docs and transient ones populate one.

Each spec also serializes to/from a manifest dict for round-tripping
through VIResult.metadata['corpus_manifest']['doc_spec'], so eval
drivers can reproduce fit-time docs from the checkpoint alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# Registry of name -> spec class, populated by subclasses. Used by
# from_manifest() to dispatch without circular imports.
_REGISTRY: dict[str, type["DocSpec"]] = {}


def _register(cls: type["DocSpec"]) -> type["DocSpec"]:
    _REGISTRY[cls.name] = cls
    return cls


@dataclass(frozen=True)
class DocSpec:
    """Base class: a strategy for deriving documents from event-level OMOP rows.

    Subclasses set `name` (class-level), declare `required_columns`, and
    implement `derive_docs` and `manifest`. The `min_doc_length` filter is
    applied after CountVectorizer in to_bow_dataframe, by length of the
    feature vector — generic across specs.
    """
    name: str = "doc_spec"  # overridden by subclasses
    required_columns: tuple[str, ...] = ()  # overridden by subclasses
    min_doc_length: int = 0

    def derive_docs(self, events_df: DataFrame) -> DataFrame:
        """Return `events_df` with a `doc_id` column added.

        Each event row may be replicated (era-spanning years) or not
        (patient-as-doc). person_id is preserved on every output row so
        downstream patient-keyed splits still work after aggregation.
        """
        raise NotImplementedError

    def manifest(self) -> dict[str, Any]:
        """JSON-serializable summary; round-trips via DocSpec.from_manifest."""
        raise NotImplementedError

    @classmethod
    def from_manifest(cls, d: dict[str, Any]) -> "DocSpec":
        """Reconstruct the right subclass from a manifest dict."""
        name = d.get("name")
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown DocSpec name {name!r} in manifest; "
                f"known names: {sorted(_REGISTRY)}"
            )
        return _REGISTRY[name]._from_manifest(d)

    @classmethod
    def _from_manifest(cls, d: dict[str, Any]) -> "DocSpec":
        """Subclass hook: build an instance from its own manifest dict."""
        raise NotImplementedError

    def validate(self, events_df: DataFrame) -> None:
        """Raise if `events_df` lacks any column this spec requires."""
        cols = set(events_df.columns)
        missing = [c for c in self.required_columns if c not in cols]
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires columns {self.required_columns}, "
                f"but events frame is missing: {missing}. "
                f"Available columns: {sorted(cols)}"
            )


@_register
@dataclass(frozen=True)
class PatientDocSpec(DocSpec):
    """One document per patient over their full event history.

    doc_id = person_id (cast to string for consistency with other specs
    that need compound keys). Reproduces the original CharmPheno pipeline
    default exactly.
    """
    name: str = field(default="patient", init=False)
    required_columns: tuple[str, ...] = field(default=("person_id",), init=False)
    min_doc_length: int = 0

    def derive_docs(self, events_df: DataFrame) -> DataFrame:
        self.validate(events_df)
        return events_df.withColumn("doc_id", F.col("person_id").cast("string"))

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "min_doc_length": self.min_doc_length}

    @classmethod
    def _from_manifest(cls, d: dict[str, Any]) -> "PatientDocSpec":
        return cls(min_doc_length=int(d.get("min_doc_length", 0)))


@_register
@dataclass(frozen=True)
class PatientYearDocSpec(DocSpec):
    """One document per (patient, calendar-year-active).

    With `replicate_eras=True` (default), each condition era contributes
    one event row per year it spans — chronic conditions populate every
    year of their span, transient eras populate one year. End dates beyond
    the current year are clipped (defensive against pathological
    future-dating in source data).

    With `replicate_eras=False`, only the era's start year is emitted.
    Less faithful to "active in year Y" semantics, but works against
    occurrence-style data that lacks an end date — set `date_start_col`
    accordingly.

    Default `min_doc_length=30` per the cutoff curve in
    analysis/cloud/doc_size_evals.ipynb: drops noise-floor patient-years
    (chronic-era padding in low-activity calendar years) before fit.
    """
    name: str = field(default="patient_year", init=False)
    required_columns: tuple[str, ...] = field(default=("person_id",), init=False)
    min_doc_length: int = 30
    replicate_eras: bool = True
    date_start_col: str = "condition_era_start_date"
    date_end_col: str = "condition_era_end_date"

    def derive_docs(self, events_df: DataFrame) -> DataFrame:
        # required_columns is conservative — actual column needs depend on
        # the replicate_eras setting, so we check explicitly here.
        needed = {"person_id", self.date_start_col}
        if self.replicate_eras:
            needed.add(self.date_end_col)
        cols = set(events_df.columns)
        missing = sorted(needed - cols)
        if missing:
            raise ValueError(
                f"PatientYearDocSpec(replicate_eras={self.replicate_eras}, "
                f"date_start_col={self.date_start_col!r}, "
                f"date_end_col={self.date_end_col!r}) requires columns "
                f"{sorted(needed)}, missing: {missing}. "
                f"Available columns: {sorted(cols)}"
            )
        if self.replicate_eras:
            # F.sequence(start_year, end_year) is inclusive on both ends.
            # F.explode turns each event into N rows (one per year_active);
            # F.least clips future-dated end years to the current year so a
            # pathological 9999-12-31 end date doesn't blow up the array.
            current_year = F.year(F.current_date())
            years_active = F.sequence(
                F.year(F.col(self.date_start_col)),
                F.least(F.year(F.col(self.date_end_col)), current_year),
            )
            exploded = events_df.withColumn(
                "year_active", F.explode(years_active),
            )
        else:
            exploded = events_df.withColumn(
                "year_active", F.year(F.col(self.date_start_col)),
            )
        return exploded.withColumn(
            "doc_id",
            F.concat_ws(":",
                F.col("person_id").cast("string"),
                F.col("year_active").cast("string")),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_doc_length": self.min_doc_length,
            "replicate_eras": self.replicate_eras,
            "date_start_col": self.date_start_col,
            "date_end_col": self.date_end_col,
        }

    @classmethod
    def _from_manifest(cls, d: dict[str, Any]) -> "PatientYearDocSpec":
        return cls(
            min_doc_length=int(d.get("min_doc_length", 30)),
            replicate_eras=bool(d.get("replicate_eras", True)),
            date_start_col=str(d.get("date_start_col", "condition_era_start_date")),
            date_end_col=str(d.get("date_end_col", "condition_era_end_date")),
        )


def doc_spec_from_cli(name: str, *, min_doc_length: int | None = None) -> DocSpec:
    """Driver-side factory: build a DocSpec from a CLI --doc-unit value.

    Centralizes the (name -> spec) mapping so cloud and local drivers
    don't drift. min_doc_length, if provided, overrides the spec's default.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"--doc-unit {name!r} not recognized; "
            f"available: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    kwargs: dict[str, Any] = {}
    if min_doc_length is not None:
        kwargs["min_doc_length"] = min_doc_length
    return cls(**kwargs)
