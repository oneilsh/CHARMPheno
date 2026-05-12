"""Canonical OMOP shape used throughout charmpheno.

Every loader in `charmpheno.omop` returns a Spark DataFrame with at least
these required columns:

    person_id:    int (Spark IntegerType or LongType) — deidentified patient identifier
    concept_id:   int (Spark IntegerType or LongType) — OMOP vocabulary concept id
    concept_name: str — human-readable concept label

The following columns are *optional* (present from some loaders, absent from
others) but type-checked when present:

    visit_occurrence_id:        int — clinical encounter id. Absent when the
        loader reads condition_era (which is event-span-based and does not
        carry visit linkage). Required only by doc specs that need it
        (PatientVisitDocSpec, not yet shipped).
    condition_start_date:       date — emitted by condition_occurrence loader.
    condition_era_start_date:   date — emitted by condition_era loader.
    condition_era_end_date:     date — emitted by condition_era loader.

Additional columns (e.g. visit_date) may be present and are passed through
unchanged. See docs/decisions/0003-explicit-omop-io.md for rationale and
docs/decisions/0018-document-unit-abstraction.md for the visit_occurrence_id
relaxation rationale.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import types as T

# Strictly required columns. Every OMOP-shaped DataFrame in CharmPheno has these.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "person_id",
    "concept_id",
    "concept_name",
)

# Columns that some loaders emit and that downstream consumers may require;
# validated for type if present, but absence is not a hard error here.
_OPTIONAL_TYPES: dict[str, tuple[type, ...]] = {
    "visit_occurrence_id":      (T.IntegerType, T.LongType),
    "condition_start_date":     (T.DateType, T.TimestampType),
    "condition_era_start_date": (T.DateType, T.TimestampType),
    "condition_era_end_date":   (T.DateType, T.TimestampType),
}

_REQUIRED_TYPES: dict[str, tuple[type, ...]] = {
    "person_id":           (T.IntegerType, T.LongType),
    "concept_id":          (T.IntegerType, T.LongType),
    "concept_name":        (T.StringType,),
}


def validate(df: DataFrame) -> None:
    """Assert `df` has the canonical OMOP columns with expected types.

    Required columns: present and typed correctly. Optional columns
    (visit_occurrence_id, condition_*_date): if present, typed correctly;
    if absent, that's OK and downstream consumers that need them will fail
    with their own specific error.

    Raises ValueError with a specific message on any mismatch.
    """
    schema = {f.name: type(f.dataType) for f in df.schema.fields}
    missing = [c for c in CANONICAL_COLUMNS if c not in schema]
    if missing:
        raise ValueError(f"OMOP DataFrame is missing required column(s): {missing}")
    for col, expected_types in _REQUIRED_TYPES.items():
        actual = schema[col]
        if not issubclass(actual, expected_types):
            allowed = " or ".join(t.__name__ for t in expected_types)
            raise ValueError(
                f"OMOP column {col!r} has wrong type: "
                f"expected {allowed}, got {actual.__name__}"
            )
    for col, expected_types in _OPTIONAL_TYPES.items():
        if col not in schema:
            continue
        actual = schema[col]
        if not issubclass(actual, expected_types):
            allowed = " or ".join(t.__name__ for t in expected_types)
            raise ValueError(
                f"OMOP optional column {col!r} has wrong type: "
                f"expected {allowed}, got {actual.__name__}"
            )
