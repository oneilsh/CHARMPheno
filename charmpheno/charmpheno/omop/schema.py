"""Canonical OMOP shape used throughout charmpheno.

Every loader in `charmpheno.omop` returns a Spark DataFrame with at least
these four columns:

    person_id:           int (Spark IntegerType or LongType) — deidentified patient identifier
    visit_occurrence_id: int (Spark IntegerType or LongType) — identifier for one clinical encounter
    concept_id:          int (Spark IntegerType or LongType) — OMOP vocabulary concept id
    concept_name:        str — human-readable concept label

Additional columns (e.g. visit_date) may be present and are passed through
unchanged. See docs/decisions/0003-explicit-omop-io.md for rationale.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import types as T

CANONICAL_COLUMNS: tuple[str, ...] = (
    "person_id",
    "visit_occurrence_id",
    "concept_id",
    "concept_name",
)

_EXPECTED_TYPES: dict[str, tuple[type, ...]] = {
    "person_id":           (T.IntegerType, T.LongType),
    "visit_occurrence_id": (T.IntegerType, T.LongType),
    "concept_id":          (T.IntegerType, T.LongType),
    "concept_name":        (T.StringType,),
}


def validate(df: DataFrame) -> None:
    """Assert `df` has the canonical OMOP columns with expected types.

    Raises ValueError with a specific message on any mismatch; extras are OK.
    """
    schema = {f.name: type(f.dataType) for f in df.schema.fields}
    missing = [c for c in CANONICAL_COLUMNS if c not in schema]
    if missing:
        raise ValueError(f"OMOP DataFrame is missing required column(s): {missing}")
    for col, expected_types in _EXPECTED_TYPES.items():
        actual = schema[col]
        if not issubclass(actual, expected_types):
            allowed = " or ".join(t.__name__ for t in expected_types)
            raise ValueError(
                f"OMOP column {col!r} has wrong type: "
                f"expected {allowed}, got {actual.__name__}"
            )
