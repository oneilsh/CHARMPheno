"""OMOP-shaped I/O and schema utilities."""
from charmpheno.omop.bigquery import load_omop_bigquery
from charmpheno.omop.doc_spec import (
    DocSpec,
    PatientDocSpec,
    PatientYearDocSpec,
    doc_spec_from_cli,
)
from charmpheno.omop.local import load_omop_parquet
from charmpheno.omop.schema import CANONICAL_COLUMNS, validate
from charmpheno.omop.topic_prep import to_bow_dataframe

__all__ = [
    "CANONICAL_COLUMNS",
    "DocSpec",
    "PatientDocSpec",
    "PatientYearDocSpec",
    "doc_spec_from_cli",
    "load_omop_bigquery",
    "load_omop_parquet",
    "to_bow_dataframe",
    "validate",
]
