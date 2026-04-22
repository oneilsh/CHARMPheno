"""OMOP-shaped I/O and schema utilities."""
from charmpheno.omop.bigquery import load_omop_bigquery
from charmpheno.omop.local import load_omop_parquet
from charmpheno.omop.schema import CANONICAL_COLUMNS, validate

__all__ = [
    "CANONICAL_COLUMNS",
    "load_omop_bigquery",
    "load_omop_parquet",
    "validate",
]
