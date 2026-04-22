"""Local-filesystem OMOP loader: read parquet into a Spark DataFrame.

Thin over `spark.read.parquet` but enforces the canonical shape via
`validate()` so a schema mismatch fails at the boundary instead of 40
iterations into a model fit.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from charmpheno.omop.schema import validate


def load_omop_parquet(path: str, *, spark: SparkSession) -> DataFrame:
    """Read an OMOP-shaped parquet file into a Spark DataFrame.

    The file must contain at least the canonical columns:
    person_id, visit_occurrence_id, concept_id, concept_name.
    """
    df = spark.read.parquet(path)
    validate(df)
    return df
