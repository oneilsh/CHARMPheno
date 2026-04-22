"""BigQuery OMOP loader (STUB).

Intended implementation: read OMOP CDM tables (condition_occurrence,
drug_exposure, procedure_occurrence) from a BigQuery dataset, project to
the canonical shape, return a Spark DataFrame via the spark-bigquery
connector (falling back to google-cloud-bigquery + pandas → Spark if
connector absent).

Bootstrap leaves this as an explicit NotImplementedError so scripts that
import it but are run in the wrong environment fail loudly with a clear
message. Real implementation is a dedicated follow-on spec.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession


def load_omop_bigquery(
    *,
    spark: SparkSession,
    cdr_dataset: str,
    concept_types: tuple[str, ...] = ("condition",),
    limit: int | None = None,
) -> DataFrame:
    """Load OMOP-shaped data from a BigQuery CDR dataset.

    Args:
        spark: active SparkSession.
        cdr_dataset: fully-qualified BQ dataset id "project.dataset".
        concept_types: which OMOP fact tables to include (condition, drug,
            procedure, measurement). Defaults to condition only.
        limit: optional row cap for development.

    Returns:
        Spark DataFrame with canonical OMOP columns.
    """
    raise NotImplementedError(
        "load_omop_bigquery is stubbed during bootstrap. See the follow-on "
        "spec in docs/superpowers/specs/ for the real implementation."
    )
