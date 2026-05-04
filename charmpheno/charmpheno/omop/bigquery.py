"""BigQuery OMOP loader via the spark-bigquery-connector.

Reads OMOP fact tables from a CDM-shaped BigQuery dataset, broadcast-joins
to the `concept` table for human-readable names, and projects to the
canonical (person_id, visit_occurrence_id, concept_id, concept_name) shape
defined in `charmpheno.omop.schema`. Returns a Spark DataFrame; nothing is
collected to the driver.

v1 supports `concept_types=("condition",)` only. drug_exposure and
procedure_occurrence will land in a follow-on once condition-only behavior
is verified end-to-end.

Connector docs: https://github.com/GoogleCloudDataproc/spark-bigquery-connector
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from charmpheno.omop.schema import CANONICAL_COLUMNS, validate

_SUPPORTED_CONCEPT_TYPES: tuple[str, ...] = ("condition",)


def load_omop_bigquery(
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    concept_types: tuple[str, ...] = ("condition",),
    person_sample_mod: int | None = None,
) -> DataFrame:
    """Load OMOP-shaped data from a BigQuery CDR dataset.

    Args:
        spark: active SparkSession with the spark-bigquery-connector available.
        cdr_dataset: fully-qualified BQ dataset id "<project>.<dataset>".
        billing_project: GCP project that owns the BQ job (read-side billing).
            Distinct from the data project encoded in `cdr_dataset` whenever
            the CDR is hosted in a separate read-only project (the AoU shape).
        concept_types: which OMOP fact tables to include. v1 supports
            ("condition",) only; anything else raises NotImplementedError.
        person_sample_mod: if set, keep rows where MOD(person_id, M) == 0.
            Whole-patient deterministic sampling — preserves each retained
            person's complete condition list, which matters for LDA.

    Returns:
        DataFrame with the canonical OMOP columns plus `condition_start_date`.
        Rows where concept_id == 0 (OMOP "no matching concept") are dropped.

    Raises:
        NotImplementedError: if concept_types contains anything other than
            "condition".
        ValueError: if cdr_dataset is malformed or person_sample_mod < 1.
    """
    if not isinstance(cdr_dataset, str) or cdr_dataset.count(".") != 1:
        raise ValueError(
            f"cdr_dataset must be '<project>.<dataset>', got {cdr_dataset!r}"
        )
    unsupported = tuple(t for t in concept_types if t not in _SUPPORTED_CONCEPT_TYPES)
    if unsupported:
        raise NotImplementedError(
            f"concept_types {unsupported} not supported in v1 "
            f"(supported: {_SUPPORTED_CONCEPT_TYPES})"
        )
    if person_sample_mod is not None and person_sample_mod < 1:
        raise ValueError(
            f"person_sample_mod must be >= 1 or None, got {person_sample_mod}"
        )

    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    cond = _read("condition_occurrence").select(
        "person_id",
        "visit_occurrence_id",
        F.col("condition_concept_id").alias("concept_id"),
        "condition_start_date",
    )
    if person_sample_mod is not None:
        # Pushed down to BQ as a predicate — full-patient sampling is the
        # right shape for LDA (per-person token bags stay intact).
        cond = cond.where((F.col("person_id") % person_sample_mod) == 0)
    cond = cond.where(F.col("concept_id") != 0)

    concept = _read("concept").select("concept_id", "concept_name")

    # Concept is ~7M rows × 2 narrow cols = small; broadcast hint avoids a
    # shuffle-join when one side dwarfs the other.
    omop = cond.join(F.broadcast(concept), on="concept_id", how="left")
    # Reorder to the canonical shape so downstream `validate()` sees a clean
    # schema position-by-position.
    omop = omop.select(*CANONICAL_COLUMNS, "condition_start_date")

    validate(omop)
    return omop
