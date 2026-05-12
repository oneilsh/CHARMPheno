"""BigQuery OMOP loader via the spark-bigquery-connector.

Reads OMOP fact tables from a CDM-shaped BigQuery dataset, joins to the
`concept` table for human-readable names, and projects to the canonical
shape defined in `charmpheno.omop.schema`. Returns a Spark DataFrame;
nothing is collected to the driver.

Two source-table modes supported via `source_table`:

- `"condition_occurrence"` (default): emits one row per condition
  occurrence with `condition_start_date` and `visit_occurrence_id`. The
  original CharmPheno loader shape.
- `"condition_era"` (added 2026-05-12, ADR 0018): emits one row per
  OMOP condition era with `condition_era_start_date` and
  `condition_era_end_date`. Eras collapse repeated condition_occurrence
  rows for the same (person, concept) under OMOP's 30-day sliding window,
  so they're the right shape for "active condition span" semantics
  (PatientYearDocSpec with era replication). Eras do not carry
  `visit_occurrence_id`.

v1 supports `concept_types=("condition",)` only. drug_exposure and
procedure_occurrence will land in a follow-on once condition-only behavior
is verified end-to-end.

Connector docs: https://github.com/GoogleCloudDataproc/spark-bigquery-connector
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from charmpheno.omop.schema import validate

_SUPPORTED_CONCEPT_TYPES: tuple[str, ...] = ("condition",)
_SUPPORTED_SOURCE_TABLES: tuple[str, ...] = ("condition_occurrence", "condition_era")


def load_omop_bigquery(
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    concept_types: tuple[str, ...] = ("condition",),
    person_sample_mod: int | None = None,
    source_table: str = "condition_occurrence",
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
        source_table: which condition fact table to read. "condition_occurrence"
            emits one row per occurrence with `condition_start_date` +
            `visit_occurrence_id`; "condition_era" emits one row per condition
            era with `condition_era_start_date` + `condition_era_end_date`
            and no visit_occurrence_id (eras span visits).

    Returns:
        DataFrame with the canonical required OMOP columns
        (person_id, concept_id, concept_name) plus source-table-specific
        date columns. Rows where concept_id == 0 (OMOP "no matching
        concept") are dropped.

    Raises:
        NotImplementedError: if concept_types contains anything other than
            "condition".
        ValueError: if cdr_dataset is malformed, person_sample_mod < 1, or
            source_table is unrecognized.
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
    if source_table not in _SUPPORTED_SOURCE_TABLES:
        raise ValueError(
            f"source_table {source_table!r} not supported "
            f"(supported: {_SUPPORTED_SOURCE_TABLES})"
        )

    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    if source_table == "condition_occurrence":
        cond = _read("condition_occurrence").select(
            "person_id",
            "visit_occurrence_id",
            F.col("condition_concept_id").alias("concept_id"),
            "condition_start_date",
        )
        extra_cols = ("visit_occurrence_id", "condition_start_date")
    else:  # condition_era
        cond = _read("condition_era").select(
            "person_id",
            F.col("condition_concept_id").alias("concept_id"),
            "condition_era_start_date",
            "condition_era_end_date",
        )
        extra_cols = ("condition_era_start_date", "condition_era_end_date")

    if person_sample_mod is not None:
        # Full-patient sampling is the right shape for LDA — per-person token
        # bags stay intact rather than getting truncated by row-level sampling.
        # Whether MOD pushes down to BQ depends on the connector version.
        cond = cond.where((F.col("person_id") % person_sample_mod) == 0)
    cond = cond.where(F.col("concept_id") != 0)

    concept = _read("concept").select("concept_id", "concept_name")

    # No broadcast hint: full OMOP `concept` (~8M rows, name strings) exceeds
    # autoBroadcastJoinThreshold, so AQE will pick shuffle-hash or sort-merge
    # at runtime. An explicit F.broadcast() here OOM'd the driver in client
    # mode — keep it implicit and let the planner choose.
    omop = cond.join(concept, on="concept_id", how="left")
    # Reorder so canonical required columns come first, then source-specific.
    omop = omop.select("person_id", "concept_id", "concept_name", *extra_cols)

    validate(omop)
    return omop
