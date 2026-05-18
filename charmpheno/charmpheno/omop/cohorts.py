"""Cohort definitions for OMOP-shaped event data.

A "cohort" here is a function that takes an OMOP events DataFrame
(person_id, concept_id, date columns) and returns a subset filtered to
a specific clinical population over a specific observation window.
Cohorts are orthogonal to DocSpecs: a cohort selects WHICH patients
and WHICH dates make it through; a DocSpec then collapses surviving
events into documents.

Currently implemented:

- ``first_cancer_year``: patients with a first malignant-cancer diagnosis
  (excluding non-melanoma skin cancer and carcinoma in situ), windowed
  to the 365 days starting at that first dx. Requires >= 365 days of
  observation_period coverage both before (to make "first" meaningful)
  and after (to make the doc window fully observed) the index date.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# Top-level SNOMED concept whose descendants define the inclusion set for
# malignant cancers. concept_ancestor(443392) returns every malignant-
# cancer condition concept in OMOP.
_CANCER_ANCESTOR = 443392

# Ancestor concepts whose descendants are EXCLUDED from the "first cancer"
# definition.
#   - NMSC (BCC/SCC) is excluded because it's enormously common,
#     clinically minor, and would otherwise dominate the cohort.
#   - Carcinoma in situ is excluded because it's pre-invasive and follows
#     a different disease trajectory than invasive cancer.
_CANCER_EXCLUSION_ANCESTORS: tuple[int, ...] = (
    4115276,  # Squamous cell carcinoma of skin
    4112744,  # Basal cell carcinoma of skin
    4180978,  # Carcinoma in situ
)

# Window after the first cancer dx that defines a patient's "document".
# Matches the existing one-year doc convention used elsewhere in the
# project for patient_year DocSpec defaults.
_WINDOW_DAYS = 365


# Names accepted by the CLI/loader. Add a new key here when adding a new
# cohort function so the registry stays the single source of truth.
SUPPORTED_COHORTS: tuple[str, ...] = ("first_cancer_year",)


def apply_cohort(
    cond_df: DataFrame,
    cohort: str,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
) -> DataFrame:
    """Dispatch on cohort name. Raises ValueError on unknown names.

    Kept as a thin registry rather than inlined in the loader so adding
    a new cohort means adding a function below + a SUPPORTED_COHORTS
    entry, without touching the loader call site.
    """
    if cohort == "first_cancer_year":
        return apply_first_cancer_year_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
        )
    raise ValueError(
        f"cohort {cohort!r} not supported (supported: {SUPPORTED_COHORTS})"
    )


def apply_first_cancer_year_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
) -> DataFrame:
    """Filter to patients with a first cancer dx + 1-year follow-up window.

    Args:
        cond_df: events DataFrame from load_omop_bigquery (must have
            ``person_id``, ``concept_id``, and ``date_col``).
        spark, cdr_dataset, billing_project: same shape as
            load_omop_bigquery — needed to read concept_ancestor +
            observation_period from the same CDR.
        date_col: name of the calendar-date column on ``cond_df`` used
            both to find the first cancer dx and to bound the doc window.
            ``condition_start_date`` for condition_occurrence,
            ``condition_era_start_date`` for condition_era.

    Returns:
        A DataFrame with the same schema as ``cond_df``, filtered to
        rows where the person had a qualifying first cancer dx and the
        row's date lies in [index_date, index_date + 365d).
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    # Build the cancer concept set as (descendants of 443392) - (descendants
    # of exclusion ancestors). Predicates on ancestor_concept_id push down
    # to BQ, so this only materializes ~thousands of concept ids, not the
    # full concept_ancestor table.
    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    included = (
        ca.where(F.col("ancestor_concept_id") == _CANCER_ANCESTOR)
          .select(F.col("descendant_concept_id").alias("concept_id"))
    )
    excluded = (
        ca.where(F.col("ancestor_concept_id").isin(
            list(_CANCER_EXCLUSION_ANCESTORS),
        ))
        .select(F.col("descendant_concept_id").alias("concept_id"))
    )
    cancer_concepts = included.subtract(excluded).distinct()

    # First cancer dx date per person. Broadcasting cancer_concepts is
    # safe — it's a few thousand integers.
    first_dx = (
        cond_df.join(F.broadcast(cancer_concepts), on="concept_id", how="inner")
               .groupBy("person_id")
               .agg(F.min(date_col).alias("index_date"))
    )

    # Observation-period filter: 365d both sides of index. "Before" makes
    # "first" mean "first in record with adequate lookback"; "after" makes
    # the doc window fully observed (so absence of a code in the window
    # is informative rather than "we just couldn't see them").
    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )
    cohort_df = (
        first_dx.join(op, on="person_id", how="inner")
                .where(F.col("index_date") >= F.date_add(
                    F.col("observation_period_start_date"), _WINDOW_DAYS,
                ))
                .where(F.date_add(F.col("index_date"), _WINDOW_DAYS)
                       <= F.col("observation_period_end_date"))
                .select("person_id", "index_date")
    )

    # Filter the events: cohort members only, in the doc window. Not
    # broadcasting cohort_df: at AoU scale a cancer cohort can run into
    # the hundreds of thousands of persons and the planner is in a
    # better position than we are to pick the join strategy.
    return (
        cond_df.join(cohort_df, on="person_id", how="inner")
               .where(F.col(date_col) >= F.col("index_date"))
               .where(F.col(date_col) < F.date_add(
                   F.col("index_date"), _WINDOW_DAYS,
               ))
               .drop("index_date")
    )
