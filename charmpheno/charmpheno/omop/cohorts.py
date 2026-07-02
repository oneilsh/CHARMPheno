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
  post-index observation_period coverage (so the doc window is fully
  observed) and, by default, >= 365 days of prior coverage (so "first" is
  meaningful); the prior lookback is configurable via ``prior_obs_days``
  (0 drops it, admitting prevalent cases). See ``_window_observed_cohort``.
- ``first_dementia_year``: patients with a first all-cause dementia
  diagnosis (Alzheimer's, vascular, Lewy body, FTD, dementia NOS — i.e.
  descendants of SNOMED "Dementia"), windowed to the 365 days starting
  at that first dx. Same observation-period bracketing as the cancer
  cohort.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
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


# Top-level SNOMED concept whose descendants define the inclusion set for
# all-cause dementia. concept_ancestor(4182210) should return AD,
# vascular dementia, DLB, FTD, dementia NOS, mixed dementia, etc.
#
# VERIFY ON FIRST RUN: a quick sanity check is to count descendants:
#   SELECT COUNT(*) FROM concept_ancestor
#   WHERE ancestor_concept_id = 4182210;
# Expect dozens-to-hundreds of descendants. If you see 0, swap for the
# correct OMOP concept_id for the SNOMED "Dementia" hierarchy in your
# vocab version.
#
# Choice rationale: we deliberately go broad (all-cause) rather than
# AD-only because EHR coding is notoriously mushy between AD vs
# "dementia NOS" vs vascular — a pure-AD cohort would silently exclude
# real-AD patients whose providers happened to code them differently.
# The post-onset phenotype cascade is also similar across dementia
# subtypes (delirium, falls, polypharmacy, behavioral disturbance,
# aspiration pneumonia, end-of-life care), so the breadth helps without
# diluting the signal.
_DEMENTIA_ANCESTOR = 4182210

# No exclusions for v1: capturing the full dementia-syndrome trajectory
# is the goal. Kept as a constant (not inlined) so adding exclusions
# later is a one-liner.
_DEMENTIA_EXCLUSION_ANCESTORS: tuple[int, ...] = ()


# Names accepted by the CLI/loader. Add a new key here when adding a new
# cohort function so the registry stays the single source of truth.
SUPPORTED_COHORTS: tuple[str, ...] = (
    "first_cancer_year",
    "first_dementia_year",
    "cancer_or_dementia",
    "population_cancer",
)

# Fixed salt for the general-population random-window assignment. Hashing
# person_id with a constant salt makes each person's sampled 1-year window
# deterministic and reproducible across runs (Spark's F.rand() is not
# resume-stable), while still spreading windows pseudo-uniformly across each
# person's observation history.
_RANDOM_WINDOW_SALT = 20260702


# User-facing metadata for each cohort. Consumed by the dashboard bundle
# builder (write into corpus_stats.json) so the UI's cohort selector has a
# label + description without having to duplicate this text in the
# frontend. Keep `label` short (fits in a dropdown); `description` is a
# one-paragraph blurb shown when the cohort is selected.
#
# The "full" entry is the unfiltered general-population corpus (i.e. the
# loader was called with cohort=None) and lets us treat the no-cohort
# case identically to the filtered ones for selector + metadata purposes.
COHORT_METADATA: dict[str, dict[str, str]] = {
    "full": {
        "id": "full",
        "label": "General Population (1 year windows)",
        "description": (
            "Unfiltered 1-year windows on 10% of AllOfUs condition data, "
            "no clinical inclusion or window constraint applied."
        ),
    },
    "first_cancer_year": {
        "id": "first_cancer_year",
        "label": "Cancer (1 year windows post-diagnosis)",
        "description": (
            "Patients with a first malignant-cancer diagnosis (SNOMED "
            "443392 and descendants), excluding non-melanoma skin cancer "
            "(BCC/SCC) and carcinoma in situ. The document window is the "
            "365 days starting at that first diagnosis. The follow-up window "
            "must be fully observed (365 days of post-index "
            "observation_period coverage); by default 'first' also requires "
            "365 days of prior coverage, relaxable via prior_obs_days."
        ),
    },
    "first_dementia_year": {
        "id": "first_dementia_year",
        "label": "Dementia (1 year windows post-diagnosis)",
        "description": (
            "Patients with a first all-cause dementia diagnosis (SNOMED "
            "4182210 and descendants — Alzheimer's, vascular dementia, "
            "Lewy body dementia, frontotemporal dementia, dementia NOS, "
            "mixed dementia). The document window is the 365 days "
            "starting at that first diagnosis, capturing the early-stage "
            "comorbidity cascade (delirium, falls, polypharmacy, "
            "behavioral disturbance, aspiration pneumonia). The follow-up "
            "window must be fully observed (365 days of post-index "
            "observation_period coverage); by default 'first' also requires "
            "365 days of prior coverage, relaxable via prior_obs_days."
        ),
    },
    "cancer_or_dementia": {
        "id": "cancer_or_dementia",
        "label": "Cancer or Dementia (combined, source-labeled)",
        "description": (
            "Union of the first-cancer-year and first-dementia-year cohorts, "
            "each document labeled by its source cohort. A patient qualifying "
            "for both contributes two documents (one per cohort). Used as an "
            "STM validation: a source_cohort covariate should produce strongly "
            "separable cancer vs dementia topic structure."
        ),
    },
    "population_cancer": {
        "id": "population_cancer",
        "label": "Population + Cancer (gated)",
        "description": (
            "The whole (sampled) population as a shared background, with a "
            "cancer subcohort carrying its own foreground topics. Disjoint, one "
            "document per person: patients with a first malignant-cancer "
            "diagnosis (SNOMED 443392 and descendants, excluding non-melanoma "
            "skin cancer and carcinoma in situ) get the 365-day post-diagnosis "
            "window and source_cohort='cancer'; every other person gets a "
            "deterministic random fully-observed 365-day window and "
            "source_cohort='general' (background-only, since 'general' has no "
            "foreground block). Trains general-population background topics "
            "against cancer-specific foreground under one gated STM."
        ),
    },
}


def cohort_metadata(cohort: str | None) -> dict[str, str]:
    """Return the user-facing metadata dict for a cohort name.

    ``cohort=None`` is treated as the ``"full"`` (unfiltered) cohort. An
    unknown name raises KeyError — callers should validate against
    ``SUPPORTED_COHORTS`` before calling this.
    """
    key = cohort if cohort is not None else "full"
    return COHORT_METADATA[key]


def apply_cohort(
    cond_df: DataFrame,
    cohort: str,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Dispatch on cohort name. Raises ValueError on unknown names.

    Kept as a thin registry rather than inlined in the loader so adding
    a new cohort means adding a function below + a SUPPORTED_COHORTS
    entry, without touching the loader call site.

    prior_obs_days is the per-cohort prior-observation lookback (default
    ``_WINDOW_DAYS`` = 365); see :func:`_window_observed_cohort`.
    """
    if cohort == "first_cancer_year":
        return apply_first_cancer_year_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "first_dementia_year":
        return apply_first_dementia_year_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "cancer_or_dementia":
        return apply_cancer_or_dementia_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    if cohort == "population_cancer":
        return apply_population_cancer_cohort(
            cond_df, spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project, date_col=date_col,
            prior_obs_days=prior_obs_days,
        )
    raise ValueError(
        f"cohort {cohort!r} not supported (supported: {SUPPORTED_COHORTS})"
    )


def _window_observed_cohort(
    first_dx: DataFrame,
    observation_period: DataFrame,
    *,
    prior_obs_days: int,
    window_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Keep the (person_id, index_date) rows that are adequately observed.

    Two observation-period gates, joined against ``observation_period``:

    - **Prior lookback**: ``index_date >= observation_period_start_date +
      prior_obs_days``. At the default 365 this makes "first dx" mean "first
      with a year of prior coverage", excluding prevalent cases whose true
      first dx predates the record. ``prior_obs_days=0`` drops the lookback
      (admitting those prevalent cases); the gate then only requires the
      index to fall within an observation period at all.
    - **Follow-up**: ``index_date + window_days <=
      observation_period_end_date``, so the document window is fully observed
      (absence of a code in the window is informative, not merely unobserved).
      Independent of ``prior_obs_days``.

    Returns ``(person_id, index_date)`` for the surviving rows.
    """
    return (
        first_dx.join(observation_period, on="person_id", how="inner")
        .where(F.col("index_date") >= F.date_add(
            F.col("observation_period_start_date"), prior_obs_days))
        .where(F.date_add(F.col("index_date"), window_days)
               <= F.col("observation_period_end_date"))
        .select("person_id", "index_date")
    )


def apply_first_cancer_year_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
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

    # Observation-period gating (prior lookback + fully-observed follow-up);
    # see _window_observed_cohort. prior_obs_days controls the lookback.
    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )
    cohort_df = _window_observed_cohort(
        first_dx, op, prior_obs_days=prior_obs_days,
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


def apply_first_dementia_year_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Filter to patients with a first dementia dx + 1-year follow-up.

    Mirrors :func:`apply_first_cancer_year_cohort` but anchored on the
    SNOMED "Dementia" hierarchy with no ancestor exclusions — all-cause
    dementia is intentional (see module-level comment on _DEMENTIA_ANCESTOR).

    Args:
        cond_df: events DataFrame from load_omop_bigquery (must have
            ``person_id``, ``concept_id``, and ``date_col``).
        spark, cdr_dataset, billing_project: same shape as
            load_omop_bigquery — needed to read concept_ancestor +
            observation_period from the same CDR.
        date_col: name of the calendar-date column on ``cond_df`` used
            both to find the first dementia event and to bound the doc
            window. ``condition_start_date`` for condition_occurrence,
            ``condition_era_start_date`` for condition_era.

    Returns:
        A DataFrame with the same schema as ``cond_df``, filtered to
        rows where the person had a qualifying first dementia dx and
        the row's date lies in [index_date, index_date + 365d).
    """
    def _read(table: str) -> DataFrame:
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id",
    )
    dementia_concepts = (
        ca.where(F.col("ancestor_concept_id") == _DEMENTIA_ANCESTOR)
          .select(F.col("descendant_concept_id").alias("concept_id"))
          .distinct()
    )
    # Exclusion subtract is a no-op for v1 (empty tuple) but kept here
    # symmetric with the cancer cohort so adding exclusions later is a
    # one-line change.
    if _DEMENTIA_EXCLUSION_ANCESTORS:
        excluded = (
            ca.where(F.col("ancestor_concept_id").isin(
                list(_DEMENTIA_EXCLUSION_ANCESTORS),
            ))
            .select(F.col("descendant_concept_id").alias("concept_id"))
        )
        dementia_concepts = dementia_concepts.subtract(excluded).distinct()

    first_event = (
        cond_df.join(F.broadcast(dementia_concepts), on="concept_id", how="inner")
               .groupBy("person_id")
               .agg(F.min(date_col).alias("index_date"))
    )

    op = _read("observation_period").select(
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    )
    cohort_df = _window_observed_cohort(
        first_event, op, prior_obs_days=prior_obs_days,
    )

    return (
        cond_df.join(cohort_df, on="person_id", how="inner")
               .where(F.col(date_col) >= F.col("index_date"))
               .where(F.col(date_col) < F.date_add(
                   F.col("index_date"), _WINDOW_DAYS,
               ))
               .drop("index_date")
    )


def _combine_cohorts(
    cancer_events: DataFrame, dementia_events: DataFrame,
) -> DataFrame:
    """Tag each cohort's events with source_cohort and union (no dedup).

    A comorbid patient's cancer-window events (tagged "cancer") and
    dementia-window events (tagged "dementia") both survive, so they become
    two distinct documents downstream via PatientCohortDocSpec.
    """
    c = cancer_events.withColumn("source_cohort", F.lit("cancer"))
    d = dementia_events.withColumn("source_cohort", F.lit("dementia"))
    return c.unionByName(d)


def apply_cancer_or_dementia_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Combined cancer-or-dementia cohort with a source_cohort label column.

    Composes the two single-disease cohorts and unions their tagged events.
    Returns cond_df's schema plus a `source_cohort` string column. Both arms
    share the same ``prior_obs_days`` lookback.
    """
    cancer = apply_first_cancer_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    dementia = apply_first_dementia_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    return _combine_cohorts(cancer, dementia)


def _random_observed_year_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    window_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Window each person to ONE deterministic random fully-observed year.

    Unlike the disease cohorts, the general population has no index event to
    anchor on. For each person we pick their longest observation_period of at
    least ``window_days`` and sample a window start uniformly in
    ``[period_start, period_end - window_days]`` — so the whole 1-year window is
    inside a single observation period (absence of a code is informative, not
    merely unobserved). The offset is ``hash(person_id) mod (span - window + 1)``
    with a fixed salt, so the assignment is deterministic and resume-stable
    (Spark's ``F.rand()`` is not).

    Returns ``cond_df``'s schema, filtered to each person's sampled window.
    Persons with no observation period spanning ``window_days`` are dropped.
    """
    op = (
        spark.read.format("bigquery")
        .option("table", f"{cdr_dataset}.observation_period")
        .option("parentProject", billing_project)
        .load()
        .select(
            "person_id",
            "observation_period_start_date",
            "observation_period_end_date",
        )
    )
    windows = _random_observed_windows(op, window_days=window_days)

    return (
        cond_df.join(windows, on="person_id", how="inner")
        .where(F.col(date_col) >= F.col("index_date"))
        .where(F.col(date_col) < F.date_add(F.col("index_date"), window_days))
        .drop("index_date")
    )


def _random_observed_windows(
    observation_period: DataFrame, *, window_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Pick one deterministic random fully-observed window start per person.

    Takes an ``observation_period`` frame (``person_id``,
    ``observation_period_start_date``, ``observation_period_end_date``) and
    returns ``(person_id, index_date)`` where the window
    ``[index_date, index_date + window_days)`` lies entirely within the person's
    longest observation period. The offset is
    ``hash(person_id, salt) mod (span - window_days + 1)`` so it is deterministic
    and reproducible (Spark's ``F.rand()`` is not resume-stable). Persons with no
    observation period spanning ``window_days`` are dropped.
    """
    op = observation_period.withColumn(
        "span",
        F.datediff(
            F.col("observation_period_end_date"),
            F.col("observation_period_start_date"),
        ),
    ).where(F.col("span") >= window_days)

    # One observation period per person (the longest; ties broken by earliest
    # start) so each person yields exactly one window.
    longest = op.withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy("person_id").orderBy(
                F.col("span").desc(),
                F.col("observation_period_start_date").asc(),
            )
        ),
    ).where(F.col("rn") == 1)

    # Deterministic pseudo-random offset in [0, span - window_days]; index_date
    # = period_start + offset days. span - window_days + 1 is >= 1 by the span
    # filter above, so the modulo is well-defined.
    offset = F.abs(F.hash(F.col("person_id"), F.lit(_RANDOM_WINDOW_SALT))) % (
        F.col("span") - F.lit(window_days) + F.lit(1)
    )
    return longest.withColumn(
        "index_date",
        F.date_add(F.col("observation_period_start_date"), offset),
    ).select("person_id", "index_date")


def apply_population_cancer_cohort(
    cond_df: DataFrame,
    *,
    spark: SparkSession,
    cdr_dataset: str,
    billing_project: str,
    date_col: str,
    prior_obs_days: int = _WINDOW_DAYS,
) -> DataFrame:
    """Whole-population background + a cancer foreground subcohort, disjoint.

    One document per person, tagged with a ``source_cohort`` column:

    - **cancer** — patients with a qualifying first cancer dx (the existing
      :func:`apply_first_cancer_year_cohort`), windowed to the 365 days after
      that diagnosis. These carry the cancer foreground topics.
    - **general** — every OTHER person (no qualifying cancer dx), windowed to a
      deterministic random fully-observed 365-day span
      (:func:`_random_observed_year_cohort`). ``source_cohort='general'`` is not
      a foreground group, so these documents resolve to background-only via
      :meth:`TopicBlockPartition.allowed_indices`.

    The arms are disjoint by person (the general arm is the ``left_anti`` of the
    cancer arm's persons), so no patient contributes two documents. Returns
    ``cond_df``'s schema plus a ``source_cohort`` string column.
    """
    cancer = apply_first_cancer_year_cohort(
        cond_df, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
        prior_obs_days=prior_obs_days,
    )
    cancer_persons = cancer.select("person_id").distinct()

    # General arm = everyone not in the cancer arm, on a random observed year.
    non_cancer = cond_df.join(cancer_persons, on="person_id", how="left_anti")
    general = _random_observed_year_cohort(
        non_cancer, spark=spark, cdr_dataset=cdr_dataset,
        billing_project=billing_project, date_col=date_col,
    )

    return (
        cancer.withColumn("source_cohort", F.lit("cancer"))
        .unionByName(general.withColumn("source_cohort", F.lit("general")))
    )
