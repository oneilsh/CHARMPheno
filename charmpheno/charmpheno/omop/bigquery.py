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

from charmpheno.omop.cohorts import SUPPORTED_COHORTS, apply_cohort
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
    cohort: str | None = None,
    prior_obs_days: int | None = None,
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
        cohort: optional cohort filter applied after the base load. None
            (default) keeps the full sampled corpus. See
            ``charmpheno.omop.cohorts.SUPPORTED_COHORTS`` for accepted names
            (e.g. "first_cancer_year").
        prior_obs_days: prior-observation lookback (days) for the cohort's
            index date. None (default) defers to the cohort default (365); 0
            drops the lookback, admitting prevalent cases. Ignored when
            ``cohort`` is None.

    Returns:
        DataFrame with the canonical required OMOP columns
        (person_id, concept_id, concept_name) plus source-table-specific
        date columns. Rows where concept_id == 0 (OMOP "no matching
        concept") are dropped.

    Raises:
        NotImplementedError: if concept_types contains anything other than
            "condition".
        ValueError: if cdr_dataset is malformed, person_sample_mod < 1,
            source_table is unrecognized, or cohort is set to an unknown
            name.
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
    if cohort is not None and cohort not in SUPPORTED_COHORTS:
        raise ValueError(
            f"cohort {cohort!r} not supported "
            f"(supported: {SUPPORTED_COHORTS})"
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

    if cohort is not None:
        # The cohort filter applies AFTER concept-name join so callers see
        # the same canonical schema regardless of cohort. The date column
        # used by the cohort logic differs across source_table modes.
        date_col = (
            "condition_start_date" if source_table == "condition_occurrence"
            else "condition_era_start_date"
        )
        # prior_obs_days=None defers to apply_cohort's default lookback, so the
        # 365-day default lives in exactly one place (cohorts._WINDOW_DAYS).
        lookback_kw = (
            {} if prior_obs_days is None else {"prior_obs_days": prior_obs_days}
        )
        omop = apply_cohort(
            omop, cohort,
            spark=spark, cdr_dataset=cdr_dataset,
            billing_project=billing_project,
            date_col=date_col,
            **lookback_kw,
        )

    validate(omop)
    return omop


def decode_sex(gender_concept_id_col):
    """Map an OMOP gender_concept_id column to a sex string M / F / Unknown.

    Standard OMOP gender concepts: 8507 = Male, 8532 = Female. Every other
    value — Unknown (8551), Other (8521), No matching concept (0), and null —
    maps to 'Unknown', NOT to 'F'. Collapsing unknowns into Female silently
    turns the sex covariate into a constant whenever gender data is absent or
    non-standard, which is a data-integrity bug (observed on exp 0027, where
    sex collapsed to a single 'F' level and dropped out of the design matrix).
    Concept IDs per the OHDSI OMOP CDM Gender vocabulary.
    """
    from pyspark.sql import functions as F

    return (
        F.when(gender_concept_id_col == 8507, "M")
        .when(gender_concept_id_col == 8532, "F")
        .otherwise("Unknown")
    )


def decode_sex_from_name(gender_concept_name_col):
    """Map an OMOP gender *concept name* to a sex string M / F / Unknown.

    Decodes from the concept NAME rather than a hard-coded concept-id list so
    the mapping is vocabulary-agnostic. The standard OMOP Gender concepts are
    8507 'MALE' / 8532 'FEMALE', but datasets routinely carry their own
    encoding: the All of Us Registered Tier `person.gender_concept_id` uses
    45878463 'Female' / 45880669 'Male' plus custom 2000000000+ concepts for
    aggregated gender-identity survey responses — none of which are 8507/8532,
    so an id-based decoder collapses every AoU person to 'Unknown' and silently
    drops C(sex) from the design matrix (exp 0027/0028). Reading the name
    handles all of these through OMOP's own vocabulary.

    Matching is on the lower-cased, trimmed name against the exact tokens
    {female, woman} -> 'F' and {male, man} -> 'M'; every other value maps to
    'Unknown', NOT to a sex. Exact-token (not substring) matching is
    deliberate: AoU's aggregated concept name 'Not man only, not woman only,
    prefer not to answer' contains 'man'/'woman' as substrings, so a substring
    rule would misclassify it — and conflating unknowns with a sex turns the
    covariate into a constant.

    Standard gender concepts per the OHDSI OMOP CDM Gender vocabulary; AoU
    gender concept ids per the All of Us CDR `person` table.
    """
    from pyspark.sql import functions as F

    norm = F.lower(F.trim(gender_concept_name_col))
    return (
        F.when(norm.isin("female", "woman"), "F")
        .when(norm.isin("male", "man"), "M")
        .otherwise("Unknown")
    )


def load_person_table(
    *,
    spark,
    cdr_dataset: str,
    billing_project: str,
    person_sample_mod: int | None = None,
    cohort: str | None = None,
) -> "DataFrame":
    """Load a per-person covariate source table from BigQuery.

    Reads the OMOP `person` table and projects it to the minimal columns
    needed for STM covariate materialization: `person_id`, `age`
    (year-of-birth based, approximate), and `sex` (M/F/Unknown string).

    Callers should pass the resulting DataFrame to
    `charmpheno.omop.covariates.build_patient_covariate_df`, which
    evaluates the formula against this projection.  If the formula
    references columns not present here (e.g. race, ethnicity), the
    BQ query in this function must be extended.

    Args:
        spark: active SparkSession with the spark-bigquery-connector.
        cdr_dataset: fully-qualified BQ dataset "<project>.<dataset>".
        billing_project: GCP project for billing.
        person_sample_mod: if set, keep rows where MOD(person_id, M) == 0.
            Should match the corpus person_sample_mod so the broadcast join
            in the driver covers the same person population.
        cohort: ignored at person-table level — the corpus load already
            restricted the person population; kept for API consistency.
            Pass None unless you want an informational cohort label column
            (which is a literal column, not a filter).

    Returns:
        Spark DataFrame with columns: person_id (long), year_of_birth
        (int), sex_at_birth_concept_id (int), sex_concept_name (string, from
        the concept vocabulary), age (double), sex (string M/F/Unknown decoded
        from the concept name). One row per person_id in the sampled
        population.

    Sex source: reads ``person.sex_at_birth_concept_id`` (standard OMOP Gender
    concepts 8507 'Male' / 8532 'Female'), NOT ``gender_concept_id``. In the
    All of Us CDR the `person` table stores *gender identity* in
    ``gender_concept_id`` (custom concepts 45878463 'Female' / 45880669 'Male'
    / 1585841 'Non-Binary' / 2000000002 'Not man only, not woman only' / ...)
    and *sex assigned at birth* in ``sex_at_birth_concept_id``. Decoding
    ``gender_concept_id`` collapsed every AoU person to a single non-standard
    level and dropped C(sex) from the design matrix (exp 0027/0028); sex at
    birth is the intended prevalence covariate here.
    """
    from pyspark.sql import functions as F

    if not isinstance(cdr_dataset, str) or cdr_dataset.count(".") != 1:
        raise ValueError(
            f"cdr_dataset must be '<project>.<dataset>', got {cdr_dataset!r}"
        )
    if person_sample_mod is not None and person_sample_mod < 1:
        raise ValueError(
            f"person_sample_mod must be >= 1 or None, got {person_sample_mod}"
        )

    def _read(table: str) -> "DataFrame":
        return (
            spark.read.format("bigquery")
            .option("table", f"{cdr_dataset}.{table}")
            .option("parentProject", billing_project)
            .load()
        )

    df = _read("person").select(
        "person_id", "year_of_birth", "sex_at_birth_concept_id"
    )

    if person_sample_mod is not None:
        df = df.where((F.col("person_id") % person_sample_mod) == 0)

    # Resolve the sex concept NAME so decoding is vocabulary-agnostic (the
    # standard OMOP 8507/8532 concepts carry human-readable 'Male'/'Female'
    # names). The `concept` table is large but only a handful of distinct sex
    # concepts participate; no broadcast hint (an explicit F.broadcast on the
    # full concept table OOM'd the driver in client mode — see
    # load_omop_bigquery), let AQE pick the join strategy.
    sex_concept = _read("concept").select(
        F.col("concept_id").alias("sex_at_birth_concept_id"),
        F.col("concept_name").alias("sex_concept_name"),
    )
    df = df.join(sex_concept, on="sex_at_birth_concept_id", how="left")

    # Approximate age from year_of_birth; 2025 is a fixed reference year
    # matching the nominal AoU CDR snapshot used at time of writing.
    df = df.withColumn("age", (F.lit(2025) - F.col("year_of_birth")).cast("double"))
    df = df.withColumn("sex", decode_sex_from_name(F.col("sex_concept_name")))
    return df
