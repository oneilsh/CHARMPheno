"""BigQuery OMOP loader via the spark-bigquery-connector.

Reads OMOP fact tables from a CDM-shaped BigQuery dataset, joins to the
`concept` table for human-readable names, and projects to the canonical
shape defined in `charmpheno.omop.schema`. Returns a Spark DataFrame;
nothing is collected to the driver.

Two condition source-table modes supported via `source_table`:

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

Multi-domain fusion (`concept_types`)
-------------------------------------
`concept_types` selects which OMOP domains contribute concept "words":
`"condition"`, `"drug"`, and `"procedure"`. Requesting more than one
domain reads each domain's fact table, projects every one to a *single
common schema* — `person_id`, `concept_id`, `concept_name`, and one
shared `event_date` column — and UNIONs them into one flat DataFrame. The
result is a single fused concept stream, so a downstream
``to_bow_dataframe`` builds ONE flat vocabulary spanning all requested
domains (a Hughes-style bag of OMOP concepts). This is the deliberately
minimal fusion: no per-domain vocab, no domain-reliability weighting — all
domains' concept_ids live in one token stream.

Domain -> fact table / concept column / event date:

- `"condition"`: `condition_occurrence` (`condition_concept_id`,
  `condition_start_date`) or `condition_era` (`condition_concept_id`,
  `condition_era_start_date`), per `source_table`.
- `"drug"`: `drug_era` (`drug_concept_id`, `drug_era_start_date`). The
  era table (not `drug_exposure`) is chosen to match the sibling
  hybrid-domain branch, which normalizes drugs to the same span-shaped
  event as `condition_era`; the empirical drug vocabulary is whatever
  concept classes the CDR's drug_era populates (no ingredient rollup).
- `"procedure"`: `procedure_occurrence` (`procedure_concept_id`,
  `procedure_date`).

Provenance note: OMOP concept_ids are globally unique across domains, so
fusion is a plain union — no id can collide between a condition, a drug,
and a procedure. A concept's originating domain is therefore always
recoverable after the fact from ``concept.domain_id`` (the same `concept`
table this loader already joins for `concept_name`); it is intentionally
NOT carried as a column here, because the fused vocabulary is flat by
design.

The single-domain default ``concept_types=("condition",)`` keeps the exact
legacy output (domain-specific date columns, `visit_occurrence_id`, cohort
support) unchanged; only the multi-domain path emits the common
`event_date` schema.

Connector docs: https://github.com/GoogleCloudDataproc/spark-bigquery-connector
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from charmpheno.omop.cohorts import SUPPORTED_COHORTS, apply_cohort
from charmpheno.omop.schema import validate

_SUPPORTED_CONCEPT_TYPES: tuple[str, ...] = ("condition", "drug", "procedure")
_SUPPORTED_SOURCE_TABLES: tuple[str, ...] = ("condition_occurrence", "condition_era")

# The single event-date column the fused multi-domain stream emits. Each
# domain's native start/event date is aliased to this shared name so the
# per-domain frames share one schema and UNION cleanly. Downstream doc specs
# that key on a date (e.g. PatientYearDocSpec) should set
# ``date_start_col="event_date"`` when consuming a fused corpus.
_FUSED_EVENT_DATE: str = "event_date"


def _domain_source_spec(domain: str, source_table: str) -> tuple[str, str, str]:
    """Map a fused-domain name to its (fact_table, concept_id_col, date_col).

    Returns the BigQuery fact table to read, the domain-specific
    ``*_concept_id`` column to alias to the common ``concept_id``, and the
    domain-specific start/event date column to alias to the common
    ``event_date`` (``_FUSED_EVENT_DATE``). ``source_table`` only selects the
    condition variant (occurrence vs era); drug and procedure have a single
    fact table each. Drug uses ``drug_era`` (span-shaped, matching the sibling
    hybrid-domain branch), procedure uses ``procedure_occurrence``. Lifts the
    condition/drug ``F.col(<domain>_concept_id).alias("concept_id")`` shape
    from that branch so the two converge cleanly on a future merge.
    """
    if domain == "condition":
        if source_table == "condition_era":
            return "condition_era", "condition_concept_id", "condition_era_start_date"
        return "condition_occurrence", "condition_concept_id", "condition_start_date"
    if domain == "drug":
        return "drug_era", "drug_concept_id", "drug_era_start_date"
    if domain == "procedure":
        return "procedure_occurrence", "procedure_concept_id", "procedure_date"
    # Unreachable: callers validate against _SUPPORTED_CONCEPT_TYPES first.
    raise NotImplementedError(f"no fused source spec for domain {domain!r}")


def _read_domain_events(read, domain: str, source_table: str) -> DataFrame:
    """Read one domain's fact table, projected to the common fused schema.

    Emits exactly (``person_id``, ``concept_id``, ``event_date``) — the
    domain's ``*_concept_id`` aliased to ``concept_id`` and its start/event
    date aliased to ``event_date`` — so every domain's frame is union-
    compatible before the shared ``concept`` name join. ``read`` is the
    table-reader seam (``_read``) so tests can inject synthetic per-table
    DataFrames without a BigQuery round-trip.
    """
    table, cid_col, date_col = _domain_source_spec(domain, source_table)
    return read(table).select(
        "person_id",
        F.col(cid_col).alias("concept_id"),
        F.col(date_col).alias(_FUSED_EVENT_DATE),
    )


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
        concept_types: which OMOP domains to fuse into the concept stream.
            Supports "condition", "drug", "procedure". A single ("condition",)
            (the default) returns the exact legacy condition output. Two or
            more domains read each domain's fact table, project to a common
            schema (person_id, concept_id, concept_name, event_date) and UNION
            into one flat stream feeding a single fused vocabulary. Anything
            outside the supported set raises NotImplementedError.
        person_sample_mod: if set, keep rows where MOD(person_id, M) == 0.
            Whole-patient deterministic sampling — preserves each retained
            person's complete concept list, which matters for LDA.
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
        (person_id, concept_id, concept_name). The single-domain
        ("condition",) path also carries the source-table-specific date
        columns (and visit_occurrence_id for condition_occurrence); the fused
        multi-domain path instead carries one common ``event_date`` column.
        Rows where concept_id == 0 (OMOP "no matching concept") are dropped.

    Raises:
        NotImplementedError: if concept_types contains a domain outside
            ("condition", "drug", "procedure").
        ValueError: if cdr_dataset is malformed, concept_types is empty,
            person_sample_mod < 1, source_table is unrecognized, cohort is set
            to an unknown name, or cohort is combined with a fused
            (multi-domain) load.
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
    # De-dupe while preserving caller order so a repeated domain can't
    # double-count its concepts into the fused bag.
    concept_types = tuple(dict.fromkeys(concept_types))
    if not concept_types:
        raise ValueError("concept_types must name at least one OMOP domain")
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

    # Single-domain condition load: preserve the exact legacy output (domain-
    # specific date columns, visit_occurrence_id, and cohort windowing)
    # byte-for-byte. Only the fused multi-domain path below reshapes to the
    # common `event_date` schema, so the default corpus is untouched.
    if concept_types == ("condition",):
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
            # Full-patient sampling is the right shape for LDA — per-person
            # token bags stay intact rather than getting truncated by row-level
            # sampling. Whether MOD pushes down to BQ depends on the connector.
            cond = cond.where((F.col("person_id") % person_sample_mod) == 0)
        cond = cond.where(F.col("concept_id") != 0)

        concept = _read("concept").select("concept_id", "concept_name")

        # No broadcast hint: full OMOP `concept` (~8M rows, name strings)
        # exceeds autoBroadcastJoinThreshold, so AQE will pick shuffle-hash or
        # sort-merge at runtime. An explicit F.broadcast() here OOM'd the driver
        # in client mode — keep it implicit and let the planner choose.
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
            # prior_obs_days=None defers to apply_cohort's default lookback, so
            # the 365-day default lives in one place (cohorts._WINDOW_DAYS).
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

    # Fused multi-domain path: union each requested domain's events into ONE
    # flat concept stream feeding a single downstream vocabulary. Cohort
    # windowing is not offered here — the cohort index date is condition-derived
    # and would be ambiguous once drug/procedure concepts share the stream.
    if cohort is not None:
        raise ValueError(
            "cohort filtering is only supported for a single-domain condition "
            f"load; got concept_types={concept_types!r} with cohort={cohort!r}. "
            "Load condition-only with cohort=, or fuse without a cohort and "
            "window downstream."
        )

    events = None
    for domain in concept_types:
        frame = _read_domain_events(_read, domain, source_table)
        events = frame if events is None else events.unionByName(frame)

    if person_sample_mod is not None:
        # Same whole-patient sampling semantics as the condition path, applied
        # once to the already-unioned stream.
        events = events.where((F.col("person_id") % person_sample_mod) == 0)
    events = events.where(F.col("concept_id") != 0)

    concept = _read("concept").select("concept_id", "concept_name")
    # Implicit join strategy — see the condition path's note on why no
    # F.broadcast() hint is used against the full `concept` table.
    omop = events.join(concept, on="concept_id", how="left")
    # One flat fused schema: canonical columns + the single common event date.
    omop = omop.select(
        "person_id", "concept_id", "concept_name", _FUSED_EVENT_DATE,
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


def filter_known_sex(person_df: "DataFrame") -> "DataFrame":
    """Keep only rows with a decoded binary sex ('M' or 'F').

    Drops persons whose sex decoded to 'Unknown' (OMOP Unknown/Other, null, or
    a non-standard/aggregated concept) so the analysis population is restricted
    to those with a known Male/Female sex-at-birth. Operates on the decoded
    `sex` column produced by decode_sex_from_name.
    """
    from pyspark.sql import functions as F

    return person_df.where(F.col("sex").isin("M", "F"))


def load_person_table(
    *,
    spark,
    cdr_dataset: str,
    billing_project: str,
    person_sample_mod: int | None = None,
    cohort: str | None = None,
    known_sex_only: bool = False,
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
        known_sex_only: when True, keep only persons whose decoded sex is
            'M' or 'F', dropping 'Unknown'/other (see filter_known_sex). The
            fit corpus is `bow ⋈ covariates` (inner), so restricting the
            covariate person set here restricts the fit population.

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

    if known_sex_only:
        df = filter_known_sex(df)
    return df
