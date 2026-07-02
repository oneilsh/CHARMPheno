"""Cohort-aware OMOP corpus loader with optional Spark-side cache.

Shared by ``lda_bigquery_cloud.py`` and ``hdp_bigquery_cloud.py``
(and any future cloud fit driver). Encapsulates the BQ load →
CountVectorizer → concept-name lookup pipeline, plus a write-through
cache keyed on the parameters that affect the resulting (BOW, vocab,
names) triple.

Cache contract (see ``_corpus_cache.py``):
  - HIT path: bypasses BQ entirely; returns the cached triple.
  - MISS path: builds + writes through.
  - cache_uri is None: skips cache entirely; builds fresh every call.

Cohort filtering (LDA's --cohort flag) is supported via an optional
``cohort`` parameter; passing ``cohort=None`` (HDP's default) yields a
cohort-unfiltered corpus and a cohort-free cache key.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from _driver_common import _phase


def load_or_build_corpus(
    spark: SparkSession,
    *,
    doc_spec,
    cdr: str,
    billing: str,
    source_table: str,
    person_mod: int,
    vocab_size: int,
    min_df: int,
    min_patient_count: int,
    cache_uri: str | None = None,
    cohort: str | None = None,
    prior_obs_days: int = 365,
    length_report_group_col: str | None = None,
) -> tuple[DataFrame, dict[int, int], dict[int, str]]:
    """Return ``(bow_df, vocab_map, name_by_id)`` with optional cache short-circuit.

    On a cache hit, skips BQ load + CountVectorizer fit entirely. On a
    miss (or when ``cache_uri`` is None), runs the BQ+vectorize pipeline,
    resolves concept names for the vocab, and writes through.

    Concept-name resolution happens here (not in the caller) so the
    cached entry is self-sufficient — downstream eval/dashboard drivers
    don't need a live ``omop`` DataFrame on the hit path.

    Parameters
    ----------
    spark : SparkSession
    doc_spec : DocSpec
        From ``charmpheno.omop.doc_spec``.
    cdr : str
    billing : str
        BigQuery dataset + billing project, threaded into
        ``charmpheno.omop.load_omop_bigquery``.
    source_table : str
    person_mod : int
    vocab_size : int
    min_df : int
    min_patient_count : int
        Standard fit-driver corpus-construction parameters.
    cache_uri : str | None
        HDFS or GCS URI for the corpus cache. None disables caching.
    cohort : str | None
        Cohort id for ``charmpheno.omop.load_omop_bigquery``; None
        yields the unfiltered general-population corpus and a cache key
        with no cohort dimension.
    prior_obs_days : int
        Prior-observation lookback (days) for the cohort index date
        (default 365). Keys the cache because it changes membership; 0
        drops the lookback. Ignored when ``cohort`` is None.
    """
    from charmpheno.omop import load_omop_bigquery, to_bow_dataframe
    from spark_vi.diagnostics.persist import assert_persisted
    from _corpus_cache import compute_cache_key, save, try_load

    cache_key: str | None = None
    if cache_uri:
        cache_key = compute_cache_key(
            source_table=source_table,
            person_mod=person_mod,
            vocab_size=vocab_size,
            min_df=min_df,
            doc_spec_manifest=doc_spec.manifest(),
            cohort=cohort,
            prior_obs_days=prior_obs_days,
        )
        with _phase(f"corpus-cache lookup ({cache_uri}/{cache_key})"):
            cached = try_load(spark, cache_uri, cache_key)
        if cached is not None:
            print(f"[driver]   corpus-cache HIT", flush=True)
            return cached
        print(f"[driver]   corpus-cache MISS, building...", flush=True)

    with _phase("BQ load + summary"):
        omop = load_omop_bigquery(
            spark=spark, cdr_dataset=cdr, billing_project=billing,
            person_sample_mod=person_mod, source_table=source_table,
            cohort=cohort, prior_obs_days=prior_obs_days,
        ).persist()
        summary = omop.agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("person_id").alias("persons"),
        ).collect()[0]
        assert_persisted(omop, name="omop")
        print(f"[driver]   OMOP: {summary['rows']} rows, "
              f"{summary['persons']} distinct persons", flush=True)

    with _phase(f"vectorize (CountVectorizer, doc_spec={doc_spec.name}, "
                f"min_doc_length={doc_spec.min_doc_length})"):
        bow_df, vocab_map = to_bow_dataframe(
            omop, doc_spec=doc_spec,
            vocab_size=vocab_size,
            min_df=min_df,
            min_patient_count=min_patient_count,
            length_report_group_col=length_report_group_col,
        )

    with _phase("concept-name lookup"):
        # Vocabulary-only lookup; small enough for the driver. dropDuplicates
        # because OMOP occasionally has multiple name variants per concept_id.
        name_rows = (
            omop.where(F.col("concept_id").isin(list(vocab_map.keys())))
                .select("concept_id", "concept_name")
                .dropDuplicates(["concept_id"])
                .collect()
        )
        name_by_id = {int(r["concept_id"]): r["concept_name"] for r in name_rows}
        print(f"[driver]   resolved {len(name_by_id)} concept names", flush=True)

    omop.unpersist()

    if cache_uri:
        with _phase(f"corpus-cache write ({cache_uri}/{cache_key})"):
            save(spark, bow_df, vocab_map, name_by_id, cache_uri, cache_key)

    return bow_df, vocab_map, name_by_id
