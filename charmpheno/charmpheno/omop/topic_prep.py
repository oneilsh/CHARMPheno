"""OMOP -> bag-of-words DataFrame conversion for topic-style models.

Sibling of `local.py`: a loader-family function that takes an OMOP-shaped
DataFrame and returns the BOW representation that OnlineLDA (and MLlib's
LDA) consume. Uses pyspark.ml.feature.CountVectorizer for battle-tested
vocab construction and SparseVector emission.

How documents are derived from the OMOP event rows is parameterized by a
DocSpec (see `charmpheno.omop.doc_spec`). The default PatientDocSpec
reproduces the pre-2026-05-12 behavior (one doc per patient over their
full event history); PatientYearDocSpec produces one doc per
(patient, calendar-year-active) for finer temporal phenotype resolution.
See ADR 0018 for the abstraction rationale.
"""
from __future__ import annotations

from pyspark.ml.feature import CountVectorizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from charmpheno.omop.doc_spec import DocSpec, PatientDocSpec


def to_bow_dataframe(
    df: DataFrame,
    *,
    doc_spec: DocSpec | None = None,
    token_col: str = "concept_id",
    vocab_size: int | None = None,
    min_df: int | float = 1,
) -> tuple[DataFrame, dict[int, int]]:
    """Group rows into bag-of-words documents and build a contiguous vocab map.

    Parameters:
        df: OMOP-shaped DataFrame; must contain the columns the doc_spec
            requires (always at least `person_id` and `token_col`).
        doc_spec: how event rows become documents. Default = PatientDocSpec()
            (one doc per patient, the pre-ADR-0018 behavior). Pass
            PatientYearDocSpec() for one doc per (patient, year-active).
        token_col: column whose values are tokens (concept_ids).
        vocab_size: cap on vocabulary; the top-N tokens by document
            frequency are kept. None leaves CountVectorizer's default
            (262144) in place — fine for the simulator, often too large
            for real OMOP.
        min_df: drop tokens that appear in fewer than this many documents.
            int = absolute count; float in (0,1) = fraction of corpus.
            Default 1 keeps every token (matches CountVectorizer default).

    Returns:
        bow_df: DataFrame[person_id, doc_id, features: SparseVector]. One row
            per document. person_id is retained so patient-keyed holdout
            splits work even when person_id != doc_id (e.g. patient_year).
        vocab_map: dict[concept_id (int), idx (int)] where idx in [0, V).

    Both paths in lda_compare consume the same SparseVector column, so
    MLlib's LDA and our OnlineLDA see byte-identical input.
    """
    if doc_spec is None:
        doc_spec = PatientDocSpec()

    # Add doc_id (may replicate event rows, e.g. era-spanning years).
    events_with_doc_id = doc_spec.derive_docs(df)

    # Group event rows into per-doc bags. We groupBy (doc_id) and pull
    # person_id along via F.first — per spec contract, person_id is constant
    # within a doc_id, so F.first is well-defined.
    grouped = (
        events_with_doc_id
        .withColumn(token_col, F.col(token_col).cast(StringType()))
        .groupBy("doc_id")
        .agg(
            F.first("person_id").alias("person_id"),
            F.collect_list(token_col).alias("tokens"),
        )
    )

    # Apply min_doc_length filter on pre-vectorize token-list length. Doing
    # this before CountVectorizer.fit means the vocab is built only on the
    # docs that will survive into training — short noise-docs don't pull
    # their token contributions into the vocab tail.
    if doc_spec.min_doc_length > 0:
        grouped = grouped.where(F.size("tokens") >= doc_spec.min_doc_length)

    cv = CountVectorizer(inputCol="tokens", outputCol="features",
                         minDF=float(min_df))
    if vocab_size is not None:
        cv = cv.setVocabSize(vocab_size)
    cv_model = cv.fit(grouped)
    bow_df = cv_model.transform(grouped).select("person_id", "doc_id", "features")

    vocab_map = {int(token): idx for idx, token in enumerate(cv_model.vocabulary)}
    return bow_df, vocab_map
