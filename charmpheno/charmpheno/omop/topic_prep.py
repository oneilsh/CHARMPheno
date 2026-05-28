"""OMOP -> bag-of-words DataFrame conversion for topic-style models.

Sibling of `local.py`: a loader-family function that takes an OMOP-shaped
DataFrame and returns the BOW representation that OnlineLDA (and MLlib's
LDA) consume. Uses a manual per-token aggregation for vocab construction
and SparseVector emission via CountVectorizerModel.

How documents are derived from the OMOP event rows is parameterized by a
DocSpec (see `charmpheno.omop.doc_spec`). The default PatientDocSpec
reproduces the pre-2026-05-12 behavior (one doc per patient over their
full event history); PatientYearDocSpec produces one doc per
(patient, calendar-year-active) for finer temporal phenotype resolution.
See ADR 0018 for the abstraction rationale.
"""
from __future__ import annotations

from pyspark.ml.feature import CountVectorizerModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from charmpheno.omop.doc_spec import DocSpec, PatientDocSpec


def _build_count_vectorizer_model(
    grouped: DataFrame,
    *,
    min_df: int | float,
    min_patient_count: int,
    vocab_size: int | None,
) -> CountVectorizerModel:
    """Build a CountVectorizerModel via manual per-token aggregation.

    Replaces CountVectorizer.fit() so we can apply both a document-frequency
    threshold (min_df) and a patient-count threshold (min_patient_count) in a
    single Spark pass.

    The `grouped` DataFrame must have columns:
        doc_id      – document key (string)
        person_id   – patient key (any type)
        tokens      – collect_list of string tokens

    Vocab ordering matches CountVectorizer's convention:
        primary:   term_count descending  (total occurrences across all docs)
        secondary: token ascending        (alphabetical, for stable tiebreak)

    Parameters
    ----------
    grouped:
        Pre-aggregated DataFrame with (doc_id, person_id, tokens) columns.
        min_doc_length filtering has already been applied by the caller.
    min_df:
        Minimum document count. int = absolute; float in (0,1) = fraction of
        total documents. Semantics match CountVectorizer's minDF parameter.
    min_patient_count:
        Minimum number of distinct patients (person_id values) a token must
        appear in to be included in the vocabulary.
    vocab_size:
        Cap on vocabulary size (top-N by term_count). None = no cap.

    Returns
    -------
    CountVectorizerModel with vocabulary set to the filtered, ordered token list.
    """
    # Resolve min_df from fraction to absolute count if needed.
    if isinstance(min_df, float) and 0.0 < min_df < 1.0:
        total_docs = grouped.count()
        min_df_int = max(1, int(min_df * total_docs))
    else:
        min_df_int = int(min_df)

    # Re-explode the token lists so we can aggregate per-token stats while
    # honouring the min_doc_length filter already applied to `grouped`.
    exploded = grouped.select(
        F.explode("tokens").alias("token"),
        "doc_id",
        "person_id",
    )

    token_stats = exploded.groupBy("token").agg(
        F.count("token").alias("term_count"),
        F.countDistinct("doc_id").alias("doc_count"),
        F.countDistinct("person_id").alias("patient_count"),
    )

    filtered = token_stats.where(
        (F.col("doc_count") >= min_df_int) &
        (F.col("patient_count") >= min_patient_count)
    )

    # Order matches CountVectorizer: descending total-count, then alphabetical
    # ascending for a stable tiebreak (CountVectorizer uses hash-bucket order
    # within ties, but alphabetical is the closest reproducible approximation).
    ordered = filtered.orderBy(
        F.col("term_count").desc(),
        F.col("token").asc(),
    )

    rows = ordered.collect()
    if vocab_size is not None:
        rows = rows[:vocab_size]

    vocabulary = [row["token"] for row in rows]

    return CountVectorizerModel.from_vocabulary(
        vocabulary,
        inputCol="tokens",
        outputCol="features",
    )


def to_bow_dataframe(
    df: DataFrame,
    *,
    doc_spec: DocSpec | None = None,
    token_col: str = "concept_id",
    vocab_size: int | None = None,
    min_df: int | float = 1,
    min_patient_count: int = 1,
    vocab: list[int] | None = None,
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
            frequency are kept. None leaves no cap in place — fine for the
            simulator, often too large for real OMOP.
        min_df: drop tokens that appear in fewer than this many documents.
            int = absolute count; float in (0,1) = fraction of corpus.
            Default 1 keeps every token.
        min_patient_count: drop tokens that appear in fewer than this many
            distinct patients (person_id values). Applied independently of
            min_df; both thresholds must be satisfied for a token to enter
            the vocabulary (AND composition). Default 1 keeps every token
            (no-op for existing callers). This threshold is useful for
            privacy-preserving corpus construction: under PatientYearDocSpec
            one patient can contribute many year-documents, so min_df alone
            is not a reliable lower-bound on patient exposure.
        vocab: optional pre-built vocabulary as a list of concept_ids in
            assignment order (``vocab[idx] = concept_id``). When provided,
            the fit step is skipped and a CountVectorizerModel is constructed
            directly from this vocab. Tokens not in the vocab are dropped
            from the output BOW. Use case: eval drivers loading vocab from a
            checkpoint's ``metadata["vocab"]`` so the eval scores against
            fit-time topic indices regardless of which input parquet is
            supplied. Combining ``vocab`` with non-default ``vocab_size``,
            ``min_df``, or ``min_patient_count`` raises ValueError (the vocab
            is fixed; those knobs have no effect).

    Returns:
        bow_df: DataFrame[person_id, doc_id, features: SparseVector]. One row
            per document. person_id is retained so downstream consumers
            that need patient-level identity (e.g. for diagnostics) can
            recover it even when person_id != doc_id (e.g. patient_year).
        vocab_map: dict[concept_id (int), idx (int)] where idx in [0, V).

    Both paths in lda_compare consume the same SparseVector column, so
    MLlib's LDA and our OnlineLDA see byte-identical input.
    """
    if doc_spec is None:
        doc_spec = PatientDocSpec()

    if vocab is not None:
        if vocab_size is not None or min_df != 1 or min_patient_count != 1:
            raise ValueError(
                "vocab=<frozen list> is incompatible with vocab_size / min_df "
                "/ min_patient_count — the vocabulary is fixed by the caller; "
                "those knobs have no effect. Drop them or drop vocab."
            )
        if any(v is None for v in vocab):
            raise ValueError(
                "frozen vocab contains None entries; saved vocab_list slots "
                "must all be filled. Checkpoint is malformed."
            )

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
    # this before vocab construction means the vocab is built only on the
    # docs that will survive into training — short noise-docs don't pull
    # their token contributions into the vocab tail. For the frozen-vocab
    # path the filter still applies (we don't want short noise-docs leaking
    # into eval reference statistics any more than into fit).
    if doc_spec.min_doc_length > 0:
        grouped = grouped.where(F.size("tokens") >= doc_spec.min_doc_length)

    if vocab is None:
        cv_model = _build_count_vectorizer_model(
            grouped,
            min_df=min_df,
            min_patient_count=min_patient_count,
            vocab_size=vocab_size,
        )
    else:
        # CountVectorizerModel.from_vocabulary takes a list[str] (token
        # strings as they appear in the input column, which we cast to
        # StringType above). The saved concept_ids are ints; stringify
        # to match the input column's runtime type.
        cv_model = CountVectorizerModel.from_vocabulary(
            [str(c) for c in vocab],
            inputCol="tokens",
            outputCol="features",
        )

    bow_df = cv_model.transform(grouped).select("person_id", "doc_id", "features")

    vocab_map = {int(token): idx for idx, token in enumerate(cv_model.vocabulary)}
    return bow_df, vocab_map
