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

import math

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
    """Construct a CountVectorizerModel from a manual vocab aggregation.

    Replaces ``CountVectorizer.fit()`` so we can jointly enforce per-document
    (``min_df``) and per-patient (``min_patient_count``) thresholds in one
    ``groupBy/agg`` over the exploded token rows. (A fractional ``min_df``
    triggers an additional ``grouped.count()`` pre-pass to resolve the
    threshold against corpus size, matching PySpark's CountVectorizer
    convention.) Vocab ordering matches CountVectorizer's convention (total
    token occurrence count descending, alphabetical tiebreak) so downstream
    consumers see the same vocab indexing they would have under the fit
    path when ``min_patient_count == 1``.

    The `grouped` DataFrame must have columns:
        doc_id      – document key (string)
        person_id   – patient key (any type)
        tokens      – collect_list of string tokens

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
        # Match PySpark CountVectorizer's behavior, which ceils fractional minDF:
        # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.CountVectorizer.html
        min_df_int = max(1, math.ceil(min_df * total_docs))
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
        F.count("*").alias("term_count"),
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


def doc_length_report(grouped: DataFrame, *, group_col: str) -> list[dict]:
    """Per-group document-length distribution BEFORE the min_doc_length filter.

    ``grouped`` must have a ``tokens`` array column (one row per document) and
    ``group_col``. Returns one dict per group with the document count, token-count
    percentiles (p10/25/50/75/90), and how many documents would survive at
    candidate thresholds (>= 5/10/20/30 tokens) — the numbers needed to choose
    ``min_doc_length`` from the real distribution rather than guessing, without
    over-dropping light-coder documents.
    """
    thresholds = (5, 10, 20, 30)
    dl = grouped.select(group_col, F.size("tokens").alias("n"))
    rows = dl.groupBy(group_col).agg(
        F.count(F.lit(1)).alias("n_docs"),
        F.percentile_approx("n", [0.1, 0.25, 0.5, 0.75, 0.9]).alias("pct"),
        *[F.sum((F.col("n") >= t).cast("long")).alias(f"ge{t}") for t in thresholds],
    ).collect()
    return [
        {
            "group": r[group_col],
            "n_docs": int(r["n_docs"]),
            "pct": [int(x) for x in (r["pct"] or [])],
            **{f"ge{t}": int(r[f"ge{t}"]) for t in thresholds},
        }
        for r in rows
    ]


def _log_doc_length_report(report: list[dict]) -> None:
    """Print a doc_length_report to the driver log (stdout)."""
    for r in sorted(report, key=lambda x: str(x["group"])):
        print(
            f"[driver]   doc-length[{r['group']}]: n={r['n_docs']} "
            f"pctiles(10/25/50/75/90)={r['pct']} "
            f"kept>=5:{r['ge5']} >=10:{r['ge10']} >=20:{r['ge20']} >=30:{r['ge30']}",
            flush=True,
        )


def group_top_codes(
    events_with_doc_id: DataFrame,
    *,
    group_col: str,
    name_col: str = "concept_name",
    top_n: int = 15,
) -> dict:
    """Top codes per group by document frequency — a content peek at what each
    cohort's documents are actually made of.

    Answers questions like "is the light-coder 'general' arm dominated by routine
    checkup/screening codes, or by varied real conditions?" without fitting a
    model. Counts each code once per document (distinct doc_id) so a code that
    repeats within a doc isn't over-weighted. Returns {group: [(name, doc_freq),
    ... top_n]}.
    """
    from pyspark.sql import Window

    dd = events_with_doc_id.select("doc_id", group_col, name_col).distinct()
    counts = dd.groupBy(group_col, name_col).agg(F.count(F.lit(1)).alias("doc_freq"))
    ranked = counts.withColumn(
        "rn",
        F.row_number().over(
            Window.partitionBy(group_col).orderBy(
                F.col("doc_freq").desc(), F.col(name_col).asc(),
            )
        ),
    ).where(F.col("rn") <= top_n)
    out: dict = {}
    for r in ranked.collect():
        out.setdefault(r[group_col], []).append((r[name_col], int(r["doc_freq"])))
    return out


def _log_group_top_codes(top: dict) -> None:
    """Print group_top_codes to the driver log (stdout)."""
    for g in sorted(top, key=str):
        codes = ", ".join(f"{name}({n})" for name, n in top[g])
        print(f"[driver]   top-codes[{g}]: {codes}", flush=True)


def to_bow_dataframe(
    df: DataFrame,
    *,
    doc_spec: DocSpec | None = None,
    token_col: str = "concept_id",
    vocab_size: int | None = None,
    min_df: int | float = 1,
    min_patient_count: int = 1,
    vocab: list[int] | None = None,
    length_report_group_col: str | None = None,
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
    # Optionally carry a group column (e.g. source_cohort) through the group-by
    # so we can report per-group doc-length distribution before filtering.
    report_col = (
        length_report_group_col
        if length_report_group_col and length_report_group_col in events_with_doc_id.columns
        else None
    )
    agg_exprs = [
        F.first("person_id").alias("person_id"),
        F.collect_list(token_col).alias("tokens"),
    ]
    if report_col:
        agg_exprs.append(F.first(report_col).alias(report_col))
    grouped = (
        events_with_doc_id
        .withColumn(token_col, F.col(token_col).cast(StringType()))
        .groupBy("doc_id")
        .agg(*agg_exprs)
    )

    # Pre-filter per-group doc-length diagnostic: how many docs each candidate
    # min_doc_length would keep, so the threshold can be chosen from the real
    # distribution. Runs only on a corpus (re)build; then drop the group col so
    # the vectorizer input is unchanged.
    if report_col:
        _log_doc_length_report(doc_length_report(grouped, group_col=report_col))
        # Content peek: top codes per group (needs concept names on the frame).
        if "concept_name" in events_with_doc_id.columns:
            _log_group_top_codes(
                group_top_codes(events_with_doc_id, group_col=report_col)
            )
        grouped = grouped.drop(report_col)

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
