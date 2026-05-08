"""OMOP -> bag-of-words DataFrame conversion for topic-style models.

Sibling of `local.py`: a loader-family function that takes an OMOP-shaped
DataFrame and returns the BOW representation that OnlineLDA (and MLlib's
LDA) consume. Uses pyspark.ml.feature.CountVectorizer for battle-tested
vocab construction and SparseVector emission.
"""
from __future__ import annotations

from pyspark.ml.feature import CountVectorizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def to_bow_dataframe(
    df: DataFrame,
    doc_col: str = "person_id",
    token_col: str = "concept_id",
    vocab_size: int | None = None,
    min_df: int | float = 1,
) -> tuple[DataFrame, dict[int, int]]:
    """Group rows into bag-of-words documents and build a contiguous vocab map.

    Parameters:
        df: OMOP-shaped DataFrame, must contain doc_col and token_col.
        doc_col: column to group on (one row per document).
        token_col: column whose values are tokens (concept_ids).
        vocab_size: cap on vocabulary; the top-N tokens by document
            frequency are kept. None leaves CountVectorizer's default
            (262144) in place — fine for the simulator, often too large
            for real OMOP.
        min_df: drop tokens that appear in fewer than this many documents.
            int = absolute count; float in (0,1) = fraction of corpus.
            Default 1 keeps every token (matches CountVectorizer default).

    Returns:
        bow_df: DataFrame[doc_col, features: SparseVector]. One row per document.
        vocab_map: dict[concept_id (int), idx (int)] where idx in [0, V).

    Both paths in lda_compare consume the same SparseVector column, so
    MLlib's LDA and our OnlineLDA see byte-identical input.
    """
    grouped = (
        df.withColumn(token_col, F.col(token_col).cast(StringType()))
          .groupBy(doc_col)
          .agg(F.collect_list(token_col).alias("tokens"))
    )

    cv = CountVectorizer(inputCol="tokens", outputCol="features",
                         minDF=float(min_df))
    if vocab_size is not None:
        cv = cv.setVocabSize(vocab_size)
    cv_model = cv.fit(grouped)
    bow_df = cv_model.transform(grouped).select(doc_col, "features")

    vocab_map = {int(token): idx for idx, token in enumerate(cv_model.vocabulary)}
    return bow_df, vocab_map
