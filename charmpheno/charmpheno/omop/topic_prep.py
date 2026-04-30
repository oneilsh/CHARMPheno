"""OMOP -> bag-of-words DataFrame conversion for topic-style models.

Sibling of `local.py`: a loader-family function that takes an OMOP-shaped
DataFrame and returns the BOW representation that VanillaLDA (and MLlib's
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
) -> tuple[DataFrame, dict[int, int]]:
    """Group rows into bag-of-words documents and build a contiguous vocab map.

    Parameters:
        df: OMOP-shaped DataFrame, must contain doc_col and token_col.
        doc_col: column to group on (one row per document).
        token_col: column whose values are tokens (concept_ids).

    Returns:
        bow_df: DataFrame[doc_col, features: SparseVector]. One row per document.
        vocab_map: dict[concept_id (int), idx (int)] where idx in [0, V).

    Both paths in lda_compare consume the same SparseVector column, so
    MLlib's LDA and our VanillaLDA see byte-identical input.
    """
    grouped = (
        df.withColumn(token_col, F.col(token_col).cast(StringType()))
          .groupBy(doc_col)
          .agg(F.collect_list(token_col).alias("tokens"))
    )

    cv = CountVectorizer(inputCol="tokens", outputCol="features")
    cv_model = cv.fit(grouped)
    bow_df = cv_model.transform(grouped).select(doc_col, "features")

    vocab_map = {int(token): idx for idx, token in enumerate(cv_model.vocabulary)}
    return bow_df, vocab_map
