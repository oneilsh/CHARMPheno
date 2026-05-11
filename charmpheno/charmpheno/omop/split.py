"""Deterministic SHA-256-hash split of a BOW DataFrame by person_id.

Application-layer helper: drivers call this between BOW build and Estimator.fit
to produce a held-out partition for coherence evaluation. Splitting is NOT a
responsibility of the estimator (MLlib idiom — see ADR 0017).

Reproducible regardless of partition state — unlike DataFrame.randomSplit,
which depends on the partition layout at call time. The SHA-256 keying is the
same pattern already used in analysis/cloud/lda_bigquery_cloud.py for ID hashing.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_BUCKET_COUNT = 10_000


def split_bow_by_person(
    bow_df: DataFrame,
    *,
    holdout_fraction: float,
    seed: int,
    person_id_col: str = "person_id",
) -> tuple[DataFrame, DataFrame]:
    """Deterministic train/holdout split of a BOW DataFrame.

    Args:
        bow_df: DataFrame with at least `person_id_col` and feature columns.
            Other columns are preserved on both sides.
        holdout_fraction: in (0, 1). Approximate fraction of distinct persons
            routed to the holdout partition.
        seed: integer mixed into the hash; changing it produces a different
            split for the same population.
        person_id_col: column name to hash on. Default 'person_id'.

    Returns:
        (train_df, holdout_df). Disjoint by person_id_col; their union is the input.
    """
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError(
            f"holdout_fraction must be in (0, 1); got {holdout_fraction}"
        )

    threshold = int(holdout_fraction * _BUCKET_COUNT)
    bucket_expr = (
        F.conv(
            F.substring(
                F.sha2(F.concat_ws("|", F.col(person_id_col).cast("string"), F.lit(str(seed))), 256),
                1, 8,
            ),
            16, 10,
        ).cast("long") % F.lit(_BUCKET_COUNT)
    )
    annotated = bow_df.withColumn("_holdout_bucket", bucket_expr)
    holdout = annotated.filter(F.col("_holdout_bucket") < threshold).drop("_holdout_bucket")
    train = annotated.filter(F.col("_holdout_bucket") >= threshold).drop("_holdout_bucket")
    return train, holdout
