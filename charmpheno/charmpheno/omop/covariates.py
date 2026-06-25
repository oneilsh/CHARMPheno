"""STM patient-covariate materialization.

build_patient_covariate_df takes a person-level Spark DataFrame and a
formula string, uses spark-vi's formula machinery (schema-frame
discovery via Spark `select distinct`) to fit a formulaic ModelSpec,
and applies the spec per row to produce a per-person `(person_id,
covariates)` DataFrame with a DenseVector column.

Decision context: docs/decisions/0025-charmpheno-covariate-sidecar-parquet.md
                  docs/decisions/0024-formulaic-in-mllib-shim-with-schema-frame-discovery.md
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField


def corpus_mean_proportions_from_covariate_df(
    cov_df: DataFrame,
    Gamma: np.ndarray,
    *,
    covariates_col: str = "covariates",
) -> np.ndarray:
    """Faithful dashboard α-equivalent (1/D) Σ_d softmax(Γᵀ x_d) from the sidecar.

    Selects only the covariate-vector column — dropping ``person_id`` so it
    never crosses into spark-vi — converts each DenseVector to a numpy array,
    and delegates the distributed reduction to spark-vi's
    ``corpus_mean_topic_proportions_rdd`` (mapPartitions+treeReduce, so only a
    K-vector + count reach the driver). Returns a length-K probability vector.
    """
    from spark_vi.mllib.topic.stm import corpus_mean_topic_proportions_rdd

    vec_rdd = cov_df.select(covariates_col).rdd.map(lambda row: row[0].toArray())
    return corpus_mean_topic_proportions_rdd(vec_rdd, Gamma)


def corpus_mean_proportions_gated_from_covariate_df(
    cov_df: DataFrame,
    Gamma: np.ndarray,
    partition,
    *,
    covariates_col: str = "covariates",
    group_col: str = "source_cohort",
) -> np.ndarray:
    """Gating-aware dashboard α-equivalent from the sidecar — distributed.

    Mirrors ``corpus_mean_proportions_from_covariate_df`` but masks each
    document's softmax to its allowed topic set (background ∪ its group's
    foreground block). Selects only the covariate-vector + group columns —
    dropping ``person_id`` so it never crosses into spark-vi — builds an RDD of
    ``(x, groups)`` pairs, and delegates to spark-vi's
    ``corpus_mean_topic_proportions_gated_rdd`` (mapPartitions+treeReduce, so
    only a K-vector + count reach the driver; no full-corpus collect). Returns a
    length-K probability vector.
    """
    from spark_vi.mllib.topic.stm import corpus_mean_topic_proportions_gated_rdd

    pair_rdd = (
        cov_df.select(covariates_col, group_col).rdd
        .map(lambda row: (row[0].toArray(), frozenset({str(row[1])})))
    )
    return corpus_mean_topic_proportions_gated_rdd(pair_rdd, Gamma, partition)


def build_patient_covariate_df(
    person_df: DataFrame,
    *,
    covariate_formula: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    key_cols: tuple[str, ...] | list[str] = ("person_id",),
    max_levels: int = 10_000,
) -> tuple[DataFrame, Any, list[str]]:
    """Materialize per-person covariates from a Spark DataFrame + formula.

    Returns:
        cov_df:      (*key_cols, covariates: DenseVector) Spark DataFrame.
                     One row per distinct key tuple in person_df.
                     Default key_cols=("person_id",) preserves the original
                     (person_id, covariates) output.
        model_spec:  formulaic ModelSpec, ready to persist + apply at
                     transform / scoring time.
        names:       list of covariate column names (length P).

    Raises:
        ValueError if covariate_formula contains a stateful transform
        (spline, standardization). See ADR 0022 / 0024.
    """
    from spark_vi.mllib.topic._formula import (
        validate_formula, fit_model_spec_from_spark, apply_model_spec,
    )

    validate_formula(covariate_formula)

    key_cols = list(key_cols)
    # De-dup the column list (a key col may also be a formula categorical,
    # e.g. source_cohort) while preserving order.
    needed_cols = list(dict.fromkeys([*key_cols, *categorical_cols, *continuous_cols]))
    projected = person_df.select(*needed_cols).dropDuplicates(key_cols)

    model_spec, names = fit_model_spec_from_spark(
        formula=covariate_formula,
        spark_df=projected,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        max_levels=max_levels,
    )

    # Extract key column type info on the driver before building closures —
    # _flush must not capture `projected` (a DataFrame), which holds a
    # SparkContext reference and cannot be serialized to workers.
    key_fields = [projected.schema[c] for c in key_cols]
    key_type_strs = {c: str(projected.schema[c].dataType) for c in key_cols}
    schema = StructType([*key_fields, StructField("covariates", VectorUDT(), False)])

    # Apply the spec per partition. ModelSpec is small (factor mappings);
    # serializing it into the closure is fine for a few-KB object.
    spec_broadcast = projected.sparkSession.sparkContext.broadcast(model_spec)
    cat_set = list(categorical_cols)
    cont_set = list(continuous_cols)

    def _apply_partition(rows_iter):
        import pandas as pd
        import numpy as _np
        spec = spec_broadcast.value
        # Buffer the partition's rows into a small pandas frame for vectorized
        # apply_model_spec — chunk size keeps memory bounded if partitions are large.
        CHUNK = 10_000
        buf = []
        for r in rows_iter:
            buf.append(r.asDict())
            if len(buf) >= CHUNK:
                yield from _flush(buf, spec, cat_set, cont_set)
                buf = []
        if buf:
            yield from _flush(buf, spec, cat_set, cont_set)

    def _flush(buf, spec, cat_set, cont_set):
        import pandas as pd
        from spark_vi.mllib.topic._formula import apply_model_spec
        from pyspark.ml.linalg import DenseVector
        pdf = pd.DataFrame(buf)
        X = apply_model_spec(spec, pdf)   # (chunk_size, P)
        for i, row in enumerate(X):
            key_vals = tuple(
                int(pdf[c].iloc[i]) if key_type_strs[c] in ("LongType()", "IntegerType()")
                else pdf[c].iloc[i]
                for c in key_cols
            )
            yield (*key_vals, DenseVector(list(row)))

    cov_df = projected.rdd.mapPartitions(_apply_partition).toDF(schema)
    return cov_df, model_spec, names
