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
from pyspark.sql.types import StructType, StructField, LongType


def build_patient_covariate_df(
    person_df: DataFrame,
    *,
    covariate_formula: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    max_levels: int = 10_000,
) -> tuple[DataFrame, Any, list[str]]:
    """Materialize per-person covariates from a Spark DataFrame + formula.

    Returns:
        cov_df:      (person_id: long, covariates: DenseVector) Spark DataFrame.
                     One row per distinct person_id in person_df.
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

    # Project only the columns the formula references + person_id.
    needed_cols = ["person_id"] + categorical_cols + continuous_cols
    projected = person_df.select(*needed_cols).dropDuplicates(["person_id"])

    model_spec, names = fit_model_spec_from_spark(
        formula=covariate_formula,
        spark_df=projected,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        max_levels=max_levels,
    )

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
        for pid, row in zip(pdf["person_id"].tolist(), X):
            yield (int(pid), DenseVector(list(row)))

    schema = StructType([
        StructField("person_id", LongType(), False),
        StructField("covariates", VectorUDT(), False),
    ])

    cov_df = projected.rdd.mapPartitions(_apply_partition).toDF(schema)
    return cov_df, model_spec, names
