"""STM patient-covariate load-or-build (mirrors _corpus_load.py).

On cache HIT: returns (cov_df, model_spec, covariate_names) from the
write-through parquet without re-fitting the ModelSpec.

On MISS (or cache_uri=None): runs build_patient_covariate_df against
the source person_df, optionally writes through the cache.

The cache key derives from (formula, person_mod, cdr, source_table, cohort)
so a cache hit is safe: same inputs => same outputs.

Decision context: docs/decisions/0024-formulaic-in-mllib-shim-with-schema-frame-discovery.md
                  docs/decisions/0025-charmpheno-covariate-sidecar-parquet.md
"""
from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession

from _driver_common import _phase


def load_or_build_covariates(
    spark: SparkSession,
    *,
    person_df: DataFrame,
    covariate_formula: str,
    categorical_cols: list[str],
    continuous_cols: list[str],
    cdr: str,
    source_table: str,
    cohort: str | None,
    person_mod: int,
    cache_uri: str | None = None,
    max_levels: int = 10_000,
) -> tuple[DataFrame, Any, list[str]]:
    """Return (cov_df, model_spec, covariate_names) for the given formula."""
    from charmpheno.omop.covariates import build_patient_covariate_df
    from _covariates_cache import compute_cache_key, try_load, save

    key: str | None = None
    if cache_uri:
        key = compute_cache_key(
            covariate_formula=covariate_formula,
            person_mod=person_mod, cdr=cdr,
            source_table=source_table, cohort=cohort,
        )
        with _phase(f"covariates-cache lookup ({cache_uri}/{key})"):
            cached = try_load(spark, cache_uri, key)
        if cached is not None:
            print("[driver]   covariates-cache HIT", flush=True)
            return cached
        print("[driver]   covariates-cache MISS, building...", flush=True)

    with _phase("build patient covariates"):
        cov_df, spec, names = build_patient_covariate_df(
            person_df,
            covariate_formula=covariate_formula,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            max_levels=max_levels,
        )

    if cache_uri:
        with _phase(f"covariates-cache write-through ({cache_uri}/{key})"):
            save(spark, cache_uri, key,
                 cov_df=cov_df, model_spec=spec, covariate_names=names)

    return cov_df, spec, names
