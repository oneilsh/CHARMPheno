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

#: The gating/label column materialized from the combined-cohort doc-spec
#: (doc_id == "{group}:{person_id}"). It is the only group column the STM
#: gating pipeline can materialize today (see stm_bigquery_cloud's guard).
DEFAULT_GROUP_COL = "source_cohort"


def validate_label_not_covariate(
    categorical_cols, continuous_cols, *, label: str = DEFAULT_GROUP_COL,
) -> None:
    """Reject a gating/label key that also appears in the covariate formula.

    ``label`` (e.g. source_cohort) is a per-document gating key, not a
    covariate — it was moved out of the formula into a pure label. If it were
    also a covariate, a person's per-(person, label) sidecar rows would carry
    *different* covariate vectors, which would (a) make the corpus join
    ambiguous and (b) corrupt the gated dashboard prevalence (each foreground
    topic's masked mean would mix the two vectors). Forbid the overlap loudly.
    """
    if label in set(categorical_cols) or label in set(continuous_cols):
        raise ValueError(
            f"{label!r} is the gating/label key and must not also appear in "
            f"the covariate formula (categorical/continuous cols); it was "
            f"moved from the formula to a pure per-document label. Drop it "
            f"from the formula.")


def covariate_key_cols(
    *, gated: bool, label: str = DEFAULT_GROUP_COL,
) -> list[str]:
    """Key columns for the covariate sidecar.

    A gated fit keys on ``(person_id, label)`` so the sidecar carries each
    document's group — the gated dashboard prevalence consumer groups by it,
    and a comorbid person (in two groups) contributes one row per group. An
    ungated fit keys on ``(person_id,)`` alone. The covariate *vector* never
    depends on ``label`` (guarded by ``validate_label_not_covariate``), so a
    comorbid person's two rows are identical apart from the key.
    """
    return ["person_id", label] if gated else ["person_id"]


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
    key_cols: tuple[str, ...] | list[str] = ("person_id",),
    prior_obs_days: int = 365,
) -> tuple[DataFrame, Any, list[str]]:
    """Return (cov_df, model_spec, covariate_names) for the given formula.

    prior_obs_days keys the cache: in composite mode the covariate person set
    is the corpus's persons, so a changed cohort lookback must not reload a
    stale covariate set. Ignored for membership (it doesn't filter person_df
    here) -- it only participates in the cache key.
    """
    from charmpheno.omop.covariates import build_patient_covariate_df
    from _covariates_cache import compute_cache_key, try_load, save

    key: str | None = None
    if cache_uri:
        key = compute_cache_key(
            covariate_formula=covariate_formula,
            person_mod=person_mod, cdr=cdr,
            source_table=source_table, cohort=cohort,
            prior_obs_days=prior_obs_days,
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
            key_cols=key_cols,
            max_levels=max_levels,
        )

    if cache_uri:
        with _phase(f"covariates-cache write-through ({cache_uri}/{key})"):
            save(spark, cache_uri, key,
                 cov_df=cov_df, model_spec=spec, covariate_names=names)

    return cov_df, spec, names
