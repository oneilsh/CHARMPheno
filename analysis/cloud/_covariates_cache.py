"""Covariate cache: parallel to _corpus_cache.py for STM patient covariates.

Cache key derives from formula text + person table source + person_mod
so a covariates cache hit guarantees identical (person_id, covariates)
output. ModelSpec is persisted alongside the parquet so the cache
hit path can return the spec without re-fitting it from BigQuery.

Layout under <cache_uri>/<key>/:
    covariates.parquet     # (person_id, covariates as DenseVector)
    model_spec.pkl         # pickled formulaic ModelSpec
    covariate_names.json   # list of P term names

Format mirrors _corpus_cache.py; see that module for the read/write helpers
this one parallels.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

from pyspark.sql import DataFrame, SparkSession


def compute_cache_key(
    *,
    covariate_formula: str,
    person_mod: int,
    cdr: str,
    source_table: str,
    cohort: str | None,
    prior_obs_days: int = 365,
) -> str:
    """Stable hex digest of the inputs that determine the covariate output.

    prior_obs_days keys the cache because in composite (e.g. cancer_or_dementia)
    mode the covariate person set is the corpus's persons, and the cohort's
    prior-observation lookback sets corpus membership. Without it, a widened
    cohort (shorter lookback) would reload the narrower covariate set and the
    corpus+covariate inner join would silently drop the new patients.

    `v` is a schema/definition version. The cohort NAME is keyed but its CODE
    version is not, so when a cohort's membership changes without a config change
    (e.g. population_cancer's general arm switching to event-anchored windows,
    which grew its person set), the old covariate cache would otherwise stay a
    hit and drop the new persons at the join. Bump `v` on any such change. v=2
    was bumped alongside corpus-cache v=5 for the population_cancer windowing fix.
    """
    payload = json.dumps({
        "covariate_formula": covariate_formula,
        "person_mod": person_mod,
        "cdr": cdr,
        "source_table": source_table,
        "cohort": cohort,
        "prior_obs_days": int(prior_obs_days),
        "v": 2,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def try_load(
    spark: SparkSession, cache_uri: str, cache_key: str,
) -> tuple[DataFrame, Any, list[str]] | None:
    """Return (cov_df, model_spec, names) on cache hit, None on miss."""
    base = f"{cache_uri.rstrip('/')}/{cache_key}"
    try:
        cov_df = spark.read.parquet(f"{base}/covariates.parquet")
    except Exception:
        return None
    # Spec + names are driver-side reads via Spark's Hadoop FS.
    try:
        spec_path = f"{base}/model_spec.pkl"
        names_path = f"{base}/covariate_names.json"
        spec_bytes = _read_bytes(spark, spec_path)
        names_bytes = _read_bytes(spark, names_path)
    except Exception:
        return None
    spec = pickle.loads(spec_bytes)
    names = json.loads(names_bytes.decode("utf-8"))
    return cov_df, spec, names


def save(
    spark: SparkSession,
    cache_uri: str,
    cache_key: str,
    *,
    cov_df: DataFrame,
    model_spec: Any,
    covariate_names: list[str],
) -> None:
    """Write through to <cache_uri>/<key>/."""
    base = f"{cache_uri.rstrip('/')}/{cache_key}"
    cov_df.write.mode("overwrite").parquet(f"{base}/covariates.parquet")
    _write_bytes(spark, f"{base}/model_spec.pkl", pickle.dumps(model_spec))
    _write_bytes(
        spark, f"{base}/covariate_names.json",
        json.dumps(covariate_names).encode("utf-8"),
    )


def _read_bytes(spark: SparkSession, path: str) -> bytes:
    """Read a small file via Spark's Hadoop FileSystem (handles GCS/HDFS/local)."""
    sc = spark.sparkContext
    jvm = sc._jvm
    fs_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = fs_path.getFileSystem(sc._jsc.hadoopConfiguration())
    in_stream = fs.open(fs_path)
    try:
        out = jvm.java.io.ByteArrayOutputStream()
        jvm.org.apache.hadoop.io.IOUtils.copyBytes(in_stream, out, sc._jsc.hadoopConfiguration())
        return bytes(out.toByteArray())
    finally:
        in_stream.close()


def _write_bytes(spark: SparkSession, path: str, data: bytes) -> None:
    sc = spark.sparkContext
    jvm = sc._jvm
    fs_path = jvm.org.apache.hadoop.fs.Path(path)
    fs = fs_path.getFileSystem(sc._jsc.hadoopConfiguration())
    out_stream = fs.create(fs_path, True)
    try:
        out_stream.write(data)
    finally:
        out_stream.close()
