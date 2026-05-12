"""Optional caching for the (BQ load + BOW build) prep phase.

The prep phase — read OMOP from BigQuery, derive doc_ids per DocSpec,
fit CountVectorizer, transform to BOW — is deterministic in
(source_table, person_mod, vocab_size, min_df, doc_spec.manifest()).
When iterating on fit-side hyperparameters, paying the prep cost on
every run is wasteful: BQ extract is network-bound, CountVectorizer
fit is a full-corpus scan.

This module lets the cloud fit drivers cache the post-BOW result under
a content-hash key. Subsequent runs that match the cache key load the
cached parquet instead of rebuilding.

Cache layout (under {cache_uri}/{key}/):
    bow.parquet/    Spark-readable parquet of (person_id, doc_id, features)
    vocab.parquet/  Spark-readable parquet of (concept_id, idx)

Cache backends are anything Spark + Hadoop FS can read/write:
    hdfs:///user/...   session-only, fast, cleaned with the cluster
    gs://bucket/...    persistent across clusters, slightly slower

When `cache_uri` is None the prep runs uncached, identical to pre-cache
behavior.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


def compute_cache_key(
    *,
    source_table: str,
    person_mod: int,
    vocab_size: int | None,
    min_df: int | float,
    doc_spec_manifest: dict,
) -> str:
    """Stable 16-hex-char hash of the inputs that determine the cached corpus.

    Schema version `v` is bumped if the cached file layout or any upstream
    transformation changes shape in a non-back-compat way; old cache entries
    silently miss and rebuild rather than load wrong data.
    """
    payload = {
        "source_table": source_table,
        "person_mod": int(person_mod),
        "vocab_size": vocab_size,
        "min_df": float(min_df),
        "doc_spec": doc_spec_manifest,
        "v": 1,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def try_load(
    spark: "SparkSession",
    cache_uri: str,
    key: str,
) -> Optional[tuple["DataFrame", dict[int, int]]]:
    """Look up the cache entry; return (bow_df, vocab_map) on hit, None on miss.

    Any read failure (path doesn't exist, schema mismatch, permission issue)
    is treated as a miss — we'd rather rebuild than crash a long-running fit.
    """
    base = f"{cache_uri.rstrip('/')}/{key}"
    try:
        bow_df = spark.read.parquet(f"{base}/bow.parquet")
        vocab_rows = spark.read.parquet(f"{base}/vocab.parquet").collect()
    except Exception:
        return None
    vocab_map = {int(r["concept_id"]): int(r["idx"]) for r in vocab_rows}
    return bow_df, vocab_map


def save(
    spark: "SparkSession",
    bow_df: "DataFrame",
    vocab_map: dict[int, int],
    cache_uri: str,
    key: str,
) -> None:
    """Persist (bow_df, vocab_map) under {cache_uri}/{key}/.

    `overwrite` mode so a stale partial write (e.g. job killed mid-save)
    doesn't poison the cache; the next successful run replaces it.
    """
    base = f"{cache_uri.rstrip('/')}/{key}"
    bow_df.write.mode("overwrite").parquet(f"{base}/bow.parquet")
    vocab_df = spark.createDataFrame(
        [(int(cid), int(idx)) for cid, idx in vocab_map.items()],
        schema="concept_id INT, idx INT",
    )
    vocab_df.write.mode("overwrite").parquet(f"{base}/vocab.parquet")
