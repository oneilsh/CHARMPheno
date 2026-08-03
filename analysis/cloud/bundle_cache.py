"""Corpus-bundle cache for the multi-domain fit driver.

The assemble phase (BigQuery load -> window -> BOW -> attest -> train/test split)
takes ~5-6 min and re-runs IDENTICALLY every time only a FIT parameter changes
(init, tpn, spectral_topo_order, spectral_proj_dim, max_iter, omega, ...). None of
those touch the corpus. This module caches the assembled ``MultiDomainBundle`` to
a gs:// root keyed by a hash of the CORPUS-affecting config, so a re-run with the
same corpus loads it back (~seconds) instead of re-assembling.

Layout under ``<cache_uri>/<key>/``:
  train.parquet/ , test.parquet/   -- the two feature DataFrames (Spark-native)
  meta/                            -- 1-row JSON DF holding the python metadata
                                      (parent_int, int2cid, cid2int, vocab_maps,
                                      name_by_id, ledger); written LAST as the
                                      completion marker.

Correctness over hit-rate: the key includes every corpus/DAG/assemble argument
(cohort, domains, window, vocab, min_n, hierarchy, rollup, n_bg, tpn, cdr, seed,
...). Pure fit-optimization params are excluded, so effrank/fit iteration hits the
cache while any corpus change (new vocab size, hierarchy fraction, ...) misses it
and re-assembles -- never a stale corpus. Only ``corpus_cache_key`` is pure/tested;
the Spark I/O is cluster-covered.
"""
from __future__ import annotations

import hashlib
import json


def corpus_cache_key(config: dict) -> str:
    """Deterministic short hash of the corpus-affecting config.

    Order-independent (keys sorted) and stable across runs. ``config`` must contain
    only JSON-serializable, corpus-affecting values; the caller is responsible for
    excluding fit-only params (init, topo order, d, max_iter, ...). Nested lists
    (e.g. the anchor set) are included as-is; sort them upstream if their order is
    not meaningful.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def bundle_dir(cache_uri: str, key: str) -> str:
    return f"{cache_uri.rstrip('/')}/{key}"


def _meta_dir(cache_dir: str) -> str:
    return f"{cache_dir}/meta"


def path_exists(spark, path: str) -> bool:
    """Hadoop-FS existence check (works for gs://, hdfs://, file://)."""
    jvm = spark._jvm
    hpath = jvm.org.apache.hadoop.fs.Path(path)
    fs = hpath.getFileSystem(spark._jsc.hadoopConfiguration())
    return bool(fs.exists(hpath))


def cache_exists(spark, cache_dir: str) -> bool:
    """A bundle is present iff its meta dir's _SUCCESS marker exists (written last)."""
    return path_exists(spark, f"{_meta_dir(cache_dir)}/_SUCCESS")


# --- metadata (de)serialization: JSON coerces int keys to str; coerce back ------

def _int_key_dict(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def _bundle_meta(bundle) -> dict:
    """Extract the JSON-able python metadata of a MultiDomainBundle (all but the
    two Spark DataFrames). Int keys survive as JSON strings; read path casts back."""
    return {
        "parent_int": {str(k): [int(x) for x in v]
                       for k, v in bundle.parent_int.items()},
        "int2cid": {str(k): int(v) for k, v in bundle.int2cid.items()},
        "cid2int": {str(k): int(v) for k, v in bundle.cid2int.items()},
        "vocab_maps": [{str(k): int(v) for k, v in vm.items()}
                       for vm in bundle.vocab_maps],
        "name_by_id": {str(k): str(v) for k, v in bundle.name_by_id.items()},
        "ledger": bundle.ledger,
    }


def _meta_to_bundle_fields(meta: dict) -> dict:
    """Inverse of _bundle_meta: JSON-string keys -> int keys."""
    return {
        "parent_int": {int(k): [int(x) for x in v]
                       for k, v in meta["parent_int"].items()},
        "int2cid": _int_key_dict({k: int(v) for k, v in meta["int2cid"].items()}),
        "cid2int": _int_key_dict({k: int(v) for k, v in meta["cid2int"].items()}),
        "vocab_maps": [{int(k): int(v) for k, v in vm.items()}
                       for vm in meta["vocab_maps"]],
        "name_by_id": {int(k): str(v) for k, v in meta["name_by_id"].items()},
        "ledger": meta["ledger"],
    }


def write_bundle(spark, bundle, cache_dir: str) -> None:
    """Persist a bundle under ``cache_dir``. Meta (the completion marker) is written
    LAST, so a crash mid-write leaves the cache absent, not half-populated."""
    bundle.train_df.write.mode("overwrite").parquet(f"{cache_dir}/train.parquet")
    bundle.test_df.write.mode("overwrite").parquet(f"{cache_dir}/test.parquet")
    meta_json = json.dumps(_bundle_meta(bundle))
    (spark.createDataFrame([(meta_json,)], ["meta"])
     .write.mode("overwrite").json(_meta_dir(cache_dir)))


def read_bundle(spark, bundle_cls, cache_dir: str):
    """Reconstruct a ``bundle_cls`` (MultiDomainBundle) from ``cache_dir``."""
    train_df = spark.read.parquet(f"{cache_dir}/train.parquet")
    test_df = spark.read.parquet(f"{cache_dir}/test.parquet")
    meta_json = spark.read.json(_meta_dir(cache_dir)).collect()[0]["meta"]
    fields = _meta_to_bundle_fields(json.loads(meta_json))
    return bundle_cls(train_df=train_df, test_df=test_df, **fields)
