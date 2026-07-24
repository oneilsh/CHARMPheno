"""Write-through cache for the CaseFindingBundle (piece-3 driver).

assemble_case_finding_corpus re-extracts from BigQuery, refits CountVectorizer,
and rebuilds/prunes the DAG on every call. This module caches the resulting
bundle under a content-hash key so repeat runs (same assembly inputs) reload
parquet + a small JSON instead of rebuilding. Mirrors analysis/cloud/
_corpus_cache.py, but the bundle is richer than (bow, vocab, names): it also
carries the DAG maps + ledger, stored as a text-serialized meta.json.

The domain module (charmpheno.omop.case_finding_assembly) stays cache-free; the
driver layer owns caching, exactly as _corpus_load wraps to_bow_dataframe.

Cache layout under {cache_uri}/{key}/:
    train.parquet/   test.parquet/    the split DataFrames
    meta/            a one-column text file holding json.dumps(python fields)
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from charmpheno.omop.case_finding_assembly import CaseFindingBundle


def _module_source_hash(module) -> str:
    try:
        return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "src-unavailable"


def compute_bundle_cache_key(*, source_table, person_mod, vocab_size, min_df,
                             min_patient_count, doc_min_length, prior_obs_days,
                             window_days, disease, min_n, holdout_frac, split_salt=None,
                             n_bg, tpn, cdr=None, strip_mode="test_only",
                             window_mode="forward", lookback_days=365,
                             label_window_days=365) -> str:
    """Stable 16-hex hash of the inputs that determine the assembled bundle.

    Folds cohort_defs_version() plus content hashes of condition_dag +
    case_finding_assembly, so any assembly-logic edit auto-invalidates the cache
    (same discipline as _corpus_cache's cohort_defs). `v` is the manual shape
    version for layout changes unrelated to that source.

    `split_salt` defaults to the assembly's own `_SPLIT_SALT` constant when not
    given, so the key stays consistent with the split the (unparameterized)
    driver actually produces — callers that don't vary the salt (the driver has
    no --split-salt) get a correct, stable key rather than a missing-arg error.

    `disease` is the corpus identity (it determines both the foreground cohort
    and the label-DAG anchors via cohorts.disease_anchors); the anchor concept-
    ids no longer appear in the key directly. cohort_defs_version() folds in the
    registry, so editing a disease's anchor set also invalidates the key.
    """
    from charmpheno.omop import condition_dag, case_finding_assembly
    from charmpheno.omop.cohorts import cohort_defs_version
    if split_salt is None:
        split_salt = case_finding_assembly._SPLIT_SALT
    payload = {
        "source_table": source_table, "person_mod": int(person_mod),
        "vocab_size": vocab_size, "min_df": float(min_df),
        "min_patient_count": int(min_patient_count),
        "doc_min_length": int(doc_min_length), "prior_obs_days": int(prior_obs_days),
        "window_days": int(window_days), "disease": str(disease), "min_n": int(min_n),
        "holdout_frac": float(holdout_frac), "split_salt": int(split_salt),
        "n_bg": int(n_bg), "tpn": int(tpn), "cdr": cdr, "strip_mode": strip_mode,
        "window_mode": window_mode, "lookback_days": int(lookback_days),
        "label_window_days": int(label_window_days),
        "cohort_defs": cohort_defs_version(),
        "dag_src": _module_source_hash(condition_dag),
        "assembly_src": _module_source_hash(case_finding_assembly),
        "v": 4,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _meta_dict(bundle) -> dict:
    # int keys are JSON-stringified; restored on load.
    return {
        "parent_int": {str(c): list(ps) for c, ps in bundle.parent_int.items()},
        "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
        "cid2int": {str(c): i for c, i in bundle.cid2int.items()},
        "vocab_map": {str(c): i for c, i in bundle.vocab_map.items()},
        "name_by_id": {str(c): n for c, n in bundle.name_by_id.items()},
        "ledger": bundle.ledger,
    }


def _restore_meta(meta: dict) -> dict:
    return {
        "parent_int": {int(c): [int(p) for p in ps]
                       for c, ps in meta["parent_int"].items()},
        "int2cid": {int(i): int(c) for i, c in meta["int2cid"].items()},
        "cid2int": {int(c): int(i) for c, i in meta["cid2int"].items()},
        "vocab_map": {int(c): int(i) for c, i in meta["vocab_map"].items()},
        "name_by_id": {int(c): n for c, n in meta["name_by_id"].items()},
        "ledger": meta["ledger"],
    }


def save(spark, bundle, cache_uri, key) -> None:
    """Persist a CaseFindingBundle under {cache_uri}/{key}/ (overwrite mode)."""
    base = f"{cache_uri.rstrip('/')}/{key}"
    bundle.train_df.write.mode("overwrite").parquet(f"{base}/train.parquet")
    bundle.test_df.write.mode("overwrite").parquet(f"{base}/test.parquet")
    meta_json = json.dumps(_meta_dict(bundle))
    (spark.createDataFrame([(meta_json,)], "value STRING")
         .coalesce(1).write.mode("overwrite").text(f"{base}/meta"))


def try_load(spark, cache_uri, key) -> Optional["CaseFindingBundle"]:
    """Return the cached CaseFindingBundle on hit, None on any miss/read failure."""
    from charmpheno.omop.case_finding_assembly import CaseFindingBundle
    base = f"{cache_uri.rstrip('/')}/{key}"
    try:
        train_df = spark.read.parquet(f"{base}/train.parquet")
        test_df = spark.read.parquet(f"{base}/test.parquet")
        meta_rows = spark.read.text(f"{base}/meta").collect()
    except Exception:
        return None
    meta = _restore_meta(json.loads(meta_rows[0]["value"]))
    return CaseFindingBundle(
        train_df=train_df, test_df=test_df, parent_int=meta["parent_int"],
        int2cid=meta["int2cid"], cid2int=meta["cid2int"],
        vocab_map=meta["vocab_map"], name_by_id=meta["name_by_id"],
        ledger=meta["ledger"])


def load_or_build_case_finding_bundle(spark, *, cache_uri=None, _assemble_fn=None,
                                      **assembly_params) -> "CaseFindingBundle":
    """Return the cached bundle on hit; otherwise assemble + write through.

    `assembly_params` are the assemble_case_finding_corpus kwargs (cdr, billing,
    anchor, person_mod, min_n, n_bg, tpn, vocab_size, ...). `_assemble_fn` is a
    seam for testing (defaults to assemble_case_finding_corpus).
    """
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    assemble = _assemble_fn or assemble_case_finding_corpus

    key = None
    if cache_uri:
        key_params = {k: assembly_params[k] for k in (
            "source_table", "person_mod", "vocab_size", "min_df",
            "min_patient_count", "doc_min_length", "prior_obs_days", "window_days",
            "disease", "min_n", "holdout_frac", "split_salt", "n_bg", "tpn", "cdr",
            "strip_mode", "window_mode", "lookback_days", "label_window_days",
        ) if k in assembly_params}
        key = compute_bundle_cache_key(**key_params)
        cached = try_load(spark, cache_uri, key)
        if cached is not None:
            print("[driver]   case-finding-cache HIT", flush=True)
            return cached
        print("[driver]   case-finding-cache MISS, building...", flush=True)

    bundle = assemble(spark, **assembly_params)
    if cache_uri:
        save(spark, bundle, cache_uri, key)
    return bundle
