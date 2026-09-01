"""Write-through cache for the CaseFindingBundle (piece-3 driver).

assemble_case_finding_corpus re-extracts from BigQuery, refits CountVectorizer,
and rebuilds/prunes the DAG on every call. This module caches the resulting
bundle under a content-hash key so repeat runs (same assembly inputs) reload
parquet + a small JSON instead of rebuilding. Mirrors analysis/cloud/
_corpus_cache.py, but the bundle is richer than (bow, vocab, names): it also
carries the DAG maps + ledger, stored as a text-serialized meta.json.

The domain module (charmpheno.omop.case_finding_assembly) stays cache-free; the
driver layer owns caching, exactly as _corpus_load wraps to_bow_dataframe.

Both bundle shapes ride the same layout. A single-domain `CaseFindingBundle` has
one `vocab_map` and a `features` column; the multi-domain `MultiDomainBundle`
(charmpheno.omop.multi_domain — the shape the Mondo path assembles) has a LIST of
`vocab_maps` and `features_0..features_{N-1}` columns. The parquet writes persist
whatever columns the frames carry, so only the meta needed teaching: it stores
`vocab_maps` for the multi-domain bundle and `vocab_map` for the single-domain
one, and `try_load` reconstructs whichever dataclass the meta describes.

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


def compute_bundle_cache_key(*, source_table=None, person_mod, vocab_size, min_df,
                             min_patient_count, doc_min_length, prior_obs_days=None,
                             window_days=None, disease, min_n, holdout_frac,
                             split_salt=None,
                             n_bg, tpn, cdr=None, strip_mode="test_only",
                             window_mode="forward", lookback_days=365,
                             label_window_days=365, emit_labels=False,
                             label_mask_mode="full", multidomain=False,
                             extra_domains=(), index_mode="disease", mondo=False,
                             mondo_version="", mondo_branch="",
                             min_positives=0, dag_collapse=False,
                             dag_collapse_version="", mondo_native=False,
                             mondo_native_version="") -> str:
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

    `multidomain=True` selects the MULTI-DOMAIN corpus identity (the
    `charmpheno.omop.multi_domain` assembler): the per-domain `extra_domains` and
    the `index_mode`, plus `mondo`/`mondo_version`/`mondo_branch`/`min_positives`
    when the label DAG is built from Mondo, plus `dag_collapse`
    (+ `dag_collapse_version`) when the ANCHOR hierarchy is put through the
    exp-0109 splice-to-fixpoint reduction, plus `mondo_native`
    (+ `mondo_native_version` and the two new modules' source hashes) when the
    label space is exp 0110's NATIVE Mondo one. Those fields are folded in
    ONLY on that path, exactly like `emit_labels` below, so every SNOMED key stays
    byte-identical and every existing cache entry stays valid — and the
    `multidomain` marker itself guarantees the new (vocab_maps) meta format can
    never land under a key an old (vocab_map) entry already occupies.

    `source_table`/`prior_obs_days`/`window_days` are required for the
    single-domain key and IGNORED (pinned to null) for the multi-domain one: that
    assembler is lookback-only and reads none of the three, so leaving them in
    would let a knob nobody consulted split the cache.
    """
    from charmpheno.omop import condition_dag, case_finding_assembly
    from charmpheno.omop.cohorts import cohort_defs_version
    if not multidomain and (source_table is None or prior_obs_days is None
                            or window_days is None):
        raise TypeError(
            "compute_bundle_cache_key requires source_table, prior_obs_days and "
            "window_days for the single-domain (SNOMED) key; they are optional "
            "only on the multidomain=True path, which pins them out of the key.")
    if split_salt is None:
        split_salt = case_finding_assembly._SPLIT_SALT
    payload = {
        "source_table": source_table, "person_mod": int(person_mod),
        "vocab_size": vocab_size, "min_df": float(min_df),
        "min_patient_count": int(min_patient_count),
        "doc_min_length": int(doc_min_length),
        "prior_obs_days": None if prior_obs_days is None else int(prior_obs_days),
        "window_days": None if window_days is None else int(window_days),
        "disease": str(disease), "min_n": int(min_n),
        "holdout_frac": float(holdout_frac), "split_salt": int(split_salt),
        "n_bg": int(n_bg), "tpn": int(tpn), "cdr": cdr, "strip_mode": strip_mode,
        "window_mode": window_mode, "lookback_days": int(lookback_days),
        "label_window_days": int(label_window_days),
        "cohort_defs": cohort_defs_version(),
        "dag_src": _module_source_hash(condition_dag),
        "assembly_src": _module_source_hash(case_finding_assembly),
        "v": 4,
    }
    # Only fold the Gated-PC label columns into the key when they are requested,
    # so the default (emit_labels=False) path keeps its existing key and existing
    # dag-placement caches stay valid; a labeled bundle gets a distinct key.
    if emit_labels:
        payload["emit_labels"] = True
        payload["label_mask_mode"] = str(label_mask_mode)
    # Same discipline for the multi-domain / Mondo identity: folded ONLY when that
    # path is in use, so a SNOMED key computed before this existed is unchanged.
    if multidomain:
        from charmpheno.omop import multi_domain
        # Order MATTERS: extra_domains[i] is domain i+1, i.e. it decides which
        # vocabulary features_{i+1} carries. Normalized to a list of str so a tuple
        # and a list of the same domains hash alike (the driver splits a CLI string
        # into a tuple; a manifest round-trips it as a JSON list).
        payload["multidomain"] = True
        payload["extra_domains"] = [str(d) for d in (extra_domains or ())]
        payload["index_mode"] = str(index_mode)
        payload["multi_domain_src"] = _module_source_hash(multi_domain)
        # The lookback-only assembler never reads the forward-window knobs; pin
        # them so a stray --source-table / --window-days cannot split the cache.
        payload["source_table"] = None
        payload["prior_obs_days"] = None
        payload["window_days"] = None
        payload["window_mode"] = "lookback"
        payload["mdv"] = 1            # multi-domain cache-format version
        if mondo:
            # The label DAG is BUILT (mondo_dag), not loaded from concept_ancestor,
            # so its identity is the build inputs plus that module's source — the
            # same auto-invalidation discipline dag_src/assembly_src give the
            # SNOMED path. `mondo_dag` is a sibling driver module, importable
            # wherever this one is.
            import mondo_dag
            payload["mondo"] = True
            payload["mondo_version"] = str(mondo_version)
            payload["mondo_branch"] = str(mondo_branch or "")
            payload["min_positives"] = int(min_positives)
            payload["mondo_src"] = _module_source_hash(mondo_dag)
            # exp 0109's splice-to-fixpoint DAG reduction: a DIFFERENT label DAG
            # (763 fewer nodes at whole-Mondo), hence a different corpus. Folded
            # ONLY when it is switched on — the same discipline as `mondo` itself
            # — so every key from a collapse-OFF run (incl. exp 0104's cached
            # record bundle) is byte-identical to what it was before this existed.
            # That is also why the reduction lives in its own module: `mondo_src`
            # would otherwise move every Mondo key on a comment edit.
            if dag_collapse:
                import mondo_collapse
                payload["dag_collapse"] = True
                # The version string is the citable record of WHICH reduction ran;
                # the source hash is the guard that no one has to remember to bump.
                payload["dag_collapse_version"] = str(dag_collapse_version)
                payload["dag_collapse_src"] = _module_source_hash(mondo_collapse)
            # exp 0110's NATIVE Mondo label space: a different attestation, a
            # different powering rule and a different DAG — a different corpus.
            # Folded ONLY when it is on, for the same reason `dag_collapse` is:
            # exp 0104's and 0109's keys (dag_source=mondo) must not move, and the
            # four hashes pinned in tests/scripts/test_case_finding_cache_mondo.py
            # are the tripwire that proves they did not. `mondo_dag`'s hash keeps
            # riding along on both paths — it is still the module the ANCHOR
            # flavour is built from, and re-scoping it would move the pinned keys.
            # `mondo_collapse` is folded here unconditionally because the native
            # build always applies its splice.
            if mondo_native:
                import mondo_collapse
                import mondo_native_dag
                import mondo_usage_core
                payload["mondo_native"] = True
                payload["mondo_native_version"] = str(mondo_native_version)
                payload["mondo_native_src"] = _module_source_hash(mondo_native_dag)
                payload["mondo_usage_core_src"] = _module_source_hash(
                    mondo_usage_core)
                payload["native_collapse_src"] = _module_source_hash(mondo_collapse)
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _meta_dict(bundle) -> dict:
    """Serialize a bundle's python-side fields (int keys are JSON-stringified).

    A multi-domain bundle carries `vocab_maps` (a LIST, one map per domain) where
    the single-domain one carries `vocab_map`; the key written here is the witness
    `try_load` reads back to decide which dataclass to reconstruct. Exactly one of
    the two is ever written, so an old single-domain entry still restores as it
    always did."""
    meta = {
        "parent_int": {str(c): list(ps) for c, ps in bundle.parent_int.items()},
        "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
        "cid2int": {str(c): i for c, i in bundle.cid2int.items()},
        "name_by_id": {str(c): n for c, n in bundle.name_by_id.items()},
        "ledger": bundle.ledger,
    }
    vocab_maps = getattr(bundle, "vocab_maps", None)
    if vocab_maps is None:
        meta["vocab_map"] = {str(c): i for c, i in bundle.vocab_map.items()}
    else:
        meta["vocab_maps"] = [{str(c): i for c, i in vm.items()} for vm in vocab_maps]
    return meta


def _restore_meta(meta: dict) -> dict:
    out = {
        "parent_int": {int(c): [int(p) for p in ps]
                       for c, ps in meta["parent_int"].items()},
        "int2cid": {int(i): int(c) for i, c in meta["int2cid"].items()},
        "cid2int": {int(c): int(i) for c, i in meta["cid2int"].items()},
        "name_by_id": {int(c): n for c, n in meta["name_by_id"].items()},
        "ledger": meta["ledger"],
    }
    if "vocab_maps" in meta:                 # multi-domain (one map per domain)
        out["vocab_maps"] = [{int(c): int(i) for c, i in vm.items()}
                             for vm in meta["vocab_maps"]]
    else:                                    # single-domain (incl. every old entry)
        out["vocab_map"] = {int(c): int(i) for c, i in meta["vocab_map"].items()}
    return out


def save(spark, bundle, cache_uri, key) -> None:
    """Persist a bundle under {cache_uri}/{key}/ (overwrite mode).

    Single- or multi-domain: the parquet writes take the frames as they are, so a
    multi-domain bundle's features_0..features_{N-1} columns persist with no extra
    handling; only `_meta_dict` distinguishes the two."""
    base = f"{cache_uri.rstrip('/')}/{key}"
    bundle.train_df.write.mode("overwrite").parquet(f"{base}/train.parquet")
    bundle.test_df.write.mode("overwrite").parquet(f"{base}/test.parquet")
    meta_json = json.dumps(_meta_dict(bundle))
    (spark.createDataFrame([(meta_json,)], "value STRING")
         .coalesce(1).write.mode("overwrite").text(f"{base}/meta"))


def try_load(spark, cache_uri, key) -> Optional["CaseFindingBundle"]:
    """Return the cached bundle on hit, None on any miss/read failure.

    The reconstructed TYPE follows the meta: a `vocab_maps` list restores a
    `MultiDomainBundle` (what the Mondo / extra-domain path assembles and what its
    consumers destructure — `bundle.vocab_maps`, per-domain features_i columns),
    anything else a `CaseFindingBundle`. Both carry the same
    parent_int/int2cid/cid2int/name_by_id/ledger bridge, so everything downstream
    of the bundle is shape-agnostic."""
    base = f"{cache_uri.rstrip('/')}/{key}"
    try:
        train_df = spark.read.parquet(f"{base}/train.parquet")
        test_df = spark.read.parquet(f"{base}/test.parquet")
        meta_rows = spark.read.text(f"{base}/meta").collect()
    except Exception:
        return None
    meta = _restore_meta(json.loads(meta_rows[0]["value"]))
    common = dict(train_df=train_df, test_df=test_df,
                  parent_int=meta["parent_int"], int2cid=meta["int2cid"],
                  cid2int=meta["cid2int"], name_by_id=meta["name_by_id"],
                  ledger=meta["ledger"])
    if "vocab_maps" in meta:
        from charmpheno.omop.multi_domain import MultiDomainBundle
        return MultiDomainBundle(vocab_maps=meta["vocab_maps"], **common)
    from charmpheno.omop.case_finding_assembly import CaseFindingBundle
    return CaseFindingBundle(vocab_map=meta["vocab_map"], **common)


# Assembly kwargs that are ALSO cache-key inputs. `extra_domains` / `index_mode`
# are multi-domain-only assembler params; compute_bundle_cache_key ignores them
# unless multidomain=True is folded in (via `_key_extra`), so listing them here
# cannot touch a SNOMED key.
_KEY_PARAM_NAMES = (
    "source_table", "person_mod", "vocab_size", "min_df",
    "min_patient_count", "doc_min_length", "prior_obs_days", "window_days",
    "disease", "min_n", "holdout_frac", "split_salt", "n_bg", "tpn", "cdr",
    "strip_mode", "window_mode", "lookback_days", "label_window_days",
    "emit_labels", "label_mask_mode", "extra_domains", "index_mode",
)


def load_or_build_case_finding_bundle(spark, *, cache_uri=None, _assemble_fn=None,
                                      _key_extra=None,
                                      **assembly_params) -> "CaseFindingBundle":
    """Return the cached bundle on hit; otherwise assemble + write through.

    `assembly_params` are the assembler's kwargs (cdr, billing, disease,
    person_mod, min_n, n_bg, tpn, vocab_size, ...). `_assemble_fn` is the assembler
    seam — a testing hook, and on the multi-domain paths the real dispatch: the
    Mondo driver binds a closure that builds the Mondo DAG + SNOMED-climb provider
    and THEN assembles, so a cache HIT never pays for a hierarchy the cached bundle
    already encodes (see gated_pc_cloud.mondo_assemble_fn).

    `_key_extra` are cache-key inputs that are NOT assembler kwargs — the
    multi-domain/Mondo identity markers (`multidomain`, `mondo`, `mondo_version`,
    `mondo_branch`, `min_positives`, `dag_collapse`, `mondo_native`) that name
    WHICH corpus this is without being something the assembler is called with.
    """
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    assemble = _assemble_fn or assemble_case_finding_corpus

    key = None
    if cache_uri:
        key_params = {k: assembly_params[k] for k in _KEY_PARAM_NAMES
                      if k in assembly_params}
        key_params.update(_key_extra or {})
        key = compute_bundle_cache_key(**key_params)
        cached = try_load(spark, cache_uri, key)
        if cached is not None:
            print("[driver]   case-finding-cache HIT", flush=True)
            return cached
        print("[driver]   case-finding-cache MISS, building...", flush=True)

    bundle = assemble(spark, **assembly_params)
    if cache_uri:
        # Write-through is a best-effort optimization, NOT a correctness step: the
        # freshly-built bundle is already valid in memory. A cache-write failure
        # (bad/unwritable cache_uri, missing GCS bucket, transient I/O) must not
        # abort a run that has already paid the assembly cost — warn and proceed
        # with the in-memory bundle (next run just misses and rebuilds). A partial
        # write is harmless: try_load reads all three artifacts and returns None on
        # any read failure, so a truncated dir simply re-triggers a rebuild.
        try:
            save(spark, bundle, cache_uri, key)
        except Exception as exc:                                # noqa: BLE001
            print(f"[driver]   WARNING: case-finding-cache write to {cache_uri} "
                  f"failed ({type(exc).__name__}: {exc}); proceeding with the "
                  f"in-memory bundle (no cache reuse next run).", flush=True)
    return bundle
