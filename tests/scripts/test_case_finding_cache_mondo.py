"""The MONDO / multi-domain corpus goes through the bundle cache like every other.

Three claims, and the third is the one that cost two recovery attempts:

  1. **The cache format carries a multi-domain bundle.** `MultiDomainBundle` has a
     LIST of per-domain vocabularies and `features_0..features_{N-1}` columns where
     the single-domain `CaseFindingBundle` has one `vocab_map` and one `features`;
     `save`/`try_load` must round-trip either and reconstruct the right dataclass,
     because everything downstream destructures the bundle it gets.

  2. **No existing key moves.** The Mondo/multi-domain identity is folded into the
     key ONLY on that path (the `emit_labels` discipline), so every SNOMED key is
     byte-identical to what it was and every cached SNOMED bundle stays findable.
     The three hashes pinned below were computed BEFORE the change.

  3. **A saved fit can be re-scored on a cluster whose cache is empty.** The
     re-readout recomputes the same key the fit used, and on a MISS re-assembles
     the corpus through the same seam the fit uses instead of erroring — after
     checking that what came back is still the corpus the λ was fit against.

Local-Spark for the round-trip and the seam; pure for the key and the drift gate.
"""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# PySpark workers inherit PYTHONPATH, not the driver's sys.path (same note as
# tests/scripts/test_readout_integration.py). Set before the session fixture builds
# the context.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
                str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import _case_finding_cache as ccache  # noqa: E402
import gated_pc_cloud as gpc  # noqa: E402
import gated_pc_readout as gpr  # noqa: E402


# --------------------------------------------------------------------------- #
# Key stability: the SNOMED keys, pinned from before the change.               #
# --------------------------------------------------------------------------- #
_SNOMED_BASE = dict(
    source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
    min_patient_count=20, doc_min_length=0, prior_obs_days=365, window_days=365,
    disease="diabetes", min_n=50, holdout_frac=0.2, split_salt=20260716,
    n_bg=2, tpn=1, cdr="p.d")
# Computed against the pre-change compute_bundle_cache_key. A cached SNOMED bundle
# lives under exactly these; if one of them moves, every cache entry in every
# bucket silently becomes unreachable and the next run rebuilds from BigQuery.
_SNOMED_KEY = "d658ce0a9a7425dd"
_SNOMED_LABELED_KEY = "3cf6c7aac6140393"          # emit_labels + closure mask
_SNOMED_LOOKBACK_KEY = "275c8e6a76283e86"         # lookback windowing + strip both


def test_existing_snomed_keys_are_byte_identical():
    """The whole compatibility claim in three hashes."""
    k = ccache.compute_bundle_cache_key
    assert k(**_SNOMED_BASE) == _SNOMED_KEY
    assert k(**_SNOMED_BASE, emit_labels=True,
             label_mask_mode="closure") == _SNOMED_LABELED_KEY
    assert k(**dict(_SNOMED_BASE, window_mode="lookback", strip_mode="both",
                    lookback_days=730, label_window_days=180)) == _SNOMED_LOOKBACK_KEY


def test_multidomain_flag_alone_changes_the_key():
    """A multi-domain corpus can never collide with a single-domain one — which is
    also what keeps the new (vocab_maps) meta format out of an old entry's dir."""
    k = ccache.compute_bundle_cache_key
    assert k(**_SNOMED_BASE, multidomain=True) != _SNOMED_KEY


_MD_BASE = dict(_SNOMED_BASE, multidomain=True, extra_domains=("drug",),
                index_mode="population", mondo=True, mondo_version="2026-06-02",
                mondo_branch="", min_positives=100)
# The MONDO key with the exp-0109 collapse OFF, pinned the same way the SNOMED
# keys above are and for the same reason: exp 0104's record bundle (~20 min of
# BigQuery to rebuild) lives under it. It folds `_module_source_hash(mondo_dag)`,
# so this hash is a deliberate TRIPWIRE — any edit to `analysis/cloud/mondo_dag.py`
# moves it and orphans every cached Mondo bundle in every bucket. That may be the
# right call (an actual hierarchy change SHOULD invalidate), but it must be a
# decision: re-pin this only alongside a note saying why the caches were dropped.
# It is also why the 0109 reduction lives in `mondo_collapse.py` instead.
_MONDO_KEY_NO_COLLAPSE = "ca958995cc1cfb17"


def test_mondo_key_with_collapse_off_is_byte_identical():
    """exp 0109 is OPT-IN, so a spec that does not ask for the collapse must key
    exactly where it keyed before the flag existed."""
    k = ccache.compute_bundle_cache_key
    assert k(**_MD_BASE) == _MONDO_KEY_NO_COLLAPSE
    assert k(**_MD_BASE, dag_collapse=False) == _MONDO_KEY_NO_COLLAPSE
    # ...and the version string is inert while the flag is off (nothing to name).
    assert k(**_MD_BASE, dag_collapse=False,
             dag_collapse_version="whatever") == _MONDO_KEY_NO_COLLAPSE


def test_dag_collapse_on_is_a_different_corpus():
    """The collapsed DAG has ~763 fewer label nodes at whole-Mondo — a different
    C, a different K, different label/frontier columns. It cannot share a bundle."""
    k = ccache.compute_bundle_cache_key
    on = k(**_MD_BASE, dag_collapse=True, dag_collapse_version="splice-fixpoint-v1")
    assert on != _MONDO_KEY_NO_COLLAPSE


def test_dag_collapse_version_is_folded_into_the_key():
    """A reduction whose OUTPUT changes must not be served a bundle built by the
    old one; bumping the version string is the manual lever that guarantees it."""
    k = ccache.compute_bundle_cache_key
    v1 = k(**_MD_BASE, dag_collapse=True, dag_collapse_version="splice-fixpoint-v1")
    v2 = k(**_MD_BASE, dag_collapse=True, dag_collapse_version="splice-fixpoint-v2")
    assert v1 != v2


def test_dag_collapse_does_not_leak_into_the_snomed_key():
    """The reduction names Mondo CLASS nodes; the SNOMED path has none, and its
    keys must stay frozen even if a caller passes the flag by accident."""
    k = ccache.compute_bundle_cache_key
    assert k(**_SNOMED_BASE, dag_collapse=True,
             dag_collapse_version="splice-fixpoint-v1") == _SNOMED_KEY


def test_multidomain_cache_key_threads_dag_collapse_from_the_spec():
    """The spec -> key_extra -> compute_bundle_cache_key path (what the fit and the
    re-readout both call), not just the raw key function."""
    base = dict(_MONDO_SPEC)
    off = gpc.multidomain_cache_key(base)
    on = gpc.multidomain_cache_key(dict(base, dag_collapse=True))
    assert off == gpc.multidomain_cache_key(dict(base, dag_collapse=False))
    assert off != on
    # a spec that predates the field behaves exactly like collapse OFF
    legacy = {k_: v for k_, v in base.items() if k_ != "dag_collapse"}
    assert gpc.multidomain_cache_key(legacy) == off


# --------------------------------------------------------------------------- #
# exp 0110: the NATIVE Mondo label space is its own corpus, and folds only when  #
# it is selected — the same discipline `dag_collapse` follows.                   #
# --------------------------------------------------------------------------- #
def test_mondo_native_is_a_different_corpus_from_the_anchor_hierarchy():
    """Different attestation (frontier, not every powered ancestor), different
    powering (closure support, not terminal counts) and a different label DAG
    (Mondo's own hierarchy, Mondo ids) — so a different C, a different K and
    different label/frontier columns. It cannot share a bundle with exp 0104's."""
    k = ccache.compute_bundle_cache_key
    native = k(**_MD_BASE, mondo_native=True,
               mondo_native_version="native-mondo-v1")
    assert native != _MONDO_KEY_NO_COLLAPSE
    assert native != k(**_MD_BASE, dag_collapse=True,
                       dag_collapse_version="splice-fixpoint-v1")


def test_mondo_native_off_leaves_every_existing_key_byte_identical():
    """The compatibility half. `mondo_native=False` (and an inert version string)
    must reproduce the pinned hashes exactly, on both the Mondo and SNOMED paths —
    which is what keeps exp 0104's and 0109's cached bundles findable."""
    k = ccache.compute_bundle_cache_key
    assert k(**_MD_BASE, mondo_native=False) == _MONDO_KEY_NO_COLLAPSE
    assert k(**_MD_BASE, mondo_native=False,
             mondo_native_version="whatever") == _MONDO_KEY_NO_COLLAPSE
    assert k(**_SNOMED_BASE, mondo_native=True,
             mondo_native_version="native-mondo-v1") == _SNOMED_KEY


def test_mondo_native_version_is_folded_into_the_key():
    """The manual lever: a build whose OUTPUT changes bumps the version, and a
    bundle from the old construction can never be served to the new one."""
    k = ccache.compute_bundle_cache_key
    v1 = k(**_MD_BASE, mondo_native=True, mondo_native_version="native-mondo-v1")
    v2 = k(**_MD_BASE, mondo_native=True, mondo_native_version="native-mondo-v2")
    assert v1 != v2


def test_mondo_native_key_threads_from_the_spec():
    """The spec -> key_extra -> compute_bundle_cache_key path the fit and the
    re-readout both call, and the marker that routes it: a mondo_native spec must
    NOT be keyed as a SNOMED corpus (that is a guaranteed MISS plus a ~20-minute
    rebuild of the wrong thing)."""
    native_spec = dict(_MONDO_SPEC, dag_source="mondo_native")
    assert gpr.spec_is_multidomain(native_spec)
    key = gpc.multidomain_cache_key(native_spec)
    assert key != gpc.multidomain_cache_key(_MONDO_SPEC)
    # and it is stable under the flag exp 0109 owns, which the native path pins off
    assert key == gpc.multidomain_cache_key(
        dict(native_spec, dag_collapse=False))


def test_mondo_native_folds_its_own_module_sources():
    """Auto-invalidation, the discipline `dag_src`/`mondo_src` give every other
    path: an edit to either NEW module changes the key, so a stale native bundle
    can never be served. Checked by monkeypatching the hash helper rather than by
    pinning a hash, because these modules are expected to keep changing."""
    import mondo_native_dag
    import mondo_usage_core

    k = ccache.compute_bundle_cache_key
    base = k(**_MD_BASE, mondo_native=True,
             mondo_native_version="native-mondo-v1")
    real = ccache._module_source_hash
    for target in (mondo_native_dag, mondo_usage_core):
        ccache._module_source_hash = (
            lambda m, _t=target: "moved" if m is _t else real(m))
        try:
            moved = k(**_MD_BASE, mondo_native=True,
                      mondo_native_version="native-mondo-v1")
        finally:
            ccache._module_source_hash = real
        assert moved != base, f"{target.__name__} is not folded into the key"


def test_native_spec_mismatch_catches_a_contradicting_override():
    """A native fit records `dag_source: mondo_native`; its int2cid values are
    plain ints, so `mondo_spec_mismatch`'s `MONDO:`-prefix witness cannot see it
    and the manifest field is the witness instead."""
    m = {"corpus_manifest": {"dag_source": "mondo_native"}}
    assert gpr.native_spec_mismatch({"dag_source": "snomed"}, m)
    assert gpr.native_spec_mismatch({"dag_source": "mondo"}, m)
    assert not gpr.native_spec_mismatch({"dag_source": "mondo_native"}, m)
    # a non-native fit is none of this function's business
    assert not gpr.native_spec_mismatch(
        {"dag_source": "snomed"}, {"corpus_manifest": {"dag_source": "mondo"}})


def test_readout_recovers_a_native_spec_from_the_manifest():
    """The recovery template: the re-readout reads `dag_source` back, keeps the
    Mondo build inputs, pins `dag_collapse` off (the splice is intrinsic to the
    native build, so asking for it again would double-apply), and recomputes the
    fit's own key."""
    m = _mondo_manifest()
    m["dag_source"] = m["corpus_manifest"]["dag_source"] = "mondo_native"
    m["corpus_manifest"]["dag_collapse"] = True     # a stale/incoherent field
    spec = gpr.corpus_spec_from_manifest(m)
    assert spec["dag_source"] == "mondo_native"
    assert spec["dag_collapse"] is False
    assert spec["index_mode"] == "population" and spec["min_n"] == 0
    assert spec["min_positives"] == 100
    assert (gpr.bundle_key_from_manifest(m)
            == gpc.multidomain_cache_key(spec))


@pytest.mark.parametrize("field,val", [
    ("extra_domains", ("drug", "procedure")),
    ("index_mode", "disease"),
    ("mondo_version", "2026-07-01"),
    ("mondo_branch", "MONDO:0004995"),
    ("min_positives", 50),
    ("mondo", False),
])
def test_every_mondo_field_moves_the_key(field, val):
    """Each of these names a DIFFERENT corpus (different DAG, different index,
    different domains), so none may alias onto another's cached bundle."""
    k = ccache.compute_bundle_cache_key
    assert k(**dict(_MD_BASE, **{field: val})) != k(**_MD_BASE)


def test_extra_domains_order_matters_but_container_type_does_not():
    """extra_domains[i] decides which vocabulary features_{i+1} carries, so the
    ORDER is corpus identity — while a tuple (the driver splits a CLI string) and a
    list (a manifest round-trips JSON) of the same domains are the same corpus."""
    k = ccache.compute_bundle_cache_key
    ab = k(**dict(_MD_BASE, extra_domains=("drug", "procedure")))
    ba = k(**dict(_MD_BASE, extra_domains=("procedure", "drug")))
    assert ab != ba
    assert ab == k(**dict(_MD_BASE, extra_domains=["drug", "procedure"]))


def test_multidomain_key_ignores_the_forward_window_knobs():
    """The multi-domain assembler is lookback-only and never reads source_table /
    prior_obs_days / window_days, so they must not be able to split its cache."""
    k = ccache.compute_bundle_cache_key
    base = k(**_MD_BASE)
    assert k(**dict(_MD_BASE, source_table="condition_occurrence")) == base
    assert k(**dict(_MD_BASE, prior_obs_days=0, window_days=1)) == base
    # ...and they may be omitted entirely (a rebuild spec need not invent them).
    lean = {k_: v for k_, v in _MD_BASE.items()
            if k_ not in ("source_table", "prior_obs_days", "window_days")}
    assert k(**lean) == base


def test_single_domain_key_still_requires_the_forward_window_knobs():
    """Making them optional must not let a SNOMED caller silently drop one."""
    lean = {k_: v for k_, v in _SNOMED_BASE.items() if k_ != "source_table"}
    with pytest.raises(TypeError, match="source_table"):
        ccache.compute_bundle_cache_key(**lean)


# --------------------------------------------------------------------------- #
# The DOC-SPEC hole (audit seam 4 / R5.3): the doc unit is hard-coded in the      #
# assembler and in the driver's provider construction and was absent from every   #
# cache key, so a driver-side doc-unit change would have produced a DIFFERENT     #
# corpus under a BYTE-IDENTICAL key. Folded now — but only when it differs from   #
# today's constant, which is what keeps every hash above where it is.             #
# --------------------------------------------------------------------------- #
def test_doc_spec_default_leaves_all_four_pinned_hashes_byte_identical():
    """The deliverable of the fix. Passing today's doc spec EXPLICITLY must key
    exactly where omitting it does, on every pinned hash — otherwise closing a
    silent-wrongness hole would orphan every cached bundle in the repo."""
    k = ccache.compute_bundle_cache_key
    d = ccache.DEFAULT_DOC_SPEC
    assert k(**_SNOMED_BASE, doc_spec=d) == _SNOMED_KEY
    assert k(**_SNOMED_BASE, emit_labels=True, label_mask_mode="closure",
             doc_spec=d) == _SNOMED_LABELED_KEY
    assert k(**dict(_SNOMED_BASE, window_mode="lookback", strip_mode="both",
                    lookback_days=730, label_window_days=180),
             doc_spec=d) == _SNOMED_LOOKBACK_KEY
    assert k(**_MD_BASE, doc_spec=d) == _MONDO_KEY_NO_COLLAPSE


def test_a_different_doc_spec_moves_the_key_on_both_paths():
    """The hole itself: an episode/patient-year corpus is a DIFFERENT corpus (a
    different doc unit means different documents, a different vocabulary, and
    different base rates) and must never be served a patient-cohort bundle."""
    k = ccache.compute_bundle_cache_key
    assert k(**_SNOMED_BASE, doc_spec="patient_year") != _SNOMED_KEY
    assert k(**_MD_BASE, doc_spec="episode") != _MONDO_KEY_NO_COLLAPSE
    # ...and two different non-default units do not alias onto each other either.
    assert (k(**_MD_BASE, doc_spec="episode")
            != k(**_MD_BASE, doc_spec="patient_year"))


def test_doc_spec_identity_is_read_off_the_class_the_driver_builds():
    """The token is derived, not written down, so swapping the driver's doc spec
    moves the key without anyone remembering to bump a string."""
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    assert gpc.doc_spec_identity() == PatientCohortDocSpec().name
    assert gpc.doc_spec_identity() == ccache.DEFAULT_DOC_SPEC


def test_doc_spec_threads_from_the_spec_and_a_legacy_spec_means_the_default():
    """spec -> key_extra -> compute_bundle_cache_key, the path the fit and the
    re-readout both take; and a spec written before the field existed keys
    byte-identically to one carrying today's value (the `dag_collapse` legacy
    pattern at :143)."""
    base = dict(_MONDO_SPEC)
    legacy = {k_: v for k_, v in base.items() if k_ != "doc_spec"}
    assert gpc.multidomain_cache_key(legacy) == gpc.multidomain_cache_key(
        dict(base, doc_spec=ccache.DEFAULT_DOC_SPEC))
    assert (gpc.multidomain_cache_key(dict(base, doc_spec="episode"))
            != gpc.multidomain_cache_key(legacy))


def test_readout_recovers_the_doc_spec_from_the_manifest():
    """Manifest -> spec -> key, on both the multi-domain and single-domain routes.
    A manifest predating the field means the default (every corpus in the repo was
    assembled under it), so an old run still recomputes its own key."""
    m = _mondo_manifest()
    assert gpr.corpus_spec_from_manifest(m)["doc_spec"] == ccache.DEFAULT_DOC_SPEC
    m["corpus_manifest"]["doc_spec"] = "episode"
    assert gpr.corpus_spec_from_manifest(m)["doc_spec"] == "episode"
    assert gpr.bundle_key_from_manifest(m) != gpr.bundle_key_from_manifest(
        _mondo_manifest())


# --------------------------------------------------------------------------- #
# Cache format: a multi-domain bundle round-trips.                             #
# --------------------------------------------------------------------------- #
_VOCAB_MAPS = [{101: 0, 102: 1, 103: 2, 104: 3}, {201: 0, 202: 1, 203: 2}]
_PARENT_INT = {0: [], 1: [0], 2: [0]}
_INT2CID = {0: -1, 1: 1001, 2: 1002}
_CID2INT = {-1: 0, 1001: 1, 1002: 2}
_NAME_BY_ID = {-1: "mondo disease root", 1001: "node a", 1002: "node b"}
_LEDGER = {"K_nodes": 2, "kept": 2, "dropped": 0}


def _md_frame(spark, n, offset=0):
    """The frame the multi-domain assembler emits: per-domain BOW columns plus the
    condition-only frontier and the Step-A dense label/mask."""
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, DoubleType, LongType, StringType,
                                   StructField, StructType)
    schema = StructType([
        StructField("doc_id", LongType(), False),
        StructField("person_id", LongType(), False),
        StructField("features_0", VectorUDT(), False),
        StructField("features_1", VectorUDT(), False),
        StructField("frontier", ArrayType(LongType()), False),
        StructField("source_cohort", StringType(), False),
        StructField("label", ArrayType(DoubleType()), False),
        StructField("labelMask", ArrayType(DoubleType()), False),
    ])
    rng = np.random.default_rng(offset + 7)
    rows = []
    for d in range(n):
        pid = offset + d
        f0 = Vectors.dense(rng.integers(0, 3, size=4).astype(float))
        f1 = Vectors.dense(rng.integers(0, 3, size=3).astype(float))
        node = 1 + (d % 2)
        rows.append((int(pid), int(pid), f0, f1, [int(node)], "general",
                     [1.0, float(node == 1), float(node == 2)], [1.0, 1.0, 1.0]))
    return spark.createDataFrame(rows, schema)


def _md_bundle(spark, n_train=12, n_test=8):
    from charmpheno.omop.multi_domain import MultiDomainBundle
    return MultiDomainBundle(
        train_df=_md_frame(spark, n_train), test_df=_md_frame(spark, n_test, 1000),
        parent_int=dict(_PARENT_INT), int2cid=dict(_INT2CID),
        cid2int=dict(_CID2INT), vocab_maps=[dict(vm) for vm in _VOCAB_MAPS],
        name_by_id=dict(_NAME_BY_ID), ledger=dict(_LEDGER))


@pytest.mark.slow
def test_multidomain_bundle_round_trips_through_the_cache(spark, tmp_path):
    """save -> try_load gives back a MultiDomainBundle, field for field, with every
    per-domain feature column intact."""
    from charmpheno.omop.multi_domain import MultiDomainBundle
    bundle = _md_bundle(spark)
    uri = f"file://{tmp_path}/cache"
    ccache.save(spark, bundle, uri, "mdkey")
    loaded = ccache.try_load(spark, uri, "mdkey")

    assert isinstance(loaded, MultiDomainBundle)
    assert loaded.vocab_maps == _VOCAB_MAPS          # a LIST, int keys restored
    assert loaded.parent_int == _PARENT_INT
    assert loaded.int2cid == _INT2CID
    assert loaded.cid2int == _CID2INT
    assert loaded.name_by_id == _NAME_BY_ID
    assert loaded.ledger == _LEDGER
    for got, want in ((loaded.train_df, bundle.train_df),
                      (loaded.test_df, bundle.test_df)):
        assert set(got.columns) == set(want.columns)
        assert "features_0" in got.columns and "features_1" in got.columns
        a = {r["person_id"]: (list(r["features_0"]), list(r["features_1"]),
                              list(r["label"])) for r in got.collect()}
        b = {r["person_id"]: (list(r["features_0"]), list(r["features_1"]),
                              list(r["label"])) for r in want.collect()}
        assert a == b


@pytest.mark.slow
def test_single_domain_bundle_round_trip_is_unchanged(spark, tmp_path):
    """The old shape still writes the old meta (`vocab_map`, no `vocab_maps`) and
    still comes back a CaseFindingBundle — nothing about an existing entry moved."""
    from charmpheno.omop.case_finding_assembly import CaseFindingBundle
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (LongType, StructField, StructType)
    schema = StructType([StructField("person_id", LongType(), False),
                         StructField("features", VectorUDT(), False)])
    df = spark.createDataFrame(
        [(int(d), Vectors.dense([float(d), 1.0])) for d in range(5)], schema)
    bundle = CaseFindingBundle(
        train_df=df, test_df=df, parent_int=dict(_PARENT_INT),
        int2cid=dict(_INT2CID), cid2int=dict(_CID2INT), vocab_map={101: 0, 102: 1},
        name_by_id=dict(_NAME_BY_ID), ledger=dict(_LEDGER))
    uri = f"file://{tmp_path}/cache1d"
    ccache.save(spark, bundle, uri, "sdkey")

    meta = json.loads(
        spark.read.text(f"{uri}/sdkey/meta").collect()[0]["value"])
    assert "vocab_map" in meta and "vocab_maps" not in meta

    loaded = ccache.try_load(spark, uri, "sdkey")
    assert isinstance(loaded, CaseFindingBundle)
    assert loaded.vocab_map == {101: 0, 102: 1}
    assert not hasattr(loaded, "vocab_maps")


def test_restore_meta_reads_a_legacy_single_vocab_map():
    """Entries written before this change have no `vocab_maps` key at all; they must
    keep restoring as single-domain."""
    legacy = {"parent_int": {"1": ["0"]}, "int2cid": {"0": "-1"},
              "cid2int": {"-1": "0"}, "vocab_map": {"101": "0"},
              "name_by_id": {"-1": "root"}, "ledger": {"K_nodes": 1}}
    out = ccache._restore_meta(legacy)
    assert out["vocab_map"] == {101: 0}
    assert "vocab_maps" not in out
    assert out["parent_int"] == {1: [0]}


# --------------------------------------------------------------------------- #
# The seam: load-or-build for the Mondo path.                                  #
# --------------------------------------------------------------------------- #
_MONDO_SPEC = {
    "dag_source": "mondo", "disease": "rare6", "cdr": "p.d", "billing": "bp",
    "source_table": "condition_era", "extra_domains": ["drug"],
    "index_mode": "population", "person_mod": 1, "vocab_size": 5000, "min_df": 20,
    "min_patient_count": 20, "doc_min_length": 10, "min_n": 0, "holdout_frac": 0.2,
    "n_bg": 8, "tpn": 1, "strip_mode": "test_only", "lookback_days": 365,
    "label_window_days": 365, "label_mask_mode": "full", "emit_labels": True,
    "window_mode": "lookback", "prior_obs_days": 0, "window_days": 0,
    "mondo_version": "2026-06-02", "mondo_branch": "", "min_positives": 100,
    "mondo_cache_dir": "data/mondo", "dag_collapse": False,
    "doc_spec": "patient_cohort",
}


class _FakeDag:
    names = {-1: "root"}

    def nodes(self):
        return [-1]


def _fake_build_inputs(spark, **kw):
    """Stands in for build_mondo_fit_inputs — the ~5 minutes of BigQuery whose
    whole point here is that a cache HIT never reaches it."""
    return _FakeDag(), object(), {1001, 1002}, {1001: 400, 1002: 250}, {
        "n_classes": 1}


@pytest.mark.slow
def test_mondo_load_or_build_misses_once_then_hits(spark, tmp_path, capsys):
    """The behaviour the mondo path never had: build once, reload forever.

    The DAG build lives INSIDE the assemble seam, so the HIT must not reach it —
    that is the ~5 min of BigQuery every mondo fit used to pay unconditionally.
    """
    built = _md_bundle(spark)
    calls = {"build": 0, "assemble": 0, "on_inputs": 0}

    def _build(spark_, **kw):
        calls["build"] += 1
        assert kw["mondo_version"] == "2026-06-02"
        assert kw["min_positives"] == 100
        assert kw["branch_root"] is None           # '' means whole Mondo
        return _fake_build_inputs(spark_, **kw)

    def _assemble(spark_, **kw):
        calls["assemble"] += 1
        # the seam must hand the assembler the DAG + provider it just built
        assert isinstance(kw["before_dag"], _FakeDag)
        assert kw["attested_provider"] is not None
        assert kw["index_mode"] == "population" and kw["min_n"] == 0
        assert kw["extra_domains"] == ("drug",)
        return built

    def _on_inputs(*, count_of, terminal_cids, reduced):
        calls["on_inputs"] += 1
        assert count_of == {1001: 400, 1002: 250}

    uri = f"file://{tmp_path}/mondo-cache"
    b1 = gpc.multidomain_load_or_build(
        spark, _MONDO_SPEC, cache_uri=uri, on_inputs=_on_inputs,
        _build_inputs=_build, _assemble=_assemble)
    assert calls == {"build": 1, "assemble": 1, "on_inputs": 1}
    assert "case-finding-cache MISS" in capsys.readouterr().out

    b2 = gpc.multidomain_load_or_build(
        spark, _MONDO_SPEC, cache_uri=uri, on_inputs=_on_inputs,
        _build_inputs=_build, _assemble=_assemble)
    assert calls == {"build": 1, "assemble": 1, "on_inputs": 1}, \
        "a HIT must not rebuild the Mondo DAG or re-assemble"
    assert "case-finding-cache HIT" in capsys.readouterr().out
    assert b2.vocab_maps == b1.vocab_maps == _VOCAB_MAPS
    assert b2.parent_int == _PARENT_INT
    assert {r["person_id"] for r in b2.train_df.collect()} == \
        {r["person_id"] for r in built.train_df.collect()}


def test_head_starvation_probe_degrades_without_the_power_counts():
    """A cache HIT skips the Mondo power-count, so the diag-only probe has no
    per-terminal +counts. It must still read the fitted head (its actual subject)
    and SAY the counts are missing rather than print a wall of zeros."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout(_PARENT_INT, n_bg=2, tpn=1)
    w = np.zeros((3, lay.K))
    w[1, :] = 1.0
    with_counts = gpc.per_node_head_report(w, lay, 3, _INT2CID, {1001: 400, 1002: 9})
    without = gpc.per_node_head_report(w, lay, 3, _INT2CID, {})
    assert "median +ct=" in with_counts
    assert "median +ct=" not in without
    assert "per-terminal +counts UNAVAILABLE" in without
    assert "heads DEAD" in without and "|w_c|:" in without


# --------------------------------------------------------------------------- #
# The re-readout: same key, load-or-BUILD, and the drift gate.                 #
# --------------------------------------------------------------------------- #
def _mondo_manifest(**over):
    """What the driver writes for a mondo fit: corpus_manifest IS the corpus spec."""
    cm = {k: v for k, v in _MONDO_SPEC.items()}
    cm["cache_uri"] = None
    cm["int2cid"] = {str(i): c for i, c in _INT2CID.items()}
    cm["name_by_id"] = {str(c): n for c, n in _NAME_BY_ID.items()}
    cm["domain_vocab_sizes"] = [len(vm) for vm in _VOCAB_MAPS]
    m = {"model_class": "gated_pc", "C": 3, "K": 5, "weight_y": 1.0,
         "max_iter": 4, "min_label_count": 0, "dag_source": "mondo",
         "extra_domains": ["drug"], "disease": "rare6", "min_n": 50,
         "strip_mode": "test_only", "label_mask_mode": "full",
         "window_mode": "lookback", "lookback_days": 365, "label_window_days": 365,
         "n_bg": 8, "tpn": 1, "corpus_manifest": cm}
    m.update(over)
    return m


def test_readout_key_for_a_mondo_manifest_matches_the_fit_key():
    """The re-readout must land on the byte-identical key the fit stored under —
    computed here three ways: from the manifest, from the spec, and directly."""
    m = _mondo_manifest()
    from_manifest = gpr.bundle_key_from_manifest(m)
    spec = gpr.corpus_spec_from_manifest(m)
    assert from_manifest == gpc.multidomain_cache_key(spec)
    assert from_manifest == ccache.compute_bundle_cache_key(
        person_mod=1, vocab_size=5000, min_df=20, min_patient_count=20,
        doc_min_length=10, disease="rare6", min_n=0, holdout_frac=0.2, n_bg=8,
        tpn=1, cdr="p.d", strip_mode="test_only", lookback_days=365,
        label_window_days=365, emit_labels=True, label_mask_mode="full",
        multidomain=True, extra_domains=("drug",), index_mode="population",
        mondo=True, mondo_version="2026-06-02", mondo_branch="", min_positives=100)


def test_readout_key_for_a_mondo_manifest_is_not_the_snomed_key():
    """Before this, the mondo manifest computed a SNOMED key — which is why the
    re-readout always missed (nothing was ever stored under it either)."""
    m = _mondo_manifest()
    snomed = ccache.compute_bundle_cache_key(**{k: gpr.corpus_spec_from_manifest(m)[k]
                                                for k in gpr._SNOMED_KEY_KEYS})
    assert gpr.bundle_key_from_manifest(m) != snomed


def test_readout_recovers_the_mondo_min_n_from_an_older_manifest():
    """A mondo fit passes min_n=0 (its DAG is already powered) while the top-level
    manifest records the CLI default. A manifest written before corpus_manifest
    carried the effective value must still recompute the fit's own key."""
    m = _mondo_manifest()
    del m["corpus_manifest"]["min_n"]
    assert gpr.corpus_spec_from_manifest(m)["min_n"] == 0
    assert gpr.bundle_key_from_manifest(m) == gpr.bundle_key_from_manifest(
        _mondo_manifest())


def test_readout_cli_overrides_supply_a_missing_mondo_spec():
    """An older mondo manifest has no Mondo build inputs at all; the overrides are
    what make such a run recoverable (they are key inputs, so a wrong one misses)."""
    m = _mondo_manifest()
    for f in ("mondo_version", "mondo_branch", "min_positives", "billing"):
        del m["corpus_manifest"][f]
    spec = gpr.corpus_spec_from_manifest(
        m, mondo_version="2026-06-02", min_positives=100, billing="bp")
    assert spec["mondo_version"] == "2026-06-02" and spec["billing"] == "bp"
    assert gpr.bundle_key_from_manifest(
        m, mondo_version="2026-06-02", min_positives=100) == \
        gpr.bundle_key_from_manifest(_mondo_manifest())


def test_dag_source_override_recovers_a_manifest_that_predates_the_field():
    """A mondo run written before the manifest recorded `dag_source` looks like an
    extra-domains SNOMED run — which would key (and REBUILD) as the wrong corpus.
    `--dag-source mondo` is what makes such a run recoverable; without it the
    rebuilt DAG would differ and the drift gate would refuse to score it."""
    old = _mondo_manifest()
    del old["dag_source"]
    del old["corpus_manifest"]["dag_source"]
    del old["corpus_manifest"]["index_mode"]
    del old["corpus_manifest"]["min_n"]
    assert gpr.corpus_spec_from_manifest(old)["dag_source"] == "snomed"
    assert gpr.bundle_key_from_manifest(old) != gpr.bundle_key_from_manifest(
        _mondo_manifest())
    fixed = gpr.corpus_spec_from_manifest(old, dag_source="mondo")
    assert fixed["index_mode"] == "population" and fixed["min_n"] == 0
    assert gpr.bundle_key_from_manifest(old, dag_source="mondo") == \
        gpr.bundle_key_from_manifest(_mondo_manifest())


def test_readout_parses_the_rebuild_flags():
    a = gpr.build_parser().parse_args(["--run-dir", "/runs/0104-x"])
    assert a.no_rebuild is False and a.dag_source is None and a.billing is None
    assert a.cache_write == "on"
    assert a.dag_collapse is None                  # tri-state: defer to the manifest
    b = gpr.build_parser().parse_args(
        ["--run-dir", "/runs/0104-x", "--no-rebuild", "--dag-source", "mondo",
         "--billing", "proj", "--min-positives", "100", "--dag-collapse", "on"])
    assert b.no_rebuild is True and b.dag_source == "mondo"
    assert b.billing == "proj" and b.min_positives == 100
    assert b.dag_collapse == "on"


def test_readout_reads_dag_collapse_from_the_manifest_and_lets_the_cli_win():
    """Same manifest-default + CLI-override shape as every other Mondo build input:
    a 0109 fit records `dag_collapse: true` and the re-readout rebuilds the SAME
    collapsed DAG; a manifest predating the field means off."""
    m = _mondo_manifest()
    m["corpus_manifest"]["dag_collapse"] = True
    assert gpr.corpus_spec_from_manifest(m)["dag_collapse"] is True
    # the CLI override wins in both directions (a fit whose manifest lies/omits)
    assert gpr.corpus_spec_from_manifest(m, dag_collapse=False)["dag_collapse"] is False
    old = _mondo_manifest()
    del old["corpus_manifest"]["dag_collapse"]
    assert gpr.corpus_spec_from_manifest(old)["dag_collapse"] is False
    assert gpr.corpus_spec_from_manifest(old, dag_collapse=True)["dag_collapse"] is True
    # ...and it lands in the key, so a wrong value MISSES rather than mis-scores.
    assert gpr.bundle_key_from_manifest(m) != gpr.bundle_key_from_manifest(old)
    assert gpr.bundle_key_from_manifest(old) == gpr.bundle_key_from_manifest(
        _mondo_manifest())


def test_snomed_manifest_still_takes_the_single_domain_path():
    """No dag_source / extra_domains -> the key it has always had."""
    m = {"disease": "rare6", "min_n": 20, "strip_mode": "both",
         "label_mask_mode": "full", "window_mode": "lookback", "lookback_days": 365,
         "label_window_days": 365, "n_bg": 40, "tpn": 5, "C": 27,
         "corpus_manifest": {
             "cdr": "p.d", "source_table": "condition_era", "person_mod": 1,
             "vocab_size": 5000, "min_df": 20, "min_patient_count": 20,
             "prior_obs_days": 0, "window_days": 365, "holdout_frac": 0.2,
             "doc_min_length": 10, "emit_labels": True}}
    assert not gpr.spec_is_multidomain(gpr.corpus_spec_from_manifest(m))
    assert gpr.bundle_key_from_manifest(m) == ccache.compute_bundle_cache_key(
        source_table="condition_era", person_mod=1, vocab_size=5000, min_df=20,
        min_patient_count=20, doc_min_length=10, prior_obs_days=0, window_days=365,
        disease="rare6", min_n=20, holdout_frac=0.2, n_bg=40, tpn=5, cdr="p.d",
        strip_mode="both", window_mode="lookback", lookback_days=365,
        label_window_days=365, emit_labels=True, label_mask_mode="full")


# --- the drift gate ------------------------------------------------------- #
class _StubBundle:
    def __init__(self, vocab_maps, int2cid):
        self.vocab_maps = vocab_maps
        self.int2cid = int2cid
        self.train_df = self.test_df = None


def test_drift_report_is_silent_on_the_corpus_the_fit_used():
    got = gpr.bundle_drift_report(
        _StubBundle([dict(vm) for vm in _VOCAB_MAPS], dict(_INT2CID)),
        _mondo_manifest(), [4, 3])
    assert got == []


def test_drift_report_catches_a_vocabulary_that_moved():
    """The λ rows are (K, V_m); a corpus whose domain-1 vocabulary grew by one
    concept cannot be scored with them — the columns no longer mean the same
    concepts. This is the check that stands between a rebuilt corpus and garbage."""
    wider = dict(_VOCAB_MAPS[1])
    wider[204] = 3                                   # one more concept than the fit
    drifted = [dict(_VOCAB_MAPS[0]), wider]
    got = gpr.bundle_drift_report(
        _StubBundle(drifted, dict(_INT2CID)), _mondo_manifest(), [4, 3])
    assert len(got) == 1
    assert "domain 1" in got[0] and "V=3" in got[0] and "4 concepts" in got[0]


def test_drift_report_catches_a_domain_count_change():
    got = gpr.bundle_drift_report(
        _StubBundle([dict(_VOCAB_MAPS[0])], dict(_INT2CID)), _mondo_manifest(), [4, 3])
    assert got and "2 domain(s)" in got[0] and "has 1" in got[0]


def test_drift_report_catches_a_relabelled_dag():
    """Engine id c is the head's row c. If the rebuilt DAG maps a different concept
    onto an id, every per-node number would be attributed to the wrong disease."""
    moved = dict(_INT2CID)
    moved[2] = 9999
    got = gpr.bundle_drift_report(
        _StubBundle([dict(vm) for vm in _VOCAB_MAPS], moved), _mondo_manifest(),
        [4, 3])
    assert got and "engine-id -> concept-id map differs at 1 node" in got[0]


def test_lambda_vocab_sizes_reads_both_lambda_shapes():
    """Multi-domain λ is a {domain: (K, V_m)} dict; single-domain is one array."""
    assert gpr.lambda_vocab_sizes(
        {"lambda": {1: np.zeros((5, 3)), 0: np.zeros((5, 4))}}) == [4, 3]
    assert gpr.lambda_vocab_sizes({"lambda": np.zeros((5, 7))}) == [7]


# --- main(): MISS -> rebuild, --no-rebuild -> fail fast, drift -> abort ----- #
def _run_dir(tmp_path, manifest, *, lam=None):
    """A finished multi-domain fit on disk: per-domain λ (lambda_0/lambda_1, the
    shape _save_fit writes) plus the manifest."""
    run = tmp_path / "0104-gated-pc-mondo"
    run.mkdir(parents=True, exist_ok=True)
    K, C = int(manifest["K"]), int(manifest["C"])
    rng = np.random.default_rng(3)
    lam = lam or {0: np.abs(rng.normal(size=(K, 4))) + 0.1,
                  1: np.abs(rng.normal(size=(K, 3))) + 0.1}
    np.savez(run / "gated_pc_result.npz",
             **{f"lambda_{m}": lam[m] for m in sorted(lam)},
             alpha=np.full(K, 1.0 / K), w_CK=rng.normal(size=(C, K)) * 0.5,
             b_CK=np.zeros(C))
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


@pytest.fixture
def patched_spark(spark, monkeypatch):
    """main() opens its own session; hand it the local fixture instead."""
    @contextmanager
    def _session(**kw):
        yield spark

    monkeypatch.setattr(gpr, "make_spark_session", _session)
    return spark


@pytest.mark.slow
def test_readout_fails_fast_on_a_miss_with_no_rebuild(patched_spark, tmp_path,
                                                      capsys):
    """--no-rebuild keeps the old behaviour for the case where a MISS is a symptom
    (wrong --cache-uri, drifted source) rather than a bill to pay."""
    run = _run_dir(tmp_path, _mondo_manifest())
    rc = gpr.main(["--run-dir", str(run), "--cache-uri", f"file://{tmp_path}/empty",
                   "--no-rebuild"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "bundle cache MISS" in out and "drop --no-rebuild" in out
    assert "rebuilding bundle" not in out


@pytest.mark.slow
def test_readout_rebuilds_on_a_miss_writes_through_and_scores(patched_spark,
                                                              tmp_path, capsys,
                                                              monkeypatch):
    """The recovery path end to end: an EMPTY cache, a saved fit, and a manifest.

    The rebuild goes through the fit's own seam, so the bundle lands under the key
    the readout just computed — which is what makes the SECOND readout a hit — and
    the run is scored without re-fitting anything.
    """
    manifest = _mondo_manifest()
    run = _run_dir(tmp_path, manifest)
    uri = f"file://{tmp_path}/cold-cache"
    built = _md_bundle(patched_spark)
    calls = {"n": 0}

    def _assemble(spark_, **kw):
        calls["n"] += 1
        return built

    def _rebuild(spark_, spec, *, cache_uri=None):
        # the real rebuild_bundle, with the BigQuery assembler stubbed out
        assert spec["dag_source"] == "mondo"
        return gpc.multidomain_load_or_build(
            spark_, spec, cache_uri=cache_uri, _build_inputs=_fake_build_inputs,
            _assemble=_assemble)

    monkeypatch.setattr(gpr, "rebuild_bundle", _rebuild)
    rc = gpr.main(["--run-dir", str(run), "--cache-uri", uri,
                   "--readout-mode", "driver", "--min-label-count", "0"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert calls["n"] == 1
    assert "cache MISS — rebuilding bundle from manifest params" in out
    assert "dag_source=mondo" in out
    assert (run / "results_readout.json").exists()

    # written through under the key the readout computed: the next run is a HIT,
    # and it is a MultiDomainBundle again.
    key = gpr.bundle_key_from_manifest(manifest)
    again = ccache.try_load(patched_spark, uri, key)
    assert again is not None and again.vocab_maps == _VOCAB_MAPS
    assert "features_1" in again.train_df.columns


@pytest.mark.slow
def test_readout_aborts_when_the_rebuilt_corpus_drifted(patched_spark, tmp_path,
                                                        capsys, monkeypatch):
    """A rebuilt corpus whose vocabulary no longer matches the saved λ must abort
    with the drift named — not report AUCs for a model scored on a corpus it was
    never fit on."""
    manifest = _mondo_manifest()
    # the saved fit's domain-1 lambda is 5 wide; the corpus has 3 concepts there
    K = int(manifest["K"])
    rng = np.random.default_rng(1)
    run = _run_dir(tmp_path, manifest,
                   lam={0: np.abs(rng.normal(size=(K, 4))) + 0.1,
                        1: np.abs(rng.normal(size=(K, 5))) + 0.1})
    built = _md_bundle(patched_spark)
    monkeypatch.setattr(gpr, "rebuild_bundle",
                        lambda spark_, spec, *, cache_uri=None: built)
    rc = gpr.main(["--run-dir", str(run),
                   "--cache-uri", f"file://{tmp_path}/cold2"])
    out = capsys.readouterr().out
    assert rc == 3, out
    assert "DRIFTED since the fit" in out
    assert "domain 1: the fit's lambda is V=5 wide" in out
    assert not (run / "results_readout.json").exists()
