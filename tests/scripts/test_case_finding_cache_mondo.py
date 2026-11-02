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
import types
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
import dag_gated_fit as gpc  # noqa: E402
import dag_gated_readout as gpr  # noqa: E402
from charmpheno.mondo.native_dag import MONDO_NATIVE_VERSION  # noqa: E402


# --------------------------------------------------------------------------- #
# Key stability: the SNOMED keys, pinned from before the change.               #
# --------------------------------------------------------------------------- #
_SNOMED_BASE = dict(
    source_table="condition_era", person_mod=10, vocab_size=5000, min_df=20,
    min_patient_count=20, doc_min_length=0, prior_obs_days=365, window_days=365,
    disease="diabetes", holdout_frac=0.2, split_salt=20260716, cdr="p.d")
# A cached SNOMED bundle lives under exactly these; if one of them moves OUTSIDE a
# deliberate cache drop, every cache entry in every bucket silently becomes
# unreachable and the next run rebuilds from BigQuery.
#
# RE-PINNED by exp-0111 WP-C (the one-blast cache drop, spec R7.1): carrying
# `index_date` through `lookback_feature_label_events` (`cohorts.py`) moves
# `cohort_defs_version()`, and the strip-claim comment fix moves `assembly_src`
# (`case_finding_assembly.py`) — both fold into EVERY key, so all four hashes here
# shifted at once. That was the announced, one-commit full cache drop; the whole
# repo re-assembles once.
#
# RE-PINNED by the 2026-10 cleanup (audit report 2026-10-26, decision Q1): removing
# the antidepressant/MDD (Hughes) block from `cohorts.py` moves
# `cohort_defs_version()`, which folds into EVERY key, so all four hashes shifted
# at once. A deliberate, announced full cache drop; the caches are HDFS-local and
# rebuilt per cluster anyway.
#
# RE-PINNED by ADR 0051 step 3 (the spark_vi.dag / charmpheno.mondo move): every
# key now folds `corpus_code` (bigquery / measurement_tokens / topic_prep, unhashed
# before) and `dag_layout_src`, `case_finding_assembly.py` changed its import of
# `frontier_from_coded`, and the payload version went 4 -> 5. Announced full drop;
# these are the post-drop values.
#
# RE-PINNED by the 2026-10-29 walkthrough: `spark_vi/dag/layout.py` (folded into
# every key as `dag_layout_src`) lost the dead `allowed(v)` and got an ancestor-walk
# `frontier_from_coded` (same output, no quadratic subtree scans). Announced full
# drop; these are the post-drop values. The same day `spark_vi/dag/ops.py` (folded
# into the native-Mondo key only) was cleaned up (shared nearest-kept walk,
# `children_and_min_depth`, generic docstrings): the native key moved again.
# And once more (commit 15bbe7c7 reworded two `ops.py` docstrings without
# re-pinning; re-pinned in the following commit): 14e647862c6f1301.
#
# RE-PINNED by WP4 of the DAG-gated user-API plan (2026-11-02, spec decision 5):
# `DagLayout` / `frontier_from_coded` are no longer exported from `spark_vi.dag`, so
# `case_finding_assembly.py` (folded into EVERY key as `assembly_src`) and
# `multi_domain.py` (`multi_domain_src`) import them from `spark_vi.dag.layout`. All
# four hashes moved at once; `spark_vi/dag/{layout,ops}.py` themselves are untouched.
# Announced full drop; these are the post-drop values.
#
# RE-PINNED by WP11 of the DAG-gated user-API plan (2026-11-02, spec 2026-10-30): the
# bundle carries the whole label DAG and each document's attested node ids (`labels`,
# `gate_labels`) instead of a pruned engine DAG, `frontier` and dense
# `label` / `labelMask`. `case_finding_assembly.py`, `condition_dag.py`,
# `multi_domain.py` and `charmpheno/mondo/native_dag.py` changed (`collapse.py`
# removed); the payload version went 5 -> 6; `min_n` / `min_positives`, `n_bg` /
# `tpn` and `emit_labels` / `label_mask_mode` left the key (not bundle inputs any
# more) and `spark_vi.dag.layout` stopped being folded (it no longer shapes the
# bundle). The second SNOMED pin was the `emit_labels` + closure-mask key; that
# option is gone, so it now pins the all-ever gate fold (`gate_attestation="ever"`).
# Announced full drop; these are the post-drop values.
_SNOMED_KEY = "94e21c0c95f5e9d4"
_SNOMED_EVER_KEY = "e872acd4ebe84669"              # gate_attestation=ever
_SNOMED_LOOKBACK_KEY = "4637ca2477bc417b"      # lookback windowing + strip both


def test_existing_snomed_keys_are_byte_identical():
    """The whole compatibility claim in three hashes."""
    k = ccache.compute_bundle_cache_key
    assert k(**_SNOMED_BASE) == _SNOMED_KEY
    assert k(**_SNOMED_BASE, gate_attestation="ever") == _SNOMED_EVER_KEY
    assert k(**dict(_SNOMED_BASE, window_mode="lookback", strip_mode="both",
                    lookback_days=730, label_window_days=180)) == _SNOMED_LOOKBACK_KEY


def test_multidomain_flag_alone_changes_the_key():
    """A multi-domain corpus can never collide with a single-domain one — which is
    also what keeps the new (vocab_maps) meta format out of an old entry's dir."""
    k = ccache.compute_bundle_cache_key
    assert k(**_SNOMED_BASE, multidomain=True) != _SNOMED_KEY


_MD_BASE = dict(_SNOMED_BASE, multidomain=True, extra_domains=("drug",),
                index_mode="population", mondo=True, mondo_version="2026-06-02",
                mondo_branch="", mondo_native=True,
                mondo_native_version="native-mondo-v1")
# The native-Mondo key, pinned the same way the SNOMED keys above are: the whole-
# Mondo record bundles live under keys of this shape. It folds the source hashes of
# `charmpheno.mondo.{native_dag,collapse,omop_mapping,ontology}` and
# `spark_vi.dag.{layout,ops}`, so it is a deliberate TRIPWIRE: any edit to those
# modules moves it, which is right (a hierarchy change must invalidate) but must be
# a decision, re-pinned here with a note.
#
# History: until ADR 0051 step 3 this was `_MONDO_NATIVE_KEY`, the anchor-Mondo
# key (exp 0104/0109), folding `analysis/cloud/mondo_dag.py`; the anchor flavour,
# the exp-0109 `dag_collapse` option, the DisMech `label_set` and E1's
# `preindex_closure` fold were removed with it (the driver had refused them since
# the 2026-10 cleanup). Post-drop value; re-pinned by WP4 (2026-11-02, the
# `spark_vi.dag.layout` import move in `case_finding_assembly` / `multi_domain`).
# Re-pinned by WP11 (2026-11-02, the subgraph handover; see the SNOMED pins' note):
# `collapse` and `spark_vi.dag.layout` no longer fold, `min_positives` left the key.
_MONDO_NATIVE_KEY = "ae054f1ef347c379"


def test_the_native_mondo_key_is_byte_identical():
    assert ccache.compute_bundle_cache_key(**_MD_BASE) == _MONDO_NATIVE_KEY


def test_the_anchor_mondo_flavour_is_refused():
    """`mondo=True` without `mondo_native` was the anchor build, removed with
    ADR 0051; keying it as if it existed would name a corpus nothing can build."""
    with pytest.raises(ValueError, match="mondo_native"):
        ccache.compute_bundle_cache_key(**dict(_MD_BASE, mondo_native=False))


def test_the_removed_key_options_are_gone_from_the_signature():
    """Including the ones spec 2026-10-30 took out of the bundle: the node-set
    threshold, the topic layout and the readout's dense-label options."""
    for gone in ("dag_collapse", "label_set", "preindex_closure", "min_n",
                 "min_positives", "n_bg", "tpn", "emit_labels", "label_mask_mode"):
        with pytest.raises(TypeError):
            ccache.compute_bundle_cache_key(**_MD_BASE, **{gone: True})


def test_mondo_native_version_is_folded_into_the_key():
    """The manual lever: a build whose OUTPUT changes bumps the version, and a
    bundle from the old construction can never be served to the new one."""
    k = ccache.compute_bundle_cache_key
    assert (k(**dict(_MD_BASE, mondo_native_version="native-mondo-v2"))
            != _MONDO_NATIVE_KEY)


def test_mondo_native_key_threads_from_the_spec():
    """The spec -> key_extra -> compute_bundle_cache_key path the fit and the
    re-readout both call, and the marker that routes it: a mondo_native spec must
    NOT be keyed as a SNOMED corpus (that is a guaranteed MISS plus a ~20-minute
    rebuild of the wrong thing)."""
    native_spec = dict(_MONDO_SPEC, dag_source="mondo_native")
    assert gpr.spec_is_multidomain(native_spec)
    key = gpc.multidomain_cache_key(native_spec)
    # the removed options' OFF values are inert...
    assert key == gpc.multidomain_cache_key(dict(native_spec, dag_collapse=False))
    # ...and asking for them is refused, never keyed as if absent (2026-10)
    for over in ({"dag_source": "mondo"}, {"dag_source": "snomed"},
                 {"dag_collapse": True}, {"preindex_closure": True},
                 {"label_set": "ribbon@x"}, {"index_mode": "external"}):
        with pytest.raises(ValueError, match="removed in the 2026-10 cleanup"):
            gpc.multidomain_cache_key(dict(native_spec, **over))


def _moves_the_key(module, **base):
    """True when replacing `module`'s source hash changes the key built from `base`."""
    k = ccache.compute_bundle_cache_key
    before = k(**base)
    real = ccache._module_source_hash
    ccache._module_source_hash = (
        lambda m, _t=module: "moved" if m is _t else real(m))
    try:
        return k(**base) != before
    finally:
        ccache._module_source_hash = real


def test_every_module_the_native_build_runs_is_folded_into_the_key():
    """Auto-invalidation: an edit to any module that shapes the label subgraph or
    the labels changes the key, so a stale native bundle can never be served.
    Checked by monkeypatching the hash helper rather than by pinning one hash per
    module."""
    from charmpheno.mondo import native_dag, omop_mapping, ontology
    from charmpheno.omop import case_finding_assembly, condition_dag, multi_domain
    from spark_vi.dag import layout, ops

    for module in (native_dag, omop_mapping, ontology, ops, case_finding_assembly,
                   condition_dag, multi_domain):
        assert _moves_the_key(module, **_MD_BASE), module.__name__
    for module in (case_finding_assembly, condition_dag):
        assert _moves_the_key(module, **_SNOMED_BASE), module.__name__
    assert not _moves_the_key(native_dag, **_SNOMED_BASE)
    # the engine layout no longer shapes the bundle (spec 2026-10-30): the gate's
    # layout is the estimator's, learned at fit time
    assert not _moves_the_key(layout, **_SNOMED_BASE)
    assert not _moves_the_key(layout, **_MD_BASE)


def test_the_corpus_shaping_modules_are_folded_into_every_key(monkeypatch):
    """bigquery / measurement_tokens / topic_prep shape every corpus and were not
    hashed before (audit 2026-10-26 §6): an edit there must move the bundle,
    corpus and covariate keys alike."""
    import _corpus_cache
    import _covariates_cache
    from charmpheno.omop import source_identity

    corpus_args = dict(source_table="condition_era", person_mod=10, vocab_size=5000,
                       min_df=20, doc_spec_manifest={}, cohort="c",
                       prior_obs_days=365)
    cov_args = dict(covariate_formula="~ age", person_mod=10, cdr="p.d",
                    source_table="condition_era", cohort="c", prior_obs_days=365)
    before = (ccache.compute_bundle_cache_key(**_SNOMED_BASE),
              ccache.compute_bundle_cache_key(**_MD_BASE),
              _corpus_cache.compute_cache_key(**corpus_args),
              _covariates_cache.compute_cache_key(**cov_args))
    monkeypatch.setattr(source_identity, "corpus_code_version", lambda: "moved")
    after = (ccache.compute_bundle_cache_key(**_SNOMED_BASE),
             ccache.compute_bundle_cache_key(**_MD_BASE),
             _corpus_cache.compute_cache_key(**corpus_args),
             _covariates_cache.compute_cache_key(**cov_args))
    assert all(b != a for b, a in zip(before, after))


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
    assert spec["index_mode"] == "population"
    for gone in ("min_n", "min_positives", "n_bg", "tpn", "emit_labels"):
        assert gone not in spec
    assert (gpr.bundle_key_from_manifest(m)
            == gpc.multidomain_cache_key(spec))


@pytest.mark.parametrize("field,val", [
    ("extra_domains", ("drug", "procedure")),
    ("index_mode", "disease"),
    ("mondo_version", "2026-07-01"),
    ("mondo_branch", "MONDO:0004995"),
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
    assert k(**_SNOMED_BASE, gate_attestation="ever",
             doc_spec=d) == _SNOMED_EVER_KEY
    assert k(**dict(_SNOMED_BASE, window_mode="lookback", strip_mode="both",
                    lookback_days=730, label_window_days=180),
             doc_spec=d) == _SNOMED_LOOKBACK_KEY
    assert k(**_MD_BASE, doc_spec=d) == _MONDO_NATIVE_KEY


def test_a_different_doc_spec_moves_the_key_on_both_paths():
    """The hole itself: an episode/patient-year corpus is a DIFFERENT corpus (a
    different doc unit means different documents, a different vocabulary, and
    different base rates) and must never be served a patient-cohort bundle."""
    k = ccache.compute_bundle_cache_key
    assert k(**_SNOMED_BASE, doc_spec="patient_year") != _SNOMED_KEY
    assert k(**_MD_BASE, doc_spec="episode") != _MONDO_NATIVE_KEY
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
# The label subgraph in Mondo curies (spec 2026-10-30): A over B and C.
_A, _B, _C = "MONDO:0001001", "MONDO:0001002", "MONDO:0001003"
_DAG = {_A: [], _B: [_A], _C: [_A]}
_NAME_BY_ID = {_A: "node a", _B: "node b", _C: "node c"}
_LEDGER = {"n_dag_nodes": 3, "train_fg_docs": 12, "test_fg_docs": 8}
_CODE_MAP = [(101, _B), (102, _C), (103, _B)]


def _md_frame(spark, n, offset=0):
    """The frame the multi-domain assembler emits: per-domain BOW columns plus the
    condition-only attested node ids (`labels`)."""
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql.types import (ArrayType, LongType, StringType, StructField,
                                   StructType)
    schema = StructType([
        StructField("doc_id", LongType(), False),
        StructField("person_id", LongType(), False),
        StructField("features_0", VectorUDT(), False),
        StructField("features_1", VectorUDT(), False),
        StructField("labels", ArrayType(StringType()), False),
        StructField("source_cohort", StringType(), False),
    ])
    rng = np.random.default_rng(offset + 7)
    rows = []
    for d in range(n):
        pid = offset + d
        f0 = Vectors.dense(rng.integers(0, 3, size=4).astype(float))
        f1 = Vectors.dense(rng.integers(0, 3, size=3).astype(float))
        node = _B if d % 2 == 0 else _C
        rows.append((int(pid), int(pid), f0, f1, [node], "general"))
    return spark.createDataFrame(rows, schema)


def _md_bundle(spark, n_train=12, n_test=8):
    from charmpheno.omop.multi_domain import MultiDomainBundle
    b = MultiDomainBundle(
        train_df=_md_frame(spark, n_train), test_df=_md_frame(spark, n_test, 1000),
        dag={n: list(ps) for n, ps in _DAG.items()},
        vocab_maps=[dict(vm) for vm in _VOCAB_MAPS],
        name_by_id=dict(_NAME_BY_ID), ledger=dict(_LEDGER))
    b.native_code_map = list(_CODE_MAP)
    return b


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
    assert loaded.dag == _DAG                        # curies stay strings
    assert loaded.name_by_id == _NAME_BY_ID
    assert loaded.ledger == _LEDGER
    assert loaded.native_code_map == _CODE_MAP
    for got, want in ((loaded.train_df, bundle.train_df),
                      (loaded.test_df, bundle.test_df)):
        assert set(got.columns) == set(want.columns)
        assert "features_0" in got.columns and "features_1" in got.columns
        a = {r["person_id"]: (list(r["features_0"]), list(r["features_1"]),
                              list(r["labels"])) for r in got.collect()}
        b = {r["person_id"]: (list(r["features_0"]), list(r["features_1"]),
                              list(r["labels"])) for r in want.collect()}
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
        train_df=df, test_df=df, dag={100: [], 200: [100]}, vocab_map={101: 0, 102: 1},
        name_by_id={100: "dm", 200: "t2"}, ledger=dict(_LEDGER))
    uri = f"file://{tmp_path}/cache1d"
    ccache.save(spark, bundle, uri, "sdkey")

    meta = json.loads(
        spark.read.text(f"{uri}/sdkey/meta").collect()[0]["value"])
    assert "vocab_map" in meta and "vocab_maps" not in meta

    loaded = ccache.try_load(spark, uri, "sdkey")
    assert isinstance(loaded, CaseFindingBundle)
    assert loaded.vocab_map == {101: 0, 102: 1}
    assert loaded.dag == {100: [], 200: [100]}       # int concept ids stay ints
    assert loaded.name_by_id == {100: "dm", 200: "t2"}
    assert not hasattr(loaded, "vocab_maps")


def test_restore_meta_keeps_each_node_ids_type():
    """The DAG and the names ride as LISTS of pairs, so an int concept id and a str
    curie both come back as given (JSON object keys would stringify them)."""
    meta = {"dag": [[100, None], [200, 100]], "vocab_map": {"101": "0"},
            "name_by_id": [[100, "dm"], [200, "t2"]], "ledger": {}}
    out = ccache._restore_meta(json.loads(json.dumps(meta)))
    assert out["dag"] == {100: [], 200: [100]}
    assert out["name_by_id"] == {100: "dm", 200: "t2"}
    assert out["vocab_map"] == {101: 0} and "vocab_maps" not in out
    cm = {"dag": [[_A, None], [_B, _A]], "vocab_maps": [{"101": 0}],
          "name_by_id": [[_A, "a"]], "ledger": {}, "native_code_map": [[101, _B]]}
    out = ccache._restore_meta(json.loads(json.dumps(cm)))
    assert out["dag"] == {_A: [], _B: [_A]}
    assert out["native_code_map"] == [(101, _B)]


@pytest.mark.slow
def test_a_pre_v6_meta_is_a_miss(spark, tmp_path):
    """An entry from before the node-set redesign (a pruned engine DAG, no `dag`)
    cannot serve this contract: `try_load` treats it as a MISS."""
    uri = f"file://{tmp_path}/old"
    df = spark.createDataFrame([(1,)], "person_id long")
    df.write.parquet(f"{uri}/k/train.parquet")
    df.write.parquet(f"{uri}/k/test.parquet")
    legacy = {"parent_int": {"1": ["0"]}, "int2cid": {"0": "-1"},
              "cid2int": {"-1": "0"}, "vocab_map": {"101": "0"},
              "name_by_id": {"-1": "root"}, "ledger": {"K_nodes": 1}}
    (spark.createDataFrame([(json.dumps(legacy),)], "value STRING")
     .coalesce(1).write.text(f"{uri}/k/meta"))
    assert ccache.try_load(spark, uri, "k") is None


# --------------------------------------------------------------------------- #
# The seam: load-or-build for the Mondo path.                                  #
# --------------------------------------------------------------------------- #
_MONDO_SPEC = {
    "dag_source": "mondo_native", "disease": "rare6", "cdr": "p.d", "billing": "bp",
    "source_table": "condition_era", "extra_domains": ["drug"],
    "index_mode": "population", "person_mod": 1, "vocab_size": 5000, "min_df": 20,
    "min_patient_count": 20, "doc_min_length": 10, "holdout_frac": 0.2,
    "strip_mode": "test_only", "lookback_days": 365,
    "label_window_days": 365,
    "window_mode": "lookback", "prior_obs_days": 0, "window_days": 0,
    "mondo_version": "2026-06-02", "mondo_branch": "",
    "mondo_cache_dir": "data/mondo", "dag_collapse": False,
    "doc_spec": "patient_cohort", "preindex_closure": False,
}


def _fake_build_inputs(spark, **kw):
    """Stands in for build_mondo_native_fit_inputs — the minutes of BigQuery whose
    whole point here is that a cache HIT never reaches it."""
    from charmpheno.mondo.native_dag import MondoLabelInputs
    return MondoLabelInputs(dag=dict(_DAG), code_map_sdf=object(),
                            node_ids=sorted(_DAG), names=dict(_NAME_BY_ID),
                            stats={"n_nodes": 3})


@pytest.fixture
def _quiet_native_reports(monkeypatch):
    from charmpheno.mondo import native_dag as mnd
    monkeypatch.setattr(mnd, "format_native_build_report", lambda s: "build")
    monkeypatch.setattr(mnd, "format_native_mapping_report", lambda s: "mapping")


@pytest.mark.slow
def test_mondo_load_or_build_misses_once_then_hits(spark, tmp_path, capsys,
                                                  _quiet_native_reports):
    """The behaviour the mondo path never had: build once, reload forever.

    The DAG build lives INSIDE the assemble seam, so the HIT must not reach it —
    that is the ~5 min of BigQuery every mondo fit used to pay unconditionally.
    """
    built = _md_bundle(spark)
    calls = {"build": 0, "assemble": 0, "on_inputs": 0}

    def _build(spark_, **kw):
        calls["build"] += 1
        assert kw["mondo_version"] == "2026-06-02"
        assert "min_positives" not in kw           # no threshold in the build
        assert kw["branch_root"] is None           # '' means whole Mondo
        return _fake_build_inputs(spark_, **kw)

    def _assemble(spark_, **kw):
        calls["assemble"] += 1
        # the seam must hand the assembler the subgraph + provider it just built
        assert kw["label_dag"] == _DAG and kw["names"] == _NAME_BY_ID
        assert kw["attested_provider"] is not None
        assert kw["index_mode"] == "population"
        for gone in ("min_n", "n_bg", "tpn", "emit_labels", "label_mask_mode"):
            assert gone not in kw
        assert kw["extra_domains"] == ("drug",)
        return built

    def _on_inputs(*, stats):
        calls["on_inputs"] += 1
        assert stats == {"n_nodes": 3}

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
    assert b2.dag == _DAG
    assert {r["person_id"] for r in b2.train_df.collect()} == \
        {r["person_id"] for r in built.train_df.collect()}



# --------------------------------------------------------------------------- #
# The re-readout: same key, load-or-BUILD, and the drift gate.                 #
# --------------------------------------------------------------------------- #
def _mondo_manifest(**over):
    """What the driver writes for a mondo fit: corpus_manifest IS the corpus spec."""
    cm = {k: v for k, v in _MONDO_SPEC.items()}
    cm["cache_uri"] = None
    cm["domain_vocab_sizes"] = [len(vm) for vm in _VOCAB_MAPS]
    m = {"model_class": "gated_pc", "C": 4, "K": 11, "model_path": "model",
         "max_iter": 4, "min_label_count": 0, "dag_source": "mondo_native",
         "extra_domains": ["drug"], "disease": "rare6",
         "strip_mode": "test_only", "label_mask_mode": "full", "labels_col": "labels",
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
        doc_min_length=10, disease="rare6", holdout_frac=0.2,
        cdr="p.d", strip_mode="test_only", lookback_days=365,
        label_window_days=365,
        multidomain=True, extra_domains=("drug",), index_mode="population",
        mondo=True, mondo_version="2026-06-02", mondo_branch="",
        mondo_native=True, mondo_native_version=MONDO_NATIVE_VERSION)


def test_readout_key_for_a_mondo_manifest_is_not_the_snomed_key():
    """Before this, the mondo manifest computed a SNOMED key — which is why the
    re-readout always missed (nothing was ever stored under it either)."""
    m = _mondo_manifest()
    snomed = ccache.compute_bundle_cache_key(**{k: gpr.corpus_spec_from_manifest(m)[k]
                                                for k in gpr._SNOMED_KEY_KEYS})
    assert gpr.bundle_key_from_manifest(m) != snomed


def test_the_fit_knobs_a_manifest_still_carries_do_not_key_the_bundle():
    """`min_n`, `min_positives`, `n_bg`, `tpn`, `label_mask_mode` are fit or readout
    settings since spec 2026-10-30: a manifest carrying them (top level or in the
    corpus spec) keys the same bundle as one without them."""
    m = _mondo_manifest()
    m["corpus_manifest"].update(min_n=50, min_positives=100, n_bg=200, tpn=3,
                                label_mask_mode="closure", emit_labels=True)
    m.update(min_n=7, n_bg=12)
    assert gpr.bundle_key_from_manifest(m) == gpr.bundle_key_from_manifest(
        _mondo_manifest())


def test_readout_cli_overrides_supply_a_missing_mondo_spec():
    """An older mondo manifest has no Mondo build inputs at all; the overrides are
    what make such a run recoverable (they are key inputs, so a wrong one misses)."""
    m = _mondo_manifest()
    for f in ("mondo_version", "mondo_branch", "billing"):
        del m["corpus_manifest"][f]
    spec = gpr.corpus_spec_from_manifest(
        m, mondo_version="2026-06-02", billing="bp")
    assert spec["mondo_version"] == "2026-06-02" and spec["billing"] == "bp"
    assert gpr.bundle_key_from_manifest(
        m, mondo_version="2026-06-02") == gpr.bundle_key_from_manifest(
            _mondo_manifest())



def test_readout_parses_the_rebuild_flags():
    a = gpr.build_parser().parse_args(["--run-dir", "/runs/0104-x"])
    assert a.no_rebuild is False and a.dag_source is None and a.billing is None
    assert a.cache_write == "on"
    b = gpr.build_parser().parse_args(
        ["--run-dir", "/runs/0104-x", "--no-rebuild", "--dag-source", "mondo_native",
         "--billing", "proj"])
    assert b.no_rebuild is True and b.dag_source == "mondo_native"
    assert b.billing == "proj"
    with pytest.raises(SystemExit):            # the threshold is the fit's now
        gpr.build_parser().parse_args(["--run-dir", "/r", "--min-positives", "100"])



def test_snomed_manifest_still_takes_the_single_domain_path():
    """No dag_source / extra_domains -> the key it has always had."""
    m = {"disease": "rare6", "strip_mode": "both",
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
        disease="rare6", holdout_frac=0.2, cdr="p.d",
        strip_mode="both", window_mode="lookback", lookback_days=365,
        label_window_days=365)


# --- the drift gate ------------------------------------------------------- #
class _StubBundle:
    def __init__(self, vocab_maps, dag):
        self.vocab_maps = vocab_maps
        self.dag = dag
        self.train_df = self.test_df = None


class _StubModel:
    """`getDag()` (the DAG the fit was given) and the node set's size."""
    def __init__(self, dag, n_heads):
        self._dag, self._n = dag, n_heads

    def getDag(self):
        return {n: list(ps) for n, ps in self._dag.items()}

    def _ns(self):
        return types.SimpleNamespace(user_ids=tuple(range(self._n)))


def test_drift_report_is_silent_on_the_corpus_the_fit_used():
    got = gpr.bundle_drift_report(
        _StubBundle([dict(vm) for vm in _VOCAB_MAPS], dict(_DAG)),
        _mondo_manifest(), [4, 3], _StubModel(_DAG, 4))
    assert got == []


def test_drift_report_catches_a_vocabulary_that_moved():
    """The λ rows are (K, V_m); a corpus whose domain-1 vocabulary grew by one
    concept cannot be scored with them — the columns no longer mean the same
    concepts. This is the check that stands between a rebuilt corpus and garbage."""
    wider = dict(_VOCAB_MAPS[1])
    wider[204] = 3                                   # one more concept than the fit
    drifted = [dict(_VOCAB_MAPS[0]), wider]
    got = gpr.bundle_drift_report(
        _StubBundle(drifted, dict(_DAG)), _mondo_manifest(), [4, 3],
        _StubModel(_DAG, 4))
    assert len(got) == 1
    assert "domain 1" in got[0] and "V=3" in got[0] and "4 concepts" in got[0]


def test_drift_report_catches_a_domain_count_change():
    got = gpr.bundle_drift_report(
        _StubBundle([dict(_VOCAB_MAPS[0])], dict(_DAG)), _mondo_manifest(), [4, 3])
    assert got and "2 domain(s)" in got[0] and "has 1" in got[0]


def test_drift_report_catches_a_label_dag_that_moved():
    """The model's node set was learned on the DAG it was given; a rebuilt corpus
    whose label DAG differs (a Mondo release re-parented a term, the code ladder
    reached another term) is not the corpus the heads' labels mean."""
    moved = dict(_DAG)
    moved[_C] = [_B]
    got = gpr.bundle_drift_report(
        _StubBundle([dict(vm) for vm in _VOCAB_MAPS], moved), _mondo_manifest(),
        [4, 3], _StubModel(_DAG, 4))
    assert got and "label DAG is not the one the model was fit with" in got[0]
    assert "1 with other parents" in got[0]


def test_drift_report_catches_a_manifest_model_mismatch():
    got = gpr.bundle_drift_report(
        _StubBundle([dict(vm) for vm in _VOCAB_MAPS], dict(_DAG)),
        _mondo_manifest(), [4, 3], _StubModel(_DAG, 6))
    assert got and "C=4" in got[0]


def test_lambda_vocab_sizes_reads_both_lambda_shapes():
    """Multi-domain λ is a {domain: (K, V_m)} dict; single-domain is one array."""
    assert gpr.lambda_vocab_sizes(
        {"lambda": {1: np.zeros((5, 3)), 0: np.zeros((5, 4))}}) == [4, 3]
    assert gpr.lambda_vocab_sizes({"lambda": np.zeros((5, 7))}) == [7]


# --- main(): MISS -> rebuild, --no-rebuild -> fail fast, drift -> abort ----- #
def _saved_model(run, lam, *, n_bg=8, tpn=1):
    """A fitted DagGatedLDAModel saved to <run>/model as the fit driver saves it: the
    node set learned from the train labels of `_md_bundle` (min 1 doc), the per-domain
    λ, the feature columns and the DAG Params. Built from a VIResult instead of a
    fit, so the readout tests do not pay for an SVI run."""
    from spark_vi.core.result import VIResult
    from spark_vi.dag.nodeset import fit_node_set
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    import dag_gated_fit as _gpc
    ns = fit_node_set(_DAG, [[_B], [_C]], min_docs_per_node=1)
    K = n_bg + len(ns.nodes) * tpn
    md = {"model_class": "DagGatedOnlineLDA",
          "dag_node_set": {**ns.to_dict(), "n_bg": n_bg, "tpn": tpn}}
    res = VIResult(global_params={"lambda": lam, "alpha": np.full(K, 1.0 / K)},
                   elbo_trace=[], n_iterations=1, converged=True, metadata=md)
    model = DagGatedLDAModel(res)
    model.setFeaturesCols(["features_0", "features_1"])
    model.setLabelsCol("labels")
    model.setDag(_DAG)
    _gpc.save_fitted_model(model, run)
    return model


def _run_dir(tmp_path, manifest, *, lam=None):
    """A finished multi-domain fit on disk: the saved model (<run>/model), the
    per-domain λ (lambda_0/lambda_1, the shape _save_fit writes) and the manifest."""
    run = tmp_path / "0104-gated-pc-mondo"
    run.mkdir(parents=True, exist_ok=True)
    K = int(manifest["K"])
    rng = np.random.default_rng(3)
    lam = lam or {0: np.abs(rng.normal(size=(K, 4))) + 0.1,
                  1: np.abs(rng.normal(size=(K, 3))) + 0.1}
    np.savez(run / "gated_pc_result.npz",
             **{f"lambda_{m}": lam[m] for m in sorted(lam)},
             alpha=np.full(K, 1.0 / K))
    _saved_model(run, lam)
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
def test_readout_rebuilds_on_a_miss_writes_through_and_scores(
        patched_spark, tmp_path, capsys, monkeypatch, _quiet_native_reports):
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
        assert spec["dag_source"] == "mondo_native"
        return gpc.multidomain_load_or_build(
            spark_, spec, cache_uri=cache_uri, _build_inputs=_fake_build_inputs,
            _assemble=_assemble)

    monkeypatch.setattr(gpr, "rebuild_bundle", _rebuild)
    rc = gpr.main(["--run-dir", str(run), "--cache-uri", uri,
                   "--min-label-count", "0"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert calls["n"] == 1
    assert "cache MISS — rebuilding bundle from manifest params" in out
    assert "dag_source=mondo_native" in out
    assert (run / "results_readout.json").exists()

    # written through under the key the readout computed: the next run is a HIT,
    # and it is a MultiDomainBundle again.
    key = gpr.bundle_key_from_manifest(manifest)
    again = ccache.try_load(patched_spark, uri, key)
    assert again is not None and again.vocab_maps == _VOCAB_MAPS
    assert "features_1" in again.train_df.columns


@pytest.mark.slow
def test_readout_reads_the_node_set_from_the_fitted_model(
        patched_spark, tmp_path, capsys, monkeypatch, _quiet_native_reports):
    """The heads, the stacked product, the bundle meta, the landed code map and the
    gated NPMI all read the FITTED model's node set (spec 2026-10-30): the targets
    are built from the bundle's `labels` on it, the bridge in bundle_meta.json is its
    engine order, and the code ladder lands on its kept nodes."""
    manifest = _mondo_manifest()
    run = _run_dir(tmp_path, manifest)
    built = _md_bundle(patched_spark)
    monkeypatch.setattr(gpr, "rebuild_bundle",
                        lambda spark_, spec, *, cache_uri=None: built)
    rc = gpr.main(["--run-dir", str(run), "--cache-uri", f"file://{tmp_path}/c3",
                   "--min-label-count", "0", "--readout-stacked",
                   "--readout-hierarchy", "flat"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "fitted model loaded: 3 kept node(s)" in out
    res = json.loads((run / "results_readout_stacked.json").read_text())
    assert "gated_pc" in res and "gated_pc_stacked" in res
    meta = json.loads((run / "bundle_meta.json").read_text())
    model = gpc.load_fitted_model(run)
    br = gpc.node_bridge(model, _NAME_BY_ID)
    assert meta["int2cid"] == {str(e): c for e, c in br["int2cid"].items()}
    assert sorted(meta["int2cid"].values()) == [-1, 1001, 1002, 1003]
    rows = (run / "code_map.tsv").read_text().splitlines()[1:]
    assert sorted(rows) == ["101\t1002", "102\t1003", "103\t1002"]
    rc = gpr.main(["--run-dir", str(run), "--cache-uri", f"file://{tmp_path}/c3",
                   "--npmi", "--npmi-top", "2"])
    assert rc == 0, capsys.readouterr().out
    npmi = json.loads((run / "npmi.json").read_text())["rows"]
    assert {r["cid"] for r in npmi if r["kind"] == "node"} == {1001, 1002, 1003}


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
