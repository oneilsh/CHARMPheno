"""Pre-index closure `R_d` — what a document already carried BEFORE its index (E1).

WHAT IT IS
----------
For each document *d*, `R_d` is the **label-style closure vector computed over the
FEATURE window** (the pre-index lookback) instead of the label window: the same
frontier→closure machinery, the same kept-node DAG, the same climb, the same
closure rule as the labels — evaluated on the other side of the index. It is not
a new definition of "known"; it is the label definition, run on the history.

It exists to separate two questions that today share one metric (spec §1):

  * **tracking** — the patient already carried the condition before the index and
    it is coded again in the label window (its workup, its drugs, its sequelae are
    all in the lookback);
  * **onset prediction** — the condition is absent from the pre-index record and
    appears in the label window.

At 0109's root prevalence of 0.9609 the blend is not a rounding effect. With
`R_d` in hand, `incident-eligible(d, c) := c ∉ R_d` (spec D2) partitions rows into
tracking and prediction cells **at eval time only** — the corpus, the index, the
windows and the fit are all untouched.

GRAIN: PER DOCUMENT
-------------------
`R_d` is keyed on **`doc_id`**, one row per document — NOT per person. Under
today's `PatientCohortDocSpec` + `index_mode=population` there is exactly one
document per person, so `doc_id` is also a person key; that coincidence is not
relied on anywhere here, and when exp 0111's episode doc unit appends the index
component to `doc_id` this frame stays correct without an edit. Three grains
exist across this program and are never conflated: per-**document** (here),
per-**(person, node)** (E4's first-attestation sidecar), per-**person** (ADR
0025's covariate sidecar).

WHY IT LIVES OUTSIDE THE ASSEMBLER (the `mondo_collapse` precedent)
-------------------------------------------------------------------
The obvious implementation is three lines inside `multi_domain`: call `_attest`
a third time on the condition FEATURE frame, `attach_frontiers`, done. That
implementation is forbidden, and not for style reasons.
`_case_finding_cache.compute_bundle_cache_key` folds
`_module_source_hash(case_finding_assembly)` and `_module_source_hash(
multi_domain)` into every bundle key, so ANY edit to either — a comment, a
docstring — moves every key and orphans every cached bundle in every bucket,
including exp 0104's record run (~20 min of BigQuery) and the four tripwire
hashes pinned in `tests/scripts/test_case_finding_cache_mondo.py`. So this module
follows `mondo_collapse.py`'s route exactly: a NEW driver-level module, applied
in `gated_pc_cloud` between the corpus build and the cache write, folded into the
key ONLY when the flag is on. Flag off ⇒ byte-identical keys.

THE DESIGN POINT THAT MAKES THAT POSSIBLE: THE INDEX IS DETERMINISTIC
---------------------------------------------------------------------
The driver does not hold the assembler's feature frame and may not edit the
assembler to get it. It does not need to. `case_finding_population_index_table`
→ `_random_event_windows` picks each person's anchor by
``min hash(person_id, event_date, _RANDOM_WINDOW_SALT)`` — explicitly
*"resume-stable"*, explicitly **not** `F.rand()` (`cohorts.py:1083-1145`, the
pick at `:1129-1143`). A pure function of (person_id, event_date, salt) and the
observation-period gate. So calling the SAME imported index builder with the SAME
arguments the assembler passes reproduces the assembler's own windows exactly,
row for row.

That is why `feature_window_condition_events` below re-derives rather than
re-implements: every name in it is **imported** from `cohorts` / `multi_domain` /
`bigquery`, none is copied, so there is no forked pick to drift from the
assembler's. `tests/../test_preindex_closure.py` asserts the reproduction against
`cohorts._random_event_windows` itself on a synthetic frame — the plan's named
key risk, checked rather than argued.

Cost: one extra full-history `condition_era` scan plus one attestation pass at
build time, paid once per cached corpus.

SPARSE, NOT DENSE (R1.5)
------------------------
`R_d` is stored as an `array<int>` of ENGINE ids — the closure index list — not
as a dense `array<double>`. `label` + `labelMask` are already 2×3,820 float64 ≈
61 KB/row at whole-Mondo; a third dense array would be +50% on the bundle
parquet. This is also why `case_finding_assembly.attach_labels` is NOT reused:
it emits dense (`:177-180`). `frontier_to_label`'s closure rule is reproduced
here on the sparse side only — one `lay.closure` fold, the same one.

THE WITNESS (R1.4)
------------------
`_meta_dict` serializes only `parent_int/int2cid/cid2int/name_by_id/ledger/
vocab_map(s)`, so an extra parquet column would ride in a cached bundle
**silently** and a mixed-vintage cache dir would hand a readout a bundle without
it, failing at `select` with a Spark column error. So the bundle carries a
`preindex_closure` witness dict `{version, col_name}`, `_case_finding_cache`
round-trips it through the meta JSON, and `require_preindex_closure` below is
what every consumer calls FIRST: it raises a message naming the key and the
cache_uri instead of letting Spark complain about a missing column.

BROADCAST DISCIPLINE (ADR 0047)
-------------------------------
Nothing here creates an `sc.broadcast`, so there is nothing to `destroy()`. The
per-document closure fold is a UDF over small picklable structures (`lay`, `C`) —
the same shape `case_finding_assembly.attach_frontiers` / `attach_labels` already
use and the same size — not an array-shaped reduction identity.
"""
from __future__ import annotations

# Bumped when this primitive's OUTPUT would change for the same inputs. Folded
# into the bundle cache key alongside this module's source hash whenever the flag
# is on: the hash is the automatic guard (nobody has to remember), the version
# string is the citable record of WHICH construction a cached bundle carries.
PREINDEX_CLOSURE_VERSION = "preindex-closure-v1"

# The column the sparse closure list rides in, and the name recorded in the
# witness. Read it from the witness, not from this constant, when consuming a
# cached bundle — the witness is what says which vintage the bundle is.
PREINDEX_CLOSURE_COL = "preindexClosure"

# The frontier column the pre-index pass builds (kept distinct from the label
# frontier so the two can coexist in one frame during the join).
PREINDEX_FRONTIER_COL = "preindexFrontier"


# --------------------------------------------------------------------------- #
# Pure core (no Spark): frontier -> sparse closure index list.                 #
# --------------------------------------------------------------------------- #
def preindex_closure_ids(frontier, lay, C) -> list:
    """The sorted ENGINE ids in the is-a closure of `frontier` — `R_d`, sparse.

    Identical in meaning to the positive support of
    `case_finding_assembly.frontier_to_label`'s dense `label` vector (`c` is set
    iff `c` is in the closure of some frontier node, and `lay.closure` includes
    the root, so a non-empty frontier always carries engine-id 0), but emitted as
    an index LIST. An empty frontier — a document whose pre-index record contains
    no code resolving to a kept node — yields `[]`, which is the correct "carried
    nothing" and makes every node incident-eligible for it.

    Pure and total: ids outside `[0, C)` are dropped, exactly as
    `frontier_to_label` drops them when writing the dense vector."""
    active = set()
    for f in frontier or ():
        active.update(lay.closure(int(f)))
    return sorted(c for c in active if 0 <= int(c) < int(C))


def is_incident_eligible(closure_ids, C):
    """The D2 eligibility vector for one document: `True` where `c ∉ R_d`.

    Returned as a numpy bool array of width `C` so a caller can `&` it with a
    label/mask row. D2 is symmetric on purpose: a prior carrier of `closure(c)`
    leaves **both** classes for node c — dropping it from the positives only
    would be a different, and wrong, estimator."""
    import numpy as np

    elig = np.ones(int(C), dtype=bool)
    for c in closure_ids or ():
        if 0 <= int(c) < int(C):
            elig[int(c)] = False
    return elig


def attach_preindex_closure(df, lay, C, *, frontier_col=PREINDEX_FRONTIER_COL,
                            col_name=PREINDEX_CLOSURE_COL):
    """Append the sparse `array<int>` closure column derived from `frontier_col`.

    The sparse sibling of `attach_labels`, and deliberately not a call to it:
    that one emits two dense `array<double>` vectors (R1.5). `lay`/`C` are small
    and picklable and ride the UDF closure exactly as they do in
    `attach_frontiers` / `attach_labels`."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, IntegerType

    def _cl(fr):
        return [int(c) for c in preindex_closure_ids(
            [int(x) for x in (fr or [])], lay, C)]

    return df.withColumn(col_name, F.udf(_cl, ArrayType(IntegerType()))(
        F.col(frontier_col)))


# --------------------------------------------------------------------------- #
# The witness (R1.4).                                                          #
# --------------------------------------------------------------------------- #
def preindex_witness(col_name=PREINDEX_CLOSURE_COL) -> dict:
    """The meta/manifest witness recording that a bundle carries the column."""
    return {"version": PREINDEX_CLOSURE_VERSION, "col_name": str(col_name)}


def bundle_preindex_witness(bundle):
    """The witness a bundle carries, or None.

    The bundle dataclasses live in source-hashed modules and cannot gain a field,
    so the witness rides as an ATTRIBUTE set by the driver post-pass and
    round-tripped by `_case_finding_cache._meta_dict` / `try_load`. `getattr`
    with a default is therefore the whole contract, and every bundle built before
    this existed answers None."""
    return getattr(bundle, "preindex_closure", None)


def require_preindex_closure(bundle, *, key=None, cache_uri=None) -> str:
    """The column name to read, or raise saying exactly what is missing and where.

    R1.4: a readout or diagnostic that asks a bundle for the pre-index column must
    fail LOUDLY on a bundle that predates it — with the cache key and uri in the
    message, so the fix ("rebuild that corpus with --preindex-closure, or point at
    the right cache") is readable off the error — rather than dying inside Spark
    on a missing column. No silent mixed-vintage cache dirs."""
    witness = bundle_preindex_witness(bundle)
    where = f" (key={key}, cache_uri={cache_uri})" if key or cache_uri else ""
    if not witness:
        raise ValueError(
            f"this cached bundle carries NO pre-index closure witness{where}: it "
            f"was built without --preindex-closure, so column "
            f"{PREINDEX_CLOSURE_COL!r} is absent and every incident-eligibility "
            f"number derived from it would be undefined. Re-run the fit (or "
            f"gated_pc_readout) for this corpus with the flag on — that is a "
            f"different bundle cache key, so nothing already cached is "
            f"invalidated.")
    col = str(witness.get("col_name") or PREINDEX_CLOSURE_COL)
    have = set(getattr(bundle.train_df, "columns", ()) or ())
    if have and col not in have:
        raise ValueError(
            f"bundle{where} claims a pre-index closure witness "
            f"({witness}) but its train frame has no column {col!r} "
            f"(columns: {sorted(have)}) — a mixed-vintage cache dir. Rebuild the "
            f"corpus with --preindex-closure.")
    return col


# --------------------------------------------------------------------------- #
# The re-derivation: the assembler's own feature window, driver-side.          #
# --------------------------------------------------------------------------- #
def feature_window_index_table(cond, *, spark, cdr, billing, date_col,
                               label_window_days, index_mode="population",
                               disease="rare6"):
    """The assembler's OWN index table, rebuilt by calling the same function.

    `multi_domain.assemble_multidomain_case_finding_corpus:378-392` calls exactly
    one of these two with exactly these arguments; the prior-observation gate is
    the intrinsic `_LOOKBACK_PRIOR_OBS_DAYS` floor, not the forward-mode
    `prior_obs_days` knob, and passing anything else here would silently produce
    a different index than the corpus was built on. Both callees are IMPORTED —
    the pick (`min hash(person_id, event_date, _RANDOM_WINDOW_SALT)`, ties broken
    by earliest date) is never reimplemented, so it cannot drift."""
    from charmpheno.omop.case_finding_assembly import _LOOKBACK_PRIOR_OBS_DAYS
    from charmpheno.omop.cohorts import (case_finding_index_table,
                                         case_finding_population_index_table)

    if index_mode == "population":
        return case_finding_population_index_table(
            cond, spark=spark, cdr_dataset=cdr, billing_project=billing,
            date_col=date_col, prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS,
            label_window_days=label_window_days)
    if index_mode == "disease":
        return case_finding_index_table(
            cond, disease=disease, spark=spark, cdr_dataset=cdr,
            billing_project=billing, date_col=date_col,
            prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS,
            label_window_days=label_window_days)
    raise ValueError(
        f"index_mode must be 'disease' or 'population', got {index_mode!r}")


def feature_window_condition_events(spark, *, cdr, billing, person_mod,
                                    lookback_days, label_window_days,
                                    index_mode="population", disease="rare6",
                                    _load=None, _cond=None):
    """The pre-index CONDITION feature frame the assembler built — re-derived.

    Mirrors `assemble_multidomain_case_finding_corpus:365-396` step for step, with
    every step an import:

      1. `load_omop_bigquery(..., source_table="condition_era")` — the same
         full-history load, the same `person_sample_mod`;
      2. `feature_window_index_table` — the same deterministic index;
      3. `lookback_feature_frames` — the same
         `[index - lookback_days, index)` split.

    Returns that frame (`person_id, concept_id, condition_era_start_date,
    source_cohort`), which is exactly what an `attested_provider` consumes.

    `_cond` / `_load` are test seams: pass a ready-made condition frame (or a
    loader) to exercise the windowing without BigQuery."""
    from charmpheno.omop.multi_domain import lookback_feature_frames

    cond_date = "condition_era_start_date"
    if _cond is None:
        load = _load
        if load is None:
            from charmpheno.omop import load_omop_bigquery as load
        _cond = load(spark=spark, cdr_dataset=cdr, billing_project=billing,
                     person_sample_mod=person_mod, source_table="condition_era")
    index_df = feature_window_index_table(
        _cond, spark=spark, cdr=cdr, billing=billing, date_col=cond_date,
        label_window_days=label_window_days, index_mode=index_mode,
        disease=disease)
    feature_frames, _label = lookback_feature_frames(
        [_cond], index_df, [cond_date], lookback_days=lookback_days,
        label_window_days=label_window_days)
    return feature_frames[0]


def preindex_closure_frame(feature_events, *, attested_provider, before_dag,
                           bundle, n_bg, tpn):
    """`(doc_id, preindexClosure)` for every document, from the FEATURE frame.

    The post-prune internals are recovered from the BUNDLE, never from assembler
    internals (R1.3): `DagLayout(bundle.parent_int, n_bg, tpn)` is the same layout
    the labels were written against, `bundle.cid2int` the same concept→engine map,
    and `keep = set(bundle.int2cid.values())` the same surviving node set. That is
    what makes `R_d` and `label` commensurable — same DAG, same ids, same closure
    rule, different window.

    `attested_provider` is the SAME callable object the corpus was assembled with
    (the native-Mondo code-map provider, or the legacy SNOMED climb): calling it
    on the feature frame is the only difference between this and the label path.
    """
    from charmpheno.omop.case_finding_assembly import attach_frontiers
    from spark_vi.models.topic.dag_placement import DagLayout

    lay = DagLayout(bundle.parent_int, n_bg=n_bg, tpn=tpn)
    C = len(bundle.int2cid)
    keep = set(bundle.int2cid.values())
    attested = attested_provider(feature_events)
    fr = attach_frontiers(attested, before_dag, keep, bundle.cid2int, lay)
    fr = fr.withColumnRenamed("frontier", PREINDEX_FRONTIER_COL)
    out = attach_preindex_closure(fr, lay, C)
    return out.select("doc_id", PREINDEX_CLOSURE_COL)


def attach_preindex_closure_to_bundle(spark, bundle, *, before_dag,
                                      attested_provider, cdr, billing,
                                      person_mod, lookback_days,
                                      label_window_days, n_bg, tpn,
                                      index_mode="population", disease="rare6",
                                      _cond=None, _feature_events=None):
    """Build `R_d` and attach it to a freshly-assembled bundle, in place.

    The driver-owned post-pass (R1.1): called in `gated_pc_cloud`'s assemble seam
    AFTER the assembler returns and BEFORE the cache write, so the sparse column
    and its witness are part of what gets cached — exactly where the
    `dag_collapse` reduction sits relative to `mondo_src`, and for the same
    reason.

    Both splits get the column by a LEFT join on `doc_id`; a document with no
    resolvable pre-index condition code gets `[]` (carried nothing ⇒ eligible
    everywhere), which is the same convention the providers use for a background
    doc's empty frontier. Returns the bundle, mutated: `train_df`/`test_df` gain
    the column and the instance gains the `preindex_closure` witness attribute
    that `_case_finding_cache` serializes."""
    from pyspark.sql import functions as F

    feature_events = _feature_events
    if feature_events is None:
        feature_events = feature_window_condition_events(
            spark, cdr=cdr, billing=billing, person_mod=person_mod,
            lookback_days=lookback_days, label_window_days=label_window_days,
            index_mode=index_mode, disease=disease, _cond=_cond)
    # Cached and deliberately NOT unpersisted here: this frame is consumed three
    # times downstream and all three are LAZY — the two joins below, and then the
    # parquet write in `_case_finding_cache.save`. Unpersisting on the way out
    # would recompute the whole attestation pass at write time, which is the one
    # cost this primitive has. It is a DataFrame cache, not an `sc.broadcast`, so
    # ADR 0047's destroy-not-unpersist rule does not apply; the session ends with
    # the build.
    frame = preindex_closure_frame(
        feature_events, attested_provider=attested_provider,
        before_dag=before_dag, bundle=bundle, n_bg=n_bg, tpn=tpn).cache()

    def _join(df):
        joined = df.join(frame, on="doc_id", how="left")
        return joined.withColumn(
            PREINDEX_CLOSURE_COL,
            F.coalesce(F.col(PREINDEX_CLOSURE_COL),
                       F.array().cast("array<int>")))

    bundle.train_df = _join(bundle.train_df)
    bundle.test_df = _join(bundle.test_df)
    bundle.preindex_closure = preindex_witness()
    return bundle


def format_preindex_report(bundle, *, n_docs=None) -> str:
    """The one-line build diagnostic, in `format_collapse_report`'s style."""
    w = bundle_preindex_witness(bundle) or {}
    docs = f", {n_docs} train doc(s)" if n_docs is not None else ""
    return (f"[preindex] pre-index closure ({w.get('version', '?')}): column "
            f"{w.get('col_name', '?')!r} attached to both splits over C="
            f"{len(bundle.int2cid)} engine nodes{docs}; eligibility is "
            f"c NOT IN R_d (spec D2), a CORPUS property — no fit output enters it")
