"""FIRST-ATTESTATION-DATE sidecar — the artifact E4's conversion analysis needs.

WHAT IT IS
----------
For every `(person_id, node_cid)` pair: the EARLIEST date on which that person has
a condition code resolving to that label node, over their WHOLE record — before the
lookback, inside it, and, crucially, AFTER the label window.

    cond ⋈ code_map on concept_id
      -> groupBy(person_id, node_cid).agg(min(condition_era_start_date))

That last part is the entire reason this file exists. E4 asks: of the documents
scored INCIDENT NEGATIVE for node c — eligible (c ∉ R_d) and not gaining
`closure(c)` in the label window — how many are diagnosed with c LATER? That is
**PU channel 1, measured**: a lower bound on how many "negatives" are really
undiagnosed-yet positives.

NOTHING DOWNSTREAM CAN ANSWER IT. Full history IS loaded with no date filter
(`load_omop_bigquery`, `bigquery.py`, applies only `person_sample_mod` and
`concept_id != 0`), but post-label-window events are DISCARDED at windowing, not
retained (`lookback_feature_label_events`, `cohorts.py:1648-1668`), and the cached
bundle holds BOW vectors, not events (`_case_finding_cache.py:21-23`). The future
date has to be captured where the full-history frame still exists, or it is gone.

GRAIN: PER (PERSON, NODE) — AND IT IS NOT THE OTHER TWO (R4.3)
--------------------------------------------------------------
Three artifact grains exist across this program and are NEVER conflated:

  * per-**document** — E1's `R_d` (`preindex_closure.py`), keyed on `doc_id`;
  * per-**(person, node)** — THIS FILE, keyed on `(person_id, node_cid)`;
  * per-**person** — ADR 0025's covariate sidecar, keyed on `person_id`.

Under today's `PatientCohortDocSpec` + `index_mode=population` there is one
document per person, so a (person, node) row joins a document row — but that is a
COINCIDENCE of the current doc unit, not a property of this frame, and exp 0111's
episode doc unit ends it. Every join that crosses a grain boundary names the
crossing in the code that does it (`conversion_analysis.py` does, at its one join).

`node_cid` is a FRONTIER node — the most-specific label node a code resolves to,
the same cid the attestation providers emit — NOT a closure. "First attestation of
`closure(c)`" is a driver-side fold at ANALYSIS time: the min over every frontier
node whose closure contains c (i.e. over c's own subtree). Storing the closure
here instead would multiply the frame by the average closure depth for a number
that is one `lay.closure` call away.

WHY A SIDECAR, AND WHY ITS OWN KEY (R4.2 / ADR 0025)
-----------------------------------------------------
The natural aggregation site is `multi_domain.py:365-367`, where `cond` is already
loaded once with full history — and that is INSIDE the source-hashed assembler.
Editing it moves `multi_domain_src`, orphans every cached bundle in the repo
including exp 0104's record bundle (~20 min of BigQuery), and breaks the four
pinned tripwire hashes. Same trap as E1, same answer: driver-owned code, applied at
the driver's own seam, cached as a SEPARATE parquet artifact per ADR 0025's
pattern (its own directory, referenced by a manifest field, joined at use time).

The sidecar carries its **own cache key**, over `(cdr, person_mod, dag identity,
code-map identity, this module's source)` — deliberately NOT the bundle key. Two
consequences, both wanted:

  * a bundle-key move (a new flag, a new assembler vintage) does not orphan the
    sidecar, which is expensive and depends on none of those things;
  * the sidecar survives readout re-runs and is shared by every run over the same
    corpus identity.

Horizons are NOT in the key: the frame stores dates, and a horizon is an arithmetic
comparison the analysis makes. Keying on the horizon set would rebuild a scan to
answer a subtraction.

RIGHT-CENSORING IS NOT HANDLED HERE (R4.4)
-------------------------------------------
This module stores WHEN, not WHETHER-OBSERVED. The conversion denominator has to
be gated on `observation_period_end_date` at each horizon or the "conversion rate"
is a censoring artifact — that gate is `observation_gate_frame` below (the
driver-side re-derivation of `_window_observed_cohort`'s follow-up clause), and it
is applied by `conversion_analysis.py`, at the point where a denominator is formed.

BROADCAST DISCIPLINE (ADR 0047)
--------------------------------
One `broadcast(code_map)` join and one `groupBy`, both Spark-native; no UDF, no
`sc.broadcast`, nothing array-shaped in a task closure.
"""
from __future__ import annotations

import hashlib
import json
import sys

# Bumped when this sidecar's OUTPUT would change for the same inputs. Folded into
# the sidecar key next to this module's source hash: the hash is the automatic
# guard (nobody has to remember), the version string is the citable record of
# WHICH construction a cached sidecar carries.
CONVERSION_SIDECAR_VERSION = "conversion-sidecar-v1"

# The parquet's columns. Named once, here, because three files read them and a
# silent rename is a wrong join rather than an error.
PERSON_COL = "person_id"
NODE_COL = "node_cid"
FIRST_DATE_COL = "first_attested_date"
SIDECAR_COLUMNS = (PERSON_COL, NODE_COL, FIRST_DATE_COL)

# The observation-gate frame's columns (`observation_gate_frame`).
OBS_END_COL = "observation_period_end_date"
INDEX_COL = "index_date"
HORIZON_COLUMNS = (PERSON_COL, INDEX_COL, OBS_END_COL)

# TWO parquets under one key, because they are TWO GRAINS and the whole of R4.3 is
# that two grains never share a schema:
#
#   first_attestation.parquet   per (person_id, node_cid)  — WHEN, ever
#   index_horizon.parquet       per person_id              — the index the corpus
#                                                            used, and how long
#                                                            that person stays
#                                                            observed after it
#
# The second exists so the analysis tool needs no BigQuery at all: the index date
# is re-derived here, at build time, by the SAME deterministic index builder the
# assembler used (`preindex_closure.feature_window_index_table` — the pick is
# `min hash(person_id, event_date, salt)`, resume-stable and explicitly not
# `F.rand()`), and the observation-period end travels with it. Recomputing either
# at analysis time would mean a second full-history scan to answer a subtraction.
FIRST_ATTESTATION_FILE = "first_attestation.parquet"
INDEX_HORIZON_FILE = "index_horizon.parquet"


def _module_source_hash(module) -> str:
    """`_case_finding_cache._module_source_hash`, duplicated rather than imported.

    Copied deliberately (three lines, no logic): importing it would make this
    module's key depend on that module's import graph, and this key exists
    precisely so the sidecar does NOT move when the bundle key does."""
    import inspect
    try:
        return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return "src-unavailable"


def conversion_sidecar_key(*, cdr, person_mod, dag_source, mondo_version="",
                           mondo_branch="", min_positives=0,
                           code_map_identity="") -> str:
    """Stable 16-hex key over what determines the sidecar's CONTENT.

    Five things and no more:

      * `cdr` + `person_mod` — which patients, which data release;
      * `dag_source` + `mondo_version` + `mondo_branch` + `min_positives` — which
        label nodes exist, since `node_cid` is one of them;
      * `code_map_identity` — an optional caller-supplied token for a code map that
        is not fully described by the four above (the native build re-resolves its
        map against the final node set, so its identity is the node set's);
      * this module's `version` + source hash — the automatic guard.

    Deliberately ABSENT: everything about the corpus that does not change which
    (person, node) pairs exist or when they were first coded — the vocabulary, the
    windows, the split, `doc_min_length`, `label_mask_mode`, the doc spec. A
    sidecar keyed on those would be rebuilt by every experiment that changes none
    of what it contains. Also absent: the horizon set (see the module docstring).
    """
    payload = {
        "artifact": "conversion_sidecar",
        "version": CONVERSION_SIDECAR_VERSION,
        "cdr": str(cdr or ""),
        "person_mod": int(person_mod),
        "dag_source": str(dag_source or ""),
        "mondo_version": str(mondo_version or ""),
        "mondo_branch": str(mondo_branch or ""),
        "min_positives": int(min_positives or 0),
        "code_map_identity": str(code_map_identity or ""),
        "src": _module_source_hash(sys.modules[__name__]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def normalize_code_map(code_map_sdf, *, concept_col, node_col):
    """Any provider's code frame -> the two columns this module joins on.

    The two label front ends name them differently — the legacy Mondo climb emits
    `(descendant_concept_id, ancestor_concept_id)` and exp 0110's native build
    emits `(std_cid, node_cid)` — and both mean "this OMOP concept attests this
    label node". Normalizing at the seam keeps that difference in ONE place
    instead of forking the aggregation."""
    from pyspark.sql import functions as F
    return (code_map_sdf
            .select(F.col(concept_col).cast("long").alias("concept_id"),
                    F.col(node_col).cast("long").alias(NODE_COL))
            .distinct())


def build_conversion_sidecar(cond, code_map, *, date_col="condition_era_start_date"):
    """`(person_id, node_cid, first_attested_date)` over the FULL history.

    `cond` is the unwindowed condition frame — `load_omop_bigquery(...,
    source_table="condition_era")`, which applies only `person_sample_mod` and
    `concept_id != 0`, so its date range is the record's. `code_map` is
    `normalize_code_map`'s output.

    An INNER join on purpose: a person with no code resolving to any label node has
    no row here, and "no row" is the correct reading of "never attested" at every
    horizon. The analysis treats a missing pair as no-conversion rather than as
    missing data, and says so where it does."""
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast
    return (cond.join(broadcast(code_map), on="concept_id", how="inner")
            .groupBy(PERSON_COL, NODE_COL)
            .agg(F.min(F.col(date_col)).alias(FIRST_DATE_COL))
            .select(*SIDECAR_COLUMNS))


def observation_gate_frame(observation_period):
    """`(person_id, observation_period_end_date)` — the LAST day each person is
    observed, for R4.4's horizon denominators.

    `_window_observed_cohort` (`cohorts.py:693-729`) applies two gates: a prior
    lookback and a follow-up `index + window <= observation_period_end_date`. E4
    needs the SECOND one, at three different horizons, against an index date the
    cohort table already fixed — so what it needs from that function is the
    max observation end per person, and the comparison is made at analysis time,
    once per horizon.

    RE-DERIVED, NOT IMPORTED, and that is the whole point: `cohorts.py` is folded
    into EVERY bundle cache key through `cohort_defs_version()`, so calling into it
    is free but EDITING it — which extending `_window_observed_cohort` to return a
    per-horizon frame would require — would orphan every cached bundle in the repo.
    E1's `preindex_closure` made the same choice for the same reason and re-derived
    the index rather than editing the index builder.

    `max` over a person's periods, matching `_window_observed_cohort`'s `distinct()`
    semantics: a person passes the follow-up gate if ANY of their observation
    periods covers the horizon, so the binding value is the latest end date."""
    from pyspark.sql import functions as F
    return (observation_period
            .groupBy(PERSON_COL)
            .agg(F.max(F.col(OBS_END_COL)).alias(OBS_END_COL)))


def build_index_horizon_frame(index_df, observation_period):
    """`(person_id, index_date, observation_period_end_date)` — the PER-PERSON half.

    A grain change happens here and is named where it happens (R4.3): `index_df` is
    per `(person_id, index_date)` and `observation_period` is one row per period, so
    the join is per-person after `observation_gate_frame` has reduced the periods to
    one max-end row each. Under today's doc unit each person has exactly one index,
    which is what makes "per person" the right grain for this file; exp 0111's
    episode unit makes it per (person, index), and this frame is then the one that
    has to gain a column rather than the one that silently fans out.

    LEFT join on the observation gate: a person with no observation-period row keeps
    a NULL end date, which the analysis reads as "not observed at any horizon" —
    excluded from the denominator, never counted as a non-converter."""
    from pyspark.sql import functions as F
    gate = observation_gate_frame(observation_period)
    return (index_df.select(PERSON_COL, F.col(INDEX_COL).alias(INDEX_COL))
            .join(gate, on=PERSON_COL, how="left")
            .select(*HORIZON_COLUMNS))


def load_observation_period(spark, *, cdr, billing):
    """The OMOP `observation_period` table, read exactly as `cohorts.py:807` reads
    it — same three columns, same connector options, so the follow-up gate this
    feeds is the assembler's gate and not a lookalike."""
    return (spark.read.format("bigquery")
            .option("table", f"{cdr}.observation_period")
            .option("parentProject", billing)
            .load()
            .select(PERSON_COL, "observation_period_start_date", OBS_END_COL))


# --------------------------------------------------------------------------- #
# Persistence (ADR 0025's pattern: its own dir, its own key, joined at use).   #
# --------------------------------------------------------------------------- #
def sidecar_dir(sidecar_uri, key) -> str:
    return f"{str(sidecar_uri).rstrip('/')}/{key}"


def sidecar_path(sidecar_uri, key, file=FIRST_ATTESTATION_FILE) -> str:
    return f"{sidecar_dir(sidecar_uri, key)}/{file}"


def save_sidecar(df, sidecar_uri, key, file=FIRST_ATTESTATION_FILE) -> str:
    """Write one of the sidecar's two parquets and return its path. Overwrite: the
    key IS the identity, so a rewrite under the same key is the same content."""
    path = sidecar_path(sidecar_uri, key, file)
    df.write.mode("overwrite").parquet(path)
    return path


def _try_read(spark, path, want, what):
    try:
        df = spark.read.parquet(path)
    except Exception:
        return None
    have = set(df.columns)
    missing = [c for c in want if c not in have]
    if missing:
        raise ValueError(
            f"the parquet at {path} is not {what}: columns {sorted(have)} are "
            f"missing {missing}. That key names a DIFFERENT artifact — do not "
            f"join it.")
    return df.select(*want)


def try_load_sidecar(spark, sidecar_uri, key):
    """The cached first-attestation frame, or None on any miss/read failure.

    Same contract as `_case_finding_cache.try_load`, and the same reason for the
    bare `except`: a miss is a normal outcome (a fresh bucket, a new key), not an
    error the caller should have to classify. A parquet that IS there but has the
    wrong columns is a different matter and raises — that is a wrong-artifact join,
    which fails silently and expensively if allowed through."""
    return _try_read(spark, sidecar_path(sidecar_uri, key),
                     SIDECAR_COLUMNS, "a first-attestation sidecar")


def try_load_index_horizon(spark, sidecar_uri, key):
    """The cached per-person index/observation frame, or None on a miss."""
    return _try_read(spark, sidecar_path(sidecar_uri, key, INDEX_HORIZON_FILE),
                     HORIZON_COLUMNS, "a sidecar index-horizon frame")


def sidecar_witness(key, uri, *, n_rows=None, n_persons=None) -> dict:
    """The witness recording which sidecar exists and what is in it.

    Written to the RUN DIR as `conversion_sidecar.json` rather than into
    `corpus_manifest`: the sidecar is keyed independently of the bundle (see the
    module docstring), and burying its key inside the corpus block would tell the
    next reader the opposite. `conversion_analysis` requires this file, exactly as
    the census requires E1's witness — no silent mixed-vintage joins."""
    out = {"version": CONVERSION_SIDECAR_VERSION, "key": str(key),
           "sidecar_uri": str(uri),
           "first_attestation": sidecar_path(uri, key),
           "index_horizon": sidecar_path(uri, key, INDEX_HORIZON_FILE),
           "grains": {"first_attestation": "per (person_id, node_cid)",
                      "index_horizon": "per person_id"}}
    if n_rows is not None:
        out["n_rows"] = int(n_rows)
    if n_persons is not None:
        out["n_persons"] = int(n_persons)
    return out


def format_sidecar_report(witness) -> str:
    """The build diagnostic, in `format_preindex_report`'s style."""
    w = witness or {}
    n = w.get("n_rows")
    rows = f", {n} (person, node) rows" if n is not None else ""
    ppl = (f", {w['n_persons']} persons" if w.get("n_persons") is not None else "")
    return "\n".join([
        f"[sidecar] first-attestation sidecar ({w.get('version', '?')}): "
        f"key={w.get('key', '?')}{rows}{ppl}",
        "[sidecar]   grains, never conflated: first_attestation is PER (PERSON, "
        "NODE); index_horizon is PER PERSON; E1's R_d is PER DOCUMENT",
        "[sidecar]   this is the ONLY artifact carrying POST-label-window dates — "
        "windowing discards them and the bundle holds BOW vectors, not events",
        "[sidecar]   its key is its OWN (cdr, person_mod, DAG identity, module "
        "source): a bundle-key move does not orphan it, and it does not move one",
    ])


# --------------------------------------------------------------------------- #
# The builder (ADR 0025's "separate make target for explicit re-builds").      #
# --------------------------------------------------------------------------- #
def build_and_save(spark, *, cdr, billing, person_mod, code_map, lookback_days,
                   label_window_days, index_mode, disease, sidecar_uri, key,
                   _cond=None, _observation_period=None):
    """Load full history once, write both parquets, return the witness.

    ONE `condition_era` scan serves both halves: the first-attestation aggregation
    and the deterministic index re-derivation (which is a pure function of the same
    frame plus the observation-period gate). That sharing is the plan's own note —
    pay one scan, not two — and it is why `_cond` is threaded rather than loaded
    twice."""
    from preindex_closure import feature_window_index_table

    cond = _cond
    if cond is None:
        from charmpheno.omop import load_omop_bigquery
        cond = load_omop_bigquery(
            spark=spark, cdr_dataset=cdr, billing_project=billing,
            person_sample_mod=person_mod, source_table="condition_era")
    cond = cond.cache()
    obs = _observation_period
    if obs is None:
        obs = load_observation_period(spark, cdr=cdr, billing=billing)

    first = build_conversion_sidecar(cond, code_map)
    n_rows = first.count()
    save_sidecar(first, sidecar_uri, key)

    index_df = feature_window_index_table(
        cond, spark=spark, cdr=cdr, billing=billing,
        date_col="condition_era_start_date",
        label_window_days=label_window_days, index_mode=index_mode,
        disease=disease)
    horizon = build_index_horizon_frame(index_df, obs)
    n_persons = horizon.count()
    save_sidecar(horizon, sidecar_uri, key, INDEX_HORIZON_FILE)
    cond.unpersist()
    return sidecar_witness(key, sidecar_uri, n_rows=n_rows, n_persons=n_persons)


def code_map_from_manifest(spark, corpus_manifest):
    """Rebuild the label front end's code map from a run's recorded corpus spec.

    The map is a by-product of the DAG build, which lives inside the MISS-ONLY
    assemble closure — so a corpus that is a cache HIT (every corpus this tool is
    pointed at) has no code map in hand and it has to be rebuilt. That costs the
    Mondo→OMOP mapping and the powering pass, minutes of BigQuery, which is why the
    sidecar is cached rather than recomputed per analysis.

    Both flavours are handled and NORMALIZED to `(concept_id, node_cid)`: the
    legacy climb keys on `(descendant_concept_id, ancestor_concept_id)`, the native
    build on `(std_cid, node_cid)`. Returns `(code_map, identity)`, where
    `identity` is the token the sidecar key folds for a map whose content the four
    DAG fields do not fully determine."""
    cm = corpus_manifest
    dag_source = str(cm.get("dag_source") or "")
    if dag_source not in ("mondo", "mondo_native"):
        raise ValueError(
            f"conversion sidecar: dag_source={dag_source!r} has no driver-side "
            "code map. E4 is defined over the Mondo label space (`mondo` or "
            "`mondo_native`); a SNOMED-anchor corpus would need its own map.")
    kwargs = dict(cdr=cm["cdr"], billing=cm["billing"],
                  mondo_version=cm.get("mondo_version") or "",
                  mondo_cache_dir=cm.get("mondo_cache_dir") or "data/mondo",
                  min_positives=int(cm.get("min_positives") or 0),
                  branch_root=(cm.get("mondo_branch") or None))
    if dag_source == "mondo_native":
        from mondo_native_dag import (MONDO_NATIVE_VERSION,
                                      build_mondo_native_fit_inputs)
        _dag, code_map_sdf, kept, _support, _stats = build_mondo_native_fit_inputs(
            spark, **kwargs)
        return (normalize_code_map(code_map_sdf, concept_col="std_cid",
                                   node_col="node_cid"),
                f"mondo_native:{MONDO_NATIVE_VERSION}:{len(kept)}")
    from mondo_dag import build_mondo_fit_inputs
    _dag, climb_sdf, terminals, _count, _red = build_mondo_fit_inputs(
        spark, **kwargs)
    return (normalize_code_map(climb_sdf,
                               concept_col="descendant_concept_id",
                               node_col="ancestor_concept_id"),
            f"mondo:{len(terminals)}")


def main(argv=None) -> int:
    """Build the sidecar for a finished run's corpus. Cluster-only (BigQuery)."""
    import argparse
    import json as _json

    from _driver_common import _phase, configure_logging, make_spark_session
    from gated_pc_readout import resolve_run_dir

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--sidecar-uri", default=None,
                   help="Root for the sidecar artifact. Defaults to "
                        "<bundle cache_uri>/conversion_sidecar — a SIBLING of the "
                        "bundle cache, never inside a bundle key's dir, because "
                        "the two keys are independent.")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even on a sidecar HIT.")
    args = p.parse_args(argv)
    configure_logging()

    run_dir = resolve_run_dir(args.run_dir)
    manifest = _json.loads((run_dir / "manifest.json").read_text())
    cm = manifest.get("corpus_manifest") or {}
    if not cm:
        print("[sidecar] ERROR: this manifest has no corpus_manifest block, so the "
              "label space cannot be rebuilt. Point at a run of the current "
              "driver.", flush=True)
        return 2
    sidecar_uri = args.sidecar_uri or f"{str(cm.get('cache_uri')).rstrip('/')}/conversion_sidecar"

    with make_spark_session(app_name="build-conversion-sidecar") as spark:
        with _phase("rebuild label code map"):
            code_map, identity = code_map_from_manifest(spark, cm)
            key = conversion_sidecar_key(
                cdr=cm.get("cdr"), person_mod=int(cm.get("person_mod") or 0),
                dag_source=cm.get("dag_source"),
                mondo_version=cm.get("mondo_version") or "",
                mondo_branch=cm.get("mondo_branch") or "",
                min_positives=int(cm.get("min_positives") or 0),
                code_map_identity=identity)
            print(f"[sidecar] key={key} uri={sidecar_uri}", flush=True)
        if not args.force and try_load_sidecar(spark, sidecar_uri, key) is not None:
            print("[sidecar] HIT — nothing to build (pass --force to rebuild).",
                  flush=True)
            witness = sidecar_witness(key, sidecar_uri)
        else:
            with _phase("first-attestation aggregation (one full-history scan)"):
                witness = build_and_save(
                    spark, cdr=cm.get("cdr"), billing=cm.get("billing"),
                    person_mod=int(cm.get("person_mod") or 0), code_map=code_map,
                    lookback_days=int(cm.get("lookback_days") or 365),
                    label_window_days=int(cm.get("label_window_days") or 365),
                    index_mode=cm.get("index_mode") or "population",
                    disease=cm.get("disease") or "rare6",
                    sidecar_uri=sidecar_uri, key=key)
        print(format_sidecar_report(witness), flush=True)
        out = run_dir / "conversion_sidecar.json"
        out.write_text(_json.dumps(witness, indent=2))
        print(f"[sidecar] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
