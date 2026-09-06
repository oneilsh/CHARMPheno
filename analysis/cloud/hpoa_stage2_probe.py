"""Stage-2 IN-WORKSPACE probe: effective HPO-profile prior support vs the corpus.

Stage 1 (`hpoa_profile_survey.py`, offline) measured what the ONTOLOGY offers:
which CV-branch nodes have an HPOA phenotype profile and how much of it carries
a SNOMED xref (report: docs/reports/2026-09-06-hpoa-profile-survey-cv-branch.md).
That number is an upper bound on what a word-side eta prior could use. What the
upcoming "random init + HPO-profile eta prior" experiment actually needs is the
EFFECTIVE support against the fitted corpus, which only the workspace can
answer, per node c:

  n_in_vocab  how many of the profile's standard Condition concepts survived
              the leakage strip + min_df + vocab cap — by construction exactly
              the ones present in the bundle's condition-domain vocab_map, so
              membership there IS the test (no strip re-implementation);
  n_pos       observed positive TRAIN docs for c (label * mask, the same cells
              `diag_sibling_support` counts);
  n_hit       of those, docs whose condition BOW carries >=1 profile token —
              the fraction of the node's own positives the prior would push
              toward, i.e. whether the prior points AT the node's documents or
              past them.

Pipeline: the stage-1 ``--emit-codes`` TSV -> BigQuery `concept` (SNOMED source
codes) -> 'Maps to' via `concept_relationship` -> standard Condition concepts
(the same join shape as `mondo_to_omop_mapping.build_mondo_to_omop`, on codes
instead of Mondo same_as xrefs) -> the cached bundle's condition vocab index
space -> ONE treeAggregate over the TRAIN split.

Bundle located exactly like `gated_pc_readout` / `diag_sibling_support`:
recompute the cache key from the run's manifest and REQUIRE a HIT (a probe
never pays a rebuild — run the fit or readout first).

Closure discipline (ADR 0047 addendum, the `diag_incident_census` template):
the per-node token-index sets ride an EXPLICIT `sc.broadcast` (plain lists +
frozensets), partial arrays are allocated executor-side, the reduction identity
is a None sentinel, and the broadcast is destroyed after the pass.

EGRESS (AoU disclosure floor 20): the per-node table (patient counts!) is
written to the run dir and is WORKSPACE-INTERNAL, never committed. stdout and
the summary .md carry only counts-of-nodes, medians, and per-node coverage
FRACTIONS where the denominator >= 100 — with a numerator at or under the
floor bounded (`safe_coverage`), never exact, since fraction x denominator
would reconstruct the cell. Pooled totals go through
`mondo_usage_core.suppress_count`.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from _driver_common import _phase, configure_logging, make_spark_session
from gated_pc_readout import bundle_key_from_manifest, resolve_run_dir
from mondo_native_dag import mondo_cid
from mondo_usage_core import suppress_count

# The AoU disclosure floor (evaluate.py:76-78 / mondo_usage_core._MIN_CELL):
# any patient-derived cell at or under it never prints exact.
MIN_CELL = 20
# Fractions are disclosed only over denominators at least this large — well
# clear of the floor, and small-numerator fractions are bounded regardless.
MIN_DENOMINATOR = 100
# The headline: counts-of-nodes whose positive-doc coverage clears each bar.
COVERAGE_THRESHOLDS = (0.10, 0.25, 0.50, 0.75, 0.90)


# --------------------------------------------------------------------------- #
# Pure logic (no Spark, no BigQuery) — unit-tested in                          #
# tests/test_hpoa_stage2_probe.py                                              #
# --------------------------------------------------------------------------- #
def profile_token_sets(codes_df, eid_by_mondo, code_to_std, vocab_map):
    """Per-node in-vocab token-index sets from the emit-codes TSV.

    POSITIVE rows only: the probe measures where a BOOST prior would point, and
    a negative (NOT/excluded) term describes documents the node should NOT
    have — counting hits on those would answer a different question (they stay
    in the TSV for the eta prior's downweight side).

    Membership in ``vocab_map`` is the whole survivorship test: the assembler
    built that map AFTER the leakage strip + min_df + vocab cap, so a concept
    absent from it was dropped by one of those and a concept present survived
    all of them — no re-implementation, no drift.

    Returns ``(per_node, skipped)``: per_node maps mondo_id -> dict(eid,
    n_profile_concepts, n_in_vocab, tokens frozenset of vocab indices); skipped
    is the sorted list of mondo_ids with no engine id (nodes the label DAG
    dropped as unpowered/collapsed — expected, counted, not an error)."""
    neg = codes_df["neg"]
    if neg.dtype != bool:  # a TSV round-trip stringifies bools
        neg = neg.astype(str).str.lower().isin(("true", "1"))
    pos = codes_df[~neg]
    per_node, skipped = {}, []
    for mid, g in pos.groupby("mondo_id"):
        eid = eid_by_mondo.get(mid)
        if eid is None:
            skipped.append(mid)
            continue
        concept_ids = set()
        for code in set(g["code"].astype(str)):
            concept_ids |= code_to_std.get(code, set())
        tokens = frozenset(vocab_map[c] for c in concept_ids if c in vocab_map)
        per_node[mid] = {"eid": int(eid),
                         "n_profile_concepts": len(concept_ids),
                         "n_in_vocab": len(tokens),
                         "tokens": tokens}
    return per_node, sorted(skipped)


def _bow_index_set(v):
    """Nonzero token indices of one BOW cell, whatever the writer stored: a
    Spark ML SparseVector (indices/values), a DenseVector/ndarray (toArray),
    or a plain sequence — the same duck-typing the label decode uses."""
    idx = getattr(v, "indices", None)
    if idx is not None:
        vals = getattr(v, "values", None)
        if vals is None:
            return {int(i) for i in idx}
        return {int(i) for i, x in zip(idx, vals) if x != 0}
    arr = np.asarray(getattr(v, "toArray", lambda: v)(), dtype=float)
    return {int(i) for i in np.nonzero(arr)[0]}


def support_partial(rows, eids, token_sets, *, label_col="label",
                    mask_col="labelMask", feat_col="features_0"):
    """Fold rows into ``(n_pos, n_hit)`` float64 arrays over the PROBED nodes,
    or nothing for an empty partition.

    Arrays are allocated HERE, executor-side, and the reduction identity is a
    None sentinel — nothing array-shaped rides the task closure (ADR 0047
    addendum; `eids`/`token_sets` arrive via the driver's explicit broadcast).

    Per doc: the BOW index set is decoded ONCE (lazily — a doc positive for no
    probed node never pays the decode), then for each probed node c observed
    positive on this doc (label*mask, `diag_sibling_support`'s cells) n_pos
    increments, and n_hit too iff the BOW intersects c's profile token set."""
    out = None
    n = len(eids)
    for r in rows:
        y = np.asarray(getattr(r[label_col], "toArray",
                               lambda: r[label_col])(), float)
        m = np.asarray(getattr(r[mask_col], "toArray",
                               lambda: r[mask_col])(), float)
        if out is None:
            out = (np.zeros(n), np.zeros(n))
        n_pos, n_hit = out
        bow = None
        for j in range(n):
            c = eids[j]
            if c >= len(y) or y[c] * m[c] <= 0:
                continue
            n_pos[j] += 1.0
            if bow is None:
                bow = _bow_index_set(r[feat_col])
            if token_sets[j] & bow:
                n_hit[j] += 1.0
    return [out] if out is not None else []


def support_combine(a, b):
    """The None-identity combiner: a missing side contributes nothing."""
    if a is None:
        return b
    if b is None:
        return a
    return (a[0] + b[0], a[1] + b[1])


def safe_coverage(n_hit, n_pos, *, min_den=MIN_DENOMINATOR,
                  min_cell=MIN_CELL) -> str:
    """Disclosure-safe coverage string for one node.

    A fraction over a KNOWN denominator reconstructs its numerator, so the
    floor applies to the numerator too: n_pos under `min_den` -> "n/a"; a
    numerator of 0 is safe ("0.00"); a nonzero numerator at or under the floor
    prints as the BOUND the floor implies ("<= min_cell/n_pos"), never the
    exact ratio; above the floor the exact fraction is fine."""
    n_hit, n_pos = int(n_hit), int(n_pos)
    if n_pos < min_den:
        return "n/a"
    if n_hit <= 0:
        return "0.00"
    if n_hit <= min_cell:
        return f"≤{min(1.0, min_cell / n_pos):.2f}"
    return f"{n_hit / n_pos:.2f}"


def build_summary(rows, meta, *, min_den=MIN_DENOMINATOR,
                  min_cell=MIN_CELL) -> str:
    """The pooled, egress-SAFE markdown summary (stdout + sidecar .md).

    Discloses ONLY: counts of nodes, medians/quantiles over nodes, coverage
    fractions via `safe_coverage` (denominator >= min_den, small numerators
    bounded), and pooled totals through `suppress_count`. Per-node patient
    counts never appear — they live in the workspace-internal TSV.

    `rows`: dicts with mondo_id, name, n_profile_concepts, n_in_vocab, n_pos,
    n_hit. `meta`: run/C/provenance strings, plus n_skipped_not_in_dag."""
    L = ["# HPOA stage-2 probe — effective prior support (egress-safe summary)",
         "",
         f"Run {meta.get('run', '?')}; bundle C={meta.get('C', '?')}; "
         f"profile codes: {meta.get('profile_codes', '?')} "
         f"({meta.get('n_codes', '?')} SNOMED codes -> "
         f"{meta.get('n_std_concepts', '?')} standard Condition concepts).",
         f"Nodes probed: **{len(rows)}** "
         f"({meta.get('n_skipped_not_in_dag', 0)} profiled branch nodes are "
         "not in the run's label DAG — unpowered/collapsed, expected).", ""]

    niv = np.array([r["n_in_vocab"] for r in rows], dtype=float)
    L += ["## Profile realization in the fitted vocabulary "
          "(counts of CODES/NODES — not patient counts)", ""]
    if len(niv):
        L += [f"- in-vocab profile tokens per node: median "
              f"{np.median(niv):.0f} (p25 {np.percentile(niv, 25):.0f}, "
              f"p75 {np.percentile(niv, 75):.0f})",
              f"- nodes with 0 in-vocab tokens: {int((niv == 0).sum())}; "
              f">=1: {int((niv >= 1).sum())}; >=5: {int((niv >= 5).sum())}; "
              f">=20: {int((niv >= 20).sum())}", ""]

    # Coverage: only nodes whose positive cell is comfortably above the floor
    # enter the fraction section at all (min_den, not the bare floor).
    covered = [r for r in rows if r["n_pos"] >= min_den and r["n_in_vocab"] > 0]
    fracs = np.array([r["n_hit"] / r["n_pos"] for r in covered], dtype=float)
    L += [f"## Coverage of observed positive TRAIN docs "
          f"(nodes with n_pos >= {min_den} and >=1 in-vocab token: "
          f"{len(covered)} of {len(rows)} probed)", ""]
    if len(covered):
        bars = " / ".join(f"{t:.2f}" for t in COVERAGE_THRESHOLDS)
        hits = " / ".join(str(int((fracs >= t).sum()))
                          for t in COVERAGE_THRESHOLDS)
        L += [f"- median coverage fraction: {np.median(fracs):.2f}",
              f"- nodes with coverage >= {bars}: {hits}", ""]
        ranked = sorted(covered, key=lambda r: r["n_hit"] / r["n_pos"])
        L += ["Example nodes (fractions only; denominators >= "
              f"{min_den}, never printed; numerators <= {min_cell} bounded):",
              "", "| node | coverage |", "|---|---|"]
        show = ranked[-5:][::-1] + (ranked[:5] if len(ranked) > 5 else [])
        for r in show:
            cov = safe_coverage(r["n_hit"], r["n_pos"],
                                min_den=min_den, min_cell=min_cell)
            L.append(f"| {r['mondo_id']} {r['name']} | {cov} |")
        L.append("")

    total_pos = int(sum(r["n_pos"] for r in rows))
    total_hit = int(sum(r["n_hit"] for r in rows))
    L += ["## Pooled totals (label cells over all probed nodes; "
          f"suppress_count floor {min_cell})", "",
          f"- observed positive cells: {suppress_count(total_pos, min_cell)}",
          f"- cells hit by >=1 profile token: "
          f"{suppress_count(total_hit, min_cell)}", "",
          "Per-node patient counts live in the workspace-internal TSV next to "
          "this file and are NOT disclosable.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# The driver.                                                                  #
# --------------------------------------------------------------------------- #
def _resolve_env(args, manifest):
    """(cdr, billing) from args > the manifest's corpus record > the sourced
    .workspace_env — the same fallback chain `gated_pc_readout`'s rebuild path
    uses, and for the same reason (a null parentProject dies as an opaque py4j
    NPE)."""
    cm = manifest.get("corpus_manifest") or {}
    cdr = args.cdr or cm.get("cdr") or os.environ.get("WORKSPACE_CDR")
    billing = (args.billing or cm.get("billing")
               or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    return cdr, billing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--profile-codes", required=True,
                   help="the stage-1 --emit-codes TSV "
                        "(mondo_id, hp_id, neg, freq, vocab, code)")
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--cdr", default=None, help="<project>.<dataset>; defaults "
                   "to the manifest's record, then WORKSPACE_CDR")
    p.add_argument("--billing", default=None, help="read-side billing project; "
                   "defaults to the manifest, then GOOGLE_CLOUD_PROJECT")
    p.add_argument("--label-col", default="label")
    p.add_argument("--mask-col", default="labelMask")
    p.add_argument("--out-dir", default=None,
                   help="where the per-node TSV + summary land "
                        "(default: the run dir)")
    args = p.parse_args(argv)
    configure_logging()

    run_dir = resolve_run_dir(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    C = int(manifest["C"])
    cm = manifest.get("corpus_manifest") or {}
    cache_uri = args.cache_uri or cm.get("cache_uri")
    key = bundle_key_from_manifest(manifest)
    cdr, billing = _resolve_env(args, manifest)
    if not cdr or not billing:
        print("[probe] ERROR: need a CDR and billing project (pass --cdr/"
              "--billing, or source .workspace_env — make setup).", flush=True)
        return 2

    codes_df = pd.read_csv(args.profile_codes, sep="\t",
                           dtype={"code": str, "vocab": str})
    snomed_codes = sorted(
        set(codes_df.loc[codes_df["vocab"] == "SNOMED", "code"].astype(str)))
    print(f"[probe] profile TSV: {len(codes_df)} rows, "
          f"{codes_df['mondo_id'].nunique()} nodes, "
          f"{len(snomed_codes)} distinct SNOMED codes", flush=True)

    with make_spark_session(app_name="hpoa-stage2-probe") as spark:
        from pyspark.sql import functions as F

        from _case_finding_cache import try_load
        from anchor_selection_cloud import _read_bq

        with _phase("load cached bundle"):
            bundle = try_load(spark, cache_uri, key)
            if bundle is None:
                print(f"[probe] ERROR: cache MISS at {cache_uri}/{key} — this "
                      "probe never rebuilds; run the fit or gated_pc_readout "
                      "first so the bundle is cached.", flush=True)
                return 2
            # Domain 0 is conditions by the multi-domain convention; a
            # single-domain bundle has the one map/column.
            if hasattr(bundle, "vocab_maps"):
                vocab_map, feat_col = bundle.vocab_maps[0], "features_0"
            else:
                vocab_map, feat_col = bundle.vocab_map, "features"
            print(f"[probe] bundle HIT: C={C}, condition vocab "
                  f"|V|={len(vocab_map)}, feature col {feat_col}", flush=True)

        # SNOMED source codes -> standard Condition concept ids: the same
        # concept -> 'Maps to' -> standard-'S'/Condition join shape as
        # `build_mondo_to_omop`, keyed on concept_code instead of Mondo
        # same_as xrefs (that function's Mondo-frame inputs don't apply here).
        with _phase("BQ concept: SNOMED source codes"):
            src_pd = (_read_bq(spark, cdr, billing, "concept")
                      .select("concept_id", "concept_code", "vocabulary_id")
                      .where((F.col("vocabulary_id") == "SNOMED")
                             & F.col("concept_code").isin(snomed_codes))
                      .toPandas())
            source_ids = [int(x) for x in src_pd["concept_id"].unique()]
            print(f"[probe] {len(src_pd)} SNOMED source concepts for "
                  f"{src_pd['concept_code'].nunique()} of "
                  f"{len(snomed_codes)} codes", flush=True)

        with _phase("BQ concept_relationship: 'Maps to'"):
            cr_pd = (_read_bq(spark, cdr, billing, "concept_relationship")
                     .select("concept_id_1", "concept_id_2", "relationship_id")
                     .where((F.col("relationship_id") == "Maps to")
                            & F.col("concept_id_1").isin(source_ids))
                     .toPandas())

        with _phase("BQ concept: standard Condition filter"):
            mapped_ids = [int(x) for x in cr_pd["concept_id_2"].unique()]
            std_pd = (_read_bq(spark, cdr, billing, "concept")
                      .select("concept_id", "standard_concept", "domain_id")
                      .where(F.col("concept_id").isin(mapped_ids)
                             & (F.col("standard_concept") == "S")
                             & (F.col("domain_id") == "Condition"))
                      .toPandas())
            std_ids = {int(x) for x in std_pd["concept_id"]}
            code_to_std: dict[str, set] = {}
            by_src = cr_pd.groupby("concept_id_1")["concept_id_2"].agg(set)
            for code, cid in zip(src_pd["concept_code"], src_pd["concept_id"]):
                std = {int(t) for t in by_src.get(int(cid), set())} & std_ids
                if std:
                    code_to_std.setdefault(str(code), set()).update(std)
            n_std = len(set().union(*code_to_std.values())) if code_to_std else 0
            print(f"[probe] {len(code_to_std)} codes map to {n_std} standard "
                  "Condition concepts", flush=True)

        with _phase("map profiles into the vocab index space"):
            eid_by_mondo = {}
            for mid in codes_df["mondo_id"].unique():
                try:
                    eid = bundle.cid2int.get(mondo_cid(mid))
                except ValueError:
                    eid = None            # not a Mondo curie — skip, counted
                if eid is not None:
                    eid_by_mondo[str(mid)] = int(eid)
            per_node, skipped = profile_token_sets(
                codes_df, eid_by_mondo, code_to_std, vocab_map)
            if not per_node:
                print("[probe] ERROR: no profiled node maps into this run's "
                      "label DAG — is this a mondo_native run? (cid2int keys "
                      "must be Mondo numeric ids)", flush=True)
                return 3
            n_zero = sum(1 for v in per_node.values() if v["n_in_vocab"] == 0)
            print(f"[probe] {len(per_node)} nodes probed; {len(skipped)} not "
                  f"in the label DAG; {n_zero} with zero in-vocab tokens",
                  flush=True)

        with _phase("train support pass (one treeAggregate)"):
            order = sorted(per_node)
            eids = [per_node[m]["eid"] for m in order]
            token_sets = [per_node[m]["tokens"] for m in order]
            sc = spark.sparkContext
            # Explicit broadcast (plain lists + frozensets), so nothing
            # array-shaped rides the task closure; destroyed below — the
            # unpersist-only lifecycle leaks the driver-side pickle
            # (SparkStatsFn's leak #2).
            bcast = sc.broadcast((eids, token_sets))
            try:
                cols = (args.label_col, args.mask_col, feat_col)

                def _local(rows, _b=bcast, _cols=cols):
                    e, t = _b.value
                    return support_partial(rows, e, t, label_col=_cols[0],
                                           mask_col=_cols[1], feat_col=_cols[2])

                out = (bundle.train_df.select(*cols).rdd
                       .mapPartitions(_local)
                       .treeAggregate(None, support_combine, support_combine,
                                      depth=2))
            finally:
                bcast.destroy()
        n_pos, n_hit = ((np.zeros(len(order)), np.zeros(len(order)))
                        if out is None else out)

        with _phase("write outputs"):
            name_by_eid = {i: bundle.name_by_id.get(c, "?")
                           for i, c in bundle.int2cid.items()}
            rows = []
            for j, mid in enumerate(order):
                v = per_node[mid]
                rows.append({
                    "mondo_id": mid, "name": name_by_eid.get(v["eid"], "?"),
                    "engine_id": v["eid"],
                    "n_profile_concepts": v["n_profile_concepts"],
                    "n_in_vocab": v["n_in_vocab"],
                    "n_pos": int(n_pos[j]), "n_hit": int(n_hit[j]),
                })
            out_dir = args.out_dir or str(run_dir)
            os.makedirs(out_dir, exist_ok=True)
            tsv_path = os.path.join(out_dir, "hpoa_stage2_support.tsv")
            # WORKSPACE-INTERNAL: per-node patient counts, many under the
            # floor. Never committed, never pasted out of the workbench.
            pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
            print(f"[probe] wrote per-node table (WORKSPACE-INTERNAL, cells "
                  f"< {MIN_CELL} not disclosable): {tsv_path}", flush=True)

            meta = {"run": run_dir.name, "C": C,
                    "profile_codes": args.profile_codes,
                    "n_codes": len(snomed_codes), "n_std_concepts": n_std,
                    "n_skipped_not_in_dag": len(skipped)}
            summary = build_summary(rows, meta)
            md_path = os.path.join(out_dir, "hpoa_stage2_summary.md")
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(summary)
            print(summary, flush=True)
            print(f"[probe] wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
