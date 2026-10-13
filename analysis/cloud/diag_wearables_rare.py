"""WEARABLES × RARE DISEASE census (side question, 2026-10-13; not a fit, no cache keys).

"How many All of Us participants have Fitbit data AND a rare-disease diagnosis, and
which rare diseases have the most wearables data?"

RARE = Mondo's rare-disease subsets. Mondo tags each disease with the sources that call
it rare (`orphanet_rare`, `gard_rare`, `nord_rare`, `inferred_rare`, `mondo_rare`, and
the union `rare`), so GARD / Orphanet / NORD are each a filter on one Mondo download.
Mondo -> OMOP goes through the repo's port of Monarch's mondo2omop
(`mondo_to_omop_mapping.build_mondo_to_omop`: Mondo `same_as` SNOMED / ICD10CM / MeSH
xref -> concept -> 'Maps to' -> standard Condition), then each mapped standard concept
is widened to its OMOP descendants (`concept_ancestor`), the same in-subtree rule the
fit's anchor powering uses.

WEARABLES = the Fitbit daily-summary tables of the CDR (`activity_summary`,
`heart_rate_summary`, `sleep_daily_summary` — whichever exist in this CDR version; the
driver says which it found). A participant's wear is the number of DISTINCT DAYS with
any daily summary, plus the first and last such day. A handful of days is not usable
data, so the headline columns require `--min-days` (default 30).

Per Mondo rare disease (pooled counts of persons):
  n_dx             persons with any in-subtree condition_occurrence
  n_wear           ... who also have >= --min-days of Fitbit data
  n_wear_before    ... whose Fitbit data STARTS on or before the first diagnosis
                   (wearables data from before the diagnosis — the early-detection case)
  n_wear_span      ... whose Fitbit span covers the first diagnosis date
Plus `group` (another mapped rare node sits under it, so its counts include its
children's) and `suspect` (n_dx above --suspect-share of the condition population: far
too common for a rare disease, usually an over-broad SNOMED xref).

EGRESS: rows with n_wear < 20 are dropped; any remaining cell in 1..19 prints as "<20".
Output goes to --out-dir (a workspace run dir), never the tree.

ADR 0047: Spark-native joins / groupBys only; the small (mondo_id, concept_id) map is a
broadcast-joined DataFrame, not a closure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

EGRESS_FLOOR = 20
_SOURCE_FLAGS = ("rare", "orphanet_rare", "gard_rare", "nord_rare", "inferred_rare",
                 "mondo_rare")
# (table, candidate date columns) — the Fitbit daily summaries of the AoU CDR.
_WEAR_TABLES = (("activity_summary", ("date",)),
                ("heart_rate_summary", ("date",)),
                ("sleep_daily_summary", ("sleep_date", "date")))


def rare_concept_map(mapping: pd.DataFrame, flags) -> pd.DataFrame:
    """(mondo_id, standard_concept_id) rows for Mondo nodes carrying ANY of `flags`,
    with one 0/1 column per known source flag (so the report can split by source)."""
    flags = list(flags)
    have = [f for f in _SOURCE_FLAGS if f in mapping.columns]
    m = mapping.copy()
    for f in have:
        m[f] = m[f].fillna(0).astype(int)
    sel = m[[f for f in flags if f in m.columns]].sum(axis=1) > 0
    out = (m.loc[sel, ["mondo_id", "standard_concept_id", *have]]
           .drop_duplicates(["mondo_id", "standard_concept_id"]))
    out["standard_concept_id"] = out["standard_concept_id"].astype("int64")
    return out.reset_index(drop=True)


def group_nodes(mondo_ids, child_adj) -> set:
    """Mapped rare nodes that have ANOTHER mapped rare node below them in Mondo: their
    counts include their descendants' patients (a grouping, not one disease)."""
    ids = set(mondo_ids)
    out = set()
    for m in ids:
        stack, seen = list(child_adj.get(m, ())), set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n in ids:
                out.add(m)
                break
            stack.extend(child_adj.get(n, ()))
    return out


def cell(n, floor=EGRESS_FLOOR) -> str:
    n = int(n)
    return "<20" if 0 < n < floor else f"{n:,}"


def format_report(rows: pd.DataFrame, overall: dict, *, min_days, top,
                  floor=EGRESS_FLOOR) -> str:
    """Markdown: the overall counts, then the top-`top` rare diseases by n_wear. Rows
    under the floor are dropped before this is called (`table_rows`)."""
    L = ["# Wearables × rare disease (All of Us)", "",
         f"Fitbit tables found: {', '.join(overall['wear_tables']) or 'NONE'} · "
         f"usable wear = >= {min_days} distinct days · rare = Mondo subsets "
         f"{', '.join(overall['flags'])} · {overall['n_rare_nodes']:,} rare Mondo nodes "
         f"mapped to {overall['n_rare_concepts']:,} standard concepts "
         f"({overall['n_rare_concepts_desc']:,} with descendants)", "",
         "| population | persons |", "|---|--:|"]
    for k, v in overall["counts"].items():
        L.append(f"| {k} | {cell(v, floor)} |")
    L += ["", f"## Rare diseases with the most wearables data (top {top}; "
          f"n_wear >= {floor})", "",
          "| Mondo | disease | n_dx | n_wear | % | wear before dx | wear spans dx | "
          "sources | note |", "|---|---|--:|--:|--:|--:|--:|---|---|"]
    for _, r in rows.head(int(top)).iterrows():
        note = ", ".join(x for x, f in (("group", r["group"]), ("SUSPECT broad", r["suspect"]))
                         if f)
        L.append(f"| {r['mondo_id']} | {str(r['name'])[:48]} | {cell(r['n_dx'], floor)} | "
                 f"{cell(r['n_wear'], floor)} | {100 * r['n_wear'] / max(r['n_dx'], 1):.1f} | "
                 f"{cell(r['n_wear_before'], floor)} | {cell(r['n_wear_span'], floor)} | "
                 f"{r['sources']} | {note} |")
    return "\n".join(L) + "\n"


def table_rows(per: pd.DataFrame, names: dict, groups: set, flags_by_node: dict, *,
               n_cond_pop, suspect_share, floor=EGRESS_FLOOR) -> pd.DataFrame:
    """Per-node rows over the floor, sorted by n_wear, with name / group / suspect /
    sources annotations."""
    d = per[per["n_wear"] >= floor].copy()
    d["name"] = d["mondo_id"].map(lambda m: names.get(m, m))
    d["group"] = d["mondo_id"].isin(groups)
    d["suspect"] = d["n_dx"] > float(suspect_share) * max(int(n_cond_pop), 1)
    d["sources"] = d["mondo_id"].map(
        lambda m: ",".join(f.replace("_rare", "") for f in flags_by_node.get(m, ())))
    return d.sort_values(["n_wear", "n_dx"], ascending=False).reset_index(drop=True)


def _wear_days(spark, cdr, billing, read_bq):
    """(person_id, fb_days, fb_start, fb_end) over the daily-summary tables found."""
    from pyspark.sql import functions as F
    parts, found = [], []
    for t, cols in _WEAR_TABLES:
        try:
            df = read_bq(spark, cdr, billing, t)
            dc = next((c for c in cols if c in df.columns), None)
            if dc is None:
                print(f"[wear] {t}: no date column among {cols} (has {df.columns[:8]}…); "
                      f"skipped", flush=True)
                continue
            parts.append(df.select("person_id", F.to_date(F.col(dc)).alias("d")))
            found.append(t)
        except Exception as e:  # table absent in this CDR version
            print(f"[wear] {t}: not readable ({type(e).__name__}); skipped", flush=True)
    if not parts:
        raise SystemExit("[wear] no Fitbit daily-summary table found in this CDR")
    u = parts[0]
    for p in parts[1:]:
        u = u.unionByName(p)
    wear = (u.where(F.col("d").isNotNull()).groupBy("person_id")
            .agg(F.countDistinct("d").alias("fb_days"), F.min("d").alias("fb_start"),
                 F.max("d").alias("fb_end")))
    return wear, found


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flags", default="rare",
                   help="comma list of Mondo rare subsets to include (any-of): "
                        + ", ".join(_SOURCE_FLAGS))
    p.add_argument("--min-days", type=int, default=30)
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--suspect-share", type=float, default=0.005,
                   help="flag a 'rare' node diagnosed in more than this share of the "
                        "persons with any condition (over-broad mapping)")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    args = p.parse_args(argv)
    cdr, billing = os.environ.get("WORKSPACE_CDR"), os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not cdr or not billing:
        raise SystemExit("WORKSPACE_CDR and GOOGLE_CLOUD_PROJECT must be set (source "
                         ".workspace_env; the make target does)")
    flags = [f.strip() for f in args.flags.split(",") if f.strip()]
    bad = [f for f in flags if f not in _SOURCE_FLAGS]
    if bad:
        raise SystemExit(f"unknown --flags {bad}; known: {_SOURCE_FLAGS}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.functions import broadcast

    from anchor_selection_cloud import _download_cached, _read_bq
    from mondo_to_omop_mapping import (_disease_child_adjacency, build_mondo_to_omop,
                                       seed_source_xrefs)
    spark = SparkSession.builder.appName("diag-wearables-rare").getOrCreate()

    # 1) Mondo rare nodes -> OMOP standard conditions (the fit's own mapping recipe).
    cache = Path(args.mondo_cache_dir)
    edges = pd.read_csv(_download_cached(args.mondo_version, "mondo_edges.tsv", cache),
                        sep="\t", low_memory=False)
    nodes = pd.read_csv(_download_cached(args.mondo_version, "mondo_nodes.tsv", cache),
                        sep="\t", low_memory=False)
    concept_pd = (_read_bq(spark, cdr, billing, "concept")
                  .select("concept_id", "concept_name", "vocabulary_id", "domain_id",
                          "concept_code", "standard_concept")
                  .where(F.col("vocabulary_id").isin("SNOMED", "ICD10CM", "MeSH"))
                  .toPandas())
    same_as = seed_source_xrefs(mondo_edges_df=edges, mondo_nodes_df=nodes,
                                restrict_mondo_ids=set(nodes["id"]))
    src = same_as.merge(concept_pd, on=["concept_code", "vocabulary_id"], how="inner")
    src_sdf = spark.createDataFrame(pd.DataFrame(
        {"concept_id_1": sorted({int(x) for x in src["concept_id"]})}))
    cr_pd = (_read_bq(spark, cdr, billing, "concept_relationship")
             .select("concept_id_1", "concept_id_2", "relationship_id")
             .where(F.col("relationship_id") == "Maps to")
             .join(broadcast(src_sdf), "concept_id_1", "inner").toPandas())
    mapping = build_mondo_to_omop(mondo_edges_df=edges, mondo_nodes_df=nodes,
                                  concept_df=concept_pd, concept_relationship_df=cr_pd)
    rmap = rare_concept_map(mapping, flags)
    names = dict(zip(nodes["id"], nodes["name"].astype(str)))
    flags_by_node = {m: [f for f in _SOURCE_FLAGS[1:] if f in g.columns and g[f].max()]
                     for m, g in rmap.groupby("mondo_id")}
    groups = group_nodes(set(rmap["mondo_id"]), _disease_child_adjacency(edges, nodes))
    print(f"[wear] rare map: {rmap['mondo_id'].nunique():,} Mondo nodes -> "
          f"{rmap['standard_concept_id'].nunique():,} standard concepts "
          f"({len(groups):,} are groupings)", flush=True)

    # 2) widen to OMOP descendants; first in-subtree diagnosis per (person, node).
    rm_sdf = spark.createDataFrame(rmap[["mondo_id", "standard_concept_id"]])
    ca = _read_bq(spark, cdr, billing, "concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id")
    node_concepts = (ca.join(broadcast(rm_sdf), ca["ancestor_concept_id"]
                             == rm_sdf["standard_concept_id"], "inner")
                     .select("mondo_id", F.col("descendant_concept_id").alias("cid"))
                     .distinct().cache())
    n_desc = node_concepts.select("cid").distinct().count()
    co = (_read_bq(spark, cdr, billing, "condition_occurrence")
          .select("person_id", F.col("condition_concept_id").alias("cid"),
                  F.to_date("condition_start_date").alias("d")))
    n_cond_pop = co.select("person_id").distinct().count()
    dx = (co.join(broadcast(node_concepts), "cid", "inner")
          .groupBy("person_id", "mondo_id").agg(F.min("d").alias("first_dx")).cache())

    # 3) Fitbit wear per person.
    wear, found = _wear_days(spark, cdr, billing, _read_bq)
    wear = wear.cache()
    ok = F.col("fb_days") >= int(args.min_days)
    j = dx.join(wear, "person_id", "left")
    per = (j.groupBy("mondo_id").agg(
        F.countDistinct("person_id").alias("n_dx"),
        F.countDistinct(F.when(ok, F.col("person_id"))).alias("n_wear"),
        F.countDistinct(F.when(ok & (F.col("fb_start") <= F.col("first_dx")),
                               F.col("person_id"))).alias("n_wear_before"),
        F.countDistinct(F.when(ok & (F.col("fb_start") <= F.col("first_dx"))
                               & (F.col("fb_end") >= F.col("first_dx")),
                               F.col("person_id"))).alias("n_wear_span"))
        .toPandas())

    # 4) overall counts (any rare, and per source subset).
    counts = {
        "persons with any condition": n_cond_pop,
        "persons with any Fitbit day": wear.count(),
        f"persons with >= {args.min_days} Fitbit days": wear.where(ok).count()}
    rare_p = dx.select("person_id").distinct()
    wear_ok = wear.where(ok).select("person_id")
    counts["persons with any rare dx"] = rare_p.count()
    counts[f"... and >= {args.min_days} Fitbit days"] = rare_p.join(wear_ok, "person_id").count()
    nongroup = dx.join(broadcast(spark.createDataFrame(
        pd.DataFrame({"mondo_id": sorted(set(rmap["mondo_id"]) - groups)}))), "mondo_id")
    counts["... rare dx on a non-grouping node, with wear"] = (
        nongroup.select("person_id").distinct().join(wear_ok, "person_id").count())
    for f in _SOURCE_FLAGS[1:4]:
        ids = sorted(rmap.loc[rmap[f] == 1, "mondo_id"].unique()) if f in rmap else []
        if not ids:
            continue
        sub = dx.join(broadcast(spark.createDataFrame(pd.DataFrame({"mondo_id": ids}))),
                      "mondo_id").select("person_id").distinct()
        counts[f"  {f} dx"] = sub.count()
        counts[f"  {f} dx with wear"] = sub.join(wear_ok, "person_id").count()

    rows = table_rows(per, names, groups, flags_by_node, n_cond_pop=n_cond_pop,
                      suspect_share=args.suspect_share)
    overall = {"wear_tables": found, "flags": flags,
               "n_rare_nodes": int(rmap["mondo_id"].nunique()),
               "n_rare_concepts": int(rmap["standard_concept_id"].nunique()),
               "n_rare_concepts_desc": int(n_desc), "counts": counts}
    text = format_report(rows, overall, min_days=args.min_days, top=args.top)
    tag = "_".join(flags) + f"_d{args.min_days}"
    (out / f"wearables_rare_{tag}.md").write_text(text)
    keep = ["mondo_id", "name", "n_dx", "n_wear", "n_wear_before", "n_wear_span",
            "group", "suspect", "sources"]
    tsv = rows[keep].copy()
    for c in ("n_dx", "n_wear", "n_wear_before", "n_wear_span"):
        tsv[c] = tsv[c].map(cell)
    tsv.to_csv(out / f"wearables_rare_{tag}.tsv", sep="\t", index=False)
    (out / f"wearables_rare_{tag}.json").write_text(json.dumps(overall, indent=1,
                                                               default=str))
    print(text, flush=True)
    print(f"[wear] wrote {out}/wearables_rare_{tag}.md/.tsv/.json (rows with n_wear < "
          f"{EGRESS_FLOOR} dropped, cells 1-19 as <20)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
