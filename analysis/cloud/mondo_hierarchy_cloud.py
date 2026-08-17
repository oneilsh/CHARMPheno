"""Whole-Mondo POWERED hierarchy — the actual DAG the model would fit (BQ-only).

Follows exp 0087 (whole-Mondo places 97.9% of coded patients). This builds the label
DAG for a whole-Mondo fit and reports its size / implied K:

  1. map WHOLE Mondo -> OMOP standard Condition anchors (faithful mondo2omop port,
     restrict_mondo_ids=None; scale-fixed with broadcast joins);
  2. POWER-COUNT each anchor (distinct persons with >=1 in-subtree condition) and keep
     those clearing --min-positives (the min-patient floor: a node no patient populates
     has no learnable topic);
  3. reduce the Mondo is-a DAG over the powered anchors to the compact branch-point
     hierarchy (anchor_hierarchy.reduce_to_anchor_hierarchy — O(#anchors) class nodes,
     not the raw ancestor closure);
  4. report #powered anchors + #class nodes => layout nodes and implied
     K = n_bg + nodes x tpn.

Patient counts are AoU small-cell suppressed (a positive count < 20 prints "<20").

Run:  make -C analysis/cloud exp ID=88   (model_class=mondo_hierarchy)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MIN_CELL = 20


def suppress(n: int) -> str:
    n = int(n)
    return f"<{_MIN_CELL}" if 0 < n < _MIN_CELL else str(n)


def format_k_report(n_anchors, n_powered, n_classes, *, n_bgs=(2, 8), tpn=1) -> str:
    """The DAG-size / implied-K summary. Pure (testable)."""
    nodes = n_powered + n_classes
    lines = [
        "=" * 74,
        "WHOLE-MONDO POWERED HIERARCHY",
        f"  mapped OMOP anchors:            {n_anchors}",
        f"  powered (>= floor):             {n_powered}",
        f"  compact class nodes (kept):     {n_classes}",
        f"  => layout nodes (powered+class):{nodes}",
    ]
    for n_bg in n_bgs:
        lines.append(f"  => K at n_bg={n_bg}, tpn={tpn}:          {n_bg + nodes * tpn}")
    lines.append("=" * 74)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--min-positives", type=int, default=100)
    p.add_argument("--min-class-size", type=int, default=2)
    p.add_argument("--max-class-fraction", type=float, default=1.0)
    p.add_argument("--tpn", type=int, default=1)
    args = p.parse_args(argv)

    import pandas as pd
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.functions import broadcast

    from charmpheno.omop.bigquery import load_omop_bigquery
    from anchor_selection_cloud import _download_cached, _read_bq
    from anchor_hierarchy import reduce_to_anchor_hierarchy
    from mondo_to_omop_mapping import (
        build_mondo_to_omop, seed_source_xrefs, _disease_child_adjacency,
        _HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY, _DISEASE_CHARACTERISTIC, _INJURY)

    spark = SparkSession.builder.appName("mondo-hierarchy").getOrCreate()

    # --- 1. whole-Mondo -> OMOP mapping (scale-fixed, restrict=None) ---
    cache = Path(args.mondo_cache_dir)
    edges_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_edges.tsv", cache),
                           sep="\t", low_memory=False)
    nodes_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_nodes.tsv", cache),
                           sep="\t", low_memory=False)
    all_ids = set(nodes_df["id"])
    concept_pd = (_read_bq(spark, args.cdr, args.billing, "concept")
                  .select("concept_id", "concept_name", "vocabulary_id", "domain_id",
                          "concept_code", "standard_concept")
                  .where(F.col("vocabulary_id").isin("SNOMED", "ICD10CM", "MeSH"))
                  .toPandas())
    same_as = seed_source_xrefs(mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
                                restrict_mondo_ids=all_ids)
    src = same_as.merge(concept_pd, on=["concept_code", "vocabulary_id"], how="inner")
    source_ids = sorted({int(x) for x in src["concept_id"]})
    src_sdf = spark.createDataFrame(pd.DataFrame({"concept_id_1": source_ids}))
    cr_pd = (_read_bq(spark, args.cdr, args.billing, "concept_relationship")
             .select("concept_id_1", "concept_id_2", "relationship_id")
             .where(F.col("relationship_id") == "Maps to")
             .join(broadcast(src_sdf), "concept_id_1", "inner")
             .toPandas())
    mapping = build_mondo_to_omop(
        mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
        concept_df=concept_pd, concept_relationship_df=cr_pd, restrict_mondo_ids=None)
    anchors = sorted({int(x) for x in mapping["standard_concept_id"]})

    # --- 2. power-count each anchor (broadcast-join; distinct persons per subtree) ---
    anchors_sdf = spark.createDataFrame(pd.DataFrame({"ancestor_concept_id": anchors}))
    ca = (_read_bq(spark, args.cdr, args.billing, "concept_ancestor")
          .select("ancestor_concept_id", "descendant_concept_id")
          .join(broadcast(anchors_sdf), "ancestor_concept_id", "inner"))
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
        source_table="condition_occurrence").select("person_id", "concept_id")
    counts = (cond.join(ca, cond.concept_id == ca.descendant_concept_id, "inner")
              .groupBy("ancestor_concept_id")
              .agg(F.countDistinct("person_id").alias("n")).toPandas())
    count_of = {int(r["ancestor_concept_id"]): int(r["n"]) for _, r in counts.iterrows()}
    powered = sorted(c for c in anchors if count_of.get(c, 0) >= args.min_positives)
    sys.stderr.write(
        f"[hier] mapped {len(anchors)} anchors; {len(powered)} clear "
        f">= {args.min_positives} positives\n")

    # --- 3. Mondo is-a parent adjacency + powered terminals, then reduce ---
    child_adj = _disease_child_adjacency(edges_df, nodes_df)   # parent -> [children]
    parent_adj: dict = {}
    for parent, children in child_adj.items():
        for c in children:
            parent_adj.setdefault(c, []).append(parent)
    anchor_name = dict(zip(mapping["standard_concept_id"].astype(int),
                           mapping["standard_concept_name"]))
    mondo_name = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}
    anchor_mondos: dict = {}
    for cid, mid in zip(mapping["standard_concept_id"].astype(int), mapping["mondo_id"]):
        if int(cid) in set(powered):
            anchor_mondos.setdefault(int(cid), []).append(str(mid))
    for cid, mids in anchor_mondos.items():
        parent_adj[f"anchor:{cid}"] = list(dict.fromkeys(mids))
    terminals = [f"anchor:{cid}" for cid in anchor_mondos]
    stop = {_HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY, _DISEASE_CHARACTERISTIC,
            _INJURY, "MONDO:0000001"}

    def label(node):
        if isinstance(node, str) and node.startswith("anchor:"):
            return anchor_name.get(int(node.split(":", 1)[1]), node)
        return mondo_name.get(str(node), str(node))

    h = reduce_to_anchor_hierarchy(
        terminals, parent_adj, stop=stop,
        min_class_size=args.min_class_size, max_class_fraction=args.max_class_fraction)

    # --- 4. report ---
    sys.stderr.write(format_k_report(
        len(anchors), len(powered), h["n_classes"], tpn=args.tpn) + "\n")
    sys.stderr.write(f"[hier] raw distinct Mondo ancestors AVOIDED as nodes: "
                     f"{h['n_raw_ancestors']}\n")
    # top class nodes by size (how many powered anchors each groups)
    for cid, info in sorted(h["classes"].items(), key=lambda kv: -kv[1]["size"])[:40]:
        sys.stderr.write(f"[class] [{info['size']:3d}] {label(cid)}\n")
    unclustered = [t for t, c in h["terminal_class"].items() if c is None]
    if unclustered:
        sys.stderr.write(f"[hier] unclustered (isolate) powered anchors: "
                         f"{len(unclustered)}\n")

    # --- artifacts ---
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "class", "id": cid, "label": label(cid), "size": info["size"],
             "parents": "|".join(h["parent_of"].get(cid, [])),
             "members": "|".join(label(m) for m in info["members"])}
            for cid, info in h["classes"].items()]
    rows += [{"type": "anchor", "id": t, "label": label(t),
              "size": 1, "n_patients": suppress(count_of.get(int(t.split(':', 1)[1]), 0)),
              "parents": "|".join(h["parent_of"].get(t, [])), "members": ""}
             for t in terminals]
    pd.DataFrame(rows).to_csv(out / "mondo_powered_hierarchy.tsv", sep="\t", index=False)
    sys.stderr.write(f"[hier] wrote {out}/mondo_powered_hierarchy.tsv "
                     f"({len(terminals)} anchors + {h['n_classes']} classes)\n")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
