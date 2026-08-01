"""Compact class hierarchy induced by a disease's anchors, from MONDO or SNOMED.

Answers "if we expand ancestors, how many class nodes do we actually get?" — the
worry being the raw ancestor closure is huge. It maps/expands the anchors'
ancestors, reduces the induced DAG to just the informative branch points
(anchor_hierarchy.reduce_to_anchor_hierarchy), and reports the kept class count
(O(#anchors), not the raw closure) plus the class tree.

Two source ontologies (``--source``), same reduction:
  mondo  - MONDO disease classification (clean, purpose-built; class nodes are
           ABSTRACT — not tokens patients carry).
  snomed - OMOP ``concept_ancestor`` over Condition-domain standard concepts (ONE
           ontology with the anchors + their sub-anchor DAG; class nodes are REAL
           concepts, so patients coded at a general level, e.g. "vasculitis",
           attach to the class node). Reuses the pipeline's concept_ancestor.

No patient data is read — output is pure ontology structure, safe to paste/commit.
Artifacts under --out-dir: anchor_hierarchy[.<source>].tsv (a ready DagLayout
parent_of) and anchor_classes[.<source>].tsv (anchor -> most-specific class).

Run (Dataproc master):
  make -C analysis/cloud anchor-hierarchy                    # MONDO (default)
  make -C analysis/cloud anchor-hierarchy SOURCE=snomed      # SNOMED, to compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from anchor_selection_cloud import _download_cached, _read_bq
from anchor_hierarchy import reduce_to_anchor_hierarchy


def _mondo_inputs(spark, args):
    """(terminals, parent_adj, stop, label_fn) from the MONDO disease DAG."""
    from charmpheno.omop.cohorts import disease_anchors
    from mondo_to_omop_mapping import (
        build_mondo_to_omop, seed_source_xrefs, _disease_child_adjacency,
        _HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY, _DISEASE_CHARACTERISTIC, _INJURY,
    )
    from pyspark.sql import functions as F

    cache = Path(args.mondo_cache_dir)
    edges_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_edges.tsv", cache),
                           sep="\t", low_memory=False)
    nodes_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_nodes.tsv", cache),
                           sep="\t", low_memory=False)
    seed = pd.read_csv(args.seed_tsv, sep="\t")
    seed_ids = set(seed["mondo_id"])
    anchor_ids = {int(c) for c in disease_anchors(args.disease)}

    xrefs = seed_source_xrefs(mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
                              restrict_mondo_ids=seed_ids)
    concept_pd = (_read_bq(spark, args.cdr, args.billing, "concept")
                  .select("concept_id", "concept_name", "vocabulary_id", "domain_id",
                          "concept_code", "standard_concept")
                  .where(F.col("vocabulary_id").isin("SNOMED", "ICD10CM", "MeSH"))
                  .toPandas())
    source_ids = (xrefs.merge(concept_pd, on=["concept_code", "vocabulary_id"], how="inner")
                  ["concept_id"].astype(int).unique().tolist())
    cr_pd = (_read_bq(spark, args.cdr, args.billing, "concept_relationship")
             .select("concept_id_1", "concept_id_2", "relationship_id")
             .where((F.col("relationship_id") == "Maps to")
                    & F.col("concept_id_1").isin(source_ids))
             .toPandas())
    mapping = build_mondo_to_omop(
        mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
        concept_df=concept_pd, concept_relationship_df=cr_pd,
        restrict_mondo_ids=seed_ids)
    mapping = mapping[mapping["standard_concept_id"].astype(int).isin(anchor_ids)]

    anchor_name = dict(zip(mapping["standard_concept_id"].astype(int),
                           mapping["standard_concept_name"]))
    mondo_name = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}
    child_adj = _disease_child_adjacency(edges_df, nodes_df)  # parent -> [children]
    parent_adj: dict = {}
    for parent, children in child_adj.items():
        for c in children:
            parent_adj.setdefault(c, []).append(parent)
    anchor_mondos: dict = {}
    for cid, mid in zip(mapping["standard_concept_id"].astype(int), mapping["mondo_id"]):
        anchor_mondos.setdefault(cid, []).append(str(mid))
    for cid, mids in anchor_mondos.items():
        parent_adj[f"anchor:{cid}"] = list(dict.fromkeys(mids))
    terminals = [f"anchor:{cid}" for cid in anchor_mondos]
    stop = {_HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY, _DISEASE_CHARACTERISTIC,
            _INJURY, "MONDO:0000001"}

    def label(node):
        if node.startswith("anchor:"):
            return anchor_name.get(int(node.split(":", 1)[1]), node)
        return mondo_name.get(node, node)

    return terminals, parent_adj, stop, label, len(anchor_ids)


def _snomed_inputs(spark, args):
    """(terminals, parent_adj, stop, label_fn) from OMOP concept_ancestor over
    Condition-domain standard concepts. Two reads: anchor->ancestors and
    class->ancestors, so the reduction can order classes by is-a specificity."""
    from charmpheno.omop.cohorts import disease_anchors
    from pyspark.sql import functions as F

    anchor_ids = sorted({int(c) for c in disease_anchors(args.disease)})
    # 1) ancestors of the anchors (drop self-pairs).
    ca_anchor = (_read_bq(spark, args.cdr, args.billing, "concept_ancestor")
                 .select("ancestor_concept_id", "descendant_concept_id")
                 .where(F.col("descendant_concept_id").isin(anchor_ids)
                        & (F.col("ancestor_concept_id") != F.col("descendant_concept_id")))
                 .toPandas())
    anc_ids = sorted(set(int(x) for x in ca_anchor["ancestor_concept_id"]))
    # keep only Condition-domain STANDARD concepts as candidate class nodes.
    concept = (_read_bq(spark, args.cdr, args.billing, "concept")
               .select("concept_id", "concept_name", "domain_id", "standard_concept")
               .where(F.col("concept_id").isin(sorted(set(anc_ids) | set(anchor_ids))))
               .toPandas())
    name = {int(c): str(n) for c, n in zip(concept["concept_id"], concept["concept_name"])}
    valid = {int(c) for c, d, s in zip(concept["concept_id"], concept["domain_id"],
                                       concept["standard_concept"])
             if d == "Condition" and s == "S"} - set(anchor_ids)
    # 2) class->ancestors among the valid class set (for specificity ordering).
    ca_class = (_read_bq(spark, args.cdr, args.billing, "concept_ancestor")
                .select("ancestor_concept_id", "descendant_concept_id")
                .where(F.col("descendant_concept_id").isin(sorted(valid))
                       & F.col("ancestor_concept_id").isin(sorted(valid))
                       & (F.col("ancestor_concept_id") != F.col("descendant_concept_id")))
                .toPandas())

    parent_adj: dict = {}
    for anc, desc in zip(ca_anchor["ancestor_concept_id"], ca_anchor["descendant_concept_id"]):
        if int(anc) in valid:
            parent_adj.setdefault(f"anchor:{int(desc)}", []).append(str(int(anc)))
    for anc, desc in zip(ca_class["ancestor_concept_id"], ca_class["descendant_concept_id"]):
        parent_adj.setdefault(str(int(desc)), []).append(str(int(anc)))
    terminals = [f"anchor:{c}" for c in anchor_ids if f"anchor:{c}" in parent_adj]
    stop = set(str(x) for x in args.stop_ids)

    def label(node):
        cid = int(node.split(":", 1)[1]) if node.startswith("anchor:") else int(node)
        return name.get(cid, node)

    return terminals, parent_adj, stop, label, len(anchor_ids)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source", choices=["mondo", "snomed"], default="mondo")
    p.add_argument("--seed-tsv", help="required for --source mondo")
    p.add_argument("--disease", default="rare_priority")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    p.add_argument("--stop-ids", default="",
                   type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                   help="snomed: OMOP concept_ids to exclude as over-general classes")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-class-size", type=int, default=2)
    p.add_argument("--max-class-fraction", type=float, default=1.0)
    args = p.parse_args(argv)
    if args.source == "mondo" and not args.seed_tsv:
        p.error("--seed-tsv is required for --source mondo")

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName(f"anchor-hierarchy-{args.source}").getOrCreate()
    try:
        if args.source == "mondo":
            terminals, parent_adj, stop, label, n_anchors = _mondo_inputs(spark, args)
        else:
            terminals, parent_adj, stop, label, n_anchors = _snomed_inputs(spark, args)
    finally:
        spark.stop()

    h = reduce_to_anchor_hierarchy(
        terminals, parent_adj, stop=stop,
        min_class_size=args.min_class_size, max_class_fraction=args.max_class_fraction)

    print("=" * 74, flush=True)
    print(f"ANCHOR HIERARCHY  source={args.source}  disease={args.disease}  "
          f"anchors_mapped={len(terminals)}/{n_anchors}", flush=True)
    print(f"raw distinct ancestors (AVOIDED as nodes): {h['n_raw_ancestors']}", flush=True)
    print(f"compact class nodes kept: {h['n_classes']}  "
          f"(min_class_size={args.min_class_size}, "
          f"max_class_fraction={args.max_class_fraction})", flush=True)
    print(f"=> layout would be n_bg + ({h['n_classes']} classes + {len(terminals)} "
          f"anchors) x tpn", flush=True)
    print("-" * 74, flush=True)
    for cid, info in sorted(h["classes"].items(), key=lambda kv: -kv[1]["size"]):
        print(f"[{info['size']:2d}] {label(cid)}  ({cid})", flush=True)
        print(f"       {', '.join(label(m) for m in info['members'])}", flush=True)
    unclustered = [t for t, c in h["terminal_class"].items() if c is None]
    if unclustered:
        print("-" * 74, flush=True)
        print(f"unclustered anchors ({len(unclustered)}): "
              + ", ".join(label(t) for t in unclustered), flush=True)
    print("=" * 74, flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sfx = f".{args.source}"
    rows = [{"type": "class", "id": cid, "label": label(cid), "size": info["size"],
             "parents": "|".join(h["parent_of"].get(cid, [])),
             "members": "|".join(label(m) for m in info["members"])}
            for cid, info in h["classes"].items()]
    rows += [{"type": "anchor", "id": t, "label": label(t), "size": 1,
              "parents": "|".join(h["parent_of"].get(t, [])), "members": ""}
             for t in terminals]
    pd.DataFrame(rows).to_csv(out / f"anchor_hierarchy{sfx}.tsv", sep="\t", index=False)
    pd.DataFrame([{"concept_id": int(t.split(":", 1)[1]), "anchor_label": label(t),
                   "class_id": h["terminal_class"][t] or "",
                   "class_label": label(h["terminal_class"][t]) if h["terminal_class"][t] else ""}
                  for t in terminals]).to_csv(
        out / f"anchor_classes{sfx}.tsv", sep="\t", index=False)
    print(f"[hier] wrote {out}/anchor_hierarchy{sfx}.tsv + anchor_classes{sfx}.tsv",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
