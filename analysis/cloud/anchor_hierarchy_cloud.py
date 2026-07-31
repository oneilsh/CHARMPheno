"""Compute the compact Mondo class hierarchy induced by a disease's anchors.

Answers "if we expand ancestors in the Mondo DAG, how many class nodes do we
actually get?" — the worry being the raw ancestor closure is huge. It maps the
anchors to MONDO, reduces the induced ancestor DAG to just the informative branch
points (anchor_hierarchy.reduce_to_anchor_hierarchy), and reports how many class
nodes result (O(#anchors), not the raw closure) plus the class tree.

No patient data is read — output is pure ontology structure, safe to paste/commit.

Two artifacts under --out-dir:
  anchor_hierarchy.tsv  - one row per kept node (class or anchor): id, label,
                          type, size, parents, members. The compact parent_of is
                          a ready DagLayout for a future hierarchical fit.
  anchor_classes.tsv    - anchor concept_id -> most-specific class (id + label);
                          the class map the within-class conditional readout uses.

Run (Dataproc master):
  make -C analysis/cloud anchor-hierarchy
  make -C analysis/cloud anchor-hierarchy ANCHOR_HIER_ARGS='--min-class-size 2 --max-class-fraction 0.6'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from anchor_selection_cloud import _download_cached, _read_bq


def _parents_adjacency(edges_df, nodes_df):
    """child -> [parents] over Mondo Disease subclass_of (invert the shared
    parent->children builder so we can walk UP to ancestors)."""
    from mondo_to_omop_mapping import _disease_child_adjacency
    child_adj = _disease_child_adjacency(edges_df, nodes_df)  # parent -> [children]
    parents: dict = {}
    for parent, children in child_adj.items():
        for c in children:
            parents.setdefault(c, []).append(parent)
    return parents


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--seed-tsv", required=True)
    p.add_argument("--disease", default="rare_priority",
                   help="registered disease whose anchors to build over")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-class-size", type=int, default=2)
    p.add_argument("--max-class-fraction", type=float, default=1.0)
    args = p.parse_args(argv)

    from pyspark.sql import SparkSession, functions as F
    from charmpheno.omop.cohorts import disease_anchors
    from mondo_to_omop_mapping import (
        build_mondo_to_omop, seed_source_xrefs,
        _HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY, _DISEASE_CHARACTERISTIC, _INJURY,
    )
    from anchor_hierarchy import reduce_to_anchor_hierarchy

    cache = Path(args.mondo_cache_dir)
    edges_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_edges.tsv", cache),
                           sep="\t", low_memory=False)
    nodes_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_nodes.tsv", cache),
                           sep="\t", low_memory=False)
    seed = pd.read_csv(args.seed_tsv, sep="\t")
    seed_ids = set(seed["mondo_id"])
    anchor_ids = {int(c) for c in disease_anchors(args.disease)}
    sys.stderr.write(f"[hier] disease={args.disease} anchors={len(anchor_ids)}\n")

    spark = SparkSession.builder.appName("anchor-hierarchy").getOrCreate()

    # concept -> MONDO mapping (mirrors anchor_selection_cloud's bounded reads).
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
    spark.stop()

    # Restrict to THIS disease's anchors; build concept<->mondo, names.
    mapping = mapping[mapping["standard_concept_id"].astype(int).isin(anchor_ids)]
    anchor_name = dict(zip(mapping["standard_concept_id"].astype(int),
                           mapping["standard_concept_name"]))
    mondo_name = {str(i): str(n) for i, n in zip(nodes_df["id"], nodes_df["name"])}
    anchor_mondos: dict = {}
    for cid, mid in zip(mapping["standard_concept_id"].astype(int), mapping["mondo_id"]):
        anchor_mondos.setdefault(cid, []).append(str(mid))

    # parent adjacency = Mondo is-a, plus each anchor(terminal) -> its MONDO ids.
    parent_adj = _parents_adjacency(edges_df, nodes_df)
    for cid, mids in anchor_mondos.items():
        parent_adj[f"anchor:{cid}"] = list(dict.fromkeys(mids))
    terminals = [f"anchor:{cid}" for cid in anchor_mondos]
    stop = {_HUMAN_DISEASE, _DISEASE_SUSCEPTIBILITY, _DISEASE_CHARACTERISTIC,
            _INJURY, "MONDO:0000001"}

    h = reduce_to_anchor_hierarchy(
        terminals, parent_adj, stop=stop,
        min_class_size=args.min_class_size, max_class_fraction=args.max_class_fraction)

    def _lab(node):
        if node.startswith("anchor:"):
            return anchor_name.get(int(node.split(":", 1)[1]), node)
        return mondo_name.get(node, node)

    # --- report (privacy-safe: ontology only) ---
    print("=" * 74, flush=True)
    print(f"ANCHOR HIERARCHY  disease={args.disease}  anchors={len(terminals)}", flush=True)
    print(f"raw distinct ancestors (AVOIDED as nodes): {h['n_raw_ancestors']}", flush=True)
    print(f"compact class nodes kept: {h['n_classes']}  "
          f"(min_class_size={args.min_class_size}, "
          f"max_class_fraction={args.max_class_fraction})", flush=True)
    print(f"=> layout would be n_bg + ({h['n_classes']} classes + {len(terminals)} "
          f"anchors) x tpn", flush=True)
    print("-" * 74, flush=True)
    for cid, info in sorted(h["classes"].items(), key=lambda kv: -kv[1]["size"]):
        members = ", ".join(_lab(m) for m in info["members"])
        print(f"[{info['size']:2d}] {_lab(cid)}  ({cid})", flush=True)
        print(f"       {members}", flush=True)
    unclustered = [t for t, c in h["terminal_class"].items() if c is None]
    if unclustered:
        print("-" * 74, flush=True)
        print(f"unclustered anchors ({len(unclustered)}): "
              + ", ".join(_lab(t) for t in unclustered), flush=True)
    print("=" * 74, flush=True)

    # --- artifacts ---
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid, info in h["classes"].items():
        rows.append({"type": "class", "id": cid, "label": _lab(cid),
                     "size": info["size"],
                     "parents": "|".join(h["parent_of"].get(cid, [])),
                     "members": "|".join(_lab(m) for m in info["members"])})
    for t in terminals:
        rows.append({"type": "anchor", "id": t, "label": _lab(t), "size": 1,
                     "parents": "|".join(h["parent_of"].get(t, [])), "members": ""})
    pd.DataFrame(rows).to_csv(out / "anchor_hierarchy.tsv", sep="\t", index=False)

    cls_rows = [{"concept_id": int(t.split(":", 1)[1]),
                 "anchor_label": _lab(t),
                 "class_id": h["terminal_class"][t] or "",
                 "class_label": _lab(h["terminal_class"][t]) if h["terminal_class"][t] else ""}
                for t in terminals]
    pd.DataFrame(cls_rows).to_csv(out / "anchor_classes.tsv", sep="\t", index=False)
    print(f"[hier] wrote {out}/anchor_hierarchy.tsv + anchor_classes.tsv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
