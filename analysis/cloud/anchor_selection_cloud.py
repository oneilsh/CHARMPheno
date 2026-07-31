"""Map the #1079 seed to OMOP anchors and count coded patients (cluster job).

Stages 2-5 of the expanded-SNOMED anchor-selection pipeline
(docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md):

  2. map    seed MONDO ids -> OMOP standard Condition concepts, via the faithful
            port of monarch-initiative/mondo2omop (mondo_to_omop_mapping.py),
            reading `concept` / `concept_relationship` slices from the CDR.
  3. expand each anchor -> its `concept_ancestor` descendant subtree.
  4. count  distinct persons with >=1 in-subtree condition_occurrence (the power
            proxy; the fit's exact positive set additionally applies the first-dx
            index + >=1yr lookback, which only shrinks these counts).
  5. filter keep anchors clearing --min-positives.

Output: a candidates table (one row per OMOP anchor: positive_count, the MONDO
ids/labels/#1079 categories mapping to it, rare flags) plus the raw mapping table
for provenance. Neighborhood assembly + the nesting rule run downstream, on the
powered survivors.

CLUSTER-COVERED: the BigQuery reads + Spark aggregation run on Dataproc and are
not unit-tested here; the pure mapping logic is (test_mondo_to_omop_mapping.py).
Run via `make -C analysis/cloud anchor-select`.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

_MONDO_RELEASE = (
    "https://github.com/monarch-initiative/mondo/releases/download/v{v}/{f}"
)


def _download_cached(version: str, filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{version}_{filename}"
    if dest.exists() and dest.stat().st_size > 0:
        sys.stderr.write(f"[mondo] cache hit {dest}\n")
        return dest
    url = _MONDO_RELEASE.format(v=version, f=filename)
    sys.stderr.write(f"[mondo] downloading {url}\n")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted Monarch release URL)
    return dest


def _read_bq(spark, cdr_dataset: str, billing_project: str, table: str):
    return (
        spark.read.format("bigquery")
        .option("table", f"{cdr_dataset}.{table}")
        .option("parentProject", billing_project)
        .load()
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cdr", required=True, help="<project>.<dataset>")
    p.add_argument("--billing", required=True, help="read-side billing project")
    p.add_argument("--seed-tsv", required=True,
                   help="anchor_selection_data/priority_seed.tsv")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    p.add_argument("--out", required=True, help="output candidates TSV path")
    p.add_argument("--min-positives", type=int, default=100)
    args = p.parse_args(argv)

    from pyspark.sql import SparkSession, functions as F

    from charmpheno.omop.bigquery import load_omop_bigquery
    from mondo_to_omop_mapping import build_mondo_to_omop, seed_source_xrefs

    # --- MONDO source frames (cached web download) ---
    cache = Path(args.mondo_cache_dir)
    edges_df = pd.read_csv(
        _download_cached(args.mondo_version, "mondo_edges.tsv", cache),
        sep="\t", low_memory=False,
    )
    nodes_df = pd.read_csv(
        _download_cached(args.mondo_version, "mondo_nodes.tsv", cache),
        sep="\t", low_memory=False,
    )

    # --- seed ---
    seed = pd.read_csv(args.seed_tsv, sep="\t")
    seed_ids = set(seed["mondo_id"])
    # per-MONDO-id label + the set of #1079 categories it seeds
    seed_cats = (
        seed.groupby("mondo_id")
        .agg(label=("label", "first"),
             categories=("category", lambda s: "|".join(sorted(set(s)))))
        .reset_index()
    )
    sys.stderr.write(f"[seed] {len(seed)} rows, {len(seed_ids)} distinct MONDO ids\n")

    spark = SparkSession.builder.appName("anchor-selection").getOrCreate()

    # --- bound the concept_relationship read to codes the seed can hit ---
    xrefs = seed_source_xrefs(
        mondo_edges_df=edges_df, mondo_nodes_df=nodes_df, restrict_mondo_ids=seed_ids
    )
    src_codes = set(xrefs["concept_code"])
    concept_pd = (
        _read_bq(spark, args.cdr, args.billing, "concept")
        .select("concept_id", "concept_name", "vocabulary_id", "domain_id",
                "concept_code", "standard_concept")
        .where(F.col("vocabulary_id").isin("SNOMED", "ICD10CM", "MeSH"))
        .toPandas()
    )
    source_ids = (
        xrefs.merge(concept_pd, on=["concept_code", "vocabulary_id"], how="inner")
        ["concept_id"].astype(int).unique().tolist()
    )
    sys.stderr.write(
        f"[map] {len(src_codes)} seed xref codes; {len(source_ids)} matched OMOP "
        f"source concepts\n"
    )
    cr_pd = (
        _read_bq(spark, args.cdr, args.billing, "concept_relationship")
        .select("concept_id_1", "concept_id_2", "relationship_id")
        .where((F.col("relationship_id") == "Maps to")
               & F.col("concept_id_1").isin(source_ids))
        .toPandas()
    )

    mapping = build_mondo_to_omop(
        mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
        concept_df=concept_pd, concept_relationship_df=cr_pd,
        restrict_mondo_ids=seed_ids,
    )
    anchors = sorted({int(x) for x in mapping["standard_concept_id"]})
    sys.stderr.write(
        f"[map] {len(mapping)} mapping rows -> {len(anchors)} distinct OMOP anchors\n"
    )

    # --- subtree expansion + patient counts ---
    ca = (
        _read_bq(spark, args.cdr, args.billing, "concept_ancestor")
        .select("ancestor_concept_id", "descendant_concept_id")
        .where(F.col("ancestor_concept_id").isin(anchors))
    )
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
        source_table="condition_occurrence",
    ).select("person_id", "concept_id")
    counts_pd = (
        cond.join(ca, cond.concept_id == ca.descendant_concept_id, "inner")
        .groupBy("ancestor_concept_id")
        .agg(F.countDistinct("person_id").alias("positive_count"))
        .toPandas()
    )

    # --- assemble one row per OMOP anchor ---
    per_anchor = (
        mapping.merge(seed_cats, on="mondo_id", how="left")
        .groupby(["standard_concept_id", "standard_concept_name"])
        .agg(
            n_mondo=("mondo_id", "nunique"),
            mondo_ids=("mondo_id", lambda s: "|".join(sorted(set(s)))),
            labels=("label", lambda s: "|".join(sorted({x for x in s if isinstance(x, str)}))),
            categories=("categories", lambda s: "|".join(sorted({c for v in s if isinstance(v, str) for c in v.split("|")}))),
            orphanet_rare_any=("orphanet_rare", "max"),
        )
        .reset_index()
        .merge(counts_pd, left_on="standard_concept_id",
               right_on="ancestor_concept_id", how="left")
    )
    per_anchor["positive_count"] = per_anchor["positive_count"].fillna(0).astype(int)
    per_anchor = per_anchor.drop(columns=["ancestor_concept_id"]).sort_values(
        "positive_count", ascending=False
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_anchor.to_csv(out, sep="\t", index=False)
    mapping.to_csv(out.with_suffix(".mapping.tsv"), sep="\t", index=False)

    powered = per_anchor[per_anchor["positive_count"] >= args.min_positives]
    sys.stderr.write(
        f"[done] {len(per_anchor)} anchors; {len(powered)} clear "
        f">= {args.min_positives} positives. wrote {out}\n"
    )
    # per-category powered survivor counts (a disease can seed several categories)
    cat_counts: dict[str, int] = {}
    for cats in powered["categories"]:
        for c in str(cats).split("|"):
            if c:
                cat_counts[c] = cat_counts.get(c, 0) + 1
    sys.stderr.write(f"[done] powered per category: {cat_counts}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
