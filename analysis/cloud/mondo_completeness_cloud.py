"""Whole-Mondo mapping-completeness + unplaced-condition diagnostic (BQ-only).

Follow-up to exp 0086 (the 760-seed coverage ladder). That run showed ~79% of coded
patients are NOT placed by the rare-disease seed; this driver asks the completeness
question for the WHOLE Mondo disease ontology (`restrict_mondo_ids=None`):

  1. MAPPING COMPLETENESS — how much of Mondo resolves to OMOP standard Condition
     concepts, and how much of the SNOMED condition space that covers.
  2. PLACEMENT LADDER — whole-population placement onto ANY Mondo node (via the
     SNOMED-climb roll-up), vs the truly-unplaced-coded residual.
  3. UNPLACED-CONDITION DIAGNOSTIC — for patients with condition codes that reach NO
     Mondo node, the top condition concepts they carry (concept_id, name, domain,
     standard flag, patient count). This is "what falls through": SNOMED disease
     concepts with no Mondo mapping + non-disease codes.

AoU SMALL-CELL SUPPRESSION: every reported patient count in (0, 20) prints as "<20"
(exact for >=20, "0"/omitted for 0). Applied to the ladder and the diagnostic table.

Reuses the faithful `mondo_to_omop_mapping.build_mondo_to_omop` port and the Mondo
auto-download from `anchor_selection_cloud`. SCALE: whole-Mondo yields ~10^4 source
concepts and ~10^4 target anchors — too many for a driver-side IN list — so the
`concept_relationship` and `concept_ancestor` filters are broadcast JOINS, not `isin`.

Run:  make -C analysis/cloud exp ID=87   (model_class=mondo_completeness)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MIN_CELL = 20


def suppress(n: int) -> str:
    """AoU small-cell rule: a positive count below the floor prints as '<20'."""
    n = int(n)
    return f"<{_MIN_CELL}" if 0 < n < _MIN_CELL else str(n)


def format_ladder(n_total, n_coded, n_placed, *, label="whole-Mondo") -> str:
    """The placement ladder, small-cell-suppressed. Pure (testable)."""
    def pct(n):
        return f"{100.0 * n / n_total:6.2f}%" if n_total else "   n/a"
    n_unplaced_coded = n_coded - n_placed
    n_nocode = n_total - n_coded
    return "\n".join([
        f"[coverage] ===== {label} placement ladder =====",
        f"[coverage]   persons (total)                {n_total:>11d}   100.00%",
        f"[coverage]   has >=1 condition code         {suppress(n_coded):>11}   "
        f"{pct(n_coded)}   (placeability ceiling)",
        f"[coverage]   placed on ANY Mondo node       {suppress(n_placed):>11}   "
        f"{pct(n_placed)}   (whole-Mondo foreground)",
        f"[coverage]   coded but UNPLACED             {suppress(n_unplaced_coded):>11}   "
        f"{pct(n_unplaced_coded)}   (falls through Mondo -> the diagnostic below)",
        f"[coverage]   residual: no condition code    {suppress(n_nocode):>11}   "
        f"{pct(n_nocode)}   (healthy/undocumented floor)",
        "[coverage] ================================================",
    ])


def format_unplaced(rows, n_unplaced_persons, n_suppressed_concepts) -> str:
    """Render the unplaced-condition diagnostic. ``rows`` = list of dicts
    (concept_id, concept_name, domain_id, standard_concept, n_patients) already
    sorted desc and filtered to the reported top-K; counts suppressed. Pure."""
    out = [f"[unplaced] top condition concepts among {suppress(n_unplaced_persons)} "
           f"coded-but-unplaced patients (what fails to reach any Mondo node):",
           f"[unplaced]   {'n_pts':>7}  {'concept_id':>9}  {'dom':>9}  std  name"]
    for r in rows:
        out.append(
            f"[unplaced]   {suppress(r['n_patients']):>7}  {int(r['concept_id']):>9}  "
            f"{str(r['domain_id'])[:9]:>9}  {str(r.get('standard_concept') or '-'):>3}  "
            f"{str(r['concept_name'])[:52]}")
    out.append(f"[unplaced]   (+ {n_suppressed_concepts} more distinct concepts, "
               f"each < {_MIN_CELL} patients)")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    p.add_argument("--out", required=True, help="output dir for the tsv artifacts")
    p.add_argument("--top-unplaced", type=int, default=100,
                   help="how many top unplaced condition concepts to list")
    args = p.parse_args(argv)

    import pandas as pd
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.functions import broadcast

    from charmpheno.omop.bigquery import load_omop_bigquery
    from anchor_selection_cloud import _download_cached, _read_bq
    from mondo_to_omop_mapping import build_mondo_to_omop, seed_source_xrefs

    spark = SparkSession.builder.appName("mondo-completeness").getOrCreate()

    # --- 1. Mondo frames (auto-download, same as anchor_selection_cloud) ---
    cache = Path(args.mondo_cache_dir)
    edges_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_edges.tsv", cache),
                           sep="\t", low_memory=False)
    nodes_df = pd.read_csv(_download_cached(args.mondo_version, "mondo_nodes.tsv", cache),
                           sep="\t", low_memory=False)
    all_ids = set(nodes_df["id"])

    # --- 2. OMOP concept slice (SNOMED/ICD/MeSH), the mapping's source universe ---
    concept_sdf = (_read_bq(spark, args.cdr, args.billing, "concept")
                   .select("concept_id", "concept_name", "vocabulary_id",
                           "domain_id", "concept_code", "standard_concept"))
    concept_pd = (concept_sdf
                  .where(F.col("vocabulary_id").isin("SNOMED", "ICD10CM", "MeSH"))
                  .toPandas())

    # --- 3. whole-Mondo source xrefs -> source OMOP concepts (pandas; Mondo-sized) ---
    same_as = seed_source_xrefs(mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
                                restrict_mondo_ids=all_ids)
    n_mondo_disease = same_as["mondo_id"].nunique()
    src = same_as.merge(concept_pd, on=["concept_code", "vocabulary_id"], how="inner")
    source_ids = sorted({int(x) for x in src["concept_id"]})

    # --- 4. concept_relationship 'Maps to' for the source concepts (JOIN, not isin) ---
    src_sdf = spark.createDataFrame(pd.DataFrame({"concept_id_1": source_ids}))
    cr_pd = (_read_bq(spark, args.cdr, args.billing, "concept_relationship")
             .select("concept_id_1", "concept_id_2", "relationship_id")
             .where(F.col("relationship_id") == "Maps to")
             .join(broadcast(src_sdf), "concept_id_1", "inner")
             .toPandas())

    # --- 5. whole-Mondo mapping -> OMOP standard Condition anchors ---
    mapping = build_mondo_to_omop(
        mondo_edges_df=edges_df, mondo_nodes_df=nodes_df,
        concept_df=concept_pd, concept_relationship_df=cr_pd,
        restrict_mondo_ids=None)
    anchors = sorted({int(x) for x in mapping["standard_concept_id"]})

    # SNOMED condition-space coverage: of all standard SNOMED Condition concepts, how
    # many are Mondo-mapped (as a target here — before roll-up).
    snomed_cond = concept_pd[(concept_pd["vocabulary_id"] == "SNOMED")
                             & (concept_pd["standard_concept"] == "S")
                             & (concept_pd["domain_id"] == "Condition")]
    n_snomed_cond = int(snomed_cond["concept_id"].nunique())
    n_snomed_cond_mapped = int(snomed_cond["concept_id"].isin(anchors).sum())

    sys.stderr.write(
        "[mapping] ===== whole-Mondo -> OMOP completeness =====\n"
        f"[mapping]   Mondo disease terms w/ >=1 SNOMED/ICD/MeSH xref: {n_mondo_disease}\n"
        f"[mapping]   -> matched OMOP source concepts:                 {len(source_ids)}\n"
        f"[mapping]   -> distinct OMOP standard Condition anchors:     {len(anchors)}\n"
        f"[mapping]   SNOMED standard Condition concepts (vocab):      {n_snomed_cond}\n"
        f"[mapping]   -> directly Mondo-mapped:                        {n_snomed_cond_mapped}"
        f" ({100.0 * n_snomed_cond_mapped / max(n_snomed_cond, 1):.1f}%)\n"
        "[mapping] ================================================\n")

    # --- 6. placement: roll patient conditions UP to any anchor (JOIN, not isin) ---
    anchors_sdf = spark.createDataFrame(
        pd.DataFrame({"ancestor_concept_id": anchors}))
    ca = (_read_bq(spark, args.cdr, args.billing, "concept_ancestor")
          .select("ancestor_concept_id", "descendant_concept_id")
          .join(broadcast(anchors_sdf), "ancestor_concept_id", "inner"))
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=args.cdr, billing_project=args.billing,
        source_table="condition_occurrence").select("person_id", "concept_id").cache()

    coded_persons = cond.select("person_id").distinct().cache()
    placed_persons = (cond.join(ca, cond.concept_id == ca.descendant_concept_id, "inner")
                      .select("person_id").distinct().cache())
    n_total = (_read_bq(spark, args.cdr, args.billing, "person")
               .select("person_id").distinct().count())
    n_coded = coded_persons.count()
    n_placed = placed_persons.count()
    sys.stderr.write(format_ladder(n_total, n_coded, n_placed) + "\n")

    # --- 7. unplaced-condition diagnostic: conditions of coded-but-unplaced patients ---
    unplaced_persons = coded_persons.join(placed_persons, "person_id", "left_anti").cache()
    n_unplaced = unplaced_persons.count()
    concept_names = concept_sdf.select(
        F.col("concept_id").alias("c_id"), "concept_name", "domain_id",
        "standard_concept")
    per_concept = (cond.join(unplaced_persons, "person_id", "inner")
                   .groupBy("concept_id")
                   .agg(F.countDistinct("person_id").alias("n_patients"))
                   .join(concept_names, F.col("concept_id") == F.col("c_id"), "left")
                   .orderBy(F.col("n_patients").desc()))
    top = per_concept.limit(args.top_unplaced).toPandas()
    n_ge = int((per_concept.where(F.col("n_patients") >= _MIN_CELL)).count())
    n_all_concepts = int(per_concept.count())
    rows = top.to_dict("records")
    sys.stderr.write(format_unplaced(
        rows, n_unplaced, max(n_all_concepts - n_ge, 0)) + "\n")

    # --- artifacts ---
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out / "mondo_omop_mapping.tsv", sep="\t", index=False)
    # suppress small cells in the persisted diagnostic too
    top = top.copy()
    top["n_patients"] = top["n_patients"].map(suppress)
    top.to_csv(out / "unplaced_top_conditions.tsv", sep="\t", index=False)
    sys.stderr.write(f"[done] wrote mapping ({len(mapping)} rows) + unplaced top "
                     f"conditions to {out}\n")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
