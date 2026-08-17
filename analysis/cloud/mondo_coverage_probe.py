"""Mondo/SNOMED placement-coverage probe (BQ-only; NO model fit).

Answers the question behind the Mondo-backbone redesign (see the use-case discussion):
if we expand the node set from the 41 rare anchors UP/OUT to a Mondo-native disease
hierarchy, how many currently-"background" patients get PLACED on a real disease node,
and how big is the irreducible truly-healthy residual?

Mechanism (the SNOMED-climb trick): a patient's standard SNOMED condition code `s` is
PLACEABLE at a target node `t` iff some `concept_ancestor` of `s` (or `s` itself) is in
the target set T. So we do NOT need a Mondo mapping for every leaf — only for SOME
ancestor. We roll patient conditions UP to T by filtering `concept_ancestor` to
`ancestor_concept_id IN T` FIRST (the O(n)-ancestors trick — the same pattern
case_finding_assembly uses), then joining on `descendant = condition_concept_id`.

Produces a COVERAGE LADDER over a person-mod sample:
  N total sampled
   ├ has >=1 standard condition code                 (complement = no-code residual)
   ├ rolls up to the SNOMED Disease hierarchy        (SNOMED placeability CEILING)
   ├ under the 41 current anchors                    (status-quo foreground)
   ├ placed on a Mondo-mapped node (roll-up)         (HEADLINE proposed foreground)
   └ placed on a Mondo TOP-LEVEL category            (background-gets-placed payoff)
plus a node-support histogram (how many target nodes clear a patient-count threshold —
sizes the tree / K), and the Mondo-vocab detection result.

Mondo source: if the CDR vocabulary already contains Mondo (vocabulary_id ILIKE
'%mondo%'), the SNOMED<->Mondo mapping is built IN-CDR from `concept_relationship`
('Maps to' from non-standard Mondo concepts to standard SNOMED). If Mondo is absent from
the vocab, Stage B is skipped and the ceiling (Stage A) still prints — stage the Mondo
SSSOM as a side table and pass --mondo-map-table to run Stage B.

Run (adapt to your cluster's spark-submit wrapper):
  spark-submit analysis/cloud/mondo_coverage_probe.py \
    --cdr <project.dataset> --billing <billing-project> --person-mod 20

This is read-only and cheap-ish: the heavy join is bounded by filtering concept_ancestor
to the target ancestor set before touching condition_occurrence.
"""
from __future__ import annotations

import argparse

# --- OMOP standard concept_ids (SNOMED hierarchy roots). Adjust if your vocab differs;
#     verify once with: SELECT concept_id, concept_name FROM concept
#     WHERE concept_code IN ('64572001','404684003') AND vocabulary_id='SNOMED'. ---
SNOMED_DISEASE = 4274025        # SNOMED "Disease" (disorder) — the disease-hierarchy root
SNOMED_CLINICAL_FINDING = 441840  # broader "Clinical finding" (incl. symptoms/findings)


def _reader(spark, cdr, billing):
    def _read(table):
        return (spark.read.format("bigquery")
                .option("table", f"{cdr}.{table}")
                .option("parentProject", billing).load())
    return _read


def _target_ancestor_set(read, target_concept_ids):
    """Rows of concept_ancestor whose ANCESTOR is a target node — the pre-filter that
    makes the roll-up cheap. Returns (descendant_concept_id, ancestor_concept_id)."""
    from pyspark.sql import functions as F
    tgt = read("concept_ancestor").select(
        F.col("descendant_concept_id"), F.col("ancestor_concept_id"))
    return tgt.where(F.col("ancestor_concept_id").isin(list(target_concept_ids)))


def _placed_persons(cond_pc, ca_filtered):
    """DISTINCT person_ids whose condition rolls up to at least one filtered ancestor.

    ``cond_pc`` = distinct (person_id, condition_concept_id); ``ca_filtered`` =
    (descendant, ancestor) already restricted to target ancestors."""
    return (cond_pc.join(
        ca_filtered,
        cond_pc.condition_concept_id == ca_filtered.descendant_concept_id, "inner")
        .select("person_id").distinct())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdr", required=True, help="BQ dataset, e.g. project.dataset")
    ap.add_argument("--billing", required=True, help="parentProject / billing project")
    ap.add_argument("--person-mod", type=int, default=20,
                    help="sample: keep persons with MOD(person_id, N)==0 (N=20 -> ~5%%)")
    ap.add_argument("--mondo-map-table", default="",
                    help="BYO Mondo mapping BQ table (CDR vocab has no Mondo). Accepts "
                         "EITHER a `snomed_code` column (bare SNOMED CT SCTID as string, "
                         "as in the Mondo SSSOM object_id — joined to concept.concept_code "
                         "here) OR a pre-resolved `snomed_concept_id` column, plus an "
                         "optional `mondo_id`. If empty, Mondo mapping is built from the "
                         "CDR concept_relationship when Mondo IS present in the vocab.")
    ap.add_argument("--anchors", default="",
                    help="comma-separated OMOP concept_ids for the current-anchor row; "
                         "defaults to cohorts._RARE_PRIORITY_ANCESTORS")
    ap.add_argument("--support-thresholds", default="20,50,100,500")
    ap.add_argument("--out-dir", default="",
                    help="optional; if set, the report is also written to "
                         "<out-dir>/mondo_coverage_report.txt (harness contract).")
    args = ap.parse_args()

    from pyspark.sql import SparkSession, functions as F
    spark = SparkSession.builder.appName("mondo-coverage-probe").getOrCreate()
    read = _reader(spark, args.cdr, args.billing)

    # --- sample persons ---
    persons = read("person").select("person_id").where(
        F.pmod(F.col("person_id"), F.lit(args.person_mod)) == 0).cache()
    n_total = persons.count()

    # --- distinct (person, standard-condition) over the sample ---
    concept = read("concept").select(
        "concept_id", "concept_name", "concept_code",
        "standard_concept", "vocabulary_id", "domain_id")
    std_cond = concept.where((F.col("domain_id") == "Condition")
                             & (F.col("standard_concept") == "S")).select(
        F.col("concept_id").alias("std_cid"))
    cond = (read("condition_occurrence")
            .select("person_id", "condition_concept_id")
            .join(persons, "person_id", "inner")
            .join(std_cond, F.col("condition_concept_id") == std_cond.std_cid, "inner")
            .select("person_id", "condition_concept_id").distinct().cache())
    n_any_condition = cond.select("person_id").distinct().count()

    # --- Stage A: SNOMED placeability ceiling (roll up to the Disease hierarchy) ---
    ca_disease = _target_ancestor_set(read, [SNOMED_DISEASE])
    n_disease = _placed_persons(cond, ca_disease).count()

    # --- current 41-anchor foreground ---
    if args.anchors.strip():
        anchor_ids = [int(x) for x in args.anchors.split(",") if x.strip()]
    else:
        from charmpheno.omop.cohorts import _RARE_PRIORITY_ANCESTORS
        anchor_ids = list(_RARE_PRIORITY_ANCESTORS)
    ca_anchor = _target_ancestor_set(read, anchor_ids)
    n_anchor = _placed_persons(cond, ca_anchor).count()

    # --- Stage B: Mondo landing. Build the target SNOMED set (SNOMED concepts that carry
    #     a Mondo mapping), then roll patient conditions up to it. ---
    mondo_targets = None          # DataFrame[t_cid]  (target SNOMED OMOP concept_ids)
    mondo_source = None
    if args.mondo_map_table.strip():
        raw = (spark.read.format("bigquery")
               .option("table", args.mondo_map_table)
               .option("parentProject", args.billing).load())
        cols = set(raw.columns)
        if "snomed_concept_id" in cols:
            # pre-resolved to OMOP concept_ids
            mondo_targets = raw.select(F.col("snomed_concept_id").alias("t_cid")).distinct()
            mondo_source = f"side table {args.mondo_map_table} (snomed_concept_id)"
        elif "snomed_code" in cols:
            # SSSOM shape: resolve the SNOMED CT code -> standard OMOP SNOMED concept_id
            snomed = concept.where((F.col("vocabulary_id") == "SNOMED")
                                   & (F.col("standard_concept") == "S")).select(
                F.col("concept_code").alias("c_code"), F.col("concept_id").alias("t_cid"))
            mondo_targets = (raw.select(F.col("snomed_code").cast("string").alias("c_code"))
                             .distinct()
                             .join(snomed, "c_code", "inner").select("t_cid").distinct())
            mondo_source = f"side table {args.mondo_map_table} (snomed_code -> concept_code)"
        else:
            raise SystemExit(
                f"--mondo-map-table {args.mondo_map_table} must have a `snomed_code` or "
                f"`snomed_concept_id` column; got {sorted(cols)}")
    else:
        # Detect Mondo in the CDR vocab.
        n_mondo_concepts = concept.where(
            F.lower(F.col("vocabulary_id")).like("%mondo%")).count()
        if n_mondo_concepts > 0:
            mondo_concepts = concept.where(
                F.lower(F.col("vocabulary_id")).like("%mondo%")).select(
                F.col("concept_id").alias("mondo_cid"))
            # non-standard Mondo -> 'Maps to' -> standard SNOMED concept
            cr = read("concept_relationship").select(
                "concept_id_1", "concept_id_2", "relationship_id").where(
                F.col("relationship_id") == "Maps to")
            mondo_targets = (cr.join(
                mondo_concepts, cr.concept_id_1 == mondo_concepts.mondo_cid, "inner")
                .select(F.col("concept_id_2").alias("t_cid"),
                        F.col("concept_id_1").alias("mondo_curie")).distinct())
            mondo_source = f"CDR vocab ({n_mondo_concepts} Mondo concepts)"

    n_mondo = None
    support_hist = {}
    if mondo_targets is not None:
        # Join-based filter (NOT collect()+isin): the Mondo target set can be tens of
        # thousands of SNOMED concepts, too large for a driver-side IN list.
        ca_all = read("concept_ancestor").select(
            "descendant_concept_id", "ancestor_concept_id")
        ca_mondo = (ca_all.join(
            mondo_targets.select("t_cid").distinct(),
            ca_all.ancestor_concept_id == F.col("t_cid"), "inner")
            .select("descendant_concept_id", "ancestor_concept_id"))
        placed = _placed_persons(cond, ca_mondo)
        n_mondo = placed.count()
        # node-support histogram: patients placed PER target node (which node, via ancestor)
        per_node = (cond.join(
            ca_mondo, cond.condition_concept_id == ca_mondo.descendant_concept_id, "inner")
            .select("person_id", "ancestor_concept_id").distinct()
            .groupBy("ancestor_concept_id").agg(F.countDistinct("person_id").alias("n")))
        for t in [int(x) for x in args.support_thresholds.split(",")]:
            support_hist[t] = per_node.where(F.col("n") >= t).count()

    # --- report ---
    def pct(n):
        return f"{100.0 * n / n_total:6.2f}%" if n_total else "   n/a"

    lines = [
        "=" * 74,
        f"MONDO/SNOMED PLACEMENT COVERAGE PROBE   sample MOD(person_id,{args.person_mod})==0",
        "=" * 74,
        f"  patients (sampled)                       {n_total:>10d}   100.00%",
        f"  has >=1 standard condition code          {n_any_condition:>10d}   "
        f"{pct(n_any_condition)}   (complement = no-code residual)",
        f"  rolls up to SNOMED Disease hierarchy     {n_disease:>10d}   {pct(n_disease)}"
        f"   (SNOMED placeability CEILING)",
        f"  under the {len(anchor_ids)} current anchors           {n_anchor:>10d}   "
        f"{pct(n_anchor)}   (status-quo foreground)",
    ]
    if n_mondo is not None:
        lines.append(
            f"  placed on a Mondo-mapped node (rollup)    {n_mondo:>10d}   {pct(n_mondo)}"
            f"   (HEADLINE: proposed foreground)   [{mondo_source}]")
        lines.append(
            f"  residual: has codes, NO Mondo mapping    {n_disease - n_mondo:>10d}   "
            f"{pct(max(n_disease - n_mondo, 0))}   (Mondo-incompleteness gap)")
    else:
        lines.append("  Mondo landing: SKIPPED — Mondo not in the CDR vocab and no "
                     "--mondo-map-table given.")
    lines.append(
        f"  residual: truly healthy (no codes)       {n_total - n_any_condition:>10d}   "
        f"{pct(n_total - n_any_condition)}   (irreducible background floor)")
    if support_hist:
        lines.append("  node support (Mondo-placed) — #target nodes with >= T placed patients:")
        lines.append("    " + "   ".join(f"T>={t}: {c}" for t, c in support_hist.items()))
    lines.append("=" * 74)

    report = "\n".join(lines)
    print(report, flush=True)
    if args.out_dir.strip():
        # harness contract: leave an artifact in out_dir (the summary is also teed
        # from stdout by run_experiment). Best-effort; both local and gs:// paths.
        try:
            dest = args.out_dir.rstrip("/") + "/mondo_coverage_report.txt"
            spark.createDataFrame([(report,)], ["report"]).coalesce(1).write.mode(
                "overwrite").text(dest)
        except Exception as e:  # noqa: BLE001 — never fail the probe on the artifact write
            print(f"[probe] (could not write report artifact to {args.out_dir}: {e})",
                  flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
