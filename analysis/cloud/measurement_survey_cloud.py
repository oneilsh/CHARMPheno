"""Privacy-safe survey of the OMOP ``measurement`` table (measurement arc, Step 1).

Answers the questions the value-aware lab-representation choice turns on (see
``docs/superpowers/specs/2026-07-31-measurement-arc-value-aware-representation-design.md``):
which labs are common, how often each carries a numeric value / a coded
qualitative value / a reference range / a unit, how bad the unit-harmonization
problem is per lab, what the coded-value vocabulary looks like, and how feasible
range-derived low/normal/high coding is (plus the abnormality base rate).

Everything emitted is an aggregate group count or a ratio over a large
denominator. The small-cell floor (``--min-persons``; ``measurement_survey.apply_floor``)
is applied on the driver BEFORE anything is written or printed, so nothing
sub-floor reaches disk or the paste-back digest (AoU small-cell rule). Runs
sampled by ``--person-mod`` for speed — fractions are stable under whole-person
sampling.

Reads the same two env vars as the other cloud drivers when ``--cdr`` /
``--billing`` are omitted:
    WORKSPACE_CDR          - "<data-project>.<dataset>", read-only
    GOOGLE_CLOUD_PROJECT   - billing/compute project for the BQ job

Run (from this directory on the Dataproc master):
    make measurement-survey
    make measurement-survey MEASUREMENT_SURVEY_ARGS='--person-mod 50 --top-concepts 300'

Outputs (under ``--out-dir``, default analysis/cloud/measurement_survey_data/):
    concepts.tsv          - per measurement_concept_id: volume + value/range/unit
                            coverage + unit spread + abnormality mix + rep tag
    concept_units.tsv     - per (measurement_concept_id, unit_concept_id): row
                            counts + within-concept share (unit harmonization)
    value_concepts.tsv    - top value_as_concept_id overall (coded-value vocab)
    globals.tsv           - one row of table-wide totals + coverage fractions
A compact digest is also printed to stdout for paste-back.
"""
from __future__ import annotations

import argparse
import os
import sys

from pyspark.sql import functions as F

from _driver_common import _phase, make_spark_session
from measurement_survey import (
    apply_floor,
    derive_concept_summary,
    classify_representation,
    summarize_representation_mix,
)

MEASUREMENT_TABLE = "measurement"
CONCEPT_TABLE = "concept"


# ── presence predicates ─────────────────────────────────────────────────────
# OMOP encodes "absent" two ways depending on column: NULL, or the sentinel
# concept_id 0 ("No matching concept"). value_as_number has no sentinel — NULL is
# the only absence (0.0 is a legitimate measured value and is counted present).
def _num_present():
    return F.col("value_as_number").isNotNull()


def _concept_present(col: str):
    return F.col(col).isNotNull() & (F.col(col) != 0)


def _range_present():
    return F.col("range_low").isNotNull() & F.col("range_high").isNotNull()


def _feasible():
    # A row is codable as low/normal/high iff it has BOTH a numeric value and a
    # reference range (the range is in the value's own unit, so no conversion).
    return _num_present() & _range_present()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OMOP measurement-table survey (arc Step 1)")
    p.add_argument("--cdr", default=os.environ.get("WORKSPACE_CDR"),
                   help="BQ dataset '<project>.<dataset>' (default $WORKSPACE_CDR)")
    p.add_argument("--billing", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                   help="BQ billing project (default $GOOGLE_CLOUD_PROJECT)")
    p.add_argument("--person-mod", type=int, default=100,
                   help="keep persons with MOD(person_id, M)==0 (whole-person "
                        "sample; default 100). 1 = full table.")
    p.add_argument("--top-concepts", type=int, default=200,
                   help="how many measurement concepts (by distinct patients) to "
                        "detail in concepts.tsv / concept_units.tsv (default 200)")
    p.add_argument("--top-value-concepts", type=int, default=60,
                   help="how many value_as_concept_id values to list (default 60)")
    p.add_argument("--min-persons", type=int, default=50,
                   help="small-cell floor: drop concepts with fewer distinct "
                        "patients, and any (concept,unit)/value cell with fewer "
                        "rows (default 50; must be >= the AoU floor of 20)")
    p.add_argument("--out-dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "measurement_survey_data"),
                   help="local dir for the TSV outputs")
    p.add_argument("--print-top", type=int, default=40,
                   help="how many concepts to show in the stdout digest (default 40)")
    args = p.parse_args(argv)
    if not args.cdr or not args.billing:
        p.error("set --cdr/--billing or WORKSPACE_CDR/GOOGLE_CLOUD_PROJECT in env")
    if args.min_persons < 20:
        p.error("--min-persons must be >= 20 (AoU small-cell floor)")
    return args


def _read(spark, cdr, billing, table):
    return (spark.read.format("bigquery")
            .option("table", f"{cdr}.{table}")
            .option("parentProject", billing)
            .load())


def _base_measurement(spark, args):
    """Sampled measurement rows projected to the columns the survey needs, with
    the presence predicates pre-evaluated as 0/1 integer columns so every
    downstream aggregate is a plain SUM."""
    m = _read(spark, args.cdr, args.billing, MEASUREMENT_TABLE).select(
        "person_id",
        F.col("measurement_concept_id").alias("concept_id"),
        F.col("value_as_number"),
        F.col("value_as_concept_id"),
        F.col("unit_concept_id"),
        F.col("range_low"),
        F.col("range_high"),
        F.col("operator_concept_id"),
    )
    if args.person_mod and args.person_mod > 1:
        m = m.where((F.col("person_id") % args.person_mod) == 0)
    m = m.where(F.col("concept_id") != 0)
    return m.select(
        "person_id", "concept_id",
        "value_as_number", "value_as_concept_id", "unit_concept_id",
        "range_low", "range_high", "operator_concept_id",
        _num_present().cast("int").alias("is_num"),
        _concept_present("value_as_concept_id").cast("int").alias("is_valconcept"),
        _concept_present("unit_concept_id").cast("int").alias("is_unit"),
        _range_present().cast("int").alias("is_range"),
        _concept_present("operator_concept_id").cast("int").alias("is_operator"),
        _feasible().cast("int").alias("is_feasible"),
        (_feasible() & (F.col("value_as_number") < F.col("range_low"))
         ).cast("int").alias("is_low"),
        (_feasible() & (F.col("value_as_number") > F.col("range_high"))
         ).cast("int").alias("is_high"),
    )


def _per_concept_agg(base):
    """One row per concept_id with the raw counts derive_concept_summary needs.
    n_distinct_units / top_unit_n are folded in later from the (concept,unit)
    table to avoid a countDistinct in this pass."""
    return base.groupBy("concept_id").agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.countDistinct("person_id").alias("n_persons"),
        F.sum("is_num").alias("n_val_number"),
        F.sum("is_valconcept").alias("n_val_concept"),
        F.sum("is_unit").alias("n_unit"),
        F.sum("is_range").alias("n_range"),
        F.sum("is_operator").alias("n_operator"),
        F.sum("is_feasible").alias("n_feasible"),
        F.sum("is_low").alias("n_low"),
        F.sum("is_high").alias("n_high"),
    )


def _concept_unit_agg(base, top_ids):
    """One row per (concept_id, unit_concept_id) for the top concepts only —
    rows carrying a real unit (is_unit=1). Feeds both concept_units.tsv and the
    n_distinct_units / top_unit_n columns of the concepts table."""
    return (base.where(F.col("concept_id").isin(list(top_ids)) & (F.col("is_unit") == 1))
            .groupBy("concept_id", "unit_concept_id")
            .agg(F.count(F.lit(1)).alias("n_rows")))


def _value_concept_agg(base):
    """One row per value_as_concept_id (coded qualitative results) over rows that
    carry one — the coded-value vocabulary for representation option 2."""
    return (base.where(F.col("is_valconcept") == 1)
            .groupBy("value_as_concept_id")
            .agg(F.count(F.lit(1)).alias("n_rows"),
                 F.countDistinct("person_id").alias("n_persons")))


def _globals(base):
    return base.agg(
        F.count(F.lit(1)).alias("n_rows"),
        F.countDistinct("person_id").alias("n_persons"),
        F.countDistinct("concept_id").alias("n_concepts"),
        F.sum("is_num").alias("n_val_number"),
        F.sum("is_valconcept").alias("n_val_concept"),
        F.sum("is_unit").alias("n_unit"),
        F.sum("is_range").alias("n_range"),
        F.sum("is_operator").alias("n_operator"),
        F.sum("is_feasible").alias("n_feasible"),
        F.sum("is_low").alias("n_low"),
        F.sum("is_high").alias("n_high"),
    ).collect()[0].asDict()


def _names_for(spark, args, ids):
    """{concept_id: concept_name} for a set of ids (small; collected)."""
    ids = [int(i) for i in ids if i is not None]
    if not ids:
        return {}
    rows = (_read(spark, args.cdr, args.billing, CONCEPT_TABLE)
            .select("concept_id", "concept_name")
            .where(F.col("concept_id").isin(ids))
            .collect())
    return {int(r["concept_id"]): r["concept_name"] for r in rows}


def _fold_unit_spread(concept_rows, cu_rows):
    """Add n_distinct_units + top_unit_n to each concept row from the
    (concept,unit) rows. Pure driver-side aggregation over already-floored data.
    NOTE: computed over units that individually clear the floor, so it is a
    lower bound on true unit diversity — reported as such in the digest."""
    by_concept: dict[int, list[int]] = {}
    for r in cu_rows:
        by_concept.setdefault(int(r["concept_id"]), []).append(int(r["n_rows"]))
    for r in concept_rows:
        counts = by_concept.get(int(r["concept_id"]), [])
        r["n_distinct_units"] = len(counts)
        r["top_unit_n"] = max(counts) if counts else 0
    return concept_rows


def _write_tsv(path, header, rows, fmt):
    with open(path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(fmt(r, c) for c in header) + "\n")


def _f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(args.out_dir, exist_ok=True)
    floor = args.min_persons

    spark = make_spark_session(app_name="measurement-survey")
    try:
        with _phase(f"read + sample measurement (person_mod={args.person_mod})"):
            base = _base_measurement(spark, args).persist()
            g = _globals(base)
        print(f"[driver]   {g['n_rows']} rows, {g['n_persons']} persons, "
              f"{g['n_concepts']} distinct measurement concepts (sampled)", flush=True)

        with _phase("per-concept aggregate"):
            concept_df = _per_concept_agg(base)
            top = (concept_df.orderBy(F.col("n_persons").desc())
                   .limit(args.top_concepts).collect())
            concept_rows = [r.asDict() for r in top]
        # Floor on distinct patients BEFORE anything leaves the driver.
        concept_rows, n_sup_c, _ = apply_floor(concept_rows, "n_persons", floor)
        top_ids = [int(r["concept_id"]) for r in concept_rows]

        with _phase("per-(concept,unit) + value-concept + names"):
            cu_rows = [r.asDict() for r in _concept_unit_agg(base, top_ids).collect()]
            cu_rows, n_sup_cu, _ = apply_floor(cu_rows, "n_rows", floor)
            vc_rows = [r.asDict() for r in
                       (_value_concept_agg(base).orderBy(F.col("n_rows").desc())
                        .limit(args.top_value_concepts).collect())]
            vc_rows, n_sup_vc, _ = apply_floor(vc_rows, "n_rows", floor)

            name_ids = set(top_ids)
            name_ids |= {int(r["unit_concept_id"]) for r in cu_rows}
            name_ids |= {int(r["value_as_concept_id"]) for r in vc_rows}
            names = _names_for(spark, args, name_ids)
        base.unpersist()

        # Derive fractions + unit spread + representation tag (all pure).
        concept_rows = _fold_unit_spread(concept_rows, cu_rows)
        summaries = []
        for r in concept_rows:
            s = derive_concept_summary(r)
            s["concept_name"] = names.get(int(s["concept_id"]), "?")
            s["representation"] = classify_representation(s)
            summaries.append(s)
        summaries.sort(key=lambda s: s["n_persons"], reverse=True)
        for r in cu_rows:
            r["unit_name"] = names.get(int(r["unit_concept_id"]), "?")
            r["share"] = round(r["n_rows"] / max(
                next((s["n_rows"] for s in summaries
                      if int(s["concept_id"]) == int(r["concept_id"])), r["n_rows"]), 1), 4)
        for r in vc_rows:
            r["value_name"] = names.get(int(r["value_as_concept_id"]), "?")

        _write_outputs(args, g, summaries, cu_rows, vc_rows)
        _print_digest(args, g, summaries, vc_rows, floor,
                      (n_sup_c, n_sup_cu, n_sup_vc))
        print(f"[driver] MEASUREMENT SURVEY COMPLETE -> {args.out_dir}", flush=True)
        return 0
    finally:
        spark.stop()


def _write_outputs(args, g, summaries, cu_rows, vc_rows):
    concept_header = [
        "concept_id", "concept_name", "n_persons", "n_rows",
        "pct_val_number", "pct_val_concept", "pct_range", "pct_unit",
        "pct_operator", "pct_feasible", "n_distinct_units", "top_unit_share",
        "frac_low", "frac_normal", "frac_high", "representation",
    ]
    _write_tsv(os.path.join(args.out_dir, "concepts.tsv"), concept_header,
               summaries, lambda r, c: _f(r.get(c, "")))
    _write_tsv(os.path.join(args.out_dir, "concept_units.tsv"),
               ["concept_id", "unit_concept_id", "unit_name", "n_rows", "share"],
               cu_rows, lambda r, c: _f(r.get(c, "")))
    _write_tsv(os.path.join(args.out_dir, "value_concepts.tsv"),
               ["value_as_concept_id", "value_name", "n_rows", "n_persons"],
               vc_rows, lambda r, c: _f(r.get(c, "")))
    gs = derive_concept_summary(g)
    _write_tsv(os.path.join(args.out_dir, "globals.tsv"),
               ["n_rows", "n_persons", "n_concepts", "pct_val_number",
                "pct_val_concept", "pct_range", "pct_unit", "pct_operator",
                "pct_feasible", "frac_low", "frac_normal", "frac_high"],
               [gs], lambda r, c: _f(r.get(c, "")))


def _print_digest(args, g, summaries, vc_rows, floor, suppressed):
    gs = derive_concept_summary(g)
    n_sup_c, n_sup_cu, n_sup_vc = suppressed
    print("\n" + "=" * 78, flush=True)
    print("MEASUREMENT SURVEY DIGEST  (sampled person_mod=%d; floor=%d)"
          % (args.person_mod, floor), flush=True)
    print("=" * 78, flush=True)
    print(f"rows={g['n_rows']}  persons={g['n_persons']}  "
          f"distinct_concepts={g['n_concepts']}", flush=True)
    print("table-wide coverage: "
          f"value_number={gs['pct_val_number']:.2f}  "
          f"value_concept={gs['pct_val_concept']:.2f}  "
          f"range={gs['pct_range']:.2f}  unit={gs['pct_unit']:.2f}  "
          f"operator={gs['pct_operator']:.2f}", flush=True)
    print(f"range-abnormality feasible over ALL rows: {gs['pct_feasible']:.2f}  "
          f"(of feasible: low={gs['frac_low']:.2f} normal={gs['frac_normal']:.2f} "
          f"high={gs['frac_high']:.2f})", flush=True)

    mix = summarize_representation_mix(summaries, weight_key="n_persons")
    print("\nrepresentation viability across top %d concepts "
          "(concepts / summed patient-count):" % len(summaries), flush=True)
    for tag in ("range-abnormality", "value-concept",
                "numeric-needs-binning", "presence-only"):
        b = mix.get(tag, {"n_concepts": 0, "weight": 0})
        print(f"  {tag:22s} {b['n_concepts']:4d}  {b['weight']:>10d}", flush=True)

    print(f"\ntop {min(args.print_top, len(summaries))} concepts by distinct patients:",
          flush=True)
    print("  %-40s %8s %5s %5s %5s %5s %4s %5s  %s"
          % ("concept", "persons", "vnum", "vcpt", "rng", "feas", "unt#",
             "u1sh", "rep"), flush=True)
    for s in summaries[:args.print_top]:
        nm = (s["concept_name"] or "?")[:40]
        print("  %-40s %8d %5.2f %5.2f %5.2f %5.2f %4d %5.2f  %s"
              % (nm, s["n_persons"], s["pct_val_number"], s["pct_val_concept"],
                 s["pct_range"], s["pct_feasible"], s["n_distinct_units"],
                 s["top_unit_share"], s["representation"]), flush=True)

    print(f"\ntop {min(20, len(vc_rows))} coded values (value_as_concept_id):", flush=True)
    for r in vc_rows[:20]:
        print("  %-40s rows=%d persons=%d"
              % ((r.get("value_name") or "?")[:40], r["n_rows"], r["n_persons"]),
              flush=True)

    print(f"\nsuppressed sub-floor cells: concepts={n_sup_c} "
          f"(concept,unit)={n_sup_cu} value-codes={n_sup_vc}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    sys.exit(main())
