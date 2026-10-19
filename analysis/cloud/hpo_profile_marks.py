#!/usr/bin/env python
"""hpo_profile_marks.py -- the HPO-profile concept table for the digest's `*`, off-YARN.

`inspect_topics --digest --credited-file TSV` marks with `*` each top-N word
that is in the node's own HPO phenotype profile. Its TSV only needs
(mondo_id, concept_id, neg): which standard Condition concepts each node's
profile maps to. The stage-2 probe (`hpoa_stage2_probe --emit-eta`) writes a
superset of that, but it is a Spark job: it loads the bundle and makes a
treeAggregate pass, all for the train-derived `coverage` and `weight` the eta
PRIOR needs. Marking words needs neither, so this tool does only the mapping:
the stage-1 `--emit-codes` TSV (SNOMED codes per node profile, public
ontology) -> targeted `bq query` (SNOMED source concept -> 'Maps to' ->
standard Condition, the probe's join shape) -> TSV. No Spark, no YARN, no
bundle, no patient data. Vocabulary membership is applied by the digest
itself against the bundle meta.

  python hpo_profile_marks.py --run-dir <dir> --profile-codes codes.tsv \
      --out profile_marks.tsv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
from pathlib import Path


def read_profile_codes(path):
    """Stage-1 `--emit-codes` TSV -> list of (mondo_id, code, neg) for the
    SNOMED rows (columns mondo_id, hp_id, neg, freq, vocab, code)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if str(r.get("vocab", "")) != "SNOMED":
                continue
            neg = str(r.get("neg", "")).strip().lower() in ("1", "true")
            rows.append((str(r["mondo_id"]), str(r["code"]).strip(), neg))
    return rows


def mapping_sql(cdr, codes):
    """SNOMED concept_code -> standard Condition concept_id, one query."""
    in_list = ",".join("'" + c.replace("'", "") + "'" for c in codes)
    return (
        f"SELECT s.concept_code AS code, t.concept_id AS std_id "
        f"FROM `{cdr}.concept` s "
        f"JOIN `{cdr}.concept_relationship` r "
        f"  ON r.concept_id_1 = s.concept_id AND r.relationship_id = 'Maps to' "
        f"JOIN `{cdr}.concept` t "
        f"  ON t.concept_id = r.concept_id_2 AND t.standard_concept = 'S' "
        f"  AND t.domain_id = 'Condition' "
        f"WHERE s.vocabulary_id = 'SNOMED' AND s.concept_code IN ({in_list})")


def bq_code_map(codes, cdr, billing, *, bq_bin="bq", batch=1000):
    """{snomed_code: {std concept_id}} via one `bq query` per batch of codes."""
    codes = sorted(set(codes))
    out = {}
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        cmd = [bq_bin, "--project_id", billing, "--format", "csv", "query",
               "--use_legacy_sql=false", "--max_rows", "1000000", "--quiet",
               mapping_sql(cdr, chunk)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"[hpo_profile_marks] bq query failed "
                             f"(batch {i // batch}): {res.stderr.strip()}")
        rdr = csv.reader(res.stdout.splitlines())
        next(rdr, None)                       # code,std_id
        for row in rdr:
            if len(row) >= 2 and row[1].strip():
                out.setdefault(row[0], set()).add(int(row[1]))
        print(f"[hpo_profile_marks] batch {i // batch + 1}: {len(chunk)} codes "
              f"-> {len(out)} mapped so far", flush=True)
    return out


def mark_rows(profile_rows, code_map):
    """(mondo_id, concept_id, neg) rows, deduplicated. A concept reached by
    both a positive and a NOT term of the same node keeps both rows; the
    digest drops neg rows."""
    seen = set()
    for mid, code, neg in profile_rows:
        for cid in code_map.get(code, ()):
            seen.add((mid, int(cid), bool(neg)))
    return sorted(seen)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="Run dir (or glob) with manifest.json, for cdr + billing")
    ap.add_argument("--profile-codes", required=True,
                    help="stage-1 hpoa_profile_survey --emit-codes TSV")
    ap.add_argument("--out", default=None,
                    help="output TSV (default <run-dir>/profile_marks.tsv)")
    ap.add_argument("--cdr", default=None)
    ap.add_argument("--billing", default=None)
    ap.add_argument("--bq-bin", default="bq")
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():
        hits = [Path(m) for m in glob.glob(str(args.run_dir))
                if (Path(m) / "manifest.json").exists()]
        if len(hits) != 1:
            raise SystemExit(f"[hpo_profile_marks] run-dir {args.run_dir!r} "
                             f"resolved to {len(hits)} dirs with a manifest")
        run_dir = hits[0]
    cm = json.loads((run_dir / "manifest.json").read_text()).get(
        "corpus_manifest", {})
    cdr = args.cdr or cm.get("cdr") or os.environ.get("WORKSPACE_CDR")
    billing = (args.billing or cm.get("billing")
               or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    if not cdr or not billing:
        raise SystemExit("[hpo_profile_marks] no cdr/billing; pass --cdr/--billing")

    prof = read_profile_codes(args.profile_codes)
    codes = {c for _, c, _ in prof}
    print(f"[hpo_profile_marks] {len(prof)} SNOMED profile rows, "
          f"{len({m for m, _, _ in prof})} nodes, {len(codes)} distinct codes",
          flush=True)
    code_map = bq_code_map(codes, cdr, billing, bq_bin=args.bq_bin,
                           batch=args.batch_size)
    rows = mark_rows(prof, code_map)
    out = Path(args.out) if args.out else run_dir / "profile_marks.tsv"
    with open(out, "w") as fh:
        fh.write("mondo_id\tconcept_id\tneg\n")
        for mid, cid, neg in rows:
            fh.write(f"{mid}\t{cid}\t{int(neg)}\n")
    print(f"[hpo_profile_marks] {len(code_map)}/{len(codes)} codes map to a "
          f"standard Condition; {len({m for m, _, n in rows if not n})} nodes "
          f"with a positive profile concept -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
