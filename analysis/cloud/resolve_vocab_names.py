#!/usr/bin/env python
"""resolve_vocab_names.py -- build a concept-name CSV for inspect_topics, off-YARN.

The topic-word view needs human names for the VOCAB features, which are not in
any saved artifact: the label DAG's name_by_id covers only label nodes, and
measurement features are not even real concept ids -- they are synthetic tokens
`measurement_concept_id * TOKEN_BASE + state_code` (charmpheno.omop.
measurement_tokens) that must be decoded to "Creatinine [high]" etc. before
naming. This tool reads the bundle meta's vocab_maps, decodes measurement
tokens, looks up the real OMOP concept names with a TARGETED `bq query` (a few
thousand ids, not the ~8M-row concept table -- a BigQuery CLIENT read, NO Spark,
NO YARN, safe to run mid-fit), and writes a CSV that
`inspect_topics --concept-names` consumes.

The CDR dataset and billing project come from the run's OWN manifest.json
(`corpus_manifest.cdr` = "<project>.<dataset>", `corpus_manifest.billing`), so
no extra input is needed. Output CSV is keyed on the VOCAB FEATURE id exactly as
it appears in vocab_maps (the token for measurement, the concept id otherwise),
so inspect_topics' flat `names.get(feature_id)` lookup names every word.

  python resolve_vocab_names.py --run-dir <dir> --bundle-meta meta.json \
      --out concept_names.csv
  # then: inspect_topics.py <dir> --bundle-meta meta.json --concept-names concept_names.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Measurement-token decode (authoritative from charmpheno; inline fallback)    #
# --------------------------------------------------------------------------- #
def _token_tools():
    """Return (TOKEN_BASE, decode(token)->(cid,state), state_label(state)->str).

    Prefer charmpheno.omop.measurement_tokens (the source of truth) so the
    decode never drifts from the encoder; fall back to the pinned constants if
    charmpheno is not importable (e.g. run outside PYTHONPATH=charmpheno).
    """
    try:
        from charmpheno.omop import measurement_tokens as mt
        return mt.TOKEN_BASE, mt.decode_token, mt.state_label
    except Exception:
        BASE = 100
        LABELS = {0: "measured", 1: "low", 2: "normal", 3: "high",
                  10: "coded-neg", 11: "coded-pos", 12: "coded-normal",
                  13: "coded-abnormal", 14: "coded-low", 15: "coded-high"}
        return (BASE, (lambda t: divmod(int(t), BASE)),
                (lambda s: LABELS.get(int(s), f"state{int(s)}")))


# --------------------------------------------------------------------------- #
# Pure logic (testable without BigQuery)                                       #
# --------------------------------------------------------------------------- #
def collect_features(vocab_maps, domain_names):
    """Map each vocab feature id to (domain, real_concept_id, state_or_None).

    vocab_maps is the meta's list of {concept_id_str: idx} per domain (the
    feature "concept id" is the id as the engine sees it -- a token in the
    measurement domain). Returns (features, real_ids) where features is a list of
    (feature_id, domain_name, real_concept_id, state) and real_ids is the set of
    real OMOP concept ids to look up.
    """
    BASE, decode, _ = _token_tools()
    features, real_ids = [], set()
    for d, vm in enumerate(vocab_maps):
        dom = domain_names[d] if d < len(domain_names) else f"dom{d}"
        is_meas = dom == "measurement"
        for cid_str in vm:
            fid = int(cid_str)
            if is_meas:
                real, state = decode(fid)
            else:
                real, state = fid, None
            features.append((fid, dom, real, state))
            real_ids.add(real)
    return features, real_ids


def build_rows(features, name_map, state_label):
    """[(feature_id, display_name)] from features + a real-concept name map.

    Measurement features get the "<real name> [<state>]" display; others get the
    concept name straight. Unknown real ids fall back to the bare id so the CSV
    still keys every feature (inspect_topics then shows cid:<id> for those).
    """
    rows = []
    for fid, dom, real, state in features:
        nm = name_map.get(real)
        if nm is None:
            continue                      # leave unnamed -> inspect_topics shows cid:
        rows.append((fid, f"{nm} [{state_label(state)}]" if state is not None else nm))
    return rows


# --------------------------------------------------------------------------- #
# BigQuery lookup (targeted, off-YARN)                                         #
# --------------------------------------------------------------------------- #
def bq_name_map(concept_ids, cdr, billing, *, bq_bin="bq", batch=1000):
    """{concept_id: concept_name} for the given ids via targeted `bq query`.

    One CLIENT-side BigQuery job per batch of ids (IN-list), reading only the
    matched rows of `<cdr>.concept`. No Spark, no YARN. `cdr` is "<project>.
    <dataset>"; the concept table is "<cdr>.concept".
    """
    ids = sorted({int(c) for c in concept_ids})
    out = {}
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        in_list = ",".join(str(c) for c in chunk)
        sql = (f"SELECT concept_id, concept_name FROM `{cdr}.concept` "
               f"WHERE concept_id IN ({in_list})")
        cmd = [bq_bin, "--project_id", billing, "--format", "csv",
               "query", "--use_legacy_sql=false", "--max_rows", str(len(chunk)),
               "--quiet", sql]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise SystemExit(f"[resolve_vocab_names] bq query failed "
                             f"(batch {i//batch}): {res.stderr.strip()}")
        rdr = csv.reader(res.stdout.splitlines())
        header = next(rdr, None)          # concept_id,concept_name
        for row in rdr:
            if len(row) >= 2 and row[0].strip():
                try:
                    out[int(row[0])] = row[1]
                except ValueError:
                    continue
        print(f"[resolve_vocab_names] batch {i//batch+1}: "
              f"{len(chunk)} ids -> {len(out)} names so far", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="Run dir with manifest.json (for cdr + billing).")
    ap.add_argument("--bundle-meta", required=True,
                    help="Bundle meta JSON (vocab_maps) -- the hdfs-cat output.")
    ap.add_argument("--out", default=None,
                    help="Output CSV (default: <run-dir>/concept_names.csv).")
    ap.add_argument("--cdr", default=None,
                    help="Override CDR '<project>.<dataset>' (else manifest).")
    ap.add_argument("--billing", default=None,
                    help="Override billing project (else manifest).")
    ap.add_argument("--bq-bin", default="bq", help="bq CLI path (default: bq).")
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--names-json", default=None,
                    help="Skip BigQuery: use this {concept_id: name} JSON as the "
                         "real-concept name map (for testing or a pre-fetched map).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the real concept-id count and the first batch's "
                         "SQL, then exit without querying.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():   # a glob like .../0111-*
        import glob as _glob
        hits = [Path(m) for m in _glob.glob(str(args.run_dir))
                if (Path(m) / "manifest.json").exists()]
        if len(hits) != 1:
            raise SystemExit(f"[resolve_vocab_names] run-dir {args.run_dir!r} "
                             f"resolved to {len(hits)} dirs with a manifest; "
                             "pass an exact dir")
        run_dir = hits[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cm = manifest.get("corpus_manifest", {})
    cdr = args.cdr or cm.get("cdr")
    billing = args.billing or cm.get("billing")
    domain_names = manifest.get("domain_names") or []

    meta = json.loads(Path(args.bundle_meta).read_text())
    if "vocab_maps" not in meta:
        raise SystemExit("[resolve_vocab_names] bundle meta has no vocab_maps "
                         "(single-domain bundle?). Nothing to resolve.")
    features, real_ids = collect_features(meta["vocab_maps"], domain_names)
    print(f"[resolve_vocab_names] {len(features)} vocab features over "
          f"{len(meta['vocab_maps'])} domains -> {len(real_ids)} real concept ids",
          flush=True)

    if args.dry_run:
        sample = ",".join(str(c) for c in sorted(real_ids)[:args.batch_size])
        print(f"[dry-run] cdr={cdr} billing={billing}")
        print(f"[dry-run] first-batch SQL:\n  SELECT concept_id, concept_name "
              f"FROM `{cdr}.concept` WHERE concept_id IN ({sample[:200]}...)")
        return

    _, _, state_label = _token_tools()
    if args.names_json:
        name_map = {int(k): v for k, v in
                    json.loads(Path(args.names_json).read_text()).items()}
    else:
        if not cdr or not billing:
            raise SystemExit("[resolve_vocab_names] no cdr/billing in manifest; "
                             "pass --cdr and --billing (or --names-json).")
        name_map = bq_name_map(real_ids, cdr, billing,
                               bq_bin=args.bq_bin, batch=args.batch_size)

    rows = build_rows(features, name_map, state_label)
    out = Path(args.out) if args.out else run_dir / "concept_names.csv"
    with open(out, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["concept_id", "concept_name"])
        wtr.writerows(rows)
    named = len(rows)
    print(f"[resolve_vocab_names] wrote {out}: {named}/{len(features)} features "
          f"named ({100*named/max(len(features),1):.0f}%)", flush=True)
    if named < len(features):
        print("  (unnamed features stay cid:<id> in the report -- typically "
              "tokens whose real concept is absent from the CDR concept table)",
          flush=True)


if __name__ == "__main__":
    main()
