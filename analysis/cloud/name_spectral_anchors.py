"""Name a run's spectral anchors: `<run>/spectral_anchors.json` (vocab ids +
from-profile flags, written by the fit since exp 0125) joined to the bundle
meta's condition-domain vocab map and a concept-names CSV.

Pure driver-side tooling, no Spark, no patient data: the output is one line
per node — its Mondo name, then each anchor's concept name with a `*` when the
anchor came from the node's HPO profile. This is the DIRECT anchor read that
exp 0124 lacked (dilated cardiomyopathy's anchor had to be inferred from its
recovered topic).

    python analysis/cloud/name_spectral_anchors.py <run_dir> \
        --bundle-meta /tmp/inspect_meta_125.json \
        --concept-names /tmp/concept_names_125.csv \
        [--grep 'dilated cardiomyopathy|^cardiomyopathy']

The names CSV is the `resolve_vocab_names` output (`RESOLVE_NAMES=1` on
`make inspect-topics`), columns (concept_id, concept_name).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def load_names(path) -> dict[int, str]:
    names: dict[int, str] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                try:
                    names[int(row[0])] = row[1]
                except ValueError:
                    continue          # header row
    return names


def name_anchors(anchors_doc: dict, vocab_map0: dict, names: dict[int, str],
                 *, grep: str | None = None) -> list[str]:
    """Render one line per node: ``<name> | w1* · w2 · ...``; `*` = from profile.

    ``vocab_map0`` is the bundle meta's domain-0 map {concept_id_str: vocab_idx};
    it is inverted here. An anchor whose concept has no name prints as
    ``cid:<id>``; an index outside the map prints as ``idx:<i>`` (a different
    bundle's meta — the shape mismatch a reader should notice, not a crash)."""
    idx2cid = {int(v): int(k) for k, v in vocab_map0.items()}
    pat = re.compile(grep, re.I) if grep else None
    lines = []
    for eid, rec in sorted(anchors_doc.get("nodes", {}).items(), key=lambda kv: int(kv[0])):
        node = rec.get("name") or f"eid{eid}"
        if pat and not pat.search(str(node)):
            continue
        words = []
        for i, f in zip(rec.get("anchors", []), rec.get("from_profile", [])):
            cid = idx2cid.get(int(i))
            w = (names.get(cid, f"cid:{cid}") if cid is not None else f"idx:{i}")
            words.append(w + ("*" if f else ""))
        lines.append(f"{node} | " + " · ".join(words))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--bundle-meta", required=True)
    ap.add_argument("--concept-names", required=True)
    ap.add_argument("--grep", default=None, help="regex on the node name (case-insensitive)")
    args = ap.parse_args(argv)
    doc = json.load(open(Path(args.run_dir) / "spectral_anchors.json"))
    meta = json.load(open(args.bundle_meta))
    vm0 = meta["vocab_maps"][0]
    for ln in name_anchors(doc, vm0, load_names(args.concept_names), grep=args.grep):
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
