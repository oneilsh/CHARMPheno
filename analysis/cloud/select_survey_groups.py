"""Pick the survey groups worth re-scoring on a nested fit (exp 0152 read 4b).

A full incident survey on whole-Mondo takes hours (~90 s per group). Groups are
scored independently, so a subset changes no group's numbers, only which groups
the pooled figures cover. The fair comparison with a flat-label-set run (0137) is
the set of groups whose members are the SAME Mondo terms in both runs: the
nested run's TOP-LEVEL members (the `--diff-members top` rule) against the flat
run's label members. This tool finds them from the two runs' readout DAGs and
writes a JSON whose keys are the chosen group names, for `--diff-groups-like`.

    python3 analysis/cloud/select_survey_groups.py <nested run> <flat run> <flat tag> \\
        --out groups.json [--max N] [--same-size]

Order: identical member sets first, then (with --same-size) equal member counts,
each by the flat run's group patient count, largest first. Only groups the flat
survey actually scored are considered. Names and ids only; no patient data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_analysis as sa  # noqa: E402


def select(nested_run, flat_run, flat_tag, *, max_groups=None, same_size=False):
    flat_js = json.loads((Path(flat_run) / f"differential_{flat_tag}.json").read_text())
    nest, ncid = sa.group_members(nested_run, rule="top", with_cids=True)
    flat, fcid = sa.group_members(flat_run, rule="all", with_cids=True)
    ident, size = [], []
    for g, r in flat_js.items():
        if g not in nest or g not in flat or not (r.get("metrics") or {}):
            continue
        a = {ncid[m] for m in nest[g]}
        b = {fcid[m] for m in flat[g]}
        if a == b:
            ident.append((g, r.get("n_eval", 0)))
        elif len(a) == len(b):
            size.append((g, r.get("n_eval", 0)))
    ident.sort(key=lambda x: -x[1])
    size.sort(key=lambda x: -x[1])
    picked = [g for g, _ in ident] + ([g for g, _ in size] if same_size else [])
    if max_groups:
        picked = picked[:max_groups]
    return picked, len(ident), len(size), len(flat_js)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("nested_run")
    ap.add_argument("flat_run")
    ap.add_argument("flat_tag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--same-size", action="store_true",
                    help="also take groups with equal member counts but different terms")
    a = ap.parse_args(argv)
    picked, n_id, n_sz, n_all = select(a.nested_run, a.flat_run, a.flat_tag,
                                       max_groups=a.max, same_size=a.same_size)
    Path(a.out).write_text(json.dumps({g: {} for g in picked}, indent=1))
    print(f"[select] flat survey scored {n_all} groups: {n_id} have IDENTICAL members "
          f"under the nested top-level rule, {n_sz} more the same member count; "
          f"picked {len(picked)} -> {a.out} (~{len(picked) * 1.5:.0f} min at ~90 s/group)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
