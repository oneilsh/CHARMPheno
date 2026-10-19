"""Exp 0152: per-node new-diagnosis AUC of two runs (and an optional raw-code arm),
matched by MONDO ID — the two fits number their nodes differently, so nodes are
joined through each run's `readout_dag.json` int2cid. Pooled numbers only.

    python3 analysis/cloud/compare_incident_runs.py \\
        <runA> <prevalence json A> <runB> <prevalence json B> [<raw json, in runA>]

Reports, on the nodes present in every arm (>= 20 incident positives each): macro
incident AUC per arm, the paired median B − A and raw − B, wins, and the same by
incident-positive band."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def by_cid(run, js_name, arm):
    rd = json.loads((Path(run) / "readout_dag.json").read_text())
    i2c = {int(k): int(v) for k, v in rd["int2cid"].items()}
    names = rd.get("name_by_id", {})
    pn = json.loads((Path(run) / js_name).read_text())[arm]["per_node"]
    out = {}
    for k, v in pn.items():
        c = i2c.get(int(k))
        if c is not None and c >= 0:
            out[c] = dict(v, name=names.get(str(c), str(c)))
    return out


def compare(A, B, R=None):
    keys = sorted(set(A) & set(B) & (set(R) if R is not None else set(A)))
    if not keys:
        return {"n": 0}
    a = np.array([A[k]["auc_incident"] for k in keys])
    b = np.array([B[k]["auc_incident"] for k in keys])
    out = {"n": len(keys), "A": float(a.mean()), "B": float(b.mean()),
           "B_minus_A_median": float(np.median(b - a)), "B_wins": int((b > a).sum())}
    if R is not None:
        r = np.array([R[k]["auc_incident"] for k in keys])
        out.update(R=float(r.mean()), R_minus_B_median=float(np.median(r - b)),
                   B_beats_R=int((b > r).sum()))
    bands = {}
    npos = np.array([A[k]["n_pos_incident"] for k in keys])
    for lo, hi in ((20, 50), (50, 100), (100, 300), (300, 1000), (1000, 10 ** 9)):
        sel = (npos >= lo) & (npos < hi)
        if sel.sum() >= 5:
            row = {"n": int(sel.sum()), "A": float(a[sel].mean()),
                   "B": float(b[sel].mean()),
                   "B_minus_A_median": float(np.median((b - a)[sel]))}
            if R is not None:
                row["R"] = float(r[sel].mean())
            bands[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = row
    out["bands"] = bands
    return out


def main(argv):
    runA, jsA, runB, jsB = argv[1:5]
    raw = argv[5] if len(argv) > 5 else None
    for arm in ("history_stacked", "history_flat"):
        A, B = by_cid(runA, jsA, arm), by_cid(runB, jsB, arm)
        R = by_cid(runA, raw, arm) if raw else None
        c = compare(A, B, R)
        print(f"## {arm} — {c['n']} nodes shared")
        if not c["n"]:
            continue
        line = (f"A {c['A']:.4f} · B {c['B']:.4f} · median B−A {c['B_minus_A_median']:+.4f}"
                f" · B wins {c['B_wins']}/{c['n']}")
        if R is not None:
            line += (f" · raw {c['R']:.4f} · median raw−B {c['R_minus_B_median']:+.4f}"
                     f" · B beats raw {c['B_beats_R']}/{c['n']}")
        print(line)
        print("| incident positives (A) | nodes | A | B | median B−A | raw |")
        print("|---|--:|--:|--:|--:|--:|")
        for band, r in c["bands"].items():
            print(f"| {band} | {r['n']} | {r['A']:.3f} | {r['B']:.3f} | "
                  f"{r['B_minus_A_median']:+.3f} | {r.get('R', float('nan')):.3f} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
