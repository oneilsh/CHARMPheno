"""Exp 0150 follow-up: read the framings survey (`differential_<tag>.json` + `.md`) and
the run's readout DAG, and report — pooled, per group — (1) groups DEDUPED by member set
(nested Mondo names often cover the same labels), (2) a chance-corrected framing-1
skill, kappa = (bal_top1 - 1/M) / (1 - 1/M), which compares groups of different size
fairly (the survey's lift = bal_top1 x M favours large M), (3) Spearman correlations of
that skill with framing 2 (gap closed, VOI - table) and framing 3 (members' new-diagnosis
AUC), and (4) the E5 reasons of the top groups by skill, so it is visible whether the
call rests on treatments (drug codes are not attestation and stay in the record).

    python3 analysis/cloud/survey_analysis.py <run dir> [tag]

Pooled, per-group numbers only (each group's metrics already rest on >= 40 patients and
reasons on members with >= 20)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


def kappa(bal, M):
    return (bal - 1.0 / M) / (1.0 - 1.0 / M) if M > 1 else float("nan")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan"), int(ok.sum())
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def group_members(run_dir):
    rd = json.loads((Path(run_dir) / "readout_dag.json").read_text())
    names = {}
    for r, c in rd["int2cid"].items():
        names[int(r)] = rd["name_by_id"].get(str(c), str(c))
    C_fit = int(rd["C_fit"])
    out = {}
    for n, srcs in enumerate(rd["pos_sources"]):
        if n == 0:
            continue
        mem = tuple(sorted({int(e) for e in srcs if 0 < int(e) < C_fit}))
        if len(mem) >= 2:
            out.setdefault(names.get(n, str(n)), mem)
    return out


def reasons_by_group(md_text):
    out, cur, in_e5 = {}, None, False
    for line in md_text.splitlines():
        m = re.match(r"^## (.+?) — \d+ members", line)
        if m:
            cur, in_e5 = m.group(1), False
            continue
        if line.startswith("E5 reasons"):
            in_e5 = True
            continue
        if in_e5 and line.startswith("- ") and cur:
            out.setdefault(cur, []).append(line[2:])
        elif in_e5 and not line.startswith("- "):
            in_e5 = False
    return out


def main(argv):
    run = Path(argv[1])
    tag = argv[2] if len(argv) > 2 else "survey"
    js = json.loads((run / f"differential_{tag}.json").read_text())
    md = (run / f"differential_{tag}.md").read_text()
    mem = group_members(run)
    load = lambda f: json.loads((run / f).read_text()) if (run / f).exists() else None
    th, rw = load("prevalence_check_history.json"), load("prevalence_check_stacked_raw2000.json")

    def f3(members, src):
        if not src:
            return float("nan")
        pn = src["history_stacked"]["per_node"]
        v = [pn[str(m)]["auc_incident"] for m in members if str(m) in pn]
        return float(np.mean(v)) if v else float("nan")
    rows, seen = [], {}
    for g, r in js.items():
        mt = r.get("metrics") or {}
        if "E6 θ + codes" not in mt or g not in mem:
            continue
        key = mem[g]
        if key in seen:
            seen[key].append(g)
            continue
        seen[key] = [g]
        M = len(key)
        bal = {e: mt[k]["bal_top1"] for e, k in (("E3", "E3 multiclass head"),
                                                ("E5", "E5 code head"),
                                                ("E6", "E6 θ + codes"))}
        best = max(bal, key=bal.get)
        nq = r.get("next_question") or {}
        if nq:
            v0, vf = nq["visible"]["p_true"], nq["full"]["p_true"]
            v5 = nq["by"]["VOI (codes)@5"]["p_true"]
            t5 = nq["by"]["population table@5"]["p_true"]
            gap = (v5 - v0) / (vf - v0) if vf > v0 else float("nan")
            dvt = v5 - t5
        else:
            gap = dvt = float("nan")
        rows.append(dict(group=g, M=M, n=r["n_eval"], best=best,
                         k_best=kappa(bal[best], M), k_e6=kappa(bal["E6"], M),
                         k_e3=kappa(bal["E3"], M), gap=gap, dvt=dvt,
                         f3t=f3(key, th), f3r=f3(key, rw)))
    rows.sort(key=lambda x: -x["k_best"])
    dup = sum(len(v) - 1 for v in seen.values())
    print(f"# survey analysis — {len(rows)} distinct member sets ({dup} duplicate names "
          f"dropped)\n")
    q = lambda k: np.nanpercentile([x[k] for x in rows], [25, 50, 75])
    print("chance-corrected skill kappa (best engine) quartiles: "
          + " / ".join(f"{v:.2f}" for v in q("k_best")))
    print("E6 − E3 kappa quartiles: " + " / ".join(
        f"{v:+.3f}" for v in np.nanpercentile([x["k_e6"] - x["k_e3"] for x in rows],
                                              [25, 50, 75])))
    print(f"groups with kappa >= 0.3: {sum(x['k_best'] >= 0.3 for x in rows)}; "
          f">= 0.5: {sum(x['k_best'] >= 0.5 for x in rows)}\n")
    print("Spearman with framing-1 skill (kappa, best engine):")
    for lab, k in (("members M", "M"), ("scored patients n", "n"),
                   ("F2 gap closed by 5 questions", "gap"), ("F2 VOI − table @5", "dvt"),
                   ("F3 new-dx AUC θ", "f3t"), ("F3 new-dx AUC raw", "f3r")):
        rho, nn = spearman([x["k_best"] for x in rows], [x[k] for x in rows])
        print(f"- {lab}: {rho:+.2f} (n={nn})")
    print("\n| group (aliases) | M | n | best | kappa | F2 gap | VOI − table | F3 θ/raw |")
    print("|---|--:|--:|---|--:|--:|--:|---|")
    for x in rows[:25]:
        al = len(seen[mem[x['group']]]) - 1
        print(f"| {x['group'][:38]}{f' (+{al})' if al else ''} | {x['M']} | {x['n']} | "
              f"{x['best']} | {x['k_best']:.2f} | {x['gap']:.2f} | {x['dvt']:+.3f} | "
              f"{x['f3t']:.2f}/{x['f3r']:.2f} |")
    rs = reasons_by_group(md)
    print("\nE5 reasons, top 10 groups by kappa (codes favouring each member):")
    for x in rows[:10]:
        print(f"\n## {x['group']}")
        for line in rs.get(x["group"], [])[:8]:
            print(f"- {line[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
