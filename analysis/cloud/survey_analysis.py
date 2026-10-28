"""Exp 0150 follow-up: read the framings survey (`differential_<tag>.json` + `.md`) and
the run's readout DAG, and report — pooled, per group — (1) groups DEDUPED by member set
(nested Mondo names often cover the same labels), (2) a chance-corrected framing-1
skill, kappa = (bal_top1 - 1/M) / (1 - 1/M), which compares groups of different size
fairly (the survey's lift = bal_top1 x M favours large M), (3) Spearman correlations of
that skill with framing 2 (gap closed, VOI - table) and framing 3 (members' new-diagnosis
AUC), and (4) the E5 reasons of the top groups by skill, so it is visible whether the
call rests on treatments (drug codes are not attestation and stay in the record).

    python3 analysis/cloud/survey_analysis.py <run dir> [tag]
    python3 analysis/cloud/survey_analysis.py <run A> <tag A> <run B> [tag B]   # paired

On a nested label space (no label set in the manifest: exp 0152) a group's members
are its MOST SPECIFIC label descendants, as the survey scored them.

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


def most_specific(members, parent_int):
    """Members that are not an ancestor of another member — the survey's own rule on
    a NESTED label space (`gated_pc_readout.most_specific_members`, exp 0152), copied
    here so this reader needs no Spark import."""
    anc = set()
    for m in members:
        stack, seen = list(parent_int.get(str(m), [])), set()
        while stack:
            a = int(stack.pop())
            if a not in seen:
                seen.add(a)
                stack.extend(parent_int.get(str(a), []))
        anc |= seen
    return tuple(m for m in members if m not in anc)


def top_level(members, parent_int, group):
    """Members with no proper ancestor among the others, the group node excluded —
    `gated_pc_readout.top_members` (the nested default from exp 0152 read 4b)."""
    mem = [m for m in members if m != group]
    ms = set(mem)
    out = []
    for m in mem:
        stack, seen, hit = list(parent_int.get(str(m), [])), set(), False
        while stack and not hit:
            a = int(stack.pop())
            if a not in seen:
                seen.add(a)
                hit = a in ms
                stack.extend(parent_int.get(str(a), []))
        if not hit:
            out.append(m)
    return tuple(out)


def group_members(run_dir, rule="legacy", with_cids=False):
    """{group name: member readout ids}. `rule`: "legacy" = what a survey without
    recorded members used (all on a label set, most-specific when nested); "top" /
    "most-specific" / "all" force one. `with_cids` also returns {readout id: cid}."""
    run_dir = Path(run_dir)
    rd = json.loads((run_dir / "readout_dag.json").read_text())
    mf = run_dir / "manifest.json"
    cm = (json.loads(mf.read_text()).get("corpus_manifest") or {}) if mf.exists() else {}
    # Names as the survey resolves them (`voi_tables.load_voi_context`): the readout
    # DAG's own name, else the fit manifest's. On a whole-Mondo fit most readout
    # nodes are named only in the manifest, and a bare cid never matches the survey.
    fallback = cm.get("name_by_id") or {}
    names = {}
    for r, c in rd["int2cid"].items():
        names[int(r)] = ((rd.get("name_by_id") or {}).get(str(c))
                         or fallback.get(str(c)) or f"cid:{c}")
    C_fit = int(rd["C_fit"])
    nested = mf.exists() and not cm.get("label_set")
    if rule == "legacy":
        rule = "most-specific" if nested else "all"
    pi = {str(k): v for k, v in rd.get("parent_int", {}).items()}
    out = {}
    for n, srcs in enumerate(rd["pos_sources"]):
        if n == 0:
            continue
        mem = tuple(sorted({int(e) for e in srcs if 0 < int(e) < C_fit}))
        if rule == "most-specific":
            mem = most_specific(mem, pi)
        elif rule == "top":
            mem = top_level(mem, pi, n)
        if len(mem) >= 2:
            out.setdefault(names.get(n, str(n)), mem)
    if with_cids:
        return out, {int(r): int(c) for r, c in rd["int2cid"].items()}
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


def framing2(r):
    """(share of the visible→full P(true) gap closed by 5 VOI questions, VOI − table @5)
    from one group's survey record; NaN without a next-question block."""
    nq = r.get("next_question") or {}
    if not nq:
        return float("nan"), float("nan")
    v0, vf = nq["visible"]["p_true"], nq["full"]["p_true"]
    v5 = nq["by"]["VOI (codes)@5"]["p_true"]
    t5 = nq["by"]["population table@5"]["p_true"]
    return ((v5 - v0) / (vf - v0) if vf > v0 else float("nan")), v5 - t5


def kappa_by_group(run, tag):
    """{group name: (M, kappa of the best engine, per-engine kappas, framing-2 pair,
    E5 code count)} — for the paired comparison."""
    js = json.loads((Path(run) / f"differential_{tag}.json").read_text())
    mem = group_members(run)
    out = {}
    for g, r in js.items():
        mt = r.get("metrics") or {}
        if r.get("members"):                      # recorded by the survey itself
            mem[g] = tuple(sorted(r["members"]))
        if "E6 θ + codes" not in mt or g not in mem:
            continue
        M = len(mem[g])
        ks = {e: kappa(mt[k]["bal_top1"], M) for e, k in
              (("E3", "E3 multiclass head"), ("E5", "E5 code head"),
               ("E6", "E6 θ + codes"))}
        out[g] = (M, max(ks.values()), ks, framing2(r), r.get("e5_n_codes"))
    return out


def paired(run_a, tag_a, run_b, tag_b):
    """Same group NAME scored in both runs (read 4 uses --diff-groups-like): the
    member sets can differ (a nested fit scores most-specific members), so M is
    printed too and kappa — chance-corrected — is the comparable number."""
    a, b = kappa_by_group(run_a, tag_a), kappa_by_group(run_b, tag_b)
    shared = sorted(set(a) & set(b))
    print(f"\n# paired by group name: {Path(run_a).name} vs {Path(run_b).name} — "
          f"{len(shared)} shared groups ({len(a)} / {len(b)} scored)")
    if not shared:
        return
    d = np.array([a[g][1] - b[g][1] for g in shared])
    ka = np.array([a[g][1] for g in shared])
    kb = np.array([b[g][1] for g in shared])
    print("kappa quartiles A: " + " / ".join(f"{v:.2f}" for v in np.percentile(ka, [25, 50, 75]))
          + " · B: " + " / ".join(f"{v:.2f}" for v in np.percentile(kb, [25, 50, 75])))
    print("A − B quartiles: " + " / ".join(f"{v:+.3f}" for v in np.percentile(d, [25, 50, 75]))
          + f" · A better in {int((d > 0).sum())} of {len(d)}; "
          f"kappa >= 0.3 A {int((ka >= 0.3).sum())} / B {int((kb >= 0.3).sum())}")
    # Per engine. Across fits, E5 (raw codes only) is the same machine, so E3 and E6
    # carry the representation; across code selections (exp 0166) on one fit, E3 is
    # the same machine and E5 / E6 carry the selection.
    for e, lab in (("E3", "θ only"), ("E5", "codes only"), ("E6", "θ + codes")):
        de = np.array([a[g][2][e] - b[g][2][e] for g in shared])
        print(f"  {e} ({lab}): A − B quartiles "
              + " / ".join(f"{v:+.3f}" for v in np.percentile(de, [25, 50, 75]))
              + f" · A better in {int((de > 0).sum())} of {len(de)} · median kappa A "
              f"{np.median([a[g][2][e] for g in shared]):.2f} / B "
              f"{np.median([b[g][2][e] for g in shared]):.2f}")
    for i, lab in ((0, "F2 gap closed by 5 questions"), (1, "F2 VOI − table @5")):
        da = np.array([a[g][3][i] - b[g][3][i] for g in shared])
        ok = np.isfinite(da)
        if ok.any():
            print(f"  {lab}: A − B quartiles "
                  + " / ".join(f"{v:+.3f}" for v in np.percentile(da[ok], [25, 50, 75]))
                  + f" · A better in {int((da[ok] > 0).sum())} of {int(ok.sum())}")
    nc = [(a[g][4], b[g][4]) for g in shared if a[g][4] is not None and b[g][4] is not None]
    if nc:
        print("  E5 codes selected, median A / B: "
              f"{np.median([x for x, _ in nc]):.0f} / {np.median([y for _, y in nc]):.0f}")
    same = [g for g in shared if a[g][0] == b[g][0]]
    print(f"same member count M in {len(same)} of {len(shared)} groups")
    if len(same) >= 4:
        ds = np.array([a[g][1] - b[g][1] for g in same])
        print("  same-M groups only: A − B quartiles "
              + " / ".join(f"{v:+.3f}" for v in np.percentile(ds, [25, 50, 75]))
              + f" · A better in {int((ds > 0).sum())} of {len(ds)}")
    print("\n| group | M A/B | kappa A | kappa B | A − B | E3 A/B | E6 A/B |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    order = sorted(shared, key=lambda g: -(a[g][1] - b[g][1]))
    show = order if len(order) <= 24 else order[:12] + order[-12:]
    for g in show:
        print(f"| {g[:38]} | {a[g][0]}/{b[g][0]} | {a[g][1]:.2f} | {b[g][1]:.2f} | "
              f"{a[g][1] - b[g][1]:+.2f} | {a[g][2]['E3']:.2f}/{b[g][2]['E3']:.2f} | "
              f"{a[g][2]['E6']:.2f}/{b[g][2]['E6']:.2f} |")


def workup_summary(js):
    """Exp 0167: framing 2 as workup, pooled over the groups that scored it — the
    coverage, ranking accuracy (R1), value of the top fact against random, the
    population ranking and hindsight (R2), calibration (R3) and decisions (R4)."""
    rows = [(g, r["workup"]) for g, r in js.items()
            if (r.get("workup") or {}).get("coverage")]
    if not rows:
        return
    skipped = sum(1 for r in js.values()
                  if (r.get("workup") or {}).get("skipped"))
    cov = [w["coverage"] for _, w in rows]
    print(f"\n## workup (exp 0167): {len(rows)} groups scored, {skipped} without "
          f"enough windows")
    print("window days (median of group medians) "
          f"{np.median([c['window_days_quartiles'][1] for c in cov]):.0f} · share with "
          f"any non-default answer, median over groups "
          f"{np.median([c['share_with_any'] for c in cov]):.2f}")
    for e in ("E6", "E3"):
        es = [(g, w[e]) for g, w in rows if w.get(e)]
        if not es:
            continue
        r1 = [x["R1_spearman_quartiles"][1] for _, x in es]
        d_rand = [x["R2"]["top1"] - x["R2"]["random"] for _, x in es]
        d_pop = [x["R2"]["top1"] - x["R2"]["population_top1"] for _, x in es]
        share = [x["R2"]["top1"] / x["R2"]["oracle"] if x["R2"]["oracle"] > 0
                 else float("nan") for _, x in es]
        slope = [x["R3"]["slope"] for _, x in es]
        chg = [x["R4"]["top1_changes_call"] for _, x in es]
        q = lambda v: " / ".join(f"{z:+.3f}" for z in np.nanpercentile(v, [25, 50, 75]))
        print(f"- {e} ({len(es)} groups): R1 median Spearman {q(r1)} · top-1 − random "
              f"{q(d_rand)} (> 0 in {sum(d > 0 for d in d_rand)}) · top-1 − "
              f"population {q(d_pop)} (> 0 in {sum(d > 0 for d in d_pop)}) · top-1 / "
              f"oracle {np.nanmedian(share):.2f} · R3 slope median "
              f"{np.nanmedian(slope):.2f} · top-1 changes the call {np.median(chg):.2f}")
    print("\n| group | n | window days | any answer | E6 R1 | E6 top-1 / random / "
          "pop / oracle | slope | E3 R1 |")
    print("|---|--:|--:|--:|--:|---|--:|--:|")
    for g, w in rows:
        e6, e3 = w.get("E6"), w.get("E3")
        if not e6:
            continue
        r2 = e6["R2"]
        print(f"| {g[:38]} | {e6['n']} | {w['coverage']['window_days_quartiles'][1]:.0f} | "
              f"{w['coverage']['share_with_any']:.2f} | "
              f"{e6['R1_spearman_quartiles'][1]:+.2f} | {r2['top1']:+.3f} / "
              f"{r2['random']:+.3f} / {r2['population_top1']:+.3f} / {r2['oracle']:+.3f}"
              f" | {e6['R3']['slope']:.2f} | "
              f"{(e3['R1_spearman_quartiles'][1] if e3 else float('nan')):+.2f} |")


def main(argv):
    run = Path(argv[1])
    tag = argv[2] if len(argv) > 2 else "survey"
    if len(argv) > 3:
        # Paired mode: <run A> <tag A> <run B> [tag B]
        paired(run, tag, Path(argv[3]), argv[4] if len(argv) > 4 else tag)
        return 0
    js = json.loads((run / f"differential_{tag}.json").read_text())
    md = (run / f"differential_{tag}.md").read_text()
    mem = group_members(run)
    from prevalence_check import framing3_sources
    th, rw = framing3_sources(run)

    def f3(members, src):
        if not src:
            return float("nan")
        pn = src["history_stacked"]["per_node"]
        v = [pn[str(m)]["auc_incident"] for m in members if str(m) in pn]
        return float(np.mean(v)) if v else float("nan")
    rows, seen = [], {}
    for g, r in js.items():
        mt = r.get("metrics") or {}
        if r.get("members"):                      # recorded by the survey itself
            mem[g] = tuple(sorted(r["members"]))
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
        gap, dvt = framing2(r)
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
    workup_summary(js)
    rs = reasons_by_group(md)
    print("\nE5 reasons, top 10 groups by kappa (codes favouring each member):")
    for x in rows[:10]:
        print(f"\n## {x['group']}")
        for line in rs.get(x["group"], [])[:8]:
            print(f"- {line[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
