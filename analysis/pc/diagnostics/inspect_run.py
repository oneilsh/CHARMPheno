"""Read a saved pc_results.json and localize the VI-PC head result -- no Spark, no refit.

Answers the cheap artifact checks from the head-optimizer diagnosis
(``head_optimizer_diagnosis.py`` proved the optimizer is sound, so a 0.52 co-fit head
is a run artifact): was this the intended run, is the head actually trained, and does
it point the right way?

Usage:
    python analysis/pc/diagnostics/inspect_run.py <run_dir_or_json> [more ...]
    python analysis/pc/diagnostics/inspect_run.py ~/workspace/dataproc-staging-*/runs/0072-*pc-vi*

A path may be a pc_results.json, a run dir (its pc_results.json is used), or a glob.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def _macro_auc(results, key):
    b = results.get(key)
    if not b or b.get("macro") is None:
        return None
    return b["macro"].get("auc")


def _find_json(path):
    if os.path.isdir(path):
        return os.path.join(path, "pc_results.json")
    return path


def inspect(path):
    jpath = _find_json(path)
    if not os.path.isfile(jpath):
        print(f"  !! no pc_results.json at {jpath}")
        return
    with open(jpath) as f:
        payload = json.load(f)
    results = payload.get("results", {})
    meta = results.get("meta", {})
    svi = meta.get("svi", {})
    conv = meta.get("vi_convergence", {})
    params = payload.get("params", {})

    print(f"\n=== {jpath} ===")
    print(f"  backend={payload.get('backend')} cohort={payload.get('cohort')}  "
          f"K={meta.get('K', params.get('K'))}  "
          f"weight_y={meta.get('weight_y', params.get('weight_y'))}  "
          f"n_train={meta.get('n_train')} n_test={meta.get('n_test')}")

    # [1] intended-run check
    gci = svi.get("grad_cavi_iters", params.get("grad_cavi_iters"))
    print(f"  [1] grad_cavi_iters={gci}  max_iter={svi.get('max_iter')}  "
          f"subsampling={svi.get('subsampling_rate')}  "
          f"warm_start={params.get('warm_start_unsup_iters')}  "
          f"head_lr_scale={svi.get('head_lr_scale')}")

    # [2] head-trained check
    wmax = conv.get("w_CK_absmax")
    trained = None if wmax is None else (wmax > 1e-3)
    tag = "?" if trained is None else ("TRAINED" if trained else "UNTRAINED (~0)")
    print(f"  [2] |w_CK|max={wmax}  -> head {tag}   "
          f"(SVI n_iter={conv.get('n_iter')} converged={conv.get('converged')})")

    # [3] direction check (present only if the run carried the new diagnostic)
    cos = conv.get("head_vs_lr_cosine")
    if cos is not None:
        vals = [x for x in cos if x is not None]
        mean_s = "n/a" if not vals else f"{sum(vals) / len(vals):+.3f}"
        per = ", ".join("n/a" if x is None else f"{x:+.2f}" for x in cos)
        print(f"  [3] head-vs-LR direction cosine: mean={mean_s}  per-label=[{per}]")
    else:
        print("  [3] head-vs-LR cosine: NOT in this JSON (pre-diagnostic run; "
              "re-run --eval-only after pulling to get it)")

    # AUCs
    pc = _macro_auc(results, "PC")
    lr = _macro_auc(results, "pc_topics_lr")
    ts = _macro_auc(results, "two_stage")
    codes = _macro_auc(results, "lr_codes")

    def s(x):
        return "  --  " if x is None else f"{x:.4f}"
    print(f"  macro AUC:  PC(head)={s(pc)}  pc_topics_lr={s(lr)}  "
          f"two_stage={s(ts)}  lr_codes={s(codes)}")

    # verdict
    print("  VERDICT:", _verdict(gci, wmax, cos, pc, lr))


def _verdict(gci, wmax, cos, pc, lr):
    if pc is None:
        return "no PC head AUC in JSON."
    if wmax is not None and wmax <= 1e-3:
        return ("head UNTRAINED (|w_CK|max~=0) -> supervised path/weight_y not applied "
                "in this run; check the fit, not the head math.")
    if lr is not None and pc >= lr - 0.02:
        return "head ~matches pc_topics_lr -> no anomaly here."
    if cos is not None:
        vals = [x for x in cos if x is not None]
        m = None if not vals else sum(vals) / len(vals)
        if m is not None and m > 0.8:
            return (f"head is TRAINED and AIMS RIGHT (mean cos={m:+.3f}) yet AUC<{lr:.3f} "
                    "-> a SCORING/scale artifact in this run, not the optimizer.")
        if m is not None and m < 0.3:
            return (f"head TRAINED but MIS-DIRECTED (mean cos={m:+.3f}) -> a real fit "
                    "problem in this run; debug the actual fit (topics/labels/mask).")
        return f"head trained; mean direction cosine={m}."
    return ("head trained but underperforms pc_topics_lr and this JSON lacks the "
            "direction cosine -> re-run --eval-only after pulling to get check [3].")


def main(argv):
    if not argv:
        print(__doc__)
        return
    paths = []
    for a in argv:
        g = glob.glob(a)
        paths.extend(sorted(g) if g else [a])
    for p in paths:
        inspect(p)


if __name__ == "__main__":
    main(sys.argv[1:])
