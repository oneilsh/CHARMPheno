"""Cloud fit+eval driver for Gated-PC rare-disease case-finding (smooshed vocab).

The forward test named in insight 0066: PC topic-shaping should help in the
HIDDEN-LOW-MASS regime — a rare phenotype an unsupervised fit spends its K
topics missing. This driver runs that test on real All-of-Us OMOP data with a
single smooshed (fused-vocab) BOW, comparing, on the SAME gated hierarchy and
per-node label:

  arm "unsup_gated"  — gated topics, NO supervision (OnlinePCLDA weightY=0). The
                       controlled incumbent: the same gated E-step + K + init as
                       the PC arm, only the head is off. (The dag_placement_cloud
                       node_affinity model is the production incumbent; this is
                       the pc_topics_lr-comparable twin.)
  arm "gated_pc"     — gate + FLAT PC head (weightY>0). Inject the hierarchy ONCE
                       (ADR 0042: gate+flat wins; gate+closure-head collapses).
  arm "dag_head"     — (optional, --with-dag-head) UNGATED + DAG-closure head, the
                       label-side-only alternative to the gate.

Headline metric: `pc_topics_lr` — a fresh post-hoc LogisticRegression on each
arm's FINAL per-doc theta against the per-node label, macro AUC over nodes
(insight 0066: this isolates representation quality from the co-fit head's own
convergence, and is directly comparable across arms). The gated_pc arm also
reports its co-fit head P(node) AUC (secondary; the head's own readout).

The per-node (label, labelMask) is emitted by the Step-A adapter
(case_finding_assembly.emit_labels): label[c]=1 iff node c is in the is-a
closure of the doc's frontier; C = len(int2cid). Mirrors dag_placement_cloud's
assemble->fit->score->save shape (cached CaseFindingBundle, npz + manifest.json).

CANNOT RUN HERE: every fact table is read from the workspace CDR via the
spark-bigquery connector, so main() only executes inside the All-of-Us Dataproc
workspace. Unit tests cover parse_args + the pure pc_topics_lr scorer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session


# --------------------------------------------------------------------------- #
# Pure scoring (no SparkSession; unit-tested numpy-level).                     #
# --------------------------------------------------------------------------- #
def pc_topics_lr_bundle(Pi_tr, y_tr, mask_tr, Pi_te, y_te, mask_te, C,
                        min_count=0):
    """`pc_topics_lr`: fit a fresh per-node LogisticRegression on TRAIN theta and
    score it on TEST theta, per-node AUC/AP macro'd over the observed cells.

    The convergence-robust representation-quality metric (insight 0066): it reads
    ONLY the final per-doc theta of an arm, so it measures whether that arm's
    topics carry per-node predictive signal — independent of any co-fit head's
    own optimization. Reuses the analysis.pc.evaluate masked-LR + masked-scorer
    so the number is identical machinery to the antidepressant driver's
    two-stage/pc_topics_lr readout. Pure (numpy in, dict out)."""
    from analysis.pc.evaluate import _bundle_masked, _lr_proba_per_label_masked

    proba = _lr_proba_per_label_masked(Pi_tr, y_tr, mask_tr, Pi_te, C)
    return _bundle_masked(proba, y_te, mask_te, C, min_count)


def precision_at_recall(y, p, targets):
    """Max precision achievable at recall >= each target (the case-finding operating
    point: 'if we must catch t% of true cases, how clean is the surfaced list?').
    NaN for a degenerate (single-class) column. Pure numpy + sklearn."""
    from sklearn.metrics import precision_recall_curve
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return {t: float("nan") for t in targets}
    prec, rec, _ = precision_recall_curve(y, p)
    out = {}
    for t in targets:
        m = rec >= t
        out[float(t)] = float(np.max(prec[m])) if np.any(m) else float("nan")
    return out


def recall_at_fdr(y, p, targets):
    """Max recall achievable while holding the false-discovery rate <= each q
    (FDR = 1 - precision): 'at a q% junk tolerance, what fraction of true cases do
    we recover?'. The FDR-controlled discovery view (insight 0064: ranking AUC !=
    FDR-controlled discovery — report both). 0.0 if no threshold meets the bound."""
    from sklearn.metrics import precision_recall_curve
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return {q: float("nan") for q in targets}
    prec, rec, _ = precision_recall_curve(y, p)
    out = {}
    for q in targets:
        m = prec >= (1.0 - q)
        out[float(q)] = float(np.max(rec[m])) if np.any(m) else 0.0
    return out


def pr_readout(proba_DC, y_DC, mask_DC, C, recall_targets, fdr_targets, min_count=0):
    """Per-node precision@recall + recall@FDR over observed test cells, macro'd over
    the non-degenerate nodes. The case-finding complement to the ranking AUC/AP: it
    answers 'at a usable operating point, how clean / how complete is each disease
    node's surfaced cohort?'. Pure; feeds off the same per-node proba as pc_topics_lr."""
    per = {}
    for c in range(C):
        rows = np.where(mask_DC[:, c].astype(bool))[0]
        yc, pc = y_DC[rows, c], proba_DC[rows, c]
        n_pos, n_neg = int(yc.sum()), int(len(yc) - yc.sum())
        if n_pos < max(min_count, 1) or n_neg < max(min_count, 1):
            per[c] = {"skipped": True, "n_pos": n_pos, "n_neg": n_neg}
            continue
        per[c] = {"skipped": False, "n_pos": n_pos, "n_neg": n_neg,
                  "par": precision_at_recall(yc, pc, recall_targets),
                  "raf": recall_at_fdr(yc, pc, fdr_targets)}
    scored = [d for d in per.values() if not d["skipped"]]

    def _mean(vals):
        vals = [v for v in vals if v == v]                 # drop NaN
        return float(np.mean(vals)) if vals else None

    macro = {
        "n_scored": len(scored),
        "par": {float(t): _mean([d["par"][float(t)] for d in scored])
                for t in recall_targets},
        "raf": {float(q): _mean([d["raf"][float(q)] for d in scored])
                for q in fdr_targets},
    }
    return {"per_node": per, "macro": macro}


def detection_readout(proba_DC, y_DC, recall_targets):
    """Case-vs-background detection: pool a per-doc case SCORE = max over disease-node
    probabilities and the foreground indicator = the root node's label (label[:,0]=1
    iff the doc has any attested disease node). Reports AUC/AP + precision@recall on
    that pooled signal — 'can we tell a rare-disease patient from a background one,
    and how precise is the surfaced case list?'. Empty/degenerate -> skipped."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if proba_DC.shape[0] == 0 or proba_DC.shape[1] < 2:
        return {"skipped": "empty or single-node"}
    y = np.asarray(y_DC[:, 0], float)                       # root = any-disease
    score = proba_DC[:, 1:].max(axis=1)                     # strongest disease node
    if len(np.unique(y)) < 2:
        return {"skipped": "degenerate foreground indicator"}
    return {"skipped": None, "prevalence": float(y.mean()),
            "auc": float(roc_auc_score(y, score)),
            "ap": float(average_precision_score(y, score)),
            "par": precision_at_recall(y, score, recall_targets)}


def readout_from_proba(proba, y_te, m_te, C, *, recall_targets, fdr_targets,
                       min_count=0):
    """Full readout from an already-computed (N_te, C) per-node probability:
    ranking (AUC/AP) + per-node precision@recall / recall@FDR + case-vs-background
    detection. Shared by the pc_topics_lr arm (proba = post-hoc LR on theta) and the
    co-fit head arm (proba = sigmoid(w_CK·theta))."""
    from analysis.pc.evaluate import _bundle_masked
    ranking = _bundle_masked(proba, y_te, m_te, C, min_count)
    pr = pr_readout(proba, y_te, m_te, C, recall_targets, fdr_targets, min_count)
    det = detection_readout(proba, y_te, recall_targets)
    return {"ranking": ranking["macro"], "pr": pr["macro"], "detection": det}


def score_arm(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, *, recall_targets,
              fdr_targets, min_count=0):
    """Full readout for one arm's theta via the pc_topics_lr proba (a fresh per-node
    LR on the shaped theta). Pure; used inline by the driver and by gated_pc_readout
    on a finished fit."""
    from analysis.pc.evaluate import _lr_proba_per_label_masked
    proba = _lr_proba_per_label_masked(Pi_tr, y_tr, m_tr, Pi_te, C)
    return readout_from_proba(proba, y_te, m_te, C, recall_targets=recall_targets,
                              fdr_targets=fdr_targets, min_count=min_count)


def format_arm_readout(name, arm):
    """Render one arm's score_arm() result as driver log lines (ranking + PR@recall +
    recall@FDR + detection)."""
    r, pr, det = arm["ranking"], arm["pr"], arm["detection"]
    auc = "n/a" if r["auc"] is None else f"{r['auc']:.4f}"
    ap = "n/a" if r["ap"] is None else f"{r['ap']:.4f}"
    par = " ".join(f"P@R{t:g}={('n/a' if v is None else f'{v:.3f}')}"
                   for t, v in sorted(pr["par"].items()))
    raf = " ".join(f"R@FDR{q:g}={('n/a' if v is None else f'{v:.3f}')}"
                   for q, v in sorted(pr["raf"].items()))
    lines = [f"{name}: macro AUC={auc} AP={ap} (over {r['n_labels_scored']} nodes)",
             f"{name}: node-macro {par}  {raf}"]
    if det.get("skipped"):
        lines.append(f"{name}: detection skipped ({det['skipped']})")
    else:
        dpar = " ".join(f"P@R{t:g}={('n/a' if v != v else f'{v:.3f}')}"
                        for t, v in sorted(det["par"].items()))
        lines.append(f"{name}: detection (case vs bg) AUC={det['auc']:.4f} "
                     f"AP={det['ap']:.4f} prev={det['prevalence']:.3f}  {dpar}")
    return "\n".join("[driver]   " + ln for ln in lines)


def _dag_children_and_depth(parent_int, C):
    """(children map {node: [children]}, depth {node: int}) over engine ids [0, C).

    ``parent_int`` maps child -> [parent engine-ids] (parentless/root -> []). Depth
    is the BFS distance from the parentless roots (the synthetic forest root is
    depth 0; the disease anchors depth 1; their Mondo/SNOMED subtypes deeper)."""
    from collections import deque
    children = {c: [] for c in range(C)}
    for child in range(C):
        for p in parent_int.get(child, []):
            p = int(p)
            if 0 <= p < C:
                children[p].append(child)
    depth = {c: None for c in range(C)}
    roots = [c for c in range(C) if not parent_int.get(c)]
    dq = deque((r, 0) for r in roots)
    while dq:
        n, d = dq.popleft()
        if depth[n] is not None and depth[n] <= d:
            continue
        depth[n] = d
        for ch in children[n]:
            dq.append((ch, d + 1))
    return children, depth


def conditional_readout(proba, y, mask, parent_int, C, *, min_count=10):
    """Conditional 'sharpening' metrics — P(child | parent-cohort), the clinician's
    'this patient has a <parent>; which <child>?' task (vs de-novo detection).

    For each DAG edge parent p -> child c, restrict to the TEST docs attested at p
    (``y[:,p]==1``) and score how well ``proba[:,c]`` ranks c's positives against
    their siblings *within that cohort*. Because the base rate among p's cohort is
    far higher than marginal prevalence, this is the metric that is NOT prevalence-
    crushed (insight 0064) — so it reveals whether the representation can subtype.

    Read the numbers HONESTLY (VOI/metrics report §2): cond_AUC is the sober,
    prevalence-INDEPENDENT discrimination number — lead with it. cond_AP and the
    'lift over marginal' mostly reflect the base rate mechanically rising when you
    condition (not skill), so they are context, not headline. Multiclass top-1 is
    reported beside its MAJORITY-CLASS baseline (predict the commonest child) and a
    balanced accuracy, because a 2-child parent has random=0.5 / majority up to ~0.7.
    A pooled ECE gauges whether P(child|parent) is calibrated (needed for VOI).

    MASK-INDEPENDENCE: pass the FULL-closure observation mask (all-ones) as ``mask``
    regardless of the run's training label_mask_mode. The `label` array is already
    mask-mode-independent (closure membership), so an all-ones eval mask fixes the
    cohort/negative sets identically across full- and closure-mask runs — otherwise
    the closure mask silently makes the conditional eval an easier sibling-only
    contrast and cross-run numbers are not comparable (exp 0079, Trap 3). Pure numpy."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    children, depth = _dag_children_and_depth(parent_int, C)
    edges, parents = [], []
    pooled_y, pooled_p = [], []                          # for a single ECE over all edges
    for p in range(C):
        kids = children[p]
        if not kids:
            continue
        cohort = np.where((y[:, p] == 1) & (mask[:, p] == 1))[0]
        if len(cohort) < max(min_count, 2):
            continue
        scored_kids = []
        for c in kids:
            rows = cohort[mask[cohort, c] == 1]
            yc = y[rows, c]
            n_pos, n_neg = int(yc.sum()), int(len(yc) - yc.sum())
            if n_pos < max(min_count, 1) or n_neg < max(min_count, 1):
                continue
            sc = proba[rows, c]
            # marginal (de-novo) AP: child vs ALL observed docs — the sharpening bar.
            mrows = np.where(mask[:, c] == 1)[0]
            my = y[mrows, c]
            marg_ap = (float(average_precision_score(my, proba[mrows, c]))
                       if 0 < my.sum() < len(my) else None)
            edges.append({
                "parent": p, "child": c, "depth": depth[p], "cohort": int(len(cohort)),
                "n_pos": n_pos, "prev": float(yc.mean()),
                "cond_auc": float(roc_auc_score(yc, sc)),
                "cond_ap": float(average_precision_score(yc, sc)),
                "marg_ap": marg_ap})
            scored_kids.append(c)
            pooled_y.append(yc); pooled_p.append(sc)
        if len(scored_kids) >= 2:
            ka = np.array(scored_kids)
            at_child = cohort[(y[cohort][:, ka] == 1).any(axis=1)]
            if len(at_child):
                pred = ka[np.argmax(proba[at_child][:, ka], axis=1)]
                correct = y[at_child, pred] == 1     # argmax child is a TRUE child of p
                # majority-class baseline: always predict the commonest child.
                child_counts = np.array([(y[at_child, c] == 1).sum() for c in ka])
                majority = float(child_counts.max() / len(at_child))
                # balanced accuracy: mean per-child recall (guards imbalance).
                recalls = []
                for c in ka:
                    truth = y[at_child, c] == 1
                    if truth.sum() > 0:
                        recalls.append(float((pred[truth] == c).mean()))
                bal_acc = float(np.mean(recalls)) if recalls else None
                parents.append({"parent": p, "depth": depth[p], "n": int(len(at_child)),
                                "n_children": len(scored_kids),
                                "top1": float(correct.mean()),
                                "majority": majority, "bal_acc": bal_acc})
    ece = None
    if pooled_y:
        ece = _ece(np.concatenate(pooled_y), np.concatenate(pooled_p))
    return {"edges": edges, "parents": parents, "ece": ece}


def _ece(y, p, n_bins=10):
    """Expected calibration error (equal-width bins): Σ_b (n_b/N)·|conf_b − acc_b|.
    The gap between predicted P(child|parent) and observed frequency — 0 = perfectly
    calibrated. Required for a real diagnostic aid and mandatory for VOI (the entropy
    H(p) must be real, not just a ranking). Pure numpy."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += abs(p[m].mean() - y[m].mean()) * (m.mean())
    return float(ece)


def calibrate_per_node(proba_tr, y_tr, m_tr, proba_te, C, *, min_pos=20):
    """Per-node ISOTONIC calibration fit on TRAIN predictions, applied to TEST.

    The head-independent path to CALIBRATED P(node) — the VOI prerequisite — without
    a well-behaved co-fit head (exp 0080: Firth couldn't tame the closure-mask head;
    this sidesteps it). Isotonic is monotone so it preserves the ranking (AUC/AP
    unchanged) while fixing the reliability curve. A node with too few observed train
    positives (or a single class) passes through UNCALIBRATED. Fit on train, applied
    to test — the standard train-fold calibration; a dedicated calibration split is a
    future refinement. Pure numpy + sklearn."""
    from sklearn.isotonic import IsotonicRegression
    out = np.asarray(proba_te, dtype=np.float64).copy()
    for c in range(C):
        tr = np.asarray(m_tr[:, c], bool)
        yc = y_tr[tr, c]
        if int(yc.sum()) < min_pos or len(np.unique(yc)) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(proba_tr[tr, c], yc)
        out[:, c] = iso.transform(proba_te[:, c])
    return out


def format_conditional_readout(cond, int2cid, name_by_id):
    """Render conditional_readout HONESTLY (VOI/metrics report §2): cond_AUC is the
    headline discrimination number (prevalence-independent); cond_AP/lift are marked
    'context' (mostly the base-rate rise). Per-parent top-1 is shown WITH its
    majority-class baseline and balanced accuracy, sorted by depth then top-1-minus-
    majority (the honest lift). A pooled ECE reports calibration. Eval is
    mask-independent (full-closure cohorts) — see conditional_readout."""
    edges, parents = cond["edges"], cond["parents"]
    if not edges:
        return "[conditional sharpening]  no parent->child edges met min_count"

    def _name(node):
        cid = int2cid.get(node)
        if cid is None or cid not in name_by_id:
            return "(root)" if node == 0 else str(cid)
        return str(name_by_id[cid])[:22]

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    ece = cond.get("ece")
    depths = sorted({e["depth"] for e in edges if e["depth"] is not None})
    lines = [f"[conditional sharpening]  P(child|parent), by DAG depth "
             f"(HEADLINE=cond_AUC; AP/lift are base-rate context)  ECE={_f(ece).strip()}",
             "  depth  #edges  cond_AUC  |  cond_AP  marg_AP  lift  (context)  top1"]
    for d in depths:
        de = [e for e in edges if e["depth"] == d]
        dp = [p for p in parents if p["depth"] == d]
        cap = _mean([e["cond_ap"] for e in de])
        map_ = _mean([e["marg_ap"] for e in de])
        lift = (cap - map_) if (cap is not None and map_ is not None) else None
        auc = _mean([e["cond_auc"] for e in de])
        top1 = _mean([p["top1"] for p in dp])
        lines.append(
            f"  {d:>5}  {len(de):>6}  {_f(auc)}  |  {_f(cap)}  {_f(map_)}"
            f"  {_f(lift)}            {_f(top1)}")
    lines.append("  per-parent multiclass top-1 vs majority-class baseline "
                 "(which child, given the parent):")
    for p in sorted(parents, key=lambda p: (p["depth"], -(p["top1"] - p["majority"]))):
        ba = "" if p.get("bal_acc") is None else f" bal_acc={p['bal_acc']:.3f}"
        lines.append(
            f"    d{p['depth']} {_name(p['parent']):<22} "
            f"top1={p['top1']:.3f} (majority={p['majority']:.3f}){ba}  "
            f"(n={p['n']}, {p['n_children']} children)")
    return "\n".join(lines)


def _f(v):
    """Format an optional float for the readout tables (n/a for None)."""
    return "  n/a " if v is None else f"{v:.4f}"


def _print_headline(results):
    """The one-glance comparison: gated_pc vs unsup_gated on the numbers that matter
    for a rare-disease surface — ranking (AUC/AP) AND the case-finding operating
    points (detection AP, node-macro precision@0.9-recall). A positive delta in the
    hidden-low-mass regime is the thesis (insight 0066); AUC alone is optimistic
    under low prevalence, so AP / P@R are the honest case-finding read (insight 0064)."""
    g = results.get("gated_pc"); u = results.get("unsup_gated")
    if not (g and u):
        return

    def _d(a, b):
        if a is None or b is None:
            return "n/a"
        return f"{a:.4f} vs {b:.4f} (Δ{a - b:+.4f})"

    def _par9(arm):
        return arm["pr"]["par"].get(0.9)

    def _det(arm, k):
        d = arm["detection"]
        return None if d.get("skipped") else d.get(k)

    print("[driver]   HEADLINE (gated_pc vs unsup_gated):", flush=True)
    print(f"[driver]     pc_topics_lr  AUC {_d(g['ranking']['auc'], u['ranking']['auc'])}",
          flush=True)
    print(f"[driver]     pc_topics_lr  AP  {_d(g['ranking']['ap'], u['ranking']['ap'])}",
          flush=True)
    print(f"[driver]     node P@R0.9        {_d(_par9(g), _par9(u))}", flush=True)
    print(f"[driver]     detection AP       {_d(_det(g, 'ap'), _det(u, 'ap'))}", flush=True)
    print("[driver]     (PC should help in the hidden-low-mass regime — insight 0066; "
          "AP/P@R are the honest case-finding read — insight 0064.)", flush=True)

    # Conditional 'sharpening' comparison: does supervision improve P(child|parent)
    # — the clinical subtyping task — over the unsupervised twin? Mean over edges.
    gc, uc = results.get("gated_pc_conditional"), results.get("unsup_gated_conditional")
    if gc and uc and gc["edges"] and uc["edges"]:
        def _mean(d, key, coll="edges"):
            vals = [r[key] for r in d[coll] if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        print("[driver]   CONDITIONAL sharpening (gated_pc vs unsup_gated):", flush=True)
        print(f"[driver]     cond AP (child|parent) {_d(_mean(gc,'cond_ap'), _mean(uc,'cond_ap'))}",
              flush=True)
        print(f"[driver]     cond AUC               {_d(_mean(gc,'cond_auc'), _mean(uc,'cond_auc'))}",
              flush=True)
        print(f"[driver]     multiclass top1        {_d(_mean(gc,'top1','parents'), _mean(uc,'top1','parents'))}",
              flush=True)


def dag_closure_parents(parent_int, C):
    """Length-C list of parent-LABEL-index lists for the ungated DAG-closure head.

    closure_parents[c] lists node c's direct parents in the [0, C) engine-id label
    space (root and any parentless node -> []); this is exactly `parent_int` (root
    omitted) densified over range(C). Feeds OnlinePCLDAEstimator.setClosureParents."""
    return [[int(p) for p in parent_int.get(c, [])] for c in range(C)]


# --------------------------------------------------------------------------- #
# Spark collectors (cluster-covered; not unit-tested).                        #
# --------------------------------------------------------------------------- #
def _collect_theta_labels(df, C, *, label_col="label", mask_col="labelMask",
                          topic_col="topicDistribution"):
    """Collect ONLY the K-dim per-doc theta + the (C,) label/mask arrays to numpy
    (never the dense BOW), so it stays on the driver's memory budget at cohort
    scale. Returns (Pi (D,K), y_DC (D,C), mask_DC (D,C), person_order). Empty df
    -> correctly-shaped zero arrays. Mirrors pc_antidepressant_cloud's
    _collect_topics_labels but with configurable label/mask column names (this
    corpus uses 'label'/'labelMask' from the Step-A adapter)."""
    rows = df.select("person_id", topic_col, label_col, mask_col).collect()
    person_order = [r["person_id"] for r in rows]
    if rows:
        Pi = np.asarray([r[topic_col].toArray() for r in rows], dtype=np.float64)
        y_DC = np.asarray([[float(v) for v in r[label_col]] for r in rows],
                          dtype=np.float64)
        mask_DC = np.asarray([[float(v) for v in r[mask_col]] for r in rows],
                             dtype=np.float64)
    else:
        Pi = np.zeros((0, 0), dtype=np.float64)
        y_DC = np.zeros((0, C), dtype=np.float64)
        mask_DC = np.zeros((0, C), dtype=np.float64)
    return Pi, y_DC, mask_DC, person_order


def _collect_head_proba(df, C, *, prob_col="probability", label_col="label",
                        mask_col="labelMask"):
    """Collect the co-fit head's per-node P(node)=sigmoid(w_CK.theta) + label/mask
    to (proba (D,C), y (D,C), mask (D,C)). Only meaningful for a supervised
    (weightY>0) transform, which appends `prob_col`. Cluster-covered."""
    rows = df.select(prob_col, label_col, mask_col).collect()
    if not rows:
        z = np.zeros((0, C), dtype=np.float64)
        return z, z, z.copy()
    proba = np.asarray([r[prob_col].toArray() for r in rows], dtype=np.float64)
    y_DC = np.asarray([[float(v) for v in r[label_col]] for r in rows],
                      dtype=np.float64)
    mask_DC = np.asarray([[float(v) for v in r[mask_col]] for r in rows],
                         dtype=np.float64)
    return proba, y_DC, mask_DC


def per_node_domain_mass(lam_dict, lay, domain_names):
    """Per DAG node, the fraction of its topic block's λ mass in each domain.

    The DIRECT test of the multi-domain PC thesis: does a node's topic block
    specialize toward the domain that predicts IT? For node u, its block rows are
    ``lay.block[u]``; domain m's mass is ``lam_dict[m][block].sum()``, and the
    per-node fractions across domains sum to 1. Returns {node_id: [frac_per_domain]}
    plus the background rows' fractions under key -1 (the shared bg topics)."""
    out = {}
    rows_by_key = {-1: list(range(lay.n_bg))}
    for u in lay.nodes:
        rows_by_key[u] = list(lay.block[u])
    for key, rows in rows_by_key.items():
        if not rows:
            continue
        per_dom = np.array([float(lam_dict[m][rows].sum()) for m in sorted(lam_dict)])
        tot = per_dom.sum()
        out[key] = (per_dom / tot).tolist() if tot > 0 else per_dom.tolist()
    return out


def format_per_node_domain_mass(mass_by_node, domain_names, int2cid, name_by_id):
    """Render per_node_domain_mass as an aligned driver-log table (node -> per-domain
    fraction), sorted by the LAST domain's fraction so the most domain-specialized
    nodes surface first — the specialist-disease story is the headline."""
    hdr = "  ".join(f"{n:>10.10}" for n in domain_names)
    lines = [f"[per-node domain λ-mass]  node{'':<20} {hdr}"]
    def _name(key):
        if key == -1:
            return "(background)"
        cid = int2cid.get(key)
        return f"{name_by_id.get(cid, cid)}"[:24]
    ordered = sorted(mass_by_node.items(), key=lambda kv: -kv[1][-1])
    for key, fracs in ordered:
        cells = "  ".join(f"{f:>10.3f}" for f in fracs)
        lines.append(f"  {_name(key):<24} {cells}")
    return "\n".join(lines)


def _macro_line(name, bundle):
    """One-line macro summary of a `_bundle_masked` result for the driver log."""
    m = bundle["macro"]
    auc = "n/a" if m["auc"] is None else f"{m['auc']:.4f}"
    ap = "n/a" if m["ap"] is None else f"{m['ap']:.4f}"
    return (f"{name}: macro AUC={auc} AP={ap} "
            f"(scored {m['n_labels_scored']}/{m['n_labels_scored'] + m['n_labels_skipped']} nodes)")


def _make_eval_logger(bundle, C, args):
    """Per-iteration callback that logs pc_topics_lr AUC + detection AP every
    `args.eval_every` iters, so the supervised shaping can be watched converge live.

    Rebuilds a scoring model from the CURRENT global_params (lambda/alpha), transforms
    train+test, and runs the same `score_arm` as the final eval. COST: two full CAVI
    transforms + an LR fit per eval — keep `eval_every` modest (e.g. 20–25 → ~4–5 evals
    over 100 iters). Wrapped so a scoring hiccup can never kill the fit."""
    rt, ft = args._recall_targets, args._fdr_targets
    every = int(args.eval_every)

    def _cb(iter_num, gp, _elbo):
        if every <= 0 or iter_num % every != 0:
            return
        try:
            from spark_vi.core.result import VIResult
            from spark_vi.mllib.topic.pc import OnlinePCLDAModel
            result = VIResult(
                global_params={k: gp[k] for k in ("lambda", "alpha", "w_CK")
                               if k in gp},
                elbo_trace=[], n_iterations=int(iter_num), converged=False)
            m = OnlinePCLDAModel(result)
            m._set(numLabels=C, caviMaxIter=args.cavi_max_iter,
                   caviTol=args.cavi_tol, gammaShape=args.gamma_shape)
            # Multi-domain: the scoring transform must read the per-domain feature
            # columns (else it defaults to `features`, which the multi-domain corpus
            # does not have). dict-λ already routes expElogbeta fusing in _transform.
            if getattr(args, "_domain_cols", None):
                m._set(featuresCols=args._domain_cols)
            Pi_tr, y_tr, mtr, _ = _collect_theta_labels(m.transform(bundle.train_df), C)
            Pi_te, y_te, mte, _ = _collect_theta_labels(m.transform(bundle.test_df), C)
            arm = score_arm(Pi_tr, y_tr, mtr, Pi_te, y_te, mte, C,
                            recall_targets=rt, fdr_targets=ft,
                            min_count=args.min_label_count)
            auc = arm["ranking"]["auc"]
            det = arm["detection"]
            detap = None if det.get("skipped") else det.get("ap")
            print(f"[driver]   eval@iter{iter_num}: pc_topics_lr AUC="
                  f"{'n/a' if auc is None else f'{auc:.4f}'}  detection AP="
                  f"{'n/a' if detap is None else f'{detap:.4f}'}", flush=True)
        except Exception as exc:                       # noqa: BLE001 — never kill the fit
            print(f"[driver]   eval@iter{iter_num}: FAILED "
                  f"({type(exc).__name__}: {exc})", flush=True)

    return _cb


def _build_pc_estimator(args, *, weight_y, gated, closure_parents=None):
    """Construct an OnlinePCLDAEstimator for one arm. `gated` injects the gated
    topic engine (gateParent set post-construction, since the Param is a JSON
    string); `closure_parents` selects the DAG-closure head. K comes from the gate
    layout when gated, else --k."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    est = OnlinePCLDAEstimator(
        featuresCol="features", labelCol="label", labelMaskCol="labelMask",
        numLabels=args._C, weightY=float(weight_y), k=args.k,
        maxIter=args.max_iter, seed=args.seed,
        subsamplingRate=args.subsampling_rate,
        learningOffset=args.tau0, learningDecay=args.kappa,
        gammaShape=args.gamma_shape, caviMaxIter=args.cavi_max_iter,
        caviTol=args.cavi_tol, gradCaviIters=args.grad_cavi_iters,
        topicTrust=args.topic_trust, weightYWarmupIters=args.weight_y_warmup_iters,
        headOptimizer=args.head_optimizer, headLr=args.head_lr,
        headNewtonRidge=args.head_newton_ridge, headL2=args.head_l2,
        headPenalty=args.head_penalty, headInnerIters=args.head_inner_iters,
        optimizeDocConcentration=args.optimize_doc_concentration,
        frontierCol="frontier", gateNBg=args.n_bg, gateTpn=args.tpn,
    )
    # Multi-domain: feed per-domain feature columns (features_0..) so the gated
    # engine carries a per-domain lambda and the topic correction scatters per
    # domain. Domain widths are read from the first row (no explicit domainBounds).
    dom_cols = getattr(args, "_domain_cols", None)
    if dom_cols:
        est._set(featuresCols=dom_cols)
    if gated:
        est.setGateParent(args._parent_int)      # JSON-encodes the DAG map
    if closure_parents is not None:
        est.setClosureParents(closure_parents)
    return est


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gated-PC rare-disease case-finding fit + pc_topics_lr eval.")
    p.add_argument("--cdr", required=True)
    p.add_argument("--billing", required=True)
    p.add_argument("--source-table", default="condition_era")
    p.add_argument("--person-mod", type=int, default=10)
    p.add_argument("--vocab-size", type=int, default=5000)
    p.add_argument("--min-df", type=int, default=20)
    p.add_argument("--min-patient-count", type=int, default=20)
    p.add_argument("--doc-min-length", type=int, default=0)
    p.add_argument("--prior-obs-days", type=int, default=365)
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--window-mode", choices=["forward", "lookback"], default="forward")
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--label-window-days", type=int, default=365)
    # assembly / DAG. `disease` selects the foreground cohort AND the label-DAG
    # anchors (cohorts.disease_anchors); rare6 = the six-anchor rare-disease forest.
    p.add_argument("--disease", default="rare6")
    p.add_argument("--extra-domains", default="",
                   help="comma-separated non-condition domains (drug, procedure) for "
                        "a MULTI-DOMAIN gated-PC fit (MixEHR-style per-domain vocab; "
                        "domain 0 is always conditions). Empty (default) = single "
                        "fused condition vocabulary. Requires --window-mode lookback. "
                        "The per-domain lambda is inspected per node in the readout.")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--strip-mode", choices=["test_only", "both"], default="test_only")
    p.add_argument("--label-mask-mode", choices=["full", "closure"], default="full",
                   help="per-node observation policy for the PC label (Step-A "
                        "adapter): 'full' (ones(C); background is a true negative "
                        "everywhere) or 'closure' (observe the active closure + DAG "
                        "siblings only).")
    # gate topic-block layout (topics per node); K is emergent = n_bg + nodes*tpn.
    p.add_argument("--n-bg", type=int, default=2)
    p.add_argument("--tpn", type=int, default=1)
    p.add_argument("--k", type=int, default=50,
                   help="K for the UNGATED --with-dag-head arm only; the gated arms "
                        "derive K from the layout (n_bg + nodes*tpn).")
    p.add_argument("--optimize-doc-concentration", action="store_true")
    # PC head + SVI
    p.add_argument("--weight-y", type=float, default=50.0,
                   help="PC prediction weight (>0). Hughes ~ tokens/doc; tune on "
                        "validation. The unsup_gated arm always refits at 0.")
    p.add_argument("--head-optimizer", choices=["sgd", "newton"], default="newton",
                   help="head optimizer; 'newton' is the settled convergent head "
                        "(ADR 0039) — the default here.")
    p.add_argument("--head-lr", type=float, default=0.5)
    p.add_argument("--head-newton-ridge", type=float, default=0.01)
    p.add_argument("--head-l2", type=float, default=1e-3,
                   help="ABSOLUTE ridge on w_CK (= Hughes lambda_w; ADR 0041). "
                        "0.0 BLOWS UP on the separable topics PC creates.")
    p.add_argument("--head-penalty", choices=["none", "firth"], default="none",
                   help="'none' (default; the fixed headL2 ridge) or 'firth' (the "
                        "Jeffreys-prior +1/2 log det H penalty — PARAMETER-FREE, bounds "
                        "|w| exactly at separation with no headL2 tuning). 'firth' needs "
                        "the flat head + newton and runs the inner-loop IRLS path.")
    p.add_argument("--head-inner-iters", type=int, default=0,
                   help="driver-side inner-loop IRLS steps converging the flat head each "
                        "SVI iter (0 = aggregated one-step Newton). Required (>0) for "
                        "--head-penalty firth; auto-enabled to 25 when 0.")
    p.add_argument("--grad-cavi-iters", type=int, default=20)
    p.add_argument("--topic-trust", type=float, default=0.1)
    p.add_argument("--weight-y-warmup-iters", type=int, default=10)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--subsampling-rate", type=float, default=0.05)
    p.add_argument("--tau0", type=float, default=64.0,
                   help="Robbins-Monro offset. On smaller cohorts try ~10-64 so the "
                        "head actually moves.")
    p.add_argument("--kappa", type=float, default=0.51)
    p.add_argument("--gamma-shape", type=float, default=100.0)
    p.add_argument("--cavi-max-iter", type=int, default=100)
    p.add_argument("--cavi-tol", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=None)
    # arms + eval
    p.add_argument("--skip-unsup-gated", action="store_true",
                   help="skip the weightY=0 gated baseline (fit only the PC arm).")
    p.add_argument("--with-dag-head", action="store_true",
                   help="also fit the UNGATED + DAG-closure-head arm (label-side "
                        "hierarchy alternative to the gate).")
    p.add_argument("--baseline-max-iter", type=int, default=-1,
                   help="cap the unsup_gated fit at N iters; <=0 reuses --max-iter "
                        "(unsupervised topics converge faster than the head).")
    p.add_argument("--min-label-count", type=int, default=20,
                   help="mask any node whose heldout column has < this many cells "
                        "of either class from the macro (AoU small-cell floor).")
    p.add_argument("--eval-every", type=int, default=0,
                   help="log pc_topics_lr AUC + detection AP every N iters of the "
                        "gated_pc fit (0 = off, only the final eval). Each eval is 2 "
                        "full CAVI transforms + an LR fit on the driver, so keep it "
                        "modest (e.g. 20–25). Lets you watch the shaping converge.")
    p.add_argument("--recall-targets", default="0.5,0.8,0.9",
                   help="comma-separated recall levels for precision@recall "
                        "(the case-finding operating points).")
    p.add_argument("--fdr-targets", default="0.1,0.25,0.5",
                   help="comma-separated FDR (=1-precision) levels for recall@FDR "
                        "(FDR-controlled discovery; insight 0064).")
    p.add_argument("--num-partitions", type=int, default=0,
                   help="repartition the corpus to this many partitions before "
                        "fitting (0 = leave as-is). Few parquet part-files pin the "
                        "job to ~2 executors under dynamic allocation; set this ≈ "
                        "total cluster executor cores to spread the per-doc autograd "
                        "and demand more executors. Pair with "
                        "CHARM_SPARK_CONF='spark.locality.wait=0s'.")
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume-from", default="",
                   help="Unused; accepted for run_experiment parity.")
    return p.parse_args(argv)


def main() -> int:
    from _case_finding_cache import load_or_build_case_finding_bundle
    from spark_vi.models.topic.dag_placement import DagLayout

    args = parse_args()
    configure_logging()
    extra_domains = tuple(d for d in args.extra_domains.split(",") if d)
    with make_spark_session(app_name="gated-pc-fit") as spark:
        if extra_domains:
            # MULTI-DOMAIN corpus (conditions + extra domains, per-domain vocab).
            # Built fresh (no cache) — a one-off comparison run; the cache key +
            # per-domain save is future. Requires lookback windowing.
            if args.window_mode != "lookback":
                raise ValueError("--extra-domains requires --window-mode lookback")
            from charmpheno.omop.multi_domain import (
                assemble_multidomain_case_finding_corpus)
            with _phase(f"assemble MULTI-DOMAIN corpus (cond + {list(extra_domains)})"):
                bundle = assemble_multidomain_case_finding_corpus(
                    spark, disease=args.disease, cdr=args.cdr, billing=args.billing,
                    extra_domains=extra_domains, person_mod=args.person_mod,
                    min_n=args.min_n, holdout_frac=args.holdout_frac,
                    vocab_size=args.vocab_size, min_df=args.min_df,
                    min_patient_count=args.min_patient_count, n_bg=args.n_bg,
                    tpn=args.tpn, doc_min_length=args.doc_min_length,
                    strip_mode=args.strip_mode, lookback_days=args.lookback_days,
                    label_window_days=args.label_window_days,
                    emit_labels=True, label_mask_mode=args.label_mask_mode)
                vocab_maps = bundle.vocab_maps
                args._domain_cols = [f"features_{i}" for i in range(len(vocab_maps))]
                args._domain_names = ["condition", *extra_domains]
        else:
            with _phase("assemble corpus (cached, emit_labels)"):
                bundle = load_or_build_case_finding_bundle(
                    spark, cache_uri=args.cache_uri,
                    cdr=args.cdr, billing=args.billing, source_table=args.source_table,
                    person_mod=args.person_mod, vocab_size=args.vocab_size,
                    min_df=args.min_df, min_patient_count=args.min_patient_count,
                    doc_min_length=args.doc_min_length, prior_obs_days=args.prior_obs_days,
                    window_days=args.window_days, disease=args.disease, min_n=args.min_n,
                    holdout_frac=args.holdout_frac, n_bg=args.n_bg, tpn=args.tpn,
                    strip_mode=args.strip_mode, window_mode=args.window_mode,
                    lookback_days=args.lookback_days,
                    label_window_days=args.label_window_days,
                    emit_labels=True, label_mask_mode=args.label_mask_mode)
                vocab_maps = [bundle.vocab_map]
                args._domain_cols = None
                args._domain_names = ["condition"]
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)

        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)
        C = len(bundle.int2cid)               # label heads = engine nodes incl. root
        # Stash on args so the estimator builder can reach them without re-threading.
        args._C = C
        args._parent_int = bundle.parent_int
        args._recall_targets = [float(x) for x in args.recall_targets.split(",") if x]
        args._fdr_targets = [float(x) for x in args.fdr_targets.split(",") if x]
        v_desc = " + ".join(f"{n}:{len(vm)}"
                            for n, vm in zip(args._domain_names, vocab_maps))
        print(f"[driver]   corpus: V=({v_desc}) vocab, "
              f"K={lay.K} gated topics ({args.n_bg} bg + {len(lay.nodes)} nodes x "
              f"{args.tpn} tpn), C={C} label heads", flush=True)

        # Parallelism: the cached bundle parquet has few partitions (~8 part-files),
        # and with dynamic allocation Spark sizes the executor pool to PENDING TASKS
        # (n_partitions / executor.cores) — so 8 partitions pins the job to ~2
        # executors while the rest of the cluster sits idle, and the heavy per-doc
        # differentiable-CAVI autograd serializes onto 8 slots. Repartition UP so the
        # fit demands (and spreads across) more executors. Set --num-partitions ≈ the
        # cluster's total executor cores (a bit over is fine; too many = task
        # overhead). Pair with CHARM_SPARK_CONF='spark.locality.wait=0s' so
        # executors added mid-job aren't starved by cache-locality. 0 = leave as-is.
        if args.num_partitions and args.num_partitions > 0:
            before = bundle.train_df.rdd.getNumPartitions()
            bundle.train_df = bundle.train_df.repartition(args.num_partitions).cache()
            bundle.test_df = bundle.test_df.repartition(args.num_partitions).cache()
            bundle.train_df.count(); bundle.test_df.count()   # materialize the spread
            print(f"[driver]   repartitioned corpus {before} -> "
                  f"{args.num_partitions} partitions (train+test cached)", flush=True)

        rt, ft = args._recall_targets, args._fdr_targets
        results = {}

        def _score(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te):
            return score_arm(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C,
                             recall_targets=rt, fdr_targets=ft,
                             min_count=args.min_label_count)

        def _score_full(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te):
            """(readout, per-node test proba) — proba reused for the conditional
            'sharpening' readout so the per-node LR is fit once per arm."""
            from analysis.pc.evaluate import _lr_proba_per_label_masked
            proba = _lr_proba_per_label_masked(Pi_tr, y_tr, m_tr, Pi_te, C)
            readout = readout_from_proba(
                proba, y_te, m_te, C, recall_targets=rt, fdr_targets=ft,
                min_count=args.min_label_count)
            return readout, proba

        def _conditional(proba_te, y_te, m_te, label):
            # Mask-INDEPENDENT eval: `label` (y) is already closure-membership
            # regardless of the training label_mask_mode, so score against an
            # all-ones observation mask — the cohort/negative sets are then identical
            # across full- and closure-mask runs (fixes exp 0079 Trap 3). m_te (the
            # training mask) is intentionally NOT used here.
            cond = conditional_readout(proba_te, y_te, np.ones_like(y_te),
                                       bundle.parent_int, C,
                                       min_count=args.min_label_count)
            print(format_conditional_readout(
                cond, bundle.int2cid, bundle.name_by_id).replace(
                    "[conditional sharpening]", f"[conditional sharpening: {label}]"),
                flush=True)
            return cond

        with _phase(f"gated_pc fit (weightY={args.weight_y}, K={lay.K})"):
            pc_est = _build_pc_estimator(args, weight_y=args.weight_y, gated=True)
            if args.eval_every > 0:
                pc_est.setOnIteration(_make_eval_logger(bundle, C, args))
            pc_model = pc_est.fit(bundle.train_df)
            # Transform each split ONCE (each transform re-runs CAVI over the split);
            # the supervised transform appends BOTH topicDistribution and probability.
            train_scored = pc_model.transform(bundle.train_df).cache()
            test_scored = pc_model.transform(bundle.test_df).cache()
            Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(train_scored, C)
            Pi_te, y_te, m_te, _ = _collect_theta_labels(test_scored, C)
            results["gated_pc"], proba_gp = _score_full(
                Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te)
            print(format_arm_readout("gated_pc (pc_topics_lr)", results["gated_pc"]),
                  flush=True)
            # Conditional 'sharpening' readout: P(child | parent-cohort) by DAG depth.
            results["gated_pc_conditional"] = _conditional(
                proba_gp, y_te, m_te, "gated_pc")
            # Post-hoc ISOTONIC calibration of the head-independent proba → calibrated
            # conditional posteriors for VOI, sidestepping the co-fit head (exp 0080:
            # Firth couldn't tame the closure-mask head). Isotonic is monotone so
            # AUC/top-1 are unchanged; only the reliability (ECE) moves.
            from analysis.pc.evaluate import _lr_proba_per_label_masked
            proba_gp_tr = _lr_proba_per_label_masked(Pi_tr, y_tr, m_tr, Pi_tr, C)
            proba_gp_cal = calibrate_per_node(proba_gp_tr, y_tr, m_tr, proba_gp, C)
            cond_cal = conditional_readout(proba_gp_cal, y_te, np.ones_like(y_te),
                                           bundle.parent_int, C,
                                           min_count=args.min_label_count)
            results["gated_pc_conditional_cal"] = cond_cal
            print(f"[driver]   conditional ECE (VOI readiness): raw="
                  f"{_f(results['gated_pc_conditional'].get('ece')).strip()} -> "
                  f"isotonic-calibrated={_f(cond_cal.get('ece')).strip()}", flush=True)
            # co-fit head's own per-node P(node) readout (secondary), from the SAME
            # scored test frame (no second CAVI pass).
            hp, hy, hm = _collect_head_proba(test_scored, C)
            results["gated_pc_head"] = readout_from_proba(
                hp, hy, hm, C, recall_targets=rt, fdr_targets=ft,
                min_count=args.min_label_count)
            print(format_arm_readout("gated_pc (co-fit head)",
                                     results["gated_pc_head"]), flush=True)
            # Conditional readout on the CO-FIT HEAD proba too — this is where Firth
            # matters: the head's calibrated P(child|parent) (its ECE) is the VOI
            # prerequisite, and the head-independent pc_topics_lr metric can't show it.
            results["gated_pc_head_conditional"] = _conditional(
                hp, hy, hm, "gated_pc co-fit head")
            print(f"[driver]   co-fit head |w_CK|max="
                  f"{float(np.abs(pc_model.headWeights()).max()):.4g} "
                  f"(head_penalty={args.head_penalty})", flush=True)
            train_scored.unpersist(); test_scored.unpersist()

        if not args.skip_unsup_gated:
            n_it = (args.baseline_max_iter if args.baseline_max_iter and
                    args.baseline_max_iter > 0 else args.max_iter)
            with _phase(f"unsup_gated fit (weightY=0, {n_it} iters)"):
                us_args = argparse.Namespace(**vars(args))
                us_args.max_iter = int(n_it)
                us_est = _build_pc_estimator(us_args, weight_y=0.0, gated=True)
                us_model = us_est.fit(bundle.train_df)
                Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(
                    us_model.transform(bundle.train_df), C)
                Pi_te, y_te, m_te, _ = _collect_theta_labels(
                    us_model.transform(bundle.test_df), C)
                results["unsup_gated"], proba_us = _score_full(
                    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te)
                print(format_arm_readout("unsup_gated (pc_topics_lr)",
                                         results["unsup_gated"]), flush=True)
                # Conditional A/B: does supervision sharpen P(child|parent) vs the
                # unsupervised twin? (The metric the clinician workflow cares about.)
                results["unsup_gated_conditional"] = _conditional(
                    proba_us, y_te, m_te, "unsup_gated")
                # Per-node domain mass on the UNSUPERVISED λ too, so the A/B tells us
                # whether the hierarchy-aligned specialization is a PC effect or a
                # property of the gated multi-domain representation itself (0078).
                us_lam = us_model.result.global_params["lambda"]
                if extra_domains and isinstance(us_lam, dict):
                    us_mass = per_node_domain_mass(us_lam, lay, args._domain_names)
                    print(format_per_node_domain_mass(
                        us_mass, args._domain_names, bundle.int2cid,
                        bundle.name_by_id).replace(
                            "[per-node domain λ-mass]",
                            "[per-node domain λ-mass: unsup_gated]"), flush=True)

        if args.with_dag_head:
            with _phase(f"dag_head fit (ungated + DAG-closure head, K={args.k})"):
                cp = dag_closure_parents(bundle.parent_int, C)
                dh_est = _build_pc_estimator(
                    args, weight_y=args.weight_y, gated=False, closure_parents=cp)
                dh_model = dh_est.fit(bundle.train_df)
                Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(
                    dh_model.transform(bundle.train_df), C)
                Pi_te, y_te, m_te, _ = _collect_theta_labels(
                    dh_model.transform(bundle.test_df), C)
                results["dag_head"] = _score(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te)
                print(format_arm_readout("dag_head (pc_topics_lr)",
                                         results["dag_head"]), flush=True)

        # Headline: does supervision improve the case-finding representation? Report
        # the pc_topics_lr delta on BOTH the ranking (AUC/AP) and the operating point
        # that matters for a rare-disease surface (detection AP + node-macro P@R0.9).
        _print_headline(results)

        gp = pc_model.result.global_params
        # Multi-domain thesis readout: per-node per-domain λ mass. Does each disease
        # node's topic block specialize toward its predictive domain?
        domain_mass = None
        if extra_domains and isinstance(gp["lambda"], dict):
            domain_mass = per_node_domain_mass(gp["lambda"], lay, args._domain_names)
            print(format_per_node_domain_mass(
                domain_mass, args._domain_names, bundle.int2cid, bundle.name_by_id),
                flush=True)

        with _phase("save"):
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            # Multi-domain: λ is a per-domain dict — save one array per domain
            # (lambda_0, lambda_1, ...) since np.savez cannot store a dict; single
            # domain saves the one `lambda` array as before.
            lam = gp["lambda"]
            if isinstance(lam, dict):
                lam_arrays = {f"lambda_{m}": lam[m] for m in sorted(lam)}
            else:
                lam_arrays = {"lambda": lam}
            np.savez(out / "gated_pc_result.npz",
                     **lam_arrays, alpha=gp["alpha"], w_CK=gp["w_CK"])
            manifest = {
                "model_class": "gated_pc",
                "disease": args.disease, "min_n": args.min_n,
                "strip_mode": args.strip_mode, "label_mask_mode": args.label_mask_mode,
                "window_mode": args.window_mode, "lookback_days": args.lookback_days,
                "label_window_days": args.label_window_days,
                "n_bg": args.n_bg, "tpn": args.tpn, "K": lay.K, "C": C,
                "weight_y": args.weight_y, "head_optimizer": args.head_optimizer,
                "head_l2": args.head_l2, "head_lr": args.head_lr,
                "weight_y_warmup_iters": args.weight_y_warmup_iters,
                "grad_cavi_iters": args.grad_cavi_iters, "topic_trust": args.topic_trust,
                "subsampling_rate": args.subsampling_rate, "tau0": args.tau0,
                "kappa": args.kappa, "max_iter": args.max_iter,
                "min_label_count": args.min_label_count,
                "recall_targets": args._recall_targets,
                "fdr_targets": args._fdr_targets,
                "with_dag_head": args.with_dag_head,
                "skip_unsup_gated": args.skip_unsup_gated,
                "extra_domains": list(extra_domains),
                "domain_names": args._domain_names,
                "domain_vocab_sizes": [len(vm) for vm in vocab_maps],
                "per_node_domain_mass": (
                    {str(k): v for k, v in domain_mass.items()} if domain_mass else None),
                "results": results, "ledger": bundle.ledger,
                # Corpus params — ALL of the bundle cache-key inputs, so a post-hoc
                # gated_pc_readout can recompute the exact key + reload the bundle
                # (doc_min_length + emit_labels are required by the key; recording
                # them here removes the lr_readout-style fragility).
                "corpus_manifest": {
                    "cdr": args.cdr, "source_table": args.source_table,
                    "person_mod": args.person_mod, "vocab_size": args.vocab_size,
                    "min_df": args.min_df, "min_patient_count": args.min_patient_count,
                    "doc_min_length": args.doc_min_length,
                    "prior_obs_days": args.prior_obs_days, "window_days": args.window_days,
                    "holdout_frac": args.holdout_frac, "emit_labels": True,
                    "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
                    "name_by_id": {str(c): n for c, n in bundle.name_by_id.items()}},
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"[driver]   saved gated_pc result to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
