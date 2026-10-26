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

That readout is fit either on the DRIVER (collect theta + label/mask, C sklearn
LRs) or DISTRIBUTED (one batched L-BFGS over all C heads on the executors, then a
lean float32/uint8 test-split collect) — see `--readout-mode` and
`resolve_readout_mode`. The metrics stack is the same object either way; only
where the per-node logistic is fit changes.

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
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session
from disk_telemetry import start_disk_telemetry
# TOP-LEVEL module name on purpose: the distributed readout's partition kernels
# reach the executors inside `mapPartitions` closures, which cloudpickle serializes
# by MODULE REFERENCE — so the name the driver imports must be the name executors
# can import. `--py-files .../distributed_readout.py` (run_experiment.py) publishes
# it as a top-level module, and spark-submit puts the driver script's own directory
# on sys.path, so `distributed_readout` resolves to the same module on both sides.
# (`analysis.pc.batched_lr` below is DRIVER-ONLY: its fold helpers are injected into
# SparkStatsFn and called driver-side around the treeAggregate, never pickled.)
import distributed_readout as _dr


def _cprint(msg, **kw) -> None:
    """print() through term_colors.highlight_line (ERROR/WARN/HIT/MISS/
    rebuild/macro-AUC/eta-boost -- see term_colors.py). Identity when color
    is disabled or nothing matches, so this is safe on every "[driver]"/
    "[episode]"/"[mondo]" status line in this module, not just the ones that
    happen to light up.

    LOCAL import, not module-level: this module is reachable transitively
    from `gated_pc_readout.py`'s top-level import (itself imported at module
    level by `hpoa_stage2_probe.py`, which rides Spark's `--py-files` for its
    own mapPartitions kernel) -- term_colors must never become a top-level
    dependency an executor could be asked to resolve. See term_colors.py's
    module docstring."""
    from term_colors import highlight_line
    print(highlight_line(msg), **kw)




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


def detection_readout(proba_DC, y_DC, recall_targets, *, doc_keys=None):
    """Case-vs-background detection: pool a per-UNIT case SCORE = max over disease-node
    probabilities and the foreground indicator = the root node's label (label[:,0]=1
    iff the unit has any attested disease node). Reports AUC/AP + precision@recall on
    that pooled signal — 'can we tell a rare-disease patient from a background one,
    and how precise is the surfaced case list?'. Empty/degenerate -> skipped.

    `doc_keys` (R5.7, spec R5.7): pass the collect's int64 DOC KEY order
    (`_doc_key_column` / `_collect_theta_labels`'s / `_collect_lean_proba`'s fourth
    return) and the pool is DEDUPED TO PERSONS before scoring — `None` (the default)
    keeps the exact pre-multi-doc per-document pool, byte-identical, which is what
    every existing single-doc-per-person fixture and recorded run (0104/0109/0110)
    is compared on. The semantic being preserved is the MAX-pooling rule this
    function already applies across disease nodes for one document: extend it one
    grain further and take the max over EVERY (document, node) cell a person owns,
    with the foreground indicator OR'd (max) the same way — a person who attested
    the disease in any one of their <=3 docs is a case. Anything else (e.g. a
    doc-count-weighted average) would let a chronic 3-doc person cast 3 votes in a
    detection rate that is supposed to answer one question per PERSON. On a
    single-doc corpus `person_of(doc_key)` is a bijection onto the existing rows, so
    passing `doc_keys` there is a no-op — proved by
    `tests/scripts/test_multidoc_seams.py::test_detection_readout_single_doc_noop`.

    CONSTANT COLUMNS ARE EXCLUDED FROM THE POOL — an EVAL bug fix, always on.
    A node whose observed train cell is single-class gets a CONSTANT fallback
    column from the readout head (763 of them at whole-Mondo, exp 0104). A
    constant column contributes the same value to every document's max, so as soon
    as one of them sits above the informative columns the per-doc max is constant
    too and the AUC pins at exactly 0.5000 — which is what every 0104 readout
    printed. The column carries no per-document information by construction, so
    dropping it cannot lose signal; it can only stop one from being masked. Note
    this touches the DETECTION pool only: the ranking and per-node metrics score
    each column against its own labels and are deliberately left untouched (a
    degenerate node is already reported there as skipped/degenerate). The
    constant-column filter runs BEFORE the person dedup — "is this node constant
    over the test split" is a corpus/fit question, unaffected by how many
    documents any one person contributes."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if proba_DC.shape[0] == 0 or proba_DC.shape[1] < 2:
        return {"skipped": "empty or single-node"}
    y = np.asarray(y_DC[:, 0], float)                       # root = any-disease
    disease = proba_DC[:, 1:]                               # node 0 is the root
    informative = np.ptp(disease, axis=0) > 0                # non-constant columns
    n_const = int(disease.shape[1] - informative.sum())
    if not informative.any():
        return {"skipped": "all disease-node scores are constant",
                "n_constant_nodes": n_const}
    score = disease[:, informative].max(axis=1)             # strongest disease node
    n_units = int(len(score))
    if doc_keys is not None:
        persons = _dr.person_of(np.asarray(doc_keys))
        # max-pool BOTH arrays onto the person grain in one pass (np.maximum.at
        # is the unbuffered scatter-max sklearn/numpy ship for exactly this —
        # ordinary fancy-index assignment would silently keep only the LAST
        # document written per person instead of the max over all of them).
        uniq_p, inv = np.unique(persons, return_inverse=True)
        p_score = np.full(uniq_p.shape[0], -np.inf)
        p_y = np.zeros(uniq_p.shape[0])
        np.maximum.at(p_score, inv, score)
        np.maximum.at(p_y, inv, y)
        score, y = p_score, p_y
        n_units = int(uniq_p.shape[0])
    if len(np.unique(y)) < 2:
        return {"skipped": "degenerate foreground indicator",
                "n_constant_nodes": n_const}
    return {"skipped": None, "prevalence": float(y.mean()),
            "auc": float(roc_auc_score(y, score)),
            "ap": float(average_precision_score(y, score)),
            "n_constant_nodes": n_const,
            "grain": "person" if doc_keys is not None else "document",
            "n_units": n_units,
            "par": precision_at_recall(y, score, recall_targets)}


def readout_from_proba(proba, y_te, m_te, C, *, recall_targets, fdr_targets,
                       min_count=0, skip_constant=False, doc_keys=None):
    """Full readout from an already-computed (N_te, C) per-node probability:
    ranking (AUC/AP) + per-node precision@recall / recall@FDR + case-vs-background
    detection. Shared by the pc_topics_lr arm (proba = post-hoc LR on theta) and the
    co-fit head arm (proba = sigmoid(w_CK·theta)).

    `skip_constant` (R2.1) forwards the RANKING axis's constant-column guard to
    `_score_label`. It defaults OFF so every prevalent arm reproduces its recorded
    numbers byte for byte (0104/0109/0110 are compared on them), and the INCIDENT
    arm turns it on — that is the arm where a train-degenerate node's constant
    column stops being all-positive and starts scoring an exact 0.5.

    `doc_keys` (R5.7) forwards ONLY to `detection_readout`. The ranking and
    per-node PR axes are per-NODE metrics (spec R5.7 is silent on them; only the
    detection pool's collapse of ALL nodes into one max-over-everything score
    turns a multi-doc person's several rows into several votes toward the SAME
    pooled AUC point). `None` (the default) is the pre-multi-doc per-document
    pool, byte-identical to every recorded run."""
    from analysis.pc.evaluate import _bundle_masked
    ranking = _bundle_masked(proba, y_te, m_te, C, min_count,
                             skip_constant=skip_constant)
    pr = pr_readout(proba, y_te, m_te, C, recall_targets, fdr_targets, min_count)
    det = detection_readout(proba, y_te, recall_targets, doc_keys=doc_keys)
    # Per-node (auc, n_pos) kept alongside the macro so the HEADLINE can split the
    # gated_pc-vs-unsup delta by node RARITY (insight 0066: PC's promised edge is the
    # hidden low-mass tail; a flat macro can hide it). auc from the ranking per_label
    # (skipped=None == scored), n_pos (test positives) from the PR per_node.
    per_node = {}
    for c in range(C):
        rl = ranking["per_label"].get(c, {})
        if rl.get("skipped") is None and rl.get("auc") is not None:
            per_node[c] = {"auc": float(rl["auc"]),
                           "ap": (None if rl.get("ap") is None else float(rl["ap"])),
                           "n_pos": int(pr["per_node"].get(c, {}).get("n_pos", 0)),
                           # top-fraction screening metrics (evaluate._score_label),
                           # so the headline can rank nodes by lift, not just AUC.
                           "prec_at_k": rl.get("prec_at_k"),
                           "recall_at_k": rl.get("recall_at_k"),
                           "lift_at_k": rl.get("lift_at_k"),
                           "topk_frac": rl.get("topk_frac")}
    return {"ranking": ranking["macro"], "pr": pr["macro"], "detection": det,
            "per_node": per_node}












# --------------------------------------------------------------------------- #
# STACKED (closure-product) arm — HSLDA's output-side hierarchy on the readout #
# heads (spec 2026-10-06 stacked-closure-readout, Part A).                     #
# --------------------------------------------------------------------------- #
# The per-node readout head is already HSLDA's conditional: under
# `label_mask_mode=closure` node c's head is fit on the rows inside its parent's
# closure with siblings as negatives, so σ(z_c) ≈ P(c | parent(c)). What the
# readout never formed is the PRODUCT down the closure,
#
#     log P_stack(c | d) = Σ_{a ∈ closure(c)} log σ(z_a(d)),
#
# closure = c plus every ancestor including the root, each counted once (the
# `DagClosureHead` convention, diamond-safe). Two consequences the flat arm does
# not have: P(child) ≤ P(parent) by construction, and the root's factor carries
# the population prior down the tree — PROVIDED the root head saw the
# background. Under the closure mask it did not (the root is observed only on
# foreground rows, all positive, so its head is the degenerate constant 1.0);
# `gated_pc_readout.observe_root_everywhere` is what makes the root head a real
# case-vs-background logistic for this arm, and `root_only_detection` below is
# that single head read on its own, so the stacked detection can be told apart
# from "a detection head would have done it".
STACKED_NAMING = ("closure-PRODUCT of the per-node readout heads (HSLDA's label-"
                  "side gating on post-hoc heads): P_stack(c) = Π_{a ∈ closure(c)} "
                  "σ(z_a), root head fit on EVERY train row (case vs background); "
                  "the heads of nodes ≥ 1 are the prevalent arm's own")


def closure_matrix(parent_int, C):
    """(C, C) float64 `M[c, a] = 1` iff `a ∈ closure(c)` — c itself plus every
    ancestor, each ONCE under diamonds, root included.

    Same construction as `DagClosureHead.__init__` (memoized DFS over the
    direct-parent lists), kept here so the readout's stacked arm and the co-fit
    closure head agree on what "closure" means without importing the engine.
    `parent_int` is the bundle's `{child: [parents]}` over engine ids."""
    C = int(C)
    parents = dag_closure_parents(parent_int, C)
    clo = [None] * C

    def _closure(node):
        if clo[node] is None:
            acc = {node}
            for p in parents[node]:
                if 0 <= p < C and p != node:
                    acc |= _closure(p)
            clo[node] = acc
        return clo[node]

    M = np.zeros((C, C), dtype=np.float64)
    for c in range(C):
        for a in _closure(c):
            M[c, a] = 1.0
    return M


def stacked_proba(proba, M, *, chunk_rows=8192, floor=1e-12):
    """`P_stack = exp(log(proba) @ M.T)` — the closure product per (doc, node).

    Row-chunked so the (D, C) log matrix never exists in full beside the float32
    input (at whole-Mondo, D_te×C float64 is ~2 GB). `floor` clips the input
    probabilities before the log: a degenerate-negative node's constant fallback
    is exactly 0.0, and log(0) would turn every descendant's product into 0
    (a constant column) through a single ancestor the solver refused to fit.
    Returns float32 (the lean collect's own dtype)."""
    proba = np.asarray(proba)
    D, C = proba.shape
    MT = np.asarray(M, dtype=np.float64).T
    out = np.empty((D, C), dtype=np.float32)
    for s in range(0, D, int(chunk_rows)):
        blk = np.log(np.clip(proba[s:s + chunk_rows].astype(np.float64),
                             floor, 1.0))
        out[s:s + chunk_rows] = np.exp(blk @ MT)
    return out


def _paired_per_node(base_per_node, new_per_node, depth):
    """Paired per-node AUC delta (new − base) over the nodes both arms scored,
    pooled and by depth: median, wins/losses/ties, n. Counts of nodes only —
    nothing per-node leaves this function."""
    shared = sorted(set(base_per_node) & set(new_per_node))
    deltas = {c: float(new_per_node[c]["auc"]) - float(base_per_node[c]["auc"])
              for c in shared}

    def _summ(cs):
        d = np.asarray([deltas[c] for c in cs], dtype=np.float64)
        if d.size == 0:
            return {"n": 0, "median": None, "mean": None,
                    "wins": 0, "losses": 0, "ties": 0}
        return {"n": int(d.size), "median": float(np.median(d)),
                "mean": float(d.mean()),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                "ties": int((d == 0).sum())}

    by_depth = {}
    for c in shared:
        by_depth.setdefault(int(depth.get(c) if depth.get(c) is not None else -1),
                            []).append(c)
    return {"all": _summ(shared),
            "by_depth": {str(k): _summ(v) for k, v in sorted(by_depth.items())}}


def _marginal_ece_by_depth(proba, stacked, y, scored_nodes, depth, n_bins=10):
    """Pooled MARGINAL calibration (ECE over ALL test docs, not the parent cohort)
    per depth, for both arms. The flat head's σ(z_c) is a P(c | parent) and is
    expected to be badly calibrated as a marginal (it never saw the background);
    the product is the arm that claims to be a marginal P(c | d). Depth 0 is the
    root head itself — the detection head's own calibration."""
    groups = {}
    for c in scored_nodes:
        groups.setdefault(int(depth.get(c) if depth.get(c) is not None else -1),
                          []).append(int(c))
    out = {}
    for d, cs in sorted(groups.items()):
        yy = np.asarray(y[:, cs], dtype=np.float64).ravel()
        out[str(d)] = {"n_nodes": len(cs),
                       "flat": _ece(yy, np.asarray(proba[:, cs], dtype=np.float64)
                                    .ravel(), n_bins),
                       "stacked": _ece(yy, np.asarray(stacked[:, cs],
                                                      dtype=np.float64).ravel(),
                                       n_bins)}
    return out


def stacked_readout(proba, y_te, m_te, parent_int, C, *, recall_targets,
                    fdr_targets, min_count=0, prevalent=None, doc_keys=None,
                    arm_label="gated_pc"):
    """The `gated_pc_stacked` results block: `readout_from_proba` on the closure
    product of the SAME per-doc probabilities the prevalent arm scored, plus the
    reads that make the product interpretable next to it:

      * `readout` — ranking / PR / detection / per_node on `P_stack`, the same
        mask and the same person-grain detection pool as the prevalent arm;
      * `root_only_detection` — `detection_readout` on the root head alone
        (column 0, which under the stacked arm is a case-vs-background logistic):
        if the stacked detection does not beat this, the product adds nothing a
        single detection head would not;
      * `paired_vs_prevalent` — per-node AUC delta (stacked − flat) over the
        shared scored nodes, pooled and by depth (counts of nodes only);
      * `marginal_ece_by_depth` — pooled marginal ECE of both arms by depth.

    Pure numpy + sklearn, off-Spark testable. `proba` is the lean collect's
    (D, C) float32; `parent_int` the bundle's engine-id parent map."""
    C = int(C)
    M = closure_matrix(parent_int, C)
    P = stacked_proba(proba, M)
    readout = readout_from_proba(
        P, y_te, m_te, C, recall_targets=recall_targets, fdr_targets=fdr_targets,
        min_count=min_count, doc_keys=doc_keys)
    root_col = np.asarray(proba[:, 0], dtype=np.float32)
    root_det = detection_readout(np.stack([root_col, root_col], axis=1), y_te,
                                 recall_targets, doc_keys=doc_keys)
    _, depth = _dag_children_and_depth(parent_int, C)
    n_terms = M.sum(axis=1)
    # HSLDA's own read — per-node AUC over ALL test docs (y_c vs everyone, no
    # cohort): the de-novo case-finding question the within-cohort ranking never
    # asks. The flat head is a P(c | parent) and is expected to collapse here (it
    # never saw the background); the product is the arm that claims a marginal.
    # Same scorer, all-ones mask, both arms, paired. Read next to the pooled
    # detection, which on a branch with ONE degenerate depth-1 node is exactly
    # the root head (P_stack(depth-1) = sigma(z_0) * 1) and says nothing per node.
    from analysis.pc.evaluate import _bundle_masked
    ones = np.ones_like(np.asarray(y_te), dtype=np.uint8)
    marg_flat = _bundle_masked(proba, y_te, ones, C, min_count)
    marg_stk = _bundle_masked(P, y_te, ones, C, min_count)

    def _scored(bm):
        return {c: r for c, r in bm["per_label"].items()
                if r.get("skipped") is None and r.get("auc") is not None}

    marginal = {
        "definition": "per-node AUC of P(c|d) over ALL test docs (all-ones mask: "
                      "y_c vs everyone), the de-novo read; same scorer and "
                      "min_count as the within-cohort ranking",
        "flat": {"macro": marg_flat["macro"], "per_node": _scored(marg_flat)},
        "stacked": {"macro": marg_stk["macro"], "per_node": _scored(marg_stk)},
    }
    marginal["paired"] = _paired_per_node(marginal["flat"]["per_node"],
                                          marginal["stacked"]["per_node"], depth)
    block = {
        "naming": STACKED_NAMING,
        "definition": "log P_stack(c|d) = sum_{a in closure(c)} log sigma(z_a(d)); "
                      "closure = c + all ancestors incl. root, each once",
        "arm_label": arm_label,
        "min_count": int(min_count),
        "closure_terms": {"mean": float(n_terms.mean()), "max": int(n_terms.max())},
        "readout": readout,
        "marginal_ranking": marginal,
        "root_only_detection": root_det,
        "marginal_ece_by_depth": _marginal_ece_by_depth(
            proba, P, y_te, sorted(set(readout["per_node"]) | {0}), depth),
    }
    if prevalent is not None:
        block["paired_vs_prevalent"] = _paired_per_node(
            prevalent["per_node"], readout["per_node"], depth)
        block["prevalent_detection"] = prevalent.get("detection")
    return block


def format_stacked_readout(block) -> str:
    """The stacked block as driver log lines: the three detection numbers side by
    side (flat max-over-nodes / root head alone / stacked max-over-nodes), the
    ranking macro, the paired per-node delta by depth, and the marginal ECE."""
    r = block["readout"]
    lines = [f"[stacked readout: {block['arm_label']}]  {block['naming']}",
             f"  {block['definition']}  (closure terms: mean="
             f"{block['closure_terms']['mean']:.2f} max={block['closure_terms']['max']})"]

    def _det(name, d):
        if not d or d.get("skipped"):
            return f"    {name:<28} skipped ({(d or {}).get('skipped')})"
        return (f"    {name:<28} AUC={d['auc']:.4f} AP={d['ap']:.4f} "
                f"prev={d['prevalence']:.3f} ({d.get('n_units')} {d.get('grain')}s)")

    lines.append("  detection (case vs bg), three reads of the SAME test split:")
    lines.append(_det("flat: max_c sigma(z_c)", block.get("prevalent_detection")))
    lines.append(_det("root head alone", block["root_only_detection"]))
    lines.append(_det("stacked: max_c P_stack(c)", r["detection"]))
    rk = r["ranking"]
    lines.append(f"  ranking (within parent cohort, same mask): macro AUC="
                 f"{_f(rk['auc']).strip()} AP={_f(rk['ap']).strip()} "
                 f"(over {rk['n_labels_scored']} nodes)")

    def _med(v):
        return "n/a" if v is None else f"{v:+.4f}"

    pv = block.get("paired_vs_prevalent")
    if pv:
        a = pv["all"]
        lines.append(f"  paired per-node AUC delta (stacked - flat): median="
                     f"{_med(a['median'])} "
                     f"wins/losses/ties={a['wins']}/{a['losses']}/{a['ties']} "
                     f"(n={a['n']})")
        for d, s in pv["by_depth"].items():
            lines.append(f"    depth {d}: n={s['n']} median={_med(s['median'])} "
                         f"wins/losses={s['wins']}/{s['losses']}")
    mg = block.get("marginal_ranking")
    if mg:
        fm, sm = mg["flat"]["macro"], mg["stacked"]["macro"]
        lines.append("  per-node AUC over ALL test docs (de-novo, y_c vs everyone; "
                     "HSLDA's read):")
        lines.append(f"    flat sigma(z_c):     macro AUC={_f(fm['auc']).strip()} "
                     f"AP={_f(fm['ap']).strip()} (over {fm['n_labels_scored']} nodes)")
        lines.append(f"    stacked P_stack(c):  macro AUC={_f(sm['auc']).strip()} "
                     f"AP={_f(sm['ap']).strip()} (over {sm['n_labels_scored']} nodes)")
        a = mg["paired"]["all"]
        lines.append(f"    paired delta (stacked - flat): median={_med(a['median'])} "
                     f"wins/losses/ties={a['wins']}/{a['losses']}/{a['ties']} "
                     f"(n={a['n']})")
        for d, st in mg["paired"]["by_depth"].items():
            lines.append(f"      depth {d}: n={st['n']} median={_med(st['median'])} "
                         f"wins/losses={st['wins']}/{st['losses']}")
    ece = block.get("marginal_ece_by_depth") or {}
    if ece:
        lines.append("  marginal ECE over ALL test docs (flat sigma(z_c) vs "
                     "stacked), by depth [depth 0 = root head]:")
        for d, s in ece.items():
            lines.append(f"    depth {d}: n_nodes={s['n_nodes']} flat="
                         f"{_f(s['flat']).strip()} stacked={_f(s['stacked']).strip()}")
    return "\n".join("[driver] " + ln if not ln.startswith("[") else ln
                     for ln in lines)


def score_arm(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, *, recall_targets,
              fdr_targets, min_count=0, doc_keys=None):
    """Full readout for one arm's theta via the pc_topics_lr proba (a fresh per-node
    LR on the shaped theta). Pure; used inline by the driver and by gated_pc_readout
    on a finished fit. `doc_keys` (R5.7) is the TEST split's collect order (the
    caller's own `_collect_theta_labels` fourth return) — forwarded to
    `readout_from_proba` so the detection pool dedups to persons; `None` keeps the
    pre-multi-doc per-document pool."""
    from analysis.pc.evaluate import _lr_proba_per_label_masked
    proba = _lr_proba_per_label_masked(Pi_tr, y_tr, m_tr, Pi_te, C)
    return readout_from_proba(proba, y_te, m_te, C, recall_targets=recall_targets,
                              fdr_targets=fdr_targets, min_count=min_count,
                              doc_keys=doc_keys)


def format_arm_readout(name, arm):
    """Render one arm's score_arm() result as driver log lines (ranking + PR@recall +
    recall@FDR + detection). The two headline lines (macro AUC=.../detection
    (case vs bg)...) are bold-green (term_colors.highlight_line, local import
    -- see `_cprint`'s docstring) so the run's bottom-line numbers are
    findable in a long log without hunting through node-macro/PR detail."""
    from term_colors import highlight_line
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
        # The excluded-column count is printed rather than silently applied: at
        # whole-Mondo it IS the degenerate-node count, so the line doubles as a
        # read on how much of the label DAG is structurally inert.
        nc = int(det.get("n_constant_nodes", 0) or 0)
        const = f"  [{nc} constant node col(s) excluded]" if nc else ""
        # R5.7: the grain is printed, not just applied, so a multi-doc run's log
        # states on its face whether "detection prevalence" means "of documents"
        # or "of persons" — the two disagree by construction under multi-doc.
        grain = det.get("grain", "document")
        n_units = det.get("n_units")
        units = f" ({n_units} {grain}s)" if n_units is not None else f" ({grain}-grain)"
        lines.append(f"{name}: detection (case vs bg){units} AUC={det['auc']:.4f} "
                     f"AP={det['ap']:.4f} prev={det['prevalence']:.3f}  {dpar}{const}")
    return "\n".join(highlight_line("[driver]   " + ln) for ln in lines)


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










def _f(v):
    """Format an optional float for the readout tables (n/a for None)."""
    return "  n/a " if v is None else f"{v:.4f}"




def _dump_partial_results(out, results, name="results_partial.json"):
    """Atomically rewrite `<out>/<name>` (default results_partial.json) with the
    arms so far.

    Called after each arm's readout lands so a multi-hour run that dies (or a
    cluster that times out) before the final manifest.json write still leaves a
    machine-readable record of every COMPLETED arm — manifest.json is written
    once, at the very end, and is lost with everything after it otherwise.
    Write-to-temp + rename so a mid-write death never leaves a torn file.

    `name` exists so the standalone re-readout (gated_pc_readout, which writes
    into a FINISHED run's dir) lands in its own `results_readout.json` instead of
    clobbering the record the fit itself left behind."""
    tmp = out / (name + ".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out / name)


def _save_fit(out, gp, C, manifest_fields, *, results=None, domain_mass=None,
              partial=None):
    """Write `<out>/gated_pc_result.npz` + `manifest.json`. Called TWICE per run.

    **Why twice.** The fit is the expensive, unrepeatable half — hours of CAVI at
    whole-Mondo K — and the readout that follows it is the FRAGILE half: it is
    where the driver collects, the second batched solve and the ECE diagnostic
    live, and where the 0104 smokes died repeatedly. Writing the model only at the
    very end meant every readout death also threw away the fit, so recovery meant
    re-running the fit instead of re-running `gated_pc_readout` against it. So the
    first call lands the npz plus a manifest carrying everything the fit itself
    determines (`results=None`, `partial="fit-only"`) the moment the fit
    completes, and the final call OVERWRITES both at the same paths with the full
    record. The final write stays authoritative; the early one is a floor.

    `partial` is the marker a reader checks before trusting `results`:
    `"fit-only"` means the arms had not been scored yet. It is set to `None` by
    the final write, so a finished run's manifest is exactly what it always was
    apart from that one explicit null.

    `gp` is `pc_model.result.global_params`. Multi-domain λ is a per-domain dict,
    which `np.savez` cannot store — it goes out as `lambda_0, lambda_1, ...`, as
    it always has; a single-domain run saves the one `lambda` array.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lam = gp["lambda"]
    if isinstance(lam, dict):
        lam_arrays = {f"lambda_{m}": lam[m] for m in sorted(lam)}
    else:
        lam_arrays = {"lambda": lam}
    # Atomic writes (tmp + os.replace). _save_fit is called mid-fit as a periodic
    # checkpoint (--fit-save-interval) with the fit still running, so a crash DURING
    # a write must not tear the npz/manifest that is the crash insurance itself —
    # the same rule the readout checkpoint (_write_readout_checkpoint) follows.
    # os.replace is atomic within a filesystem; the run dir is the driver's local
    # workspace mount. np.savez gets a FILE OBJECT so it does not append '.npz' to
    # the tmp name (the same gotcha noted at the readout-checkpoint site).
    npz_path = out / "gated_pc_result.npz"
    npz_tmp = out / "gated_pc_result.npz.tmp"
    with open(npz_tmp, "wb") as _f:
        np.savez(_f, **lam_arrays, alpha=gp["alpha"], w_CK=gp["w_CK"],
                 b_CK=np.asarray(gp.get("b_CK", np.zeros(C)), dtype=np.float64))
    os.replace(npz_tmp, npz_path)
    manifest = dict(manifest_fields)
    manifest["per_node_domain_mass"] = (
        {str(k): v for k, v in domain_mass.items()} if domain_mass else None)
    manifest["results"] = results
    manifest["partial"] = partial
    man_tmp = out / "manifest.json.tmp"
    man_tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(man_tmp, out / "manifest.json")
    return manifest


def dag_closure_parents(parent_int, C):
    """Length-C list of parent-LABEL-index lists for the ungated DAG-closure head.

    closure_parents[c] lists node c's direct parents in the [0, C) engine-id label
    space (root and any parentless node -> []); this is exactly `parent_int` (root
    omitted) densified over range(C). Feeds OnlinePCLDAEstimator.setClosureParents."""
    return [[int(p) for p in parent_int.get(c, [])] for c in range(C)]


# --------------------------------------------------------------------------- #
# The int64 doc-key seam, driver side (exp 0111 WP-A1, spec R5.4).             #
# --------------------------------------------------------------------------- #
# Every readout collect below identifies a row by an int64 DOC KEY rather than
# by the raw `person_id`/`doc_id`. The synthesis and its inverse live once, in
# `distributed_readout` (bare-importable on executors, already on every submit's
# `--py-files`); this file imports them (`_dr.synthesize_doc_key`, `_dr.person_
# of`, radix/guard constants) so there is exactly one definition of the map.
# `_doc_key_column` builds the key as a Spark Column at the collect's own select;
# `_assert_unique_doc_keys` is the corpus-level tripwire the driver runs once the
# keys have been materialized (see below).
def _doc_key_column(df, person_col="person_id"):
    """`doc_key = person_id * RADIX + episode_no` as a guarded Spark Column.

    The DRIVER-side half of the doc-key seam: the readout collects add this
    column and select it as the row id, so `_lean_eval_kernel` / the A/B
    alignment / the calibration split all speak one int64 that yields the person
    back through `_dr.person_of` (spec R5.4).

    `episode_no` is the bounded WITHIN-CORPUS document index. Today's corpora are
    single-doc — one document per person, no `episode_no` column — so it defaults
    to literal 0 and `doc_key = person_id * 64`, i.e. the current int64 person
    ids scaled order-preservingly (nothing about the byte-identical single-doc
    path moves). When WP-D2 wires the episode corpus it must materialize a
    bounded per-person document index (0 .. cap-1) under this name — NOT WP-D1's
    unbounded chronological `episode_no` (`episode_index.py`), which would carry
    past the radix and collide; see the seam comment in `distributed_readout`.

    The `[0, 2**57)` person-id and `[0, RADIX)` episode-no guards are folded into
    the column with `raise_error`, so they cost nothing extra (they evaluate in
    the collect's own scan) and a violation surfaces AT the row that carries it,
    before the silent int64 overflow / block carry it would otherwise become.
    """
    from pyspark.sql import functions as F

    person = F.col(person_col).cast("long")
    if "episode_no" in df.columns:
        ep = F.col("episode_no").cast("long")
    else:
        ep = F.lit(0).cast("long")
    radix = F.lit(int(_dr.DOC_KEY_RADIX))
    return (
        F.when(person.isNull() | (person < 0)
               | (person >= F.lit(int(_dr.DOC_KEY_MAX_PERSON_ID))),
               F.raise_error(F.concat(
                   F.lit("doc_key: person_id out of [0, 2**57): "),
                   F.coalesce(person.cast("string"), F.lit("null")))))
        .when((ep < 0) | (ep >= radix),
              F.raise_error(F.concat(
                  F.lit("doc_key: episode_no out of [0, "
                        f"{int(_dr.DOC_KEY_RADIX)}) — this is the bounded "
                        "within-corpus document index, not WP-D1's unbounded "
                        "chronological ordinal: "),
                  ep.cast("string"))))
        .otherwise(person * radix + ep)
        .cast("long")
        .alias("doc_key"))


def _assert_unique_doc_keys(ids, *, where):
    """Corpus-level uniqueness tripwire on the materialized doc keys (spec R5.4).

    Run where the keys are FIRST materialized on the driver as a full array — the
    tail of each readout collect (`_collect_theta_labels`, `_densify_lean_blocks`)
    — because that is the cheapest honest point: the ids are already in hand, so
    the check is one `set` build, no extra Spark pass. A duplicate here means two
    documents share a key, which is exactly the failure the radix is chosen to
    prevent (a person carrying > RADIX documents, or WP-D1's raw ordinal leaking
    into the low bits): the A/B alignment dict would silently overwrite one with
    the other (the seam-6 bug, spec R5.5) and the per-doc eval arrays would be
    indexed ambiguously. Better a named raise than a wrong number.
    """
    n = len(ids)
    if n != len(set(int(i) for i in ids)):
        raise ValueError(
            f"doc-key collision in {where}: {n} rows carry "
            f"{len(set(int(i) for i in ids))} distinct doc keys. Two documents "
            "synthesized the same person_id*RADIX+episode_no — check the "
            "per-person document cap (<= RADIX) and that episode_no is the "
            "bounded within-corpus index, not the raw chronological ordinal.")




# --------------------------------------------------------------------------- #
# Spark collectors (cluster-covered; not unit-tested).                        #
# --------------------------------------------------------------------------- #
def _collect_theta_labels(df, C, *, label_col="label", mask_col="labelMask",
                          topic_col="topicDistribution", sample_frac=1.0, seed=0):
    """Collect ONLY the K-dim per-doc theta + the (C,) label/mask arrays to numpy
    (never the dense BOW), so it stays on the driver's memory budget at cohort
    scale. Returns (Pi (D,K), y_DC (D,C), mask_DC (D,C), doc_key_order). Empty df
    -> correctly-shaped zero arrays. Mirrors pc_antidepressant_cloud's
    _collect_topics_labels but with configurable label/mask column names (this
    corpus uses 'label'/'labelMask' from the Step-A adapter).

    This is the DRIVER readout's input and the reason `--readout-mode distributed`
    exists: θ and both (D,C) float64 label/mask arrays are 8*D*(K+2C) bytes per
    split before any proba array is built (see `resolve_readout_mode`).

    The fourth return is the row order as int64 DOC KEYS (`_doc_key_column`), not
    raw person ids (spec R5.4): the A/B gate aligns this driver collect against
    the distributed lean collect on DOCUMENT identity, and any person-grain step
    recovers the person via `_dr.person_of`. On single-doc corpora a doc key is
    `person_id * RADIX`, so the order is the person order rescaled — unchanged in
    meaning.

    `sample_frac` (<1.0) row-subsamples BEFORE the collect — the readout collects
    per-doc theta AND builds (N, C) proba arrays on the driver, both O(N); at
    whole-Mondo K/C over the whole population that is multi-GB, so bounding N here
    bounds the readout's driver footprint (the per-node LR needs only enough rows to
    fit, not all of them)."""
    if sample_frac < 1.0:
        df = df.sample(withReplacement=False, fraction=float(sample_frac),
                       seed=int(seed))
    rows = df.select(_doc_key_column(df), topic_col, label_col, mask_col).collect()
    doc_key_order = [int(r["doc_key"]) for r in rows]
    _assert_unique_doc_keys(doc_key_order, where="_collect_theta_labels")
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
    return Pi, y_DC, mask_DC, doc_key_order




# sklearn's own `LogisticRegression` default tol, and the oracle this readout must
# reproduce stops there too. Below ~3e-6 the gradient inf-norm of a SUMMED loss is
# unreachable (roundoff floor ~1e-16*n), so a tighter gtol only buys stalled nodes
# and wasted distributed passes.
_READOUT_GTOL = 1e-4
# Iteration budget for ONE batched readout solve. At C=437 a cold solve spends all
# 200 of them (~1,200 distributed passes, ~30 min), so this is a real cap, not a
# safety net: the returned point is "wherever 200 iterations got to" and every
# iteration is a full pass over the train split. `--readout-max-iter` exposes it
# because that makes the dev loop affordable (CHARM_DEV caps it at 60, where the
# macro AUC RANKING is already stable — see run_experiment._apply_dev_profile);
# the default keeps every production run exactly where it was.
_READOUT_MAX_ITER = 200
# Fit size (C*K float64 parameter bytes) above which `_fit_readout_heads` pays for
# the θ mass-coverage measurement. 64 MB is C=K≈2,900 — comfortably above every
# cardiovascular-scale run (C=444: 1.6 MB) and comfortably below whole-Mondo
# (C=K≈3,827: 117 MB), which is exactly the population where the top-m lever is a
# live question and one extra cheap pass is noise against a multi-hour solve.
_COVERAGE_MIN_FIT_BYTES = 64 * 1024 * 1024


def write_bundle_meta(run_dir, bundle, name="bundle_meta.json"):
    """`<run>/bundle_meta.json`: the bundle's cache meta (DAG bridge, names,
    vocab maps, ledger — ids and ontology/concept names only, no patient data).
    The HDFS cache dies with the cluster; the run dir does not, so
    `inspect-topics` reads this first and never needs the bundle rebuilt to
    render words, depths or own codes."""
    from _case_finding_cache import _meta_dict
    p = Path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_meta_dict(bundle)))
    return p


def write_code_map(run_dir, pairs, name="code_map.tsv"):
    """`<run>/code_map.tsv`: `std_cid\tnode_cid` rows of the native attestation
    map (WP-C'). Ids only — no patient data."""
    p = Path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        fh.write("std_cid\tnode_cid\n")
        for a, b in pairs:
            fh.write(f"{int(a)}\t{int(b)}\n")
    return p










def _densify_lean_blocks(blocks, C):
    """Per-partition lean blocks -> `(proba f32, y u8, mask u8, ids, elig u8|None)`.

    The driver-side half of `distributed_readout._lean_eval_kernel`. Peak driver
    memory is the whole point of the v2.1 refinement, so it is worth being explicit:
    the destination arrays are (4 + 1 + 1) = **6 bytes per (doc, node) cell** —
    ~1.6 GB at D_te=80k, C=3,300 — against the driver path's 8*D*(K + 3C) for the
    same eval. Each block is dropped as soon as it is copied, so the transient
    overshoot is the collected blocks (float32 p + index lists, ~4-5 bytes/cell)
    rather than a second full copy of everything.

    `y`/`mask` arrive as CSR-style index runs; `mask` may be `None`, which means
    "every doc observes every node" (`--label-mask-mode full`).

    THE FOURTH RUN (E2/WP4). When the collect selected E1's pre-index closure
    column, each block carries a fifth CSR run holding `R_d` — the engine ids the
    document ALREADY CARRIED before its index — and this returns the complementary
    `elig` matrix, `elig[d,c] = c ∉ R_d` (spec D2), at **+1 byte/cell**. The
    complement is taken here rather than on the executors because `R_d` is the
    sparse side and eligibility the dense one (0109 root prevalence 0.9609): the
    wire carries the small set, the driver materializes the big one. Blocks without
    the run (a bundle built before E1, or a collect that did not ask) yield
    `elig=None`, which every consumer reads as "no eligibility information" and
    which makes the incident arm SKIP rather than silently score the prevalent
    cells twice. Legacy 6-tuple blocks are accepted for the same reason."""
    C = int(C)
    D = sum(int(b[1].shape[0]) for b in blocks)
    proba = np.zeros((D, C), dtype=np.float32)
    y = np.zeros((D, C), dtype=np.uint8)
    mask = np.zeros((D, C), dtype=np.uint8)
    ids = np.zeros(D, dtype=np.int64)
    # Allocated lazily: a run without the pre-index column must not pay (D,C) bytes
    # for an all-ones array nobody asked for.
    elig = None
    at = 0
    for j in range(len(blocks)):
        b = blocks[j]
        b_ids, P, y_idx, y_ptr, m_idx, m_ptr = b[:6]
        e_idx, e_ptr = (b[6], b[7]) if len(b) > 6 else (None, None)
        n = int(P.shape[0])
        proba[at:at + n] = P
        ids[at:at + n] = b_ids
        y[np.repeat(np.arange(at, at + n), np.diff(y_ptr)), y_idx] = 1
        if m_idx is None:
            mask[at:at + n] = 1
        else:
            mask[np.repeat(np.arange(at, at + n), np.diff(m_ptr)), m_idx] = 1
        if e_idx is not None:
            if elig is None:
                # Eligible by default; the run below SUBTRACTS the prior carriers.
                elig = np.ones((D, C), dtype=np.uint8)
            elig[np.repeat(np.arange(at, at + n), np.diff(e_ptr)), e_idx] = 0
        at += n
        blocks[j] = None                      # free the partition's block eagerly
    id_list = ids.tolist()
    # Corpus-level uniqueness tripwire, at the point the doc keys are first a full
    # driver array (spec R5.4). The lean eval arrays above are row-indexed and the
    # A/B gate aligns on these keys, so a collision here would be a silently
    # mis-attributed row rather than an error — assert instead.
    _assert_unique_doc_keys(id_list, where="_densify_lean_blocks")
    return proba, y, mask, id_list, elig


def _collect_lean_proba(scored_df, C, V=None, b_raw=None, *, degenerate=None,
                        const=None, score_col="topicDistribution",
                        label_col="label", mask_col="labelMask",
                        id_col="person_id", theta_topm=0, elig_col=None):
    """LEAN readout collect: `(proba (D,C) f32, y u8, mask u8, doc_key_order, elig)`.

    Plan §3 (v2.1): once the FIT is distributed, the driver eval needs only the test
    split's per-node probabilities and labels — no θ, no float64 (D,C) arrays, no
    per-node LR on the driver. This collector is the whole reason the existing eval
    stack (`readout_from_proba`, `conditional_readout`, the quartile split) can stay
    byte-identical at whole-Mondo scale: it hands them the SAME arrays in a
    6-bytes-per-cell dress.

    Two callers, one kernel: with `(V, b_raw)` the executors score raw θ from the
    batched fit; with `V=None` and `score_col="probability"` the co-fit head's own
    per-doc (C,) probability column is packed as-is (no fit involved).

    `degenerate`/`const` apply the ORACLE's fallback for nodes whose observed TRAIN
    set was empty or single-class — `_lr_proba_per_label_masked` predicts the lone
    class value (0.0 when nothing was observed) rather than fitting, and macro means
    are only comparable across paths if this reproduces it exactly.

    `theta_topm` must MATCH the `_fit_readout_heads` call that produced `(V, b_raw)`:
    the fitted coefficients belong to the truncated design matrix, so scoring full θ
    with them would evaluate a model on features it was never fit on. It is ignored
    (and must be left 0) on the `V is None` branch, where `score_col` is an
    already-computed probability rather than a feature vector.

    `elig_col` (E2/WP4) names E1's per-document pre-index closure column
    (`preindexClosure`) and adds it to the select as a FOURTH CSR run — the same
    partition kernel, one extra column, +1 byte/cell on the returned `elig` matrix.
    Left `None` (the default) the collect is byte-identical to what it was and
    `elig` comes back `None`: the incident arm is then SKIPPED, never guessed. The
    caller is responsible for checking the bundle's E1 WITNESS first
    (`preindex_closure.bundle_preindex_witness`) — asking for the column against a
    bundle that lacks it is a Spark AnalysisException, which is exactly the
    mixed-vintage failure R1.4 exists to turn into a sentence.

    `id_col` names the PERSON-ID column the row's int64 doc key is synthesized
    from (`_doc_key_column`, spec R5.4); the collect materializes a `doc_key`
    column and hands the kernel THAT, never the raw id — so the returned
    `doc_key_order` is document identity (one entry per row) and a person-grain
    consumer recovers the person via `_dr.person_of`. On single-doc corpora
    `doc_key == person_id * RADIX`, order-preserving, so nothing about the
    existing path's meaning changes."""
    C = int(C)
    theta_topm = int(theta_topm)
    # Synthesize the int64 doc key at the seam and select it as the row id, so the
    # kernel packs an int64 (a raw string doc_id would raise) and every row stays
    # person-derivable. `id_col` is the person-id SOURCE column, not the id itself.
    scored_df = scored_df.withColumn("doc_key", _doc_key_column(scored_df, id_col))
    key_col = "doc_key"
    cols = ((key_col, score_col, label_col, mask_col) if elig_col is None
            else (key_col, score_col, label_col, mask_col, elig_col))
    _rows = _dr._row_quads if elig_col is None else _dr._row_quints
    sc = scored_df.sparkSession.sparkContext

    def _collect():
        # Broadcast + collect inside ONE retried closure: this is a driver-blocking
        # action on the readout path, and it runs right after the multi-hour solve,
        # so a preemption wave landing here would throw away the fit's whole
        # readout. The kernel is a pure function of the test split, so re-running
        # the collect returns the same blocks (`_retry_spark_action`); the
        # broadcast is rebuilt per attempt and unpersisted per attempt, exactly as
        # in `SparkStatsFn.__call__`.
        if V is None:
            bcast = None

            def _block(rows, _C=C, _cols=cols, _rw=_rows):
                return [_dr._lean_eval_kernel(_rw(rows, *_cols), _C)]
        else:
            bcast = sc.broadcast((np.ascontiguousarray(V, dtype=np.float64),
                                  np.ascontiguousarray(b_raw, dtype=np.float64)))

            def _block(rows, _b=bcast, _C=C, _cols=cols, _m=theta_topm, _rw=_rows):
                V_, b_ = _b.value
                return [_dr._lean_eval_kernel(
                    _rw(rows, *_cols, topm=_m), _C, V_, b_)]

        try:
            return scored_df.select(*cols).rdd.mapPartitions(_block).collect()
        finally:
            if bcast is not None:
                # destroy, not unpersist: reclaims the driver-local temp file too
                # (see distributed_readout._destroy_broadcast — the leak that
                # filled the master's disk at one broadcast per solver pass).
                _dr._destroy_broadcast(bcast)

    blocks = _dr._retry_spark_action(_collect, label="lean eval collect")
    proba, y, mask, ids, elig = _densify_lean_blocks(blocks, C)
    if degenerate is not None and bool(np.any(degenerate)):
        deg = np.asarray(degenerate, dtype=bool)
        proba[:, deg] = np.asarray(const, dtype=np.float32)[deg]
    return proba, y, mask, ids, elig


_CKPT_VERSION = "readout-ckpt-v2"


def _readout_ckpt_fingerprint(C, K, n_obs, n_pos, theta_topm=0):
    """Digest identifying the PROBLEM a solver checkpoint belongs to.

    Hashes only the EXACTLY-REPRODUCIBLE identity of the problem: `(C, K,
    theta_topm)` plus the per-node observed/positive COUNTS. Counts are integer
    sums, and integer addition is exact in float64 in any order — so `n_obs`/
    `n_pos` come back bit-identical from every run over the same rows and mask,
    and they are also exactly what distinguishes a different arm (labels change
    `n_pos`), a different corpus/split (`n_obs`), or a different degenerate mask.

    What is deliberately NOT hashed: the standardization moments `(mu, sd)`.
    v1 hashed their bytes on the theory that the moments pass is deterministic —
    and it is, in VALUE, but not in BITS: `treeAggregate` combines partials in
    task-completion order, float addition is not associative, and the low-order
    bits differ on every run. In production that made every cross-run
    fingerprint a mismatch, which silently disabled the entire resume feature
    (exp 0104, 2026-08-29: an iter-50 checkpoint from the identical solve was
    rejected and 100 minutes were re-paid). The residual risk of not pinning the
    basis is bounded by the solver's own contract: `x0` changes the PATH, never
    the answer — a warm start whose basis drifted by float noise costs at most a
    few extra iterations, while a byte-exact check costs the whole feature.
    """
    h = hashlib.sha256()
    h.update(f"{_CKPT_VERSION}|C={int(C)}|K={int(K)}|topm={int(theta_topm)}|"
             .encode())
    for arr in (n_obs, n_pos):
        a = np.rint(np.ascontiguousarray(arr, dtype=np.float64)).astype(np.int64)
        h.update(f"|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _write_readout_ckpt(path, W_std, b_std, it, fingerprint):
    """Atomically write the solver checkpoint to `path` (tmp + replace).

    Same discipline as `_dump_partial_results`, and for a sharper reason: the run
    dir is gcsfuse, where a write is an object mutation and a process death
    mid-write leaves a TORN file — which here would be worse than no checkpoint
    at all, since the thing that reads it is a recovery run. Write to a sibling
    tmp then rename, so the visible file is always a complete one.

    Failures are swallowed and reported: a checkpoint is insurance, and a full
    disk or a gcsfuse hiccup must not kill a solve that is otherwise healthy.
    Returns True on success.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            # File OBJECT, not a name: np.savez appends `.npz` to a str path,
            # which would make the tmp name (and thus the rename) a guess.
            np.savez(fh, W_std=np.ascontiguousarray(W_std, dtype=np.float64),
                     b_std=np.ascontiguousarray(b_std, dtype=np.float64),
                     iter=np.int64(it), fingerprint=np.str_(fingerprint))
        tmp.replace(path)
        return True
    except Exception as exc:                     # pragma: no cover - I/O failure
        _cprint(f"[driver]   readout checkpoint write FAILED ({exc}); the solve "
              "continues uncheckpointed", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _read_readout_ckpt(path, fingerprint):
    """Load `(W_std, b_std, iter)` from `path` if it matches `fingerprint`.

    Returns `None` when there is nothing usable — no file, an unreadable/torn
    file, or a fingerprint mismatch — and PRINTS why in the mismatch case, which
    is the one that means something (a checkpoint left by a different arm or a
    different corpus). A mismatch does not delete the file: the fresh solve's
    own first checkpoint overwrites it, and until then a stale-but-explainable
    file on disk is more useful for diagnosis than a silent deletion.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            got = str(z["fingerprint"].item())
            if got != fingerprint:
                _cprint(f"[driver]   readout checkpoint {path.name} IGNORED: "
                      f"fingerprint {got[:12]} != {fingerprint[:12]} — it belongs "
                      "to a different arm/corpus/basis. Starting cold.", flush=True)
                return None
            return (np.asarray(z["W_std"], dtype=np.float64),
                    np.asarray(z["b_std"], dtype=np.float64),
                    int(z["iter"]))
    except Exception as exc:
        _cprint(f"[driver]   readout checkpoint {path.name} UNREADABLE ({exc}); "
              "starting cold", flush=True)
        return None


# --------------------------------------------------------------------------- #
# Persisted readout HEADS (the fitted per-node scoring params, saved for reuse) #
# --------------------------------------------------------------------------- #
# Distinct from the SOLVER checkpoint above: the checkpoint is a mid-solve
# resume point that is DELETED once the solve lands, whereas this is the
# COMPLETED fit's raw-θ scoring params, kept beside the run's other record files.
# It exists so a later scoring-only consumer (conversion_analysis --deciles on)
# can turn each doc's θ into per-node proba with ONE mapPartitions, instead of
# re-running the whole batched readout solve just to get per-doc scores — the
# re-fit that filled the worker local disks and cascaded ~12 nodes dead on 0110.
_HEADS_VERSION = "readout-heads-v1"


def _readout_heads_path(run_dir, label):
    """Path of the persisted heads npz for arm `label` in `run_dir`.

    Keyed by `label` for the same reason the solver checkpoint is
    (`readout_heads_gated_pc.npz`, `..._unsup_gated.npz`, `..._dag_head.npz`): a
    run fits several arms into one dir and each is a different scoring model."""
    return Path(run_dir) / f"readout_heads_{label or 'arm'}.npz"


def _readout_heads_fingerprint(C, K, theta_topm, label, degenerate):
    """Identity of a fitted arm's heads — the arm shape plus its fittable mask.

    Mirrors `_readout_ckpt_fingerprint`'s doctrine: hash only the
    exactly-reproducible integers, never the standardization moments' float bits
    (which `treeAggregate` combines in nondeterministic order). `degenerate` is a
    bool mask derived from integer observed/positive COUNTS, so it is bit-stable
    across runs of the same arm/corpus and is a compact signature of them."""
    h = hashlib.sha256()
    h.update(f"{_HEADS_VERSION}|label={label or 'arm'}|C={int(C)}|K={int(K)}|"
             f"topm={int(theta_topm)}|".encode())
    d = np.ascontiguousarray(np.asarray(degenerate, dtype=bool))
    h.update(f"|{d.shape}|".encode())
    h.update(d.tobytes())
    return h.hexdigest()


def _write_readout_heads(run_dir, label, V, b_raw, const, degenerate, C, K,
                         theta_topm, W_std=None):
    """Persist a COMPLETED readout fit's raw-θ scoring params to the run dir.

    Everything a scoring pass needs to turn a doc's θ into per-node proba —
    `(V, b_raw)` (raw-θ, so no scaler travels), plus the `degenerate`/`const`
    oracle fallback and the truncation width `theta_topm` the coefficients belong
    to. Written as a small npz (~60 MB at 0110 scale: C×K×8) beside the fit's
    other record files, once the solve has RETURNED.

    Additive: it changes no existing number. Written only when the arm has a
    durable run dir (`distributed_score_arm`'s `checkpoint_dir`), which is exactly
    the fit driver's readout path and `gated_pc_readout`'s recovery path — never
    the row-SAMPLED A/B fit or the calibration sub-fit, which have no business
    owning this file. Atomic (tmp + replace) for the same gcsfuse torn-file reason
    as `_write_readout_ckpt`; failures are swallowed and reported, because the
    fit's own results are the record and a missing sidecar only means
    `conversion_analysis --deciles on` asks for a one-time re-readout. Returns
    True on success.

    NOTE: pre-existing fits (e.g. 0110's) predate this file and lack it until a
    readout is re-run under this code — `make gated-pc-readout ID=<n>` once.

    `W_std` (optional, (C,K)): the solve's STANDARDIZED weights, stored beside V
    so the off-cluster loadings read (`inspect_topics`) has an honest importance
    scale once the solver checkpoint is gone — V is W_std/sd and explodes on
    low-variance (starved) topics. Additive; older sidecars simply lack it."""
    path = _readout_heads_path(run_dir, label)
    tmp = path.with_name(path.name + ".tmp")
    fp = _readout_heads_fingerprint(C, K, theta_topm, label, degenerate)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            # File OBJECT, not a name: np.savez appends `.npz` to a str path,
            # which would make the tmp name (and thus the rename) a guess.
            extra = ({"W_std": np.ascontiguousarray(W_std, dtype=np.float64)}
                     if W_std is not None else {})
            np.savez(fh,
                     version=np.str_(_HEADS_VERSION),
                     label=np.str_(label or "arm"),
                     V=np.ascontiguousarray(V, dtype=np.float64),
                     **extra,
                     b_raw=np.ascontiguousarray(b_raw, dtype=np.float64),
                     const=np.ascontiguousarray(const, dtype=np.float64),
                     degenerate=np.ascontiguousarray(degenerate, dtype=bool),
                     C=np.int64(int(C)), K=np.int64(int(K)),
                     theta_topm=np.int64(int(theta_topm)),
                     fingerprint=np.str_(fp))
        tmp.replace(path)
        _cprint(f"[driver]   wrote readout heads {path.name} "
              f"(C={int(C)} K={int(K)} topm={int(theta_topm)}) — "
              "conversion_analysis --deciles on scores from this, no re-fit",
              flush=True)
        return True
    except Exception as exc:                         # pragma: no cover - I/O failure
        _cprint(f"[driver]   readout heads write FAILED ({exc}); the fit is "
              "unaffected, but conversion_analysis --deciles on will need a "
              "re-readout to persist them", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _read_readout_heads(run_dir, label, *, C=None, K=None, theta_topm=None):
    """Load persisted readout heads for `label`, or None with a printed reason.

    Returns `(V, b_raw, const, degenerate, C, K, theta_topm)` — the exact tuple a
    scoring pass (`_collect_lean_proba`) consumes — when a heads npz is present,
    readable, of a known version, and (where the caller passes them) consistent
    with the manifest's `C`/`K`/`theta_topm`. Any miss returns None so the caller
    can degrade to the overall (non-decile) deliverable, NEVER fall back to a
    re-fit. `K=None`/`theta_topm=None` skip that particular consistency check (an
    old manifest may not record K); the V-shape check against the STORED C/K still
    runs regardless."""
    path = _readout_heads_path(run_dir, label)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            ver = str(z["version"].item())
            if ver != _HEADS_VERSION:
                _cprint(f"[driver]   readout heads {path.name} IGNORED: version "
                      f"{ver!r} != {_HEADS_VERSION!r}", flush=True)
                return None
            gC, gK, gm = int(z["C"]), int(z["K"]), int(z["theta_topm"])
            V = np.asarray(z["V"], dtype=np.float64)
            b_raw = np.asarray(z["b_raw"], dtype=np.float64)
            const = np.asarray(z["const"], dtype=np.float64)
            degenerate = np.asarray(z["degenerate"], dtype=bool)
    except Exception as exc:
        _cprint(f"[driver]   readout heads {path.name} UNREADABLE ({exc})",
              flush=True)
        return None
    for name, want, got in (("C", C, gC), ("K", K, gK),
                            ("theta_topm", theta_topm, gm)):
        if want is not None and int(want) != got:
            _cprint(f"[driver]   readout heads {path.name} IGNORED: {name}={got} "
                  f"!= manifest {name}={int(want)} — a different arm/run",
                  flush=True)
            return None
    if V.shape != (gC, gK):
        _cprint(f"[driver]   readout heads {path.name} IGNORED: V shape {V.shape} "
              f"!= (C={gC}, K={gK})", flush=True)
        return None
    return V, b_raw, const, degenerate, gC, gK, gm


def _apply_feature_mask(stats_fn, feature_mask):
    """Wrap a batched-LR `stats_fn` so the DATA gradient is zero outside a (C,K)
    boolean `feature_mask` — the driver-side seam for a per-node FEATURE ABLATION
    (own block only / drop own block / background only ...).

    Why zeroing the gradient is a correct projected solve, not a hack: the
    solver minimizes F_c = loss_c(w) + 0.5·l2·‖w‖² per node by L-BFGS from
    w = 0. On a coordinate whose data gradient is always 0, the ridge gradient
    l2·w is also 0 while w stays 0, so the full gradient is 0 there; the two-loop
    direction is a linear combination of gradients and of past steps, both of
    which are 0 on that coordinate, so the step is 0 there, so w stays exactly
    0 — by induction, for every iteration and every trial point of the line
    search. The unmasked coordinates therefore solve exactly the reduced problem
    over the kept features (same objective, restricted to the subspace), which is
    what an ablation means. No closure payload: the mask lives on the driver and
    is applied to the returned (C,K) gradient; `node_mask` passes straight
    through as in `_fittable_stats`."""
    fm = np.ascontiguousarray(np.asarray(feature_mask, dtype=bool))

    def _masked(W_std, b_std, node_mask=None, _f=stats_fn, _fm=fm):
        loss, gW, gb = _f(W_std, b_std, node_mask=node_mask)
        return loss, np.where(_fm, gW, 0.0), gb
    return _masked


def _fit_readout_heads(train_scored, C, K, *, l2=1.0, gtol=_READOUT_GTOL,
                       max_iter=_READOUT_MAX_ITER, history=6, label="", depth=None,
                       warm_start=None, theta_topm=0, checkpoint_path=None,
                       checkpoint_every=10,
                       topic_col="topicDistribution",
                       label_col="label", mask_col="labelMask",
                       feature_mask=None):
    """Fit all C per-node readout heads with ONE batched distributed L-BFGS.

    `feature_mask` (optional, (C,K) bool): per-node FEATURE ABLATION — the head
    of node c may load only on the topics where `feature_mask[c]` is True (see
    `_apply_feature_mask` for why gradient masking is an exact reduced solve).
    Folded into the checkpoint fingerprint, so an ablation solve never resumes a
    full-feature checkpoint of the same problem, and applied to any warm start /
    resumed point. The standardization moments are unaffected (a masked feature
    simply carries w = 0, which folds to V = 0).

    Returns `(V (C,K), b_raw (C,), const (C,), degenerate (C,) bool, info)` — the
    raw-θ scoring parameters, so nothing but `(V, b_raw)` has to travel to score a
    split (plan §1: the standardization is folded away, "scoring needs no scaler").

    Three driver-side steps, each doing exactly what the plan's §1 prescribes:

      1. `masked_moments` — one pass for the per-node masked mean/sd (the sklearn
         oracle standardizes on each node's OWN observed train rows) plus the
         `n_obs`/`n_pos` counts that identify degenerate nodes for free.
      2. DEGENERATE MASKING. A node whose observed train set is empty or
         single-class has no finite optimum (the intercept runs to ±inf), so the
         oracle refuses to fit it and predicts the lone class. We reproduce that by
         zeroing its loss/gradient rows before the solver sees them: its gradient is
         then exactly 0, `solve_batched_lr` freezes it at iteration 0, and the
         constant is applied at scoring time. Wrapping the stats_fn (rather than
         re-deriving the moments off a masked frame) keeps it to one data pass.
      3. `solve_batched_lr` at `l2=1.0` — sklearn's `C=1.0` on the SUMMED log-loss,
         intercept unpenalized — then fold back to raw-θ coordinates.

    `warm_start=(V, b_raw)` starts the solve from ANOTHER fit's raw-θ scoring
    params — used for the two solves that are perturbations of the arm's main fit
    (the 75/25 calibration split, the A/B harness's row sample) rather than new
    problems. It travels in RAW coordinates by necessity: standardized params
    only mean something next to the `(mu, sd)` that produced them, and this fit's
    moments come from ITS rows, so the start is unfolded here through the moments
    computed in step 1. Two things it must not do, both handled below:

      - resurrect a node this fit masks. A degenerate node's data term is zeroed,
        leaving the bare ridge, whose gradient `l2*w` vanishes only at `w = 0` —
        a stale warm-start row would therefore make the solver walk it back to
        zero over real distributed passes, for a node whose probability is
        overwritten by the constant fallback anyway. Its `x0` row is zeroed, which
        restores the exact iteration-0 no-op the cold path has;
      - change the answer. It cannot: the per-node objective is convex, so the
        start moves the path and not the optimum. What it buys is the iterate at
        a CAPPED budget (`max_iter`), which is the budget these solves run under.

    `theta_topm > 0` fits on TOP-M TRUNCATED θ (see `distributed_readout`'s module
    docstring). It changes the DESIGN MATRIX, not the procedure: the moments pass,
    every L-BFGS pass and — by the caller's obligation — the scoring pass all read
    the same truncated features, so what comes back is the exact readout of a
    narrower model rather than an approximation of the wide one. The mass that
    truncation drops is measured and logged BEFORE the fit (`theta_topm_coverage`),
    because the whole premise is an empirical claim about θ's concentration.

    `checkpoint_path` turns the solve into a RESUMABLE one, which at whole-Mondo
    scale is the difference between losing 15 minutes and losing an afternoon:
    exp 0104's 2026-08-28 recovery died 9,112s into the solve (a spot-preemption
    wave starved the scheduler and Spark aborted the job) with nothing saved.
    Every `checkpoint_every` iterations the progress hook writes the current
    standardized iterate — the solver's entire resumable state, since `x0` takes
    a POINT — atomically, next to a FINGERPRINT of `(C, K, mu, sd, n_obs, n_pos)`
    that a resume must match. On the next run a matching fingerprint is fed back
    as `x0` (with the degenerate rows re-zeroed, exactly as the `warm_start` path
    does, or the bare ridge would spend real distributed passes walking them back
    to zero); a mismatch warns and starts cold. What a resume does NOT carry is
    the L-BFGS curvature history, so its first iterations re-learn the scaling
    and are slower than the ones that preceded the crash — expected, logged, and
    still overwhelmingly cheaper than restarting at iteration 0 (insight 0074: a
    warm start supplies a point, not curvature). The file is deleted once the
    solve returns: the fit results are the record, and a checkpoint that outlives
    its solve is only a way to confuse a later run (which the fingerprint would
    catch, but a deleted file cannot even raise the question).

    When BOTH `warm_start` and a matching checkpoint are present the checkpoint
    wins: it is a point on THIS problem's own optimization path, strictly further
    along than another fit's answer mapped in.
    """
    from analysis.pc.batched_lr import (fold_standardization, solve_batched_lr,
                                        standardized_grad_from_raw,
                                        unfold_standardization)

    C, K = int(C), int(K)
    theta_topm = int(theta_topm)
    if depth is None:
        # Same driver-burst sizing rule as spark_vi.core.runner._agg_depth: a
        # per-partition partial here is ~two (C, K) float64 arrays (moments) /
        # one plus vectors (stats), so above ~128 MB the depth-2 tree would land
        # ~sqrt(P) of them on the driver JVM at once — the burst that OOM'd the
        # exp 0104 smoke's FIT aggregate at whole-Mondo scale. One extra combine
        # round is noise next to the pass itself.
        depth = 3 if 2 * C * K * 8 > 128 * 1024 * 1024 else 2
    tag = f"{label}: " if label else ""
    if C * K * 8 > _COVERAGE_MIN_FIT_BYTES:
        # Only measured where truncation is on the table. Below ~64 MB of (C,K)
        # parameters the dense fit is a couple of seconds a pass and the θ-width
        # lever is not a decision anyone is making, so the extra pass would be pure
        # cost. Above it, this is the number that says whether top-m is a cheap
        # reparameterization or a lobotomy — reported on the TRAIN frame, before the
        # moments pass, whether or not `theta_topm` is set, so the run of record
        # carries the evidence for (or against) the setting it used.
        cov = _dr.theta_topm_coverage(train_scored, K, topic_col=topic_col,
                                      depth=depth)
        _cprint(f"[driver]   {tag}theta top-m mass: "
              + " ".join(f"m={m}:{mean:.3f}/{p10:.3f}"
                         for m, (mean, p10) in sorted(cov.items()))
              + " (mean/p10)", flush=True)
    mu, sd, n_obs, n_pos = _dr.masked_moments(
        train_scored, C, K, topic_col=topic_col, label_col=label_col,
        mask_col=mask_col, depth=depth, topm=theta_topm)
    # Same three cases as `_lr_proba_per_label_masked`'s `np.unique(yc).size < 2`:
    # nothing observed (-> 0.0), all-negative (-> 0.0), all-positive (-> 1.0).
    degenerate = (n_obs <= 0) | (n_pos <= 0) | (n_pos >= n_obs)
    const = np.where((n_obs > 0) & (n_pos >= n_obs), 1.0, 0.0)
    keep = ~degenerate
    fmask = None
    if feature_mask is not None:
        fmask = np.ascontiguousarray(np.asarray(feature_mask, dtype=bool))
        if fmask.shape != (C, K):
            raise ValueError(f"feature_mask shape {fmask.shape} != ({C}, {K})")
        _cprint(f"[driver]   {tag}FEATURE ABLATION: heads restricted to a per-node "
              f"topic mask (mean {fmask.mean() * K:.1f} of {K} topics per node)",
              flush=True)

    def _proj(W):
        # Pin masked coordinates at exactly 0 on any externally supplied point.
        return W if fmask is None else np.where(fmask, W, 0.0)
    x0 = None
    if warm_start is not None:
        W0, b0 = unfold_standardization(warm_start[0], warm_start[1], mu, sd)
        x0 = (_proj(np.where(keep[:, None], W0, 0.0)), np.where(keep, b0, 0.0))
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = None
    if ckpt_path is not None:
        # Derived AFTER the moments pass because the counts come from it —
        # exact-integer sums, the reproducible identity of this arm's problem
        # (see _readout_ckpt_fingerprint for why mu/sd bytes are NOT hashed).
        fingerprint = _readout_ckpt_fingerprint(C, K, n_obs, n_pos, theta_topm)
        if fmask is not None:
            # An ablation is a different problem on the same rows: fold the mask
            # so a masked solve never resumes (or is resumed by) the full one.
            _h = hashlib.sha256((fingerprint + "|feature_mask|").encode())
            _h.update(fmask.tobytes())
            fingerprint = _h.hexdigest()
        resumed = _read_readout_ckpt(ckpt_path, fingerprint)
        if resumed is not None:
            W_ck, b_ck, ck_iter = resumed
            # Same zeroing rule as the warm-start path above, for the same reason:
            # a degenerate node's data term is zeroed, so its objective is the bare
            # ridge and any nonzero row would be walked back to 0 over real
            # distributed passes. (A checkpoint written by THIS problem already has
            # them at zero — the fingerprint pins the degenerate mask — so this is
            # a belt-and-braces restatement of the contract, not a fixup.)
            x0 = (_proj(np.where(keep[:, None], W_ck, 0.0)),
                  np.where(keep, b_ck, 0.0))
            _cprint(f"[driver]   {tag}resuming batched solve from checkpoint "
                  f"(iter {ck_iter} recorded); curvature history is not carried, "
                  "early iterations re-learn it", flush=True)
    _cprint(f"[driver]   {tag}distributed readout fit: C={C} K={K}, "
          f"{int(keep.sum())} fittable nodes, {int(degenerate.sum())} degenerate "
          f"(constant fallback), observed train cells={int(n_obs.sum())}"
          f"{' (warm start)' if x0 is not None else ''}"
          f"{f' (theta top-m={theta_topm})' if theta_topm > 0 else ''}", flush=True)

    with _dr.make_spark_stats_fn(
            train_scored, C, K, mu, sd,
            fold_standardization=fold_standardization,
            standardized_grad_from_raw=standardized_grad_from_raw,
            topic_col=topic_col, label_col=label_col, mask_col=mask_col,
            depth=depth, topm=theta_topm) as _raw_stats_fn:
        stats_fn = (_apply_feature_mask(_raw_stats_fn, fmask)
                    if fmask is not None else _raw_stats_fn)

        def _fittable_stats(W_std, b_std, node_mask=None, _f=stats_fn, _keep=keep):
            # `node_mask` passes STRAIGHT THROUGH to the pass (the solver uses it
            # to skip nodes whose stats it already holds) and composes with the
            # degenerate zeroing without interacting with it: masking decides
            # which cells are READ, `keep` decides which rows are ZEROED, and a
            # zeroed row is zero either way. Dropping the argument here would
            # silently re-inflate every trial pass back to all C heads.
            loss, gW, gb = _f(W_std, b_std, node_mask=node_mask)
            return (np.where(_keep, loss, 0.0),
                    np.where(_keep[:, None], gW, 0.0),
                    np.where(_keep, gb, 0.0))

        # Heartbeat: each stats call is a full treeAggregate over the train
        # split, so a whole-Mondo solve is minutes/iteration and looks hung
        # without it. Log every iteration for the first few (the cold-start
        # steps are the slow, uncertain ones), then every 5th, then the finish
        # line from the summary print below.
        t0 = time.time()

        every = max(1, int(checkpoint_every))

        # Window state for the heartbeat: per-iteration RATES (s/iter, passes/
        # iter, newly converged) read the solve's health far better than the
        # cumulative totals alone — a line search going pathological shows up as
        # passes/iter climbing within two heartbeats instead of a slowly bending
        # cumulative curve. `_prev` holds the totals at the last PRINTED line.
        _prev = {"iter": 0, "passes": 0, "conv": 0, "t": t0}

        def _progress(p, _t0=t0, _tag=tag, _ndeg=int(degenerate.sum()),
                      _nfit=int(keep.sum()), _path=ckpt_path, _fp=fingerprint,
                      _every=every, _prev=_prev):
            if _path is not None and p["iter"] % _every == 0:
                # BEFORE the logging gate, not after it: the two cadences are
                # independent (log every 5th iteration, checkpoint every 10th by
                # default), and hanging the checkpoint off the log's early return
                # would silently tie them together. `W_std`/`b_std` are live views
                # into the solver's iterate — `_write_readout_ckpt` copies them
                # into the npz, which is the copy the contract requires.
                _write_readout_ckpt(_path, p["W_std"], p["b_std"], p["iter"], _fp)
            if p["iter"] > 3 and p["iter"] % 5:
                return
            # Degenerate nodes converge (grad exactly 0) at iteration 0; report
            # progress over the FITTABLE nodes so the fraction reads as work left.
            # `nodes/pass` is the cost that moves now that trial passes are masked
            # to the nodes still searching: a deep line search shows up as extra
            # passes over FEW nodes, which reads very differently from extra
            # passes over all C.
            now = time.time()
            conv = max(0, p["n_converged"] - _ndeg)
            d_iter = max(1, p["iter"] - _prev["iter"])
            d_pass = p["n_stats_calls"] - _prev["passes"]
            _cprint(f"[driver]   {_tag}batched L-BFGS iter {p['iter']}: "
                  f"{(now - _prev['t']) / d_iter:.1f}s/iter, "
                  f"{d_pass / d_iter:.1f} passes/iter "
                  f"(avg {p['n_node_evals'] / max(p['n_stats_calls'], 1):.0f} "
                  f"nodes/pass), "
                  f"{conv}/{_nfit} converged (+{conv - _prev['conv']}), "
                  f"{p['n_active']} active, max|grad|={p['max_grad_inf_norm']:.3g}, "
                  f"{p['n_stats_calls']} passes/{now - _t0:.0f}s total",
                  flush=True)
            _prev.update(iter=p["iter"], passes=p["n_stats_calls"], conv=conv,
                         t=now)

        W_std, b_std, info = solve_batched_lr(
            _fittable_stats, C, K, l2=l2, max_iter=max_iter, history=history,
            gtol=gtol, x0=x0, progress_fn=_progress)
    if ckpt_path is not None:
        # The solve landed, so `(V, b_raw)` — and the results computed from them —
        # are the record; the checkpoint's only remaining effect would be to warm
        # a LATER run from a point it did not earn. (The fingerprint would refuse a
        # genuinely different problem, but an identical re-run resumed from a
        # finished solve would silently report "resuming from iter N" for a fit
        # that is already done.) Best-effort: a failed unlink is not worth failing
        # a completed fit over.
        try:
            ckpt_path.unlink(missing_ok=True)
        except OSError as exc:                   # pragma: no cover - I/O failure
            _cprint(f"[driver]   {tag}could not remove readout checkpoint "
                  f"{ckpt_path.name} ({exc})", flush=True)
    V, b_raw = fold_standardization(W_std, b_std, mu, sd)
    gmax = float(info["grad_inf_norm"][keep].max()) if keep.any() else 0.0
    # `converged` = gtol OR the principled numerical stall; at gtol=1e-4 (sklearn's
    # own tol) every node should stop on the gradient, so a nonzero stalled count is
    # the diagnostic that the summed-loss roundoff floor was hit first.
    _cprint(f"[driver]   {tag}batched L-BFGS: {int(info['n_stats_calls'])} data passes "
          f"({int(info['n_node_evals'])} node-passes, "
          f"{info['n_node_evals'] / max(int(info['n_stats_calls']), 1):.0f} avg), "
          f"{int(info['converged'][keep].sum())}/{int(keep.sum())} converged "
          f"({int(info['converged_gtol'][keep].sum())} gtol, "
          f"{int(info['stalled'][keep].sum())} stalled), max|grad|={gmax:.3g}, "
          f"max iters={int(info['n_iter'].max())}, "
          f"line-search failures={int(info['line_search_failures'])}", flush=True)
    # The STANDARDIZED weights ride along for the heads sidecar: V = W_std/sd is
    # the scoring decoder but inflates on low-variance (starved) topics, so any
    # loadings/importance read off-cluster needs W_std (inspect_topics).
    info["W_std"] = np.asarray(W_std, dtype=np.float64)
    return V, b_raw, const, degenerate, info


def distributed_score_arm(train_scored, test_scored, C, K, *, recall_targets,
                          fdr_targets, min_count=0, label="", l2=1.0,
                          gtol=_READOUT_GTOL, max_iter=_READOUT_MAX_ITER,
                          history=6, depth=None, warm_start=None, theta_topm=0,
                          checkpoint_dir=None, checkpoint_every=10,
                          topic_col="topicDistribution", label_col="label",
                          mask_col="labelMask", id_col="person_id",
                          elig_col=None, feature_mask=None):
    """`score_arm` without the driver-side θ collect — the distributed twin.

    Returns `(readout, proba_te (D_te,C) f32, y_te u8, m_te u8, doc_key_order,
    (V, b_raw), elig u8|None)`; the fifth is the per-DOCUMENT int64 key order
    (`_doc_key_column`, spec R5.4). The first four mirror `_score_full`'s
    `(readout, proba)` contract plus the two label arrays the driver path used to
    get from `_collect_theta_labels`, so every downstream consumer (`_conditional`,
    the rarity quartile split, the headline) is unchanged. The raw-θ fit
    params at index 5 are what a LATER solve on a near-identical problem warm-starts
    from (`readout_ab_report`'s row-sampled fit); callers that only want the readout
    index it out (`[0]`) and are unaffected by its presence.

    `elig_col` (E2/WP4) names E1's pre-index closure column; it rides the SAME lean
    collect as a fourth CSR run and comes back as the trailing `elig` matrix, which
    the caller feeds to `incident_readout`. `None` (the default) keeps the collect
    exactly as it was and returns `elig=None`. It is APPENDED to the tuple, so
    `[:5]` and `[5]` keep meaning what they meant.

    Same three ingredients as the driver readout, moved: fit per-node LRs on the
    train split's θ (now one batched L-BFGS on the executors), score the test split
    (now one mapPartitions), macro the per-node metrics (still `readout_from_proba`,
    on the driver, byte-identical). `readout_sample_frac` deliberately does NOT
    appear: it existed to bound a driver collect that no longer happens, and uniform
    row sampling guts the rare tail the Q1 quartile split reports on.

    `theta_topm > 0` runs the whole arm — fit AND score — on top-m truncated θ. It is
    threaded through both calls from here rather than defaulted separately in each,
    because a mismatch between them is silent (the numbers come out plausible and
    wrong), and this function is the one place that owns both halves of the arm.

    `checkpoint_dir` (normally the RUN dir, alongside `results_partial.json`) makes
    the solve resumable across a driver death — see `_fit_readout_heads`. The file
    name is keyed by `label` (`readout_ckpt_gated_pc.npz`, `..._unsup_gated.npz`,
    `..._dag_head.npz`) because a run fits several arms one after another into the
    same dir and each is a different problem; the fingerprint would refuse a
    cross-arm resume anyway, but sharing one path would make each arm's first
    checkpoint clobber the previous arm's — turning three recoverable solves into
    one. Left None by the A/B gate's internal fit, which is a row-SAMPLED
    perturbation carrying the same label and has no business owning that path.

    Callable with a plain SparkSession + two DataFrames (no argparse), which is what
    makes it testable against `score_arm` on a local Spark fixture."""
    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / f"readout_ckpt_{label or 'arm'}.npz"
    V, b_raw, const, degenerate, _info = _fit_readout_heads(
        train_scored, C, K, l2=l2, gtol=gtol, max_iter=max_iter, history=history,
        depth=depth, label=label, warm_start=warm_start, theta_topm=theta_topm,
        checkpoint_path=ckpt_path, checkpoint_every=checkpoint_every,
        topic_col=topic_col, label_col=label_col, mask_col=mask_col,
        feature_mask=feature_mask)
    if checkpoint_dir is not None:
        # PART 1 keystone: the COMPLETED fit's raw-θ scoring params are the record
        # that conversion_analysis --deciles on scores from (one mapPartitions),
        # instead of re-running this whole solve just to get per-doc scores. Both
        # readout paths that have a run dir — the fit driver's and
        # `gated_pc_readout`'s recovery — reach the fit through here with a
        # `checkpoint_dir`, so this one site persists for both; the sampled A/B
        # fit and the calibration sub-fit (no checkpoint_dir) are correctly
        # excluded. Additive.
        _write_readout_heads(checkpoint_dir, label, V, b_raw, const, degenerate,
                             C, K, theta_topm, W_std=_info.get("W_std"))
    proba, y_te, m_te, doc_keys, elig = _collect_lean_proba(
        test_scored, C, V, b_raw, degenerate=degenerate, const=const,
        score_col=topic_col, label_col=label_col, mask_col=mask_col, id_col=id_col,
        theta_topm=theta_topm, elig_col=elig_col)
    # R5.7: this collect's own doc_keys dedups the detection pool to persons — the
    # distributed path's test split is exactly as multi-doc as the driver path's.
    readout = readout_from_proba(proba, y_te, m_te, C, recall_targets=recall_targets,
                                 fdr_targets=fdr_targets, min_count=min_count,
                                 doc_keys=doc_keys)
    return readout, proba, y_te, m_te, doc_keys, (V, b_raw), elig


def _converge_localized_head(Pi, y, obs, support_cols, C, *, head_l2,
                             head_newton_ridge, n_iters, head_lr=1.0,
                             ridge_mode="relative", fixed_ridge=1.0, intercept=False):
    """Converge a per-node localized ridge-Newton logistic on FROZEN θ — the ENGINE's
    own head math (support-restricted, w=0 off support, Newton solve), run to
    convergence (full Newton from w=0) instead of one damped step/iter against a
    moving θ. Toggles step the engine formulation toward the sklearn oracle so the
    head-formulation LADDER can isolate WHICH difference costs the co-fit head its AUC:

      ridge_mode='relative' (shipped): ridge = head_l2 + head_newton_ridge·mean(diag
        H) — the RELATIVE conditioner VANISHES near separation (H→0 as p→0/1), which
        lets |w| explode (the |w_CK|=273 blowup).
      ridge_mode='fixed':   ridge = `fixed_ridge` (an ABSOLUTE L2, sklearn-style) —
        does not vanish, so it bounds |w|.
      intercept=True: add an UNPENALIZED per-node bias (θ sums to 1, so a rare node's
        marginal must otherwise be fit through ridge-penalized topic weights — the
        bias frees them for the within-sibling contrast).

    `obs` = the per-node observed (closure) mask; each node fits on its own observed
    rows, as in training. Returns (w (C,K), b (C,)); b is 0 when intercept=False.
    Pure numpy (driver-side)."""
    K = Pi.shape[1] if Pi.ndim == 2 and Pi.shape[0] else 0
    w = np.zeros((C, K), dtype=np.float64)
    b = np.zeros(C, dtype=np.float64)
    if Pi.shape[0] == 0:
        return w, b
    for _ in range(int(n_iters)):
        for c in range(C):
            s = support_cols[c]
            if s.size == 0:
                continue
            rows = np.where(obs[:, c] > 0)[0]
            if rows.size == 0:
                continue
            yc = y[rows, c]
            if np.unique(yc).size < 2:
                continue
            X = Pi[np.ix_(rows, s)]
            d = int(s.size)
            # design = [support topics | 1] when an intercept is fit; the bias column
            # is the last coordinate and is left UNPENALIZED in the ridge below.
            Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) if intercept else X
            wc = np.concatenate([w[c, s], [b[c]]]) if intercept else w[c, s].copy()
            z = np.clip(Xa @ wc, -50.0, 50.0)
            p = 1.0 / (1.0 + np.exp(-z))
            g = Xa.T @ (p - yc)
            H = (Xa * (p * (1.0 - p))[:, None]).T @ Xa
            if ridge_mode == "fixed":
                r = float(fixed_ridge)
            else:
                r = head_l2 + head_newton_ridge * (float(np.trace(H[:d, :d])) / d) + 1e-10
            rdiag = np.full(Xa.shape[1], r, dtype=np.float64)
            if intercept:
                rdiag[-1] = 0.0                              # unpenalized bias
            A = H + np.diag(rdiag)
            rhs = g + rdiag * wc
            try:
                delta = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(A, rhs, rcond=None)[0]
            wc = wc - head_lr * delta
            w[c, s] = wc[:d]
            if intercept:
                b[c] = wc[d]
    return w, b


def _localized_head_proba(Pi, w, b=None):
    """Per-node P(node) = σ(w_c·θ + b_c) for the localized head (w is 0 off-support,
    so the full dot IS the localized prediction; b is the optional per-node bias).
    Returns (N, C)."""
    C = w.shape[0]
    if Pi.shape[0] == 0:
        return np.zeros((0, C), dtype=np.float64)
    z = Pi @ w.T
    if b is not None:
        z = z + b[None, :]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def per_node_head_report(w_CK, lay, C, int2cid, count_of, *, dead_eps=1e-6):
    """FAST head-STARVATION diagnostic (exp 0092 hypothesis): the aggregate |w_CK|max
    hides the DISTRIBUTION. What decides the whole-Mondo neutral-PC question is: of the
    C localized heads, how many have `|w_c|≈0` (never trained), and does that track each
    node's POSITIVE count? A localized head shapes topics only through its own `w_c`
    (grad_topics ∝ w_CK); a node with too few positives has a degenerate per-node Fisher
    → `w_c≈0` → no shaping → corr≈0. This is pure driver-side arithmetic on the fitted
    `w_CK` (C,K) + the per-node support `lay.allowed_with_siblings(c)` + the terminal
    positive counts `count_of` — near-free, no Spark, no θ collect.

    Returns a formatted multi-line string: a |w_c| bucket histogram, and — split by
    trained-vs-dead — the median positive count, so 'dead heads are the low-positive
    ones' is read off directly.

    `count_of` may be EMPTY: the Mondo corpus now comes from the bundle cache, and a
    HIT skips the DAG build + power-count that produces those counts (the fit needs
    only `bundle.parent_int`). The |w_c| half of the probe — the part that reads the
    fitted head — is unaffected, so it still prints, with the +count columns
    suppressed and said so rather than shown as a wall of zeros."""
    lines = ["[head-starvation probe] per-node |w_c| on localized support:"]
    have_counts = bool(count_of)
    wnorm = np.empty(C, dtype=np.float64)
    sizes = np.empty(C, dtype=np.int64)
    pos = np.full(C, -1, dtype=np.int64)          # -1 = class node (no positive count)
    for c in range(C):
        sup = np.asarray(sorted(lay.allowed_with_siblings(c)), dtype=int)
        sizes[c] = sup.size
        wnorm[c] = float(np.linalg.norm(np.asarray(w_CK)[c, sup])) if sup.size else 0.0
        cid = int2cid.get(c)
        if have_counts and cid is not None and cid > 0:
            pos[c] = int(count_of.get(cid, 0))
    # |w_c| buckets
    edges = [(0.0, dead_eps, "dead  (≈0)"), (dead_eps, 1e-2, "tiny  (<1e-2)"),
             (1e-2, 1.0, "small (<1)"), (1.0, 1e2, "ok    (<100)"),
             (1e2, np.inf, "big   (≥100)")]
    for lo, hi, lbl in edges:
        sel = (wnorm >= lo) & (wnorm < hi)
        n = int(sel.sum())
        if n == 0:
            lines.append(f"    {lbl:14s} {n:5d} nodes")
            continue
        pos_sel = pos[sel & (pos >= 0)]
        if not have_counts:
            pmed = ""
        elif pos_sel.size:
            pmed = f"median +ct={int(np.median(pos_sel))}"
        else:
            pmed = "class nodes"
        lines.append(f"    {lbl:14s} {n:5d} nodes   {pmed}".rstrip())
    dead = wnorm < dead_eps
    trained = ~dead
    lines.append(f"    -> {int(dead.sum())}/{C} heads DEAD (|w_c|<{dead_eps:g}), "
                 f"{int(trained.sum())} trained")
    tp = pos[trained & (pos >= 0)]
    dp = pos[dead & (pos >= 0)]
    if tp.size and dp.size:
        lines.append(f"    -> terminal +count: trained median={int(np.median(tp))} "
                     f"vs DEAD median={int(np.median(dp))}  "
                     f"(starvation ⇔ dead≪trained)")
    lines.append(f"    -> |w_c|: min={wnorm.min():.2g} median={np.median(wnorm):.2g} "
                 f"max={wnorm.max():.2g}   support size: min={sizes.min()} "
                 f"median={int(np.median(sizes))} max={sizes.max()}")
    if not have_counts:
        lines.append("    -> per-terminal +counts UNAVAILABLE (the corpus came from "
                     "the bundle cache, which skips the Mondo power-count); the "
                     "dead-vs-trained split by rarity needs a cache MISS or a "
                     "different --cache-uri.")
    return "\n".join(lines)


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





def _build_pc_estimator(args, *, weight_y, gated, closure_parents=None):
    """Construct an OnlinePCLDAEstimator for one arm. `gated` injects the gated
    topic engine (gateParent set post-construction, since the Param is a JSON
    string); `closure_parents` selects the DAG-closure head. K comes from the gate
    layout when gated, else --k."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator

    est = OnlinePCLDAEstimator(
        featuresCol="features", labelCol="label", labelMaskCol="labelMask",
        numLabels=args._C, weightY=float(weight_y), k=int(getattr(args, "k", 50)),
        maxIter=args.max_iter, seed=args.seed,
        subsamplingRate=args.subsampling_rate,
        learningOffset=args.tau0, learningDecay=args.kappa,
        gammaShape=args.gamma_shape, caviMaxIter=args.cavi_max_iter,
        caviTol=args.cavi_tol, gradCaviIters=args.grad_cavi_iters,
        topicTrust=args.topic_trust,
        headTrustMove=float(getattr(args, "head_trust_move", 0.0)),
        weightYWarmupIters=args.weight_y_warmup_iters,
        headOptimizer=args.head_optimizer, headLr=args.head_lr,
        headInnerIters=int(getattr(args, "head_inner_iters", 3)),
        headHistoryReset=bool(getattr(args, "head_history_reset", True)),
        headNewtonRidge=args.head_newton_ridge, headL2=args.head_l2,
        headIntercept=bool(getattr(args, "head_intercept", False)),
        headStandardize=bool(getattr(args, "head_standardize", False)),
        optimizeDocConcentration=args.optimize_doc_concentration,
        frontierCol="frontier", gateNBg=args.n_bg, gateTpn=args.tpn,
        # Leaves-first gate schedule (report 2026-10-19; exp 0153/0154): 0 = the
        # single-round gate, byte-identical. Fit-only — no cache key moves.
        gateWarm=int(getattr(args, "gate_warm", 0) or 0),
        gateOpenSeed=True,
        localizeHead=bool(getattr(args, "localize_head", False)),
        headSupport=str(getattr(args, "head_support", "siblings")),
        # Spectral (anchor-word) lambda seed for the gated engine (insight 0079
        # deep-node flat-start remedy). Default "random" preserves prior behavior;
        # getattr guards a manifest/older-driver that never set these.
        init=str(getattr(args, "init", "random")),
        spectralMethod=str(getattr(args, "spectral_method", "auto")),
        spectralMaxVocab=int(getattr(args, "spectral_max_vocab", 8000)),
        spectralD=int(getattr(args, "spectral_d", 0)),
        spectralMarginalFloor=str(getattr(args, "spectral_marginal_floor", "none")),
        spectralMinDocFreq=int(getattr(args, "spectral_min_doc_freq", 5)),
        spectralBgAnchors=int(getattr(args, "spectral_bg_anchors", 0) or 0),
        anchorScope=str(getattr(args, "anchor_scope", "closure")),
        spectralTopoOrder=str(getattr(args, "spectral_topo_order", "forward")),
        countTransform=str(getattr(args, "count_transform", "none")),
        alphaInit=str(getattr(args, "alpha_init", "uniform")),
        # Profile-eta word-side prior (plan WP-3): the four Params are the
        # provenance record (path + knobs); the boost itself is built
        # driver-side (profile_eta.build_profile_eta_boost, from the TSV +
        # bundle meta) and handed over via setEtaBoost below — the spectral
        # precedent's split between "what the estimator records" and "what the
        # driver computes", except the data cannot be derived from the RDD so
        # it arrives pre-built (plan D4). getattr guards older namespaces.
        profileEta=str(getattr(args, "profile_eta", "") or ""),
        profileEtaStrength=float(getattr(args, "profile_eta_strength", 1.0)),
        profileEtaTopics=int(getattr(args, "profile_eta_topics", 1)),
        profileEtaMinCoverage=float(
            getattr(args, "profile_eta_min_coverage", 0.0)),
    )
    # Multi-domain: feed per-domain feature columns (features_0..) so the gated
    # engine carries a per-domain lambda and the topic correction scatters per
    # domain. Domain widths are read from the first row (no explicit domainBounds).
    dom_cols = getattr(args, "_domain_cols", None)
    if dom_cols:
        est._set(featuresCols=dom_cols)
    dc = getattr(args, "doc_concentration", None)
    if dc is not None and dc > 0:
        # scalar Dirichlet alpha for the gated doc-topic prior. The default 1/K is razor-
        # small at whole-Mondo K and collapses theta so the shaping-gradient CAVI Jacobian
        # underflows (no PC shaping); ~0.5 lifts it out of the collapse regime.
        est._set(docConcentration=[float(dc)])
    if gated:
        est.setGateParent(args._parent_int)      # JSON-encodes the DAG map
        gsp = getattr(args, "_gate_seed_profile", None)
        if gsp:
            est.setGateSeedProfile(gsp)
        if bool(getattr(args, "gate_seed_leaves", False)):
            est.setParams(gateSeedLeaves=True)
        if str(getattr(args, "gate_seed_mode", "subtree")) == "balanced":
            est.setParams(gateSeedMode="balanced")
            est.setGateSeedOwnCodes(getattr(args, "_gate_seed_own", None) or {})
    if closure_parents is not None:
        est.setClosureParents(closure_parents)
    return est


# --------------------------------------------------------------------------- #
# Multi-domain / Mondo corpus seam (shared by the fit and gated_pc_readout).   #
# --------------------------------------------------------------------------- #
# The `dag_source` values a manifest may carry. Only `mondo_native` is built since
# the 2026-10 cleanup (`_refuse_removed_spec_options`); `gated_pc_readout` imports
# this to recognize a Mondo run's manifest.
_MONDO_DAG_SOURCES = ("mondo", "mondo_native")

# The multi-domain assembler's kwargs, derived from a corpus SPEC — the dict the
# fit writes into `manifest["corpus_manifest"]` and the re-readout reads back out.
# One derivation, two callers, so a re-readout's cache key cannot drift from the
# key the fit stored the bundle under.
#
# `doc_spec` is a corpus-identity field that is NOT in this tuple, on purpose:
# every name here is forwarded to the assembler, and neither assembler accepts a
# doc-spec argument (both hard-code `PatientCohortDocSpec` internally). It is a
# key-only input and travels in `key_extra` instead — see `doc_spec_identity`.
_SPEC_ASSEMBLY_KEYS = (
    "disease", "cdr", "billing", "person_mod", "min_n", "holdout_frac",
    "vocab_size", "min_df", "min_patient_count", "n_bg", "tpn", "doc_min_length",
    "strip_mode", "lookback_days", "label_window_days", "label_mask_mode",
    "index_mode",
)


def doc_spec_identity() -> str:
    """The DOC-UNIT identity token this driver's corpora are assembled under.

    Read off the class rather than written down, so that swapping the driver's
    doc spec (the `EpisodeDocSpec` of exp 0111, say) moves the token — and
    therefore the bundle cache key — without anyone having to remember to. That
    is the whole content of the fix: `doc_spec` is hard-coded in two places
    (`multi_domain.py:408`, and the provider construction in `mondo_assemble_fn`
    below) and was absent from every cache key, so a driver-side doc-unit change
    would have produced a DIFFERENT corpus under a BYTE-IDENTICAL key — silent
    cache poisoning, not a rebuild cost (audit seam 4; spec R5.3, which requires
    this closed before any doc-unit work starts).

    `min_doc_length`, the spec's only other identity-bearing parameter, is
    already folded into the key as `doc_min_length`, so the class name is the
    whole remaining identity. `compute_bundle_cache_key` folds this only when it
    differs from `DEFAULT_DOC_SPEC`, which is why closing the hole moves no
    existing key."""
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    return PatientCohortDocSpec().name


def multidomain_corpus_spec(args, extra_domains) -> dict:
    """The corpus_manifest block for a native-Mondo multi-domain fit.

    Everything the bundle's cache key needs AND everything re-assembling it from
    scratch needs, in one dict — including `billing` and the Mondo build inputs,
    which a post-hoc rebuild cannot invent. `min_n` is 0 and `index_mode` is
    `population` (the DAG arrives already powered; there is no single disease).

    The removed options keep their OFF values in the spec (`dag_collapse`,
    `label_set`, `preindex_closure`), so a new spec has the same shape — and the
    same cache key — as every native spec written before the 2026-10 cleanup."""
    return {
        "dag_source": args.dag_source,
        "disease": args.disease, "cdr": args.cdr, "billing": args.billing,
        "source_table": args.source_table,
        "extra_domains": list(extra_domains),
        "index_mode": "population",
        "person_mod": args.person_mod, "vocab_size": args.vocab_size,
        "min_df": args.min_df, "min_patient_count": args.min_patient_count,
        "doc_min_length": args.doc_min_length,
        "min_n": 0,
        "holdout_frac": args.holdout_frac, "n_bg": args.n_bg, "tpn": args.tpn,
        "strip_mode": args.strip_mode, "lookback_days": args.lookback_days,
        "label_window_days": args.label_window_days,
        "label_mask_mode": args.label_mask_mode, "emit_labels": True,
        "window_mode": args.window_mode,
        "prior_obs_days": args.prior_obs_days, "window_days": args.window_days,
        "mondo_version": args.mondo_version,
        "mondo_branch": args.mondo_branch or "",
        "min_positives": args.min_positives,
        "mondo_cache_dir": args.mondo_cache_dir,
        "label_set": "", "label_set_path": "",
        "dag_collapse": False,
        # The doc unit (R5.3), a cache-key input.
        "doc_spec": doc_spec_identity(),
        # Label redesign R2 (exp 0146): the topic GATE read from all-ever
        # attestation. Present ONLY when asked, so a label-year bundle keys
        # byte-identically to before.
        **({"gate_attestation": "ever"}
           if str(getattr(args, "gate_attestation", "label")) == "ever" else {}),
        "preindex_closure": False,
    }


# Spec options the 2026-10 cleanup removed (commit fff7bf4067 has them). A spec
# that asks for one is refused rather than keyed as if it had not: silently
# dropping a key fold would let a different corpus land under this key.
def _refuse_removed_spec_options(spec):
    removed = []
    if str(spec.get("dag_source", "snomed")) != "mondo_native":
        removed.append(f"dag_source={spec.get('dag_source', 'snomed')!r}")
    if str(spec.get("index_mode")) == "external" or spec.get("episode_sampling"):
        removed.append("an episode/random index arm")
    if spec.get("gate_frontier_mode"):
        removed.append("gate_frontier_mode")
    for opt in ("dag_collapse", "label_set", "preindex_closure"):
        if spec.get(opt):
            removed.append(opt)
    if removed:
        raise ValueError(
            "corpus spec asks for options removed in the 2026-10 cleanup ("
            + ", ".join(removed) + "); re-read this run from commit fff7bf4067")


def _multidomain_params(spec):
    """`(assembly_params, key_extra)` for one native-Mondo corpus spec.

    `key_extra` carries the identity markers that are NOT assembler kwargs — the
    `multidomain` / `mondo` / `mondo_native` flags, the Mondo build inputs, the doc
    unit and the gate attestation — which is what folds them into the cache key."""
    _refuse_removed_spec_options(spec)
    assembly = {k: spec[k] for k in _SPEC_ASSEMBLY_KEYS}
    assembly["extra_domains"] = tuple(spec.get("extra_domains") or ())
    assembly["emit_labels"] = True
    # `doc_spec` is a key-only identity (the assembler builds its own spec and
    # takes no such kwarg), so it rides key_extra.
    key_extra = {"multidomain": True,
                 "doc_spec": str(spec.get("doc_spec") or doc_spec_identity())}
    # Label redesign R2: the all-ever train gate, folded only when on.
    if spec.get("gate_attestation") and str(spec["gate_attestation"]) != "label":
        key_extra["gate_attestation"] = str(spec["gate_attestation"])
    from mondo_native_dag import MONDO_NATIVE_VERSION
    key_extra.update(mondo=True, mondo_version=spec["mondo_version"],
                     mondo_branch=spec.get("mondo_branch") or "",
                     min_positives=spec["min_positives"])
    key_extra.update(mondo_native=True, mondo_native_version=MONDO_NATIVE_VERSION)
    return assembly, key_extra


def multidomain_cache_key(spec) -> str:
    """The bundle cache key for a multi-domain / Mondo corpus spec."""
    from _case_finding_cache import _KEY_PARAM_NAMES, compute_bundle_cache_key
    assembly, key_extra = _multidomain_params(spec)
    key_params = {k: assembly[k] for k in _KEY_PARAM_NAMES if k in assembly}
    key_params.update(key_extra)
    return compute_bundle_cache_key(**key_params)


def _gate_ever_attested(docs, events, provider):
    """Per TRAIN document, the label nodes its person attests ANYWHERE in the record.

    `docs` = the bundle frame (`person_id`, `doc_id`); `events` = the full-history
    condition frame (`person_id`, `concept_id`, ...). Each event inherits its
    person's doc(s) — `source_cohort` is the first `:` component of the population
    doc_id (`PatientCohortDocSpec`) — and goes through the SAME `provider` the label
    used (same code map, same most-specific reduction on the flat ribbon), so gate
    and label differ only in WHEN, never in HOW a code becomes a node. A doc whose
    person attests nothing keeps a present, empty set."""
    from pyspark.sql import functions as F
    roster = docs.select("person_id", "doc_id").distinct().withColumn(
        "source_cohort", F.element_at(F.split(F.col("doc_id"), ":"), 1))
    ev = events.select("person_id", "concept_id").join(
        roster.select("person_id", "source_cohort").distinct(), "person_id")
    att = provider(ev).select("doc_id", "attested_cids")
    return (roster.join(att, "doc_id", "left")
            .withColumn("attested_cids", F.coalesce(
                F.col("attested_cids"), F.array().cast("array<bigint>"))))


def _gate_count_receipt(before, after):
    """Pooled R2.3 receipt: gated (doc, node) cells and gated docs, label vs ever,
    and the median per-node ratio of gated train docs."""
    from pyspark.sql import functions as F

    def per_node(df):
        return {int(r["n"]): int(r["c"]) for r in
                df.select(F.explode("frontier").alias("n")).groupBy("n")
                  .agg(F.count("*").alias("c")).collect()}
    b, a = per_node(before), per_node(after)
    ratios = sorted(a.get(n, 0) / c for n, c in b.items() if c > 0)
    return {"cells_label": sum(b.values()), "cells_ever": sum(a.values()),
            "docs_gated_label": before.where(F.size("frontier") > 0).count(),
            "docs_gated_ever": after.where(F.size("frontier") > 0).count(),
            "median_node_ratio": (ratios[len(ratios) // 2] if ratios else float("nan")),
            "n_nodes": len(b)}


def _attach_gate_ever(spark, bundle, *, provider, before_dag, cdr, billing,
                      person_mod, n_bg, tpn, _events=None):
    """Label redesign R2 (spec 2026-10-13): overwrite each TRAIN document's `frontier`
    with the roll-up of its person's ALL-EVER attestations; label/labelMask and the
    test split untouched. MISS-only, inside the assemble closure, for the reason
    `_attach_gate_frontier` gives (`before_dag` exists only there). One full-history
    `condition_era` read (the sidecar's load; ~70 s on 0137's corpus). The roll-up
    and the join integrity are `attach_frontiers` and `_overwrite_frontier`, called
    as-is.

    Why the future may gate a TRAIN document: the gate shapes topics on the train
    split only; test documents are folded in ungated, so no test outcome reaches
    them (spec 2026-10-13, Why)."""
    from spark_vi.models.topic.dag_placement import DagLayout
    from charmpheno.omop.case_finding_assembly import attach_frontiers

    keep = set(bundle.cid2int)
    lay = DagLayout(bundle.parent_int, n_bg=n_bg, tpn=tpn)
    events = _events
    if events is None:
        from charmpheno.omop import load_omop_bigquery
        events = load_omop_bigquery(spark=spark, cdr_dataset=cdr,
                                    billing_project=billing,
                                    person_sample_mod=person_mod,
                                    source_table="condition_era")
    before = bundle.train_df
    attested = _gate_ever_attested(before, events, provider)
    gate_fr = attach_frontiers(attested, before_dag, keep, bundle.cid2int, lay)
    after = _overwrite_frontier(before, gate_fr).cache()
    rec = _gate_count_receipt(before, after)
    print(f"[gate-ever] train gate, label year -> ever: gated (doc, node) cells "
          f"{rec['cells_label']:,} -> {rec['cells_ever']:,}; docs with any gate "
          f"{rec['docs_gated_label']:,} -> {rec['docs_gated_ever']:,}; median "
          f"per-node ratio of gated docs {rec['median_node_ratio']:.2f} over "
          f"{rec['n_nodes']} nodes", flush=True)
    bundle.train_df = after
    bundle.gate_ever_receipt = rec
    return bundle


def _overwrite_frontier(df, gate_frontier):
    """Overwrite `df`'s `frontier` with `gate_frontier`'s, joined on the doc key,
    refusing a missing / duplicate / mismatched join LOUDLY (WP-D3 join integrity).

    `gate_frontier` carries `(person_id, doc_id, frontier)`, one row per document.
    The guards, in order: (1) the gate frame must be UNIQUE on (person_id, doc_id) —
    an ambiguous frontier is a raise, never a silent pick; (2) a LEFT join must not
    change the document count — a duplicate gate key that inflated it is a raise;
    (3) NO document may come back without a gate — a missing join surfaces as a NULL
    (distinct from an EMPTY gate's present `[]`) and is a hard error, never an
    empty-/stale-frontier fallback. Only `frontier` is replaced; every other column
    (`label`/`labelMask`/`features`/`episode_no`/…) rides through untouched."""
    from pyspark.sql import functions as F

    gate = gate_frontier.select(
        "person_id", "doc_id",
        F.col("frontier").alias("_gate_frontier")).cache()
    try:
        n_gate = gate.count()
        n_keys = gate.select("person_id", "doc_id").distinct().count()
        if n_gate != n_keys:
            raise ValueError(
                f"exp 0111 D13 gate: {n_gate - n_keys} duplicate (person_id, "
                "doc_id) rows in the gate frame — refusing an ambiguous frontier "
                "overwrite (one frontier per document is required).")
        n_before = df.count()
        joined = (df.drop("frontier")
                  .join(gate, on=["person_id", "doc_id"], how="left"))
        n_after = joined.count()
        if n_after != n_before:
            raise ValueError(
                f"exp 0111 D13 gate: the gate join changed the document count "
                f"({n_before} -> {n_after}) — a duplicate gate key inflated the "
                "bundle; refusing.")
        n_missing = joined.where(F.col("_gate_frontier").isNull()).count()
        if n_missing:
            raise ValueError(
                f"exp 0111 D13 gate: {n_missing} bundle document(s) received no "
                "gate frontier — a missing join is a hard error, never an "
                "empty-frontier fallback.")
        return (joined.withColumn("frontier", F.col("_gate_frontier"))
                .drop("_gate_frontier"))
    finally:
        gate.unpersist()


def mondo_assemble_fn(spec, *, on_inputs=None, _build_inputs=None, _assemble=None,
                      cache_uri=None):
    """A MISS-ONLY assembler for the native Mondo corpus: build the DAG, THEN assemble.

    The native label DAG costs a whole-Mondo -> OMOP mapping, closure-support
    powering and a code climb — minutes of BigQuery — and the fit consumes NONE of
    it directly: the DagLayout comes from `bundle.parent_int`, and the frontier is
    baked into the cached `frontier`/`label` columns. So the build lives inside the
    assemble seam, where `load_or_build_case_finding_bundle` reaches it only after
    `try_load` has missed. A HIT pays for none of it.

    `mondo_native_dag` builds the label DAG out of Mondo's own is-a graph (closure-
    support powering, induced Hasse, splice) and hands back a per-code attestation
    map; the assembler takes it through the `before_dag` / `attested_provider` pair.
    `spec["gate_attestation"] == "ever"` (exp 0146) then overwrites each TRAIN
    document's gate with its person's all-ever attestations.

    `on_inputs(count_of=..., terminal_cids=..., reduced=...)` hands back the build
    by-products; it is NOT called on a HIT.

    The anchor-Mondo / SNOMED fronts, the episode index, the pre-index closure and
    the fit-time label set were removed in the 2026-10 cleanup (commit fff7bf4067
    has them); `_multidomain_params` refuses a spec that asks for one.
    """
    def _assemble_mondo_native(spark, **assembly_params):
        from charmpheno.omop.doc_spec import PatientCohortDocSpec
        from charmpheno.omop.multi_domain import (
            assemble_multidomain_case_finding_corpus)
        from mondo_native_dag import (
            build_mondo_native_fit_inputs, format_native_build_report,
            format_native_powering_report, make_mondo_native_attested_provider)
        build = _build_inputs or build_mondo_native_fit_inputs
        assemble = _assemble or assemble_multidomain_case_finding_corpus
        branch = spec.get("mondo_branch") or ""
        with _phase(f"build native Mondo label DAG (branch={branch or 'ALL'})"):
            before_dag, code_map_sdf, kept_cids, support_of, stats = build(
                spark, cdr=spec["cdr"], billing=spec["billing"],
                mondo_version=spec["mondo_version"],
                mondo_cache_dir=spec.get("mondo_cache_dir") or "data/mondo",
                min_positives=spec["min_positives"],
                branch_root=(branch or None),
                label_set=None, label_set_name="")
            provider = make_mondo_native_attested_provider(
                code_map_sdf, doc_spec=PatientCohortDocSpec(),
                ancestor_pairs=stats.get("flat_ancestor_pairs"))
            # BOTH lines before any fit: C and K are measured here, not guessed.
            print(format_native_powering_report(stats), flush=True)
            print(format_native_build_report(stats), flush=True)
            if on_inputs is not None:
                # `count_of` is the per-node CLOSURE support, `terminal_cids` the
                # final node set.
                on_inputs(count_of=support_of, terminal_cids=kept_cids,
                          reduced={"n_classes": stats["n_final_nodes"]
                                   - stats["n_coded_kept"], "native": stats})
        bundle = assemble(spark, before_dag=before_dag,
                          attested_provider=provider, index_df=None,
                          doc_spec=None, **assembly_params)
        if spec.get("gate_attestation") == "ever":
            with _phase("label redesign R2: all-ever topic gate (train, MISS-only)"):
                bundle = _attach_gate_ever(
                    spark, bundle, provider=provider, before_dag=before_dag,
                    cdr=spec["cdr"], billing=spec["billing"],
                    person_mod=assembly_params["person_mod"],
                    n_bg=int(assembly_params["n_bg"]), tpn=int(assembly_params["tpn"]))
        # WP-C' (exp 0136): the `(std_cid, node_cid)` attestation map rides the
        # bundle as an ATTRIBUTE, is persisted in the cache meta, and is written to
        # the run dir by the driver (the census's own-code criterion). Ontology ids
        # and standard concept ids only; small (one row per attesting code).
        _collect = getattr(code_map_sdf, "collect", None)   # a test stub has none
        if _collect is not None:
            bundle.native_code_map = sorted(
                (int(r["std_cid"]), int(r["node_cid"])) for r in _collect())
        return bundle

    return _assemble_mondo_native


def multidomain_load_or_build(spark, spec, *, cache_uri=None, on_inputs=None,
                              _build_inputs=None, _assemble=None):
    """Cached multi-domain / Mondo corpus: HIT reloads, MISS assembles + writes through.

    The single entry point for both the fit driver and the standalone re-readout —
    same key, same assembler, same write-through — so a fit's bundle is findable by
    a later readout and a readout's rebuild is reusable by a later fit.
    """
    from _case_finding_cache import load_or_build_case_finding_bundle

    assembly, key_extra = _multidomain_params(spec)
    assemble_fn = mondo_assemble_fn(spec, on_inputs=on_inputs,
                                    _build_inputs=_build_inputs,
                                    _assemble=_assemble, cache_uri=cache_uri)
    return load_or_build_case_finding_bundle(
        spark, cache_uri=cache_uri, _assemble_fn=assemble_fn,
        _key_extra=key_extra, **assembly)


def parse_args(argv=None):
    # allow_abbrev=False: a removed flag must fail loudly, not prefix-match a
    # surviving one (`--k 4` would otherwise parse as `--kappa 4`).
    p = argparse.ArgumentParser(
        description="Gated-LDA fit on the Mondo label DAG; ends at the saved fit "
                    "(every readout is gated_pc_readout.py).", allow_abbrev=False)
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
    p.add_argument("--localize-head", action="store_true",
                   help="LOCALIZED head: each node's logistic reads only its topic "
                        "support (gated block + ancestors), not all K — the whole-Mondo "
                        "scale fix (insight 0071). Only affects newton head.")
    p.add_argument("--head-support",
                   choices=["siblings", "path_cousins", "path_cousins_kids"],
                   default="siblings",
                   help="localized-head support neighborhood: 'siblings' (closure + immediate "
                        "siblings, default); 'path_cousins' (also the siblings of every ancestor "
                        "on the root-path); 'path_cousins_kids' (also v's own children's blocks, "
                        "the subtype signal). All bounded; exact Newton kept.")
    # Spectral (anchor-word) lambda init for the gated engine (insight 0079 deep-node
    # flat-start remedy). Default 'random' preserves prior behavior; 'spectral' seeds
    # each node's block from block-aligned anchor recovery (gated_init.py, Arora 2013).
    p.add_argument("--init", choices=["random", "spectral"], default="random",
                   help="gated-engine lambda init: 'random' (default) or 'spectral' "
                        "(block-aligned anchor-word seed — breaks the deep-node "
                        "flat-start starvation trap, insight 0079). Requires the gate.")
    p.add_argument("--spectral-method", choices=["auto", "dense", "scalable"],
                   default="auto",
                   help="spectral routing: auto (dense below --spectral-max-vocab, "
                        "scalable at/above), dense (driver V×V, exact, small V), or "
                        "scalable (distributed random-projection, ADR 0032).")
    p.add_argument("--spectral-max-vocab", type=int, default=8000,
                   help="'auto' threshold: dense below this total vocab, scalable "
                        "at/above (a concatenated multi-domain V crosses it easily).")
    p.add_argument("--spectral-d", type=int, default=0,
                   help="random-projection dim for scalable init (0 = auto).")
    p.add_argument("--spectral-min-doc-freq", type=int, default=5,
                   help="min within-group doc frequency for a scalable anchor candidate.")
    p.add_argument("--spectral-bg-anchors", type=int, default=0,
                   help="how many of the n_bg background topics the spectral seed "
                        "ANCHORS (0 = all, the historical behaviour). The rest are "
                        "seeded by the random Gamma init: a wide shared background "
                        "(exp 0134, n_bg=1200) is not anchorable — the greedy and the "
                        "per-word recovery both scale with the anchor count — and "
                        "0132 showed a flat background forms its strata from a random "
                        "start. The anchored few are the deflation seeds (0133's 8).")
    p.add_argument("--anchor-scope", choices=["closure", "frontier"], default="closure",
                   help="docs feeding each node's anchor set: 'closure' (default) or "
                        "'frontier' (node only from docs where it is most-specific — "
                        "stops ancestors/background stealing a descendant's anchor).")
    p.add_argument("--spectral-topo-order", choices=["forward", "reverse"],
                   default="forward",
                   help="deflation order: 'forward' (ancestors-first, node = increment "
                        "over ancestors) or 'reverse' (leaves-first, node = residual "
                        "after descendants).")
    p.add_argument("--alpha-init", choices=["uniform", "equalized"], default="uniform",
                   help="initial doc-concentration policy for the gated engine: "
                        "'uniform' (doc_concentration everywhere) or 'equalized' "
                        "(children-first: equal TOTAL prior pseudo-count per topic "
                        "block across the corpus, alpha ∝ 1/N_docs_seeing_block, "
                        "rescaled to the doc_concentration mean; exp 0121). With "
                        "--optimize-doc-concentration the empirical-Bayes step takes "
                        "over from it — an init, not a strength knob.")
    p.add_argument("--count-transform", choices=["none", "binary", "log1p"],
                   default="none",
                   help="per-token BOW count transform before seed+fit: 'none' (raw "
                        "counts), 'binary' (min(count,1) = per-doc presence; de-biases "
                        "anchors/evidence off utilization volume, insight 0077 extended "
                        "to all domains), or 'log1p' (softer damping). In-memory; the "
                        "cached bundle stays raw (no cache-key change).")
    # Profile-eta word-side prior (plan 2026-09-06 D1-D5, exp 0116). The TSV is
    # hpoa_stage2_probe's --emit-eta output (concept ids, bundle-agnostic,
    # WORKSPACE-INTERNAL — its coverage column is train-derived); the driver
    # maps it into the bundle's condition vocab / gate blocks and hands the
    # built boost to the estimator. A fit parameter only: no bundle/corpus
    # cache-key change. Default '' preserves prior behavior byte-identically.
    p.add_argument("--spectral-marginal-floor", choices=["none", "guided", "all"],
                   default="none",
                   help="scalable spectral candidate floor (exps 0125/0128): 'guided' "
                        "= profile candidates must be at least as common as the node's "
                        "average word; 'all' = every candidate must. Derived, no knob.")
    p.add_argument("--dag-source", choices=["mondo_native"], default="mondo_native",
                   help="the label space: exp 0110's NATIVE Mondo terms, powered by "
                        "is-a CLOSURE support, the DAG Mondo's own hierarchy "
                        "transitively reduced over the kept set (the only source "
                        "since the 2026-10 cleanup).")
    p.add_argument("--mondo-branch", default="",
                   help="mondo: restrict to one body-system Mondo subtree (e.g. "
                        "MONDO:0004995 = cardiovascular disorder) — the Step-A template. "
                        "'' = whole Mondo.")
    p.add_argument("--min-positives", type=int, default=100,
                   help="mondo: keep anchors with >= this many whole-pop patients "
                        "(the K dial; exp 0088 used 100).")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    # exp 0111 (WP-D3 / D13): separate the estimator's GATE from the outcome LABEL.
    # When > 0 (and an --index-arm is set) each document's `frontier` gate is swapped
    # to its [index, index+N-day) PRESENTATION window while `label`/`labelMask` stay
    # the full 365-day frame the assembler baked — the D13 separation. A MISS-only
    # driver post-pass; the gate mode (`gateNd`) folds into the bundle cache key, so
    # a gated bundle is a distinct artifact from the un-gated one. 0 (default) keeps
    # the 365-day frontier as the gate, so every existing key stays byte-identical.
    p.add_argument("--gate-warm", type=int, default=0, metavar="N",
                   help="Leaves-first gate schedule: fit in rounds by node height, N "
                        "iterations per level (leaves first), then the full gate. 0 "
                        "(default) = single round. Each block is seeded when it opens "
                        "(see --gate-no-open-seed). Fit-only: bundle/corpus caches "
                        "unaffected. Report 2026-10-19-leaves-first-gate-sim.")
    p.add_argument("--gate-seed-mode", choices=("subtree", "balanced"),
                   default="subtree",
                   help="With --gate-warm: the open-time seed. 'subtree' (default) = "
                        "the word counts of the documents a block admits; 'balanced' = "
                        "each child's subtree share equally plus the parent's own coded "
                        "documents, descendants' own codes removed (exp 0160).")
    p.add_argument("--gate-seed-leaves", action="store_true",
                   help="With --gate-warm: seed every leaf block at iteration 0 from "
                        "its own documents' word counts (exp 0162). With every node "
                        "block data-seeded, use --init random: the spectral seed's "
                        "per-node scans are then pure cost.")
    p.add_argument("--gate-seed-profile", default="", metavar="TSV",
                   help="With --gate-warm: an external phenotype profile table "
                        "(mondo_id, concept_id, neg; e.g. hpo_profile_marks.py output) "
                        "that REPLACES a profiled node's condition-domain seed (leaves "
                        "at iteration 0, later blocks at opening); unprofiled nodes "
                        "keep the histogram seed. Fit-only.")
    p.add_argument("--gate-attestation", choices=("label", "ever"), default="label",
                   help="Label redesign R2 (spec 2026-10-13): which attestations GATE a "
                        "TRAIN document's topic blocks. 'label' (default, unchanged): X "
                        "coded in the label window. 'ever': X coded anywhere in the "
                        "person's record (full-history condition_era through the same "
                        "code map and provider). label/labelMask and the TEST frontier "
                        "are untouched. Native-Mondo population path only; a distinct "
                        "bundle key.")
    p.add_argument("--optimize-doc-concentration", action="store_true")
    # PC head + SVI
    p.add_argument("--weight-y", type=float, default=50.0,
                   help="PC prediction weight (>0). Hughes ~ tokens/doc; tune on "
                        "validation. The unsup_gated arm always refits at 0.")
    p.add_argument("--head-optimizer", choices=["sgd", "newton", "lbfgs"],
                   default="newton",
                   help="head optimizer; 'newton' is the settled convergent head "
                        "(ADR 0039) — the default here. 'lbfgs' is the matrix-free "
                        "batched-L-BFGS co-fit head (O(C*K) shuffle, no newton "
                        "O(C*K^2) collect — the whole-Mondo strong head); it needs "
                        "the distributed re-scoring provider, injected here.")
    p.add_argument("--head-inner-iters", type=int, default=3,
                   help="head_optimizer=lbfgs: batched-L-BFGS solver passes per outer "
                        "SVI iter (amortized against a moving theta). Inert otherwise.")
    p.add_argument("--head-history-reset", type=lambda s: s.lower() != "false",
                   default=True,
                   help="head_optimizer=lbfgs: reset the L-BFGS (s,y) curvature "
                        "history each outer iter (default true; 'false' carries it). "
                        "Inert otherwise.")
    p.add_argument("--head-lr", type=float, default=0.5)
    p.add_argument("--head-newton-ridge", type=float, default=0.01)
    p.add_argument("--head-l2", type=float, default=1e-3,
                   help="ABSOLUTE ridge on w_CK (= Hughes lambda_w; ADR 0041). "
                        "0.0 BLOWS UP on the separable topics PC creates.")
    p.add_argument("--grad-cavi-iters", type=int, default=20)
    p.add_argument("--topic-trust", type=float, default=0.1)
    p.add_argument("--head-trust-move", type=float, default=0.0,
                   help="scale-free trust-region radius on the supervised lambda "
                        "move: cap the EG step when its relative move exceeds this "
                        "(0.0=off; e.g. 0.03 holds corr_relDlambda in the healthy "
                        "band regardless of weight_y/K — the K-invariant fix for "
                        "the over-drive exp 0099/0118/0119 hit at large K).")
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
    p.add_argument("--fit-save-interval", type=int, default=0,
                   help="checkpoint the fit-only result (gated_pc_result.npz + "
                        "manifest.json, partial='fit-only') every N SVI iters (0 = "
                        "off, only the end-of-fit save). Crash insurance for long "
                        "co-fit runs: a death at iter M leaves iter-M lambda on disk, "
                        "re-scoreable with gated_pc_readout — no resume needed (and "
                        "profile-eta fits cannot resume anyway, D5). Writes are "
                        "atomic; the cost is one npz+manifest dump per interval, so "
                        "keep it modest (e.g. 5).")
    p.add_argument("--num-partitions", type=int, default=0,
                   help="repartition the corpus to this many partitions before "
                        "fitting (0 = leave as-is). Few parquet part-files pin the "
                        "job to ~2 executors under dynamic allocation; set this ≈ "
                        "total cluster executor cores to spread the per-doc autograd "
                        "and demand more executors. Pair with "
                        "CHARM_SPARK_CONF='spark.locality.wait=0s'.")
    p.add_argument("--head-intercept", action="store_true",
                   help="newton head: fit a per-node UNPENALIZED intercept (base rate).")
    p.add_argument("--head-standardize", action="store_true",
                   help="newton head: z-score θ per topic before the logistic (the big "
                        "conditioning lever; requires --head-intercept). Local realistic "
                        "validation: co-fit head 0.55->0.84 and readout Δ+0.29 vs unsup.")
    p.add_argument("--doc-concentration", type=float, default=None,
                   help="scalar Dirichlet alpha for the gated doc-topic prior (default "
                        "1/K). The 1/K default is razor-small at whole-Mondo K (~0.0022) "
                        "and collapses theta so the supervised-shaping CAVI Jacobian "
                        "d(theta)/d(eb) UNDERFLOWS (~1e-90) — PC shaping dies upstream of "
                        "the head. ~0.5 keeps the shaping gradient alive at any "
                        "grad_cavi_iters.")
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main() -> int:
    from spark_vi.models.topic.dag_placement import DagLayout

    args = parse_args()
    configure_logging()
    # The run dir exists from minute zero and every [driver] line is teed to a
    # durable, sanitized driver_log.md inside it — open-append-close per line,
    # so results survive wrapper death, cluster timeout, and summary.md
    # truncation (exp 0103's smoke lost 4h of readout output to exactly that;
    # `gated_pc_report --summary <run>/driver_log.md` digests this file too).
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from _driver_common import install_stdout_tee
    install_stdout_tee(out / "driver_log.md")
    extra_domains = tuple(d for d in args.extra_domains.split(",") if d)
    corpus_spec = None
    with make_spark_session(app_name="gated-pc-fit") as spark:
        # Driver-disk telemetry, IN-BAND (see disk_telemetry's docstring): the
        # fit phase is the other half of the ~100 GB that ADR 0047's destroy
        # fixes do not account for, and a watcher writing to the master's local
        # disk dies with the cluster. Printing from the driver puts the disk
        # history in the persisted job log and in driver_log.md.
        start_disk_telemetry(
            extra_dirs=[d for d in spark.sparkContext.getConf()
                        .get("spark.local.dir", "").split(",") if d],
            log=lambda msg: _cprint(f"[driver] {msg}", flush=True))
        if args.dag_source != "mondo_native":
            raise ValueError(
                f"--dag-source {args.dag_source!r}: the gated fit runs on the native "
                "Mondo label space only (the SNOMED and anchor-Mondo paths were "
                "removed in the 2026-10 cleanup; last at commit fff7bf4067)")
        # The multi-domain native-Mondo corpus (per-domain vocabularies,
        # features_0..N-1), cached through one seam; the DAG build and the code
        # climb live inside the assemble closure, so a cache HIT skips them.
        if args.window_mode != "lookback":
            raise ValueError(
                f"--dag-source {args.dag_source} / --extra-domains requires "
                f"--window-mode lookback (got {args.window_mode})")
        corpus_spec = multidomain_corpus_spec(args, extra_domains)

        def _stash_mondo_inputs(*, count_of, terminal_cids, reduced):
            # Only reached on a cache MISS; the diag-only probe degrades
            # gracefully without these (per_node_head_report).
            args._count_of = count_of

        with _phase(f"assemble MONDO_NATIVE corpus (cached, cond + "
                    f"{list(extra_domains)})"):
            bundle = multidomain_load_or_build(
                spark, corpus_spec, cache_uri=args.cache_uri,
                on_inputs=_stash_mondo_inputs)
            vocab_maps = bundle.vocab_maps
            args._domain_cols = [f"features_{i}" for i in range(len(vocab_maps))]
            args._domain_names = ["condition", *extra_domains]
        _cprint(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)
        _bmp = write_bundle_meta(out, bundle)
        _cprint(f"[driver]   wrote bundle meta -> {_bmp}", flush=True)
        _ncm = getattr(bundle, "native_code_map", None)
        if _ncm:
            # WP-C': <run>/code_map.tsv (std concept id -> Mondo node cid), the
            # census's own-code identity criterion. Present on every native
            # bundle built or reloaded from here on; older cache entries lack it.
            _cmp = write_code_map(out, _ncm)
            _cprint(f"[driver]   wrote native code map: {len(_ncm)} (code, node) "
                  f"pairs over {len({n for _, n in _ncm})} nodes -> {_cmp}",
                  flush=True)
        lay = DagLayout(bundle.parent_int, n_bg=args.n_bg, tpn=args.tpn)
        C = len(bundle.int2cid)               # label heads = engine nodes incl. root
        # Stash on args so the estimator builder can reach them without re-threading.
        args._C = C
        args._parent_int = bundle.parent_int

        # Profile seed for the leaves-first schedule (exp 0157/0158): an external
        # phenotype profile (the marks TSV from hpo_profile_marks.py) replaces a
        # profiled node's condition-domain seed. Fit-only; counts-only stats.
        args._gate_seed_profile = None
        if getattr(args, "gate_seed_profile", ""):
            if int(getattr(args, "gate_warm", 0) or 0) <= 0:
                raise ValueError("--gate-seed-profile requires --gate-warm > 0")
            with _phase("build the profile seed (driver-side)"):
                import pandas as pd
                from mondo_native_dag import mondo_cid
                from profile_eta import build_profile_seed
                gs_df = pd.read_csv(args.gate_seed_profile, sep="\t")
                gs_eid = {}
                for mid in gs_df["mondo_id"].astype(str).unique():
                    try:
                        eid = bundle.cid2int.get(mondo_cid(mid))
                    except ValueError:
                        eid = None
                    if eid is not None:
                        gs_eid[str(mid)] = int(eid)
                gs, gs_stats = build_profile_seed(
                    gs_df, eid_by_mondo=gs_eid, vocab_index_by_concept=vocab_maps[0])
                if not gs:
                    raise ValueError("--gate-seed-profile seeded NO node (none of the "
                                     "profiled nodes has a specific profile concept "
                                     "in this run's vocabulary) — check the TSV")
                args._gate_seed_profile = gs
                args._gate_seed_profile_stats = gs_stats
                _cprint(f"[driver]   profile seed: {json.dumps(gs_stats)}", flush=True)
        # Balanced open-time seed (exp 0160): each node's own attesting codes in the
        # condition vocab, removed from its ancestors' seeds. From the native code
        # map (std_cid, node_cid) pairs; ids only.
        args._gate_seed_own = None
        if str(getattr(args, "gate_seed_mode", "subtree")) == "balanced":
            if int(getattr(args, "gate_warm", 0) or 0) <= 0:
                raise ValueError("--gate-seed-mode balanced requires --gate-warm > 0")
            _ncm = getattr(bundle, "native_code_map", None) or []
            _vm0 = vocab_maps[0]
            own = {}
            for std, node in _ncm:
                eid = bundle.cid2int.get(int(node))
                vi = _vm0.get(int(std), _vm0.get(str(int(std))))
                if eid is not None and vi is not None:
                    own.setdefault(int(eid), set()).add(int(vi))
            args._gate_seed_own = {k: sorted(v) for k, v in own.items()}
            _cprint(f"[driver]   balanced seed: own codes in the condition vocab for "
                    f"{len(own)} node(s) ({sum(len(v) for v in own.values())} codes)",
                    flush=True)
        v_desc = " + ".join(f"{n}:{len(vm)}"
                            for n, vm in zip(args._domain_names, vocab_maps))
        _cprint(f"[driver]   corpus: V=({v_desc}) vocab, "
              f"K={lay.K} gated topics ({args.n_bg} bg + {len(lay.nodes)} nodes x "
              f"{args.tpn} tpn), C={C} label heads", flush=True)
        # PRE-FLIGHT cost profile at the data-build boundary: fan-out / support sizes /
        # head matrix memory+compute (dense vs localized) so a big fit's cost is visible
        # BEFORE compute is committed (esp. the whole-Mondo run — high-fan-out parents
        # inflate the localized support). Logged for every gated fit.
        _v_total = sum(len(vm) for vm in vocab_maps)
        _, _prof = lay.cost_report(C, vocab_size=_v_total,
                                   localized=bool(getattr(args, "localize_head", False)))
        print("\n".join("[cost] " + ln for ln in _prof.splitlines()), flush=True)
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
            _cprint(f"[driver]   repartitioned corpus {before} -> "
                  f"{args.num_partitions} partitions (train+test cached)", flush=True)

        # Every manifest field the FIT determines, built once and reused by both
        # saves below — the early fit-only write and the final authoritative one.
        # Keeping it in one dict is what makes "the same manifest, plus results"
        # true by construction rather than by two lists staying in sync.
        manifest_fields = {
            "model_class": "gated_pc",
            "disease": args.disease, "min_n": 0,
            "strip_mode": args.strip_mode, "label_mask_mode": args.label_mask_mode,
            "window_mode": args.window_mode, "lookback_days": args.lookback_days,
            "label_window_days": args.label_window_days,
            "n_bg": args.n_bg, "tpn": args.tpn, "K": lay.K, "C": C,
            "weight_y": args.weight_y, "head_optimizer": args.head_optimizer,
            "head_l2": args.head_l2, "head_lr": args.head_lr,
            "weight_y_warmup_iters": args.weight_y_warmup_iters,
            "grad_cavi_iters": args.grad_cavi_iters, "topic_trust": args.topic_trust,
            "head_trust_move": float(getattr(args, "head_trust_move", 0.0)),
            "head_inner_iters": int(getattr(args, "head_inner_iters", 3)),
            "head_history_reset": bool(getattr(args, "head_history_reset", True)),
            "subsampling_rate": args.subsampling_rate, "tau0": args.tau0,
            "kappa": args.kappa, "max_iter": args.max_iter,
            "alpha_init": str(getattr(args, "alpha_init", "uniform")),
            "dag_source": args.dag_source,
            "extra_domains": list(extra_domains),
            "domain_names": args._domain_names,
            "domain_vocab_sizes": [len(vm) for vm in vocab_maps],
            "ledger": bundle.ledger,
            # Corpus params — ALL of the bundle cache-key inputs, so a post-hoc
            # gated_pc_readout can recompute the exact key + reload the bundle
            # (doc_min_length + emit_labels are required by the key; recording
            # them here removes the lr_readout-style fragility). On the
            # multi-domain / Mondo paths this block is the corpus SPEC verbatim —
            # the same dict `multidomain_load_or_build` was called with, including
            # `billing` and the Mondo build inputs — so the re-readout can not only
            # recompute the key but REBUILD the bundle from it when the cache is
            # cold (a fresh cluster, a cleared bucket). `min_n`/`index_mode` are the
            # EFFECTIVE assembly values, which on the Mondo path differ from the
            # top-level CLI ones (min_n=0: that DAG is already powered).
            "corpus_manifest": {
                "cdr": args.cdr, "billing": args.billing,
                "cache_uri": args.cache_uri, "source_table": args.source_table,
                "person_mod": args.person_mod, "vocab_size": args.vocab_size,
                "min_df": args.min_df, "min_patient_count": args.min_patient_count,
                "doc_min_length": args.doc_min_length,
                "prior_obs_days": args.prior_obs_days, "window_days": args.window_days,
                "holdout_frac": args.holdout_frac, "emit_labels": True,
                "dag_source": args.dag_source,
                "extra_domains": list(extra_domains),
                "index_mode": (corpus_spec or {}).get("index_mode", "disease"),
                "min_n": corpus_spec["min_n"],
                "disease": args.disease, "n_bg": args.n_bg, "tpn": args.tpn,
                "strip_mode": args.strip_mode, "window_mode": args.window_mode,
                "lookback_days": args.lookback_days,
                "label_window_days": args.label_window_days,
                "label_mask_mode": args.label_mask_mode,
                "mondo_version": (corpus_spec or {}).get("mondo_version", ""),
                "mondo_branch": (corpus_spec or {}).get("mondo_branch", ""),
                "min_positives": (corpus_spec or {}).get("min_positives", 0),
                "mondo_cache_dir": (corpus_spec or {}).get("mondo_cache_dir", ""),
                "dag_collapse": (corpus_spec or {}).get("dag_collapse", False),
                # Spec 2026-10-07 §D1: the ribbon identity (a key input) and the
                # TSV path (a rebuild input); '' on every non-ribbon run.
                "label_set": (corpus_spec or {}).get("label_set", ""),
                "label_set_path": (corpus_spec or {}).get("label_set_path", ""),
                # The doc unit (R5.3). Recorded on BOTH paths — the single-domain
                # assembler hard-codes the same spec — so a re-readout of either
                # recomputes the fit's own key.
                "doc_spec": ((corpus_spec or {}).get("doc_spec")
                             or doc_spec_identity()),
                # E1: whether this run's corpus carries the pre-index closure
                # column. A cache-key input, so a re-readout that reads it back
                # lands on the fit's own bundle — and the WITNESS the census and
                # every incident readout check before touching the column.
                "preindex_closure": (corpus_spec or {}).get(
                    "preindex_closure", False),
                "domain_vocab_sizes": [len(vm) for vm in vocab_maps],
                "int2cid": {str(i): c for i, c in bundle.int2cid.items()},
                "name_by_id": {str(c): n for c, n in bundle.name_by_id.items()}},
        }
        # Init provenance: the spectral seed's settings, so `inspect_topics
        # --digest` can say how a block was seeded without the experiment doc.
        # Added ONLY for spectral fits (random-init manifests stay byte-identical).
        if str(getattr(args, "init", "random")) == "spectral":
            manifest_fields["init"] = {
                "init": "spectral",
                "spectral_method": str(getattr(args, "spectral_method", "auto")),
                "spectral_d": int(getattr(args, "spectral_d", 0)),
                "anchor_scope": str(getattr(args, "anchor_scope", "closure")),
                "spectral_topo_order": str(getattr(args, "spectral_topo_order",
                                                   "forward"))}
        if int(getattr(args, "gate_warm", 0) or 0) > 0:
            # Fit parameter; added only when set (byte-identical manifests otherwise).
            manifest_fields["gate_schedule"] = {
                "warm": int(args.gate_warm),
                "open_seed": True}
            manifest_fields["gate_schedule"]["seed_mode"] = str(
                getattr(args, "gate_seed_mode", "subtree"))
            manifest_fields["gate_schedule"]["seed_leaves"] = bool(
                getattr(args, "gate_seed_leaves", False))
            if getattr(args, "_gate_seed_profile", None):
                manifest_fields["gate_schedule"]["seed_profile"] = {
                    "path": args.gate_seed_profile,
                    **getattr(args, "_gate_seed_profile_stats", {})}
        if int(getattr(args, "spectral_bg_anchors", 0) or 0) > 0:
            manifest_fields["spectral_bg_anchors"] = int(args.spectral_bg_anchors)


        with _phase(f"gated_pc fit (weightY={args.weight_y}, K={lay.K})"):
            pc_est = _build_pc_estimator(args, weight_y=args.weight_y, gated=True)
            # Per-iteration hooks, composed: the optional eval logger (eval_every)
            # and the optional fit checkpoint (fit_save_interval). Both share the
            # estimator's single on_iteration seam, so build a list and fan out.
            _on_iters = []
            if args.fit_save_interval > 0 and out is not None:
                def _fit_ckpt(it, gp, _elbo, _out=out, _C=C, _mf=manifest_fields,
                              _every=int(args.fit_save_interval)):
                    # VIRunner calls this after each iter with that iter's params;
                    # dump the fit-only result every _every iters so a crash leaves
                    # a readout-able lambda. Atomic (see _save_fit). Never mutates gp.
                    if it % _every == 0:
                        _save_fit(_out, gp, _C, _mf, partial="fit-only")
                        _cprint(f"[driver]   fit checkpoint @ iter {it} "
                              "(re-scoreable with gated_pc_readout)", flush=True)
                _on_iters.append(_fit_ckpt)
            if _on_iters:
                def _on_iter_fanout(it, gp, elbo, _cbs=tuple(_on_iters)):
                    for _cb in _cbs:
                        _cb(it, gp, elbo)
                pc_est.setOnIteration(_on_iter_fanout)
            # lbfgs co-fit head: the engine's M-step re-scores the CURRENT θ each
            # outer iter through a DISTRIBUTED provider (the same treeAggregate seam
            # the frozen-θ readout uses, re-derived per iter because θ moves). We
            # build it driver-side and inject via the estimator's factory hook; the
            # engine (spark-vi) never sees Spark. The holder lets us close() the last
            # outer iter's persisted scored df after the fit. Inert for sgd/newton.
            _cofit_provider = {}
            if args.head_optimizer == "lbfgs":
                from spark_vi.models.topic.batched_lr import (
                    fold_standardization, standardized_grad_from_raw)

                def _cofit_factory(ctx, _hold=_cofit_provider):
                    engine = ctx["engine"]
                    prov = _dr.make_cofit_head_stats_provider(
                        ctx["dataset"], ctx["C"], engine.K,
                        grad_cavi_iters=engine.grad_cavi_iters,
                        head_standardize=engine.head_standardize,
                        head_std_floor=engine.head_std_floor,
                        expElogbeta_fn=engine._expElogbeta_from_lambda,
                        fold_standardization=fold_standardization,
                        standardized_grad_from_raw=standardized_grad_from_raw,
                        features_col=ctx["features_col"],
                        features_cols=ctx["features_cols"],
                        domain_sizes=ctx["domain_sizes"],
                        label_col=ctx["label_col"], mask_col=ctx["label_mask_col"])
                    _hold["provider"] = prov
                    return prov

                pc_est.setHeadStatsProviderFactory(_cofit_factory)
            try:
                pc_model = pc_est.fit(bundle.train_df)
            finally:
                if _cofit_provider.get("provider") is not None:
                    _cofit_provider["provider"].close()
            # Per-node spectral anchors -> <run>/spectral_anchors.json (vocab
            # indices + ontology names only; no patient data). Written for every
            # spectral fit, guided or not, so a block's anchor word is on record
            # (exp 0124 had to infer DCM's anchor from its recovered topic).
            _ast = getattr(pc_est, "_anchor_stats", None)
            if _ast:
                from spark_vi.models.topic.gated_init import anchor_dump_payload
                _payload = anchor_dump_payload(
                    _ast, int2cid=bundle.int2cid, name_by_id=bundle.name_by_id)
                Path(out).mkdir(parents=True, exist_ok=True)   # out = the run dir
                _ap = Path(out) / "spectral_anchors.json"
                with open(_ap, "w") as _fh:
                    # Anchor ids index the seed's vocabulary: for a multi-domain
                    # fit the CONCATENATED [domain 0; domain 1; ...] space.
                    json.dump({"vocab_domain": 0,
                               "vocab_space": ("concat" if len(vocab_maps) > 1
                                               else "domain0"),
                               "domain_vocab_sizes": [len(vm) for vm in vocab_maps],
                               "nodes": _payload}, _fh)
                _cprint(f"[driver]   wrote spectral anchors for {len(_payload)} node(s) "
                        f"-> {_ap}", flush=True)
            # EARLY SAVE, before any readout work touches the cluster: the fit is
            # the hours-long unrepeatable half and the readout is where runs die,
            # so the model reaches durable storage the moment it exists. The final
            # save overwrites these same two paths with the full record; until it
            # does, `partial="fit-only"` says so and `gated_pc_readout` can score
            # this run from the npz alone.
            _save_fit(out, pc_model.result.global_params, C, manifest_fields,
                      partial="fit-only")
            _cprint(f"[driver]   saved FIT-ONLY result to {out} (readout pending; "
                  "re-scoreable with gated_pc_readout)", flush=True)
            # The fit ends here (cleanup 2026-10): every readout and evaluation is
            # `gated_pc_readout.py` against this saved fit. Per-node per-domain λ
            # mass needs no readout, so it is still reported with the fit.
            gp = pc_model.result.global_params
            if extra_domains and isinstance(gp["lambda"], dict):
                print(format_per_node_domain_mass(
                    per_node_domain_mass(gp["lambda"], lay, args._domain_names),
                    args._domain_names, bundle.int2cid, bundle.name_by_id), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
