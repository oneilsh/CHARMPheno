"""Cloud fit driver for the DAG-gated LDA on the Mondo label DAG (model_class gated_pc).

Assembles (or reloads from cache) the multi-domain native-Mondo corpus, fits
`DagGatedLDAEstimator` (ADR 0050: one estimator, named for its DAG gate; the
supervised head and the DAG spectral init were removed), and saves the fit:
`<run>/gated_pc_result.npz` (lambda / lambda_0.., alpha) + `manifest.json`. The fit
ends at that save; every readout and evaluation is `gated_pc_readout.py` against
it. The file and model-class names keep "gated_pc" for now (renaming them is ADR
0051's business).

The per-node (label, labelMask) the readout scores against is emitted by the
Step-A adapter (case_finding_assembly.emit_labels): label[c]=1 iff node c is in the
is-a closure of the doc's frontier; C = len(int2cid).

CANNOT RUN HERE: every fact table is read from the workspace CDR via the
spark-bigquery connector, so main() only executes inside the All-of-Us Dataproc
workspace. Unit tests cover parse_args and the pure helpers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from _driver_common import _phase, configure_logging, make_spark_session
from disk_telemetry import start_disk_telemetry
# The readout's generic machinery (kernels, both backends, the solver, the closure
# product) lives in spark_vi.readout (ADR 0051) and ships in the spark-vi zip; this
# driver keeps the clinical seams: the doc key (`doc_key.py`, driver-side only),
# the metrics and their formatting.
import doc_key as _dk
from spark_vi.dag import children_and_depth
from spark_vi.readout import (LocalReadoutData, SparkReadoutData, closure_matrix,
                              collect_lean_proba, fit_heads, stacked_proba,
                              write_heads)
from spark_vi.readout.heads import READOUT_GTOL, READOUT_MAX_ITER


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
    (`_doc_key_column` / `_collect_lean_proba`'s fourth
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
        persons = _dk.person_of(np.asarray(doc_keys))
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
# memoized-DFS convention, diamond-safe). Two consequences the flat arm does
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
    _, depth = children_and_depth(parent_int, C)
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
    caller's own collect order) — forwarded to
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
        np.savez(_f, **lam_arrays, alpha=gp["alpha"])
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


# --------------------------------------------------------------------------- #
# The int64 doc-key seam, driver side (exp 0111 WP-A1, spec R5.4).             #
# --------------------------------------------------------------------------- #
# Every readout collect below identifies a row by an int64 DOC KEY rather than
# by the raw `person_id`/`doc_id`. The synthesis and its inverse live once, in
# `doc_key.py`; this file imports them (`_dk.synthesize_doc_key`, `_dk.person_of`,
# radix/guard constants) so there is exactly one definition of the map.
# `_doc_key_column` builds the key as a Spark Column at the collect's own select;
# `spark_vi.readout.kernels.assert_unique_ids` is the uniqueness tripwire the
# collect runs once the keys reach the driver.
def _doc_key_column(df, person_col="person_id"):
    """`doc_key = person_id * RADIX + episode_no` as a guarded Spark Column.

    The DRIVER-side half of the doc-key seam: the readout collects add this
    column and select it as the row id, so `lean_eval_kernel` / the A/B
    alignment / the calibration split all speak one int64 that yields the person
    back through `_dk.person_of` (spec R5.4).

    `episode_no` is the bounded WITHIN-CORPUS document index. Today's corpora are
    single-doc — one document per person, no `episode_no` column — so it defaults
    to literal 0 and `doc_key = person_id * 64`, i.e. the current int64 person
    ids scaled order-preservingly (nothing about the byte-identical single-doc
    path moves). When WP-D2 wires the episode corpus it must materialize a
    bounded per-person document index (0 .. cap-1) under this name — NOT WP-D1's
    unbounded chronological `episode_no` (`episode_index.py`), which would carry
    past the radix and collide; see the seam comment in `doc_key.py`.

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
    radix = F.lit(int(_dk.DOC_KEY_RADIX))
    return (
        F.when(person.isNull() | (person < 0)
               | (person >= F.lit(int(_dk.DOC_KEY_MAX_PERSON_ID))),
               F.raise_error(F.concat(
                   F.lit("doc_key: person_id out of [0, 2**57): "),
                   F.coalesce(person.cast("string"), F.lit("null")))))
        .when((ep < 0) | (ep >= radix),
              F.raise_error(F.concat(
                  F.lit("doc_key: episode_no out of [0, "
                        f"{int(_dk.DOC_KEY_RADIX)}) — this is the bounded "
                        "within-corpus document index, not WP-D1's unbounded "
                        "chronological ordinal: "),
                  ep.cast("string"))))
        .otherwise(person * radix + ep)
        .cast("long")
        .alias("doc_key"))


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


def _collect_lean_proba(scored_df, C, V=None, b_raw=None, *, degenerate=None,
                        const=None, score_col="topicDistribution",
                        label_col="label", mask_col="labelMask",
                        id_col="person_id", theta_topm=0):
    """`readout.spark.collect_lean_proba` keyed by the clinical doc key:
    `(proba (D,C) f32, y u8, mask u8, doc_key_order)`.

    `id_col` names the PERSON-ID column the row's int64 doc key is synthesized from
    (`_doc_key_column`, spec R5.4); the returned order is document identity, and a
    person-grain consumer recovers the person via `_dk.person_of`."""
    df = scored_df.withColumn("doc_key", _doc_key_column(scored_df, id_col))
    return collect_lean_proba(df, C, V, b_raw, degenerate=degenerate, const=const,
                              score_col=score_col, label_col=label_col,
                              mask_col=mask_col, id_col="doc_key",
                              theta_topm=theta_topm)


def readout_arm(train_scored, test_scored, C, K, *, recall_targets, fdr_targets,
                min_count=0, label="", l2=1.0, gtol=READOUT_GTOL,
                max_iter=READOUT_MAX_ITER, history=6, warm_start=None, theta_topm=0,
                checkpoint_dir=None, checkpoint_every=10,
                topic_col="topicDistribution", label_col="label",
                mask_col="labelMask", id_col="person_id", feature_mask=None,
                engine="spark"):
    """Fit the per-node heads on the train split, score the test split, macro the
    metrics: `(readout, proba_te (D_te,C) f32, y_te u8, m_te u8, doc_key_order,
    (V, b_raw))`.

    `engine` picks the readout backend (ADR 0051 §2): "spark" runs every L-BFGS pass
    as a treeAggregate over the scored frame (`SparkReadoutData`); "local" collects
    each split ONCE in compact form (the executors pack and top-m truncate the rows)
    and runs every pass in driver memory (`LocalReadoutData`), the same kernels, so
    the two agree up to summation order. Local needs the train split's packed rows
    in driver memory: ~(12m + 8·nnz) bytes per row truncated, 8K + 8·nnz dense.

    `theta_topm > 0` runs the whole arm, fit AND score, on top-m truncated θ; it is
    threaded through both from here because a mismatch between them is silent.

    `checkpoint_dir` (normally the RUN dir) makes the solve resumable across a
    driver death and receives the heads npz once it returns; both are keyed by
    `label`, because a run fits several arms into one dir.

    Callable with two DataFrames (no argparse), which is what makes it testable
    against the sklearn oracle `score_arm` on a local Spark fixture."""
    cols = dict(topic_col=topic_col, label_col=label_col, mask_col=mask_col)
    tr = train_scored.withColumn("doc_key", _doc_key_column(train_scored, id_col))
    te = test_scored.withColumn("doc_key", _doc_key_column(test_scored, id_col))
    if engine == "local":
        _cprint(f"[driver]   {label or 'arm'}: LOCAL readout engine (splits collected "
                "once, passes in driver memory)", flush=True)
        data_tr = LocalReadoutData.from_spark(tr, C, K, topm=theta_topm, **cols)
        data_te = LocalReadoutData.from_spark(te, C, K, topm=theta_topm, **cols)
    elif engine == "spark":
        data_tr = SparkReadoutData(tr, C, K, topm=theta_topm, **cols)
        data_te = SparkReadoutData(te, C, K, topm=theta_topm, **cols)
    else:
        raise ValueError(f"readout engine must be 'spark' or 'local', got {engine!r}")
    ckpt_path = None
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / f"readout_ckpt_{label or 'arm'}.npz"
    V, b_raw, const, degenerate, info = fit_heads(
        data_tr, l2=l2, gtol=gtol, max_iter=max_iter, history=history, label=label,
        warm_start=warm_start, checkpoint_path=ckpt_path,
        checkpoint_every=checkpoint_every, feature_mask=feature_mask)
    if checkpoint_dir is not None:
        write_heads(checkpoint_dir, label, V, b_raw, const, degenerate, C, K,
                    theta_topm, W_std=info.get("W_std"))
    proba, y_te, m_te, doc_keys = data_te.lean_proba(V, b_raw, degenerate=degenerate,
                                                     const=const)
    readout = readout_from_proba(proba, y_te, m_te, C, recall_targets=recall_targets,
                                 fdr_targets=fdr_targets, min_count=min_count,
                                 doc_keys=doc_keys)
    return readout, proba, y_te, m_te, doc_keys, (V, b_raw)


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


def _build_estimator(args):
    """Construct the DagGatedLDAEstimator for the fit (ADR 0050). The gate (the
    run's label DAG, `args._parent_int`) is set post-construction, since the Param
    is a JSON string; K comes from the layout (n_bg + nodes * tpn)."""
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator

    est = DagGatedLDAEstimator(
        featuresCol="features", frontierCol="frontier",
        maxIter=args.max_iter, seed=args.seed,
        subsamplingRate=args.subsampling_rate,
        learningOffset=args.tau0, learningDecay=args.kappa,
        gammaShape=args.gamma_shape, caviMaxIter=args.cavi_max_iter,
        caviTol=args.cavi_tol,
        optimizeDocConcentration=bool(args.optimize_doc_concentration),
        gateNBg=args.n_bg, gateTpn=args.tpn,
        # Leaves-first gate schedule (report 2026-10-19; exp 0153/0154): 0 = the
        # single-round gate. Fit-only — no cache key moves.
        gateWarm=int(getattr(args, "gate_warm", 0) or 0),
        gateOpenSeed=True,
        gateSeedMode=str(getattr(args, "gate_seed_mode", "subtree")),
        gateSeedLeaves=bool(getattr(args, "gate_seed_leaves", False)),
    )
    # Multi-domain: per-domain feature columns (features_0..) so the engine carries
    # a per-domain lambda. Domain widths are read from the first row.
    dom_cols = getattr(args, "_domain_cols", None)
    if dom_cols:
        est._set(featuresCols=dom_cols)
    dc = getattr(args, "doc_concentration", None)
    if dc is not None and dc > 0:
        # Scalar Dirichlet alpha for the gated doc-topic prior; the default 1/K is
        # razor-small at whole-Mondo K.
        est._set(docConcentration=[float(dc)])
    est.setGateParent(args._parent_int)      # JSON-encodes the DAG map
    gsp = getattr(args, "_gate_seed_profile", None)
    if gsp:
        est.setGateSeedProfile(gsp)
    if str(getattr(args, "gate_seed_mode", "subtree")) == "balanced":
        est.setGateSeedOwnCodes(getattr(args, "_gate_seed_own", None) or {})
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
    from charmpheno.mondo.native_dag import MONDO_NATIVE_VERSION
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
    from spark_vi.dag import DagLayout
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

    `charmpheno.mondo.native_dag` builds the label DAG out of Mondo's own is-a graph (closure-
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
        from charmpheno.mondo.native_dag import (
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
                        "its own documents' word counts (exp 0162).")
    p.add_argument("--gate-seed-profile", default="", metavar="TSV",
                   help="With --gate-warm: an external phenotype profile table "
                        "(mondo_id, concept_id, neg; e.g. hpo_profile_marks.py output) "
                        "that REPLACES a profiled node's primary-domain seed (leaves "
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
    p.add_argument("--doc-concentration", type=float, default=None,
                   help="scalar Dirichlet alpha for the gated doc-topic prior (default "
                        "1/K, razor-small at whole-Mondo K ~0.0022; the central "
                        "configuration sets 0.5).")
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main() -> int:
    from spark_vi.dag import DagLayout

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
            # gracefully without these.
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
        # profiled node's primary-domain seed (domain 0, the condition vocabulary
        # here). Fit-only; counts-only stats.
        args._gate_seed_profile = None
        if getattr(args, "gate_seed_profile", ""):
            if int(getattr(args, "gate_warm", 0) or 0) <= 0:
                raise ValueError("--gate-seed-profile requires --gate-warm > 0")
            with _phase("build the profile seed (driver-side)"):
                import pandas as pd
                from charmpheno.mondo.native_dag import mondo_cid
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
        # PRE-FLIGHT size profile at the data-build boundary (K, fan-out, lambda
        # size), so a big fit's shape is visible BEFORE compute is committed.
        _v_total = sum(len(vm) for vm in vocab_maps)
        _, _prof = lay.cost_report(vocab_size=_v_total)
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
            "subsampling_rate": args.subsampling_rate, "tau0": args.tau0,
            "kappa": args.kappa, "max_iter": args.max_iter,
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

        with _phase(f"gated_pc fit (DagGatedLDAEstimator, K={lay.K})"):
            pc_est = _build_estimator(args)
            # Per-iteration hook: the optional fit checkpoint (fit_save_interval).
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
                pc_est.setOnIteration(_fit_ckpt)
            pc_model = pc_est.fit(bundle.train_df)
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
