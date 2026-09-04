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
                           "n_pos": int(pr["per_node"].get(c, {}).get("n_pos", 0))}
    return {"ranking": ranking["macro"], "pr": pr["macro"], "detection": det,
            "per_node": per_node}


# --------------------------------------------------------------------------- #
# INCIDENT arm (spec E2 / plan WP4): the same readout on the incident cohort.  #
# --------------------------------------------------------------------------- #
# The D7 / C2.4 naming rule, verbatim, carried by every incident output. The heads
# were fit on the PREVALENT problem (`_fit_readout_heads` standardizes per node on
# that node's own observed TRAIN rows), so what the incident arm measures is a
# prevalent-fit model evaluated on an incident cohort — a legitimate quantity, but
# NOT "the incident AUC". Train-time incident masking is a deliberate non-goal
# (spec §9); this string is the price of that deferral, paid in labelling.
INCIDENT_NAMING = ("a PREVALENT-FIT model evaluated on an INCIDENT COHORT "
                   "(spec D7 / audit C2.4) — heads are standardized and fit on "
                   "each node's own observed TRAIN rows under the prevalent mask; "
                   "this is NOT 'the incident AUC'")


def incident_eval_mask(y, mask, elig):
    """The D2/D3/D4 evaluation mask for the incident arm: `elig & (y | mask)`.

    Per node c and document d, with `elig[d,c] = c ∉ R_d` (D2 — a prior carrier of
    `closure(c)` leaves **BOTH** classes, which is the whole point: dropping it from
    the positives only would be a different and wrong estimator):

      * **incident POSITIVE** (D3) — eligible AND `label[d,c] == 1`. Deliberately
        NOT gated on the observation mask: a positive is ATTESTED (the document
        gained `closure(c)` in the label window), which is a fact about the label
        window, not about whether this run chose to observe the cell.
      * **incident NEGATIVE** (D4) — eligible AND observed AND `label[d,c] == 0`.
        The mask is what makes a negative a NEGATIVE rather than an unobserved
        cell: under `label_mask_mode="closure"` a node is observed only on rows
        inside its parent's closure, so an unmasked zero is "not asked", not
        "asked and answered no".

    Union those two and you get `elig & (y | mask)`, which is what a masked ranking
    readout wants — the rows that carry a class.

    THIS IS THE SAME ARITHMETIC AS `diag_incident_census.census_partial`
    (`n_ipos = elig*y`, `n_ineg = elig*m*(1-y)`), and the agreement is asserted on a
    shared synthetic fixture in `tests/scripts/test_incident_readout.py`. It has to
    be: the census is the corpus probe that GATED this arm, and if the two disagreed
    about who is eligible, the gate would have been passed on a population the
    readout does not score.

    Pure numpy. `elig=None` (no E1 column) returns None — "no eligibility
    information", which every caller reads as SKIP the incident arm."""
    if elig is None:
        return None
    y = np.asarray(y)
    mask = np.asarray(mask)
    elig = np.asarray(elig).astype(bool)
    has_class = (y != 0) | (mask != 0)
    return (elig & has_class).astype(np.uint8)


def _macro_over(per_node, nodes):
    """Macro AUC/AP over an EXPLICIT node set (spec R2.2), from a `per_node` dict.

    `readout_from_proba`'s `per_node` holds exactly the SCORED nodes (skipped ones
    never enter it), so `set(per_node)` is that arm's full scoreable set and the
    intersection of two arms' sets is the SHARED set. Averaging over a passed-in
    node list rather than over whatever each arm happened to score is the entire
    point of R2.2: a prevalent-vs-incident delta computed across different node
    sets is not a comparison."""
    nodes = [c for c in nodes if c in per_node]
    aucs = [per_node[c]["auc"] for c in nodes]
    aps = [per_node[c]["ap"] for c in nodes if per_node[c].get("ap") is not None]
    return {"auc": float(np.mean(aucs)) if aucs else None,
            "ap": float(np.mean(aps)) if aps else None,
            "n_nodes": len(nodes)}


def incident_readout(proba, y_te, m_te, elig, C, *, recall_targets, fdr_targets,
                     min_count=0, prevalent=None, arm_label="gated_pc",
                     doc_keys=None):
    """The `gated_pc_incident` results block: the second `readout_from_proba` call.

    Everything E2 asks for, in one dict:

      * **the incident readout itself** — `readout_from_proba` on `m_incident`
        (D2/D3/D4 via `incident_eval_mask`), with R2.1's constant-column guard ON
        (`skip_constant=True`);
      * **R2.2, four macro lines** — prevalent/full, prevalent/shared,
        incident/full, incident/shared, where "shared" is the both-arms-scoreable
        node set. Reported together because the honest headline is the SHARED pair
        and the full pair is what each arm can say on its own;
      * **R2.1's three skip counts, never summed** (spec §8.5): degenerate test
        column, small test column, constant prediction column;
      * **R2.4 / D7** — `INCIDENT_NAMING` in the block itself, so a table rendered
        from the JSON cannot lose it;
      * **spec §8's four tags** on every number.

    `prevalent` is the SAME arm's already-computed prevalent readout (the one
    already in `results["gated_pc"]`), used only for the shared node set — no
    re-scoring, no second fit, and structurally impossible for a run's own output to
    enter the eligibility definition (R2.3: eligibility arrives from the BUNDLE).

    `doc_keys` (R5.7) threads through to `readout_from_proba`'s own `doc_keys` —
    the incident cohort is a row subset of the same test split the prevalent arm
    scored, so it carries the same multi-doc-person risk in its detection pool.

    Pure numpy + sklearn: given the four arrays it is fully unit-testable off-Spark,
    which is where the constant-column fixture lives."""
    m_incident = incident_eval_mask(y_te, m_te, elig)
    if m_incident is None:
        return None
    readout = readout_from_proba(
        proba, y_te, m_incident, C, recall_targets=recall_targets,
        fdr_targets=fdr_targets, min_count=min_count,
        # R2.1: the ranking axis gets the guard the detection pool has had since
        # exp 0104. It is ON here and OFF on the prevalent arm on purpose — see
        # `_score_label`'s `skip_constant`.
        skip_constant=True, doc_keys=doc_keys)
    return _assemble_incident_block(
        readout, prevalent, min_count, arm_label,
        n_eligible_cells=int(np.asarray(elig).astype(bool).sum()),
        n_scored_cells=int(m_incident.sum()))


def _assemble_incident_block(readout, prevalent, min_count, arm_label, *,
                             n_eligible_cells, n_scored_cells):
    """The incident results block, assembled from an already-scored incident readout.

    Factored out of `incident_readout` so BOTH the driver path (which scores an
    incident-masked (D,C) proba through `readout_from_proba`) and the
    eval_path=distributed path (which builds `readout` from
    `per_node_metric_arms_rows`' incident per-node dict, WP-B) produce the SAME block
    — the R2.2 macro-by-node-set logic, the three-reason skip counts and the D7
    naming live in one place, so the parity gate compares two numbers, not two
    structures. `readout` must carry `per_node` (the scored incident nodes) and
    `ranking.skipped_by_reason`; the cell counts are passed in because the driver
    reads them off `m_incident`/`elig` arrays the distributed path never materializes
    (it counts them distributed, or leaves them None). `readout` itself is embedded
    on the driver path (the full ranking/pr/detection) and omitted (None) when the
    distributed path has only the ranking axis."""
    inc_nodes = set(readout["per_node"])
    prev_nodes = set((prevalent or {}).get("per_node", {}))
    shared = sorted(inc_nodes & prev_nodes)
    macros = {
        "incident_full": _macro_over(readout["per_node"], sorted(inc_nodes)),
        "incident_shared": _macro_over(readout["per_node"], shared),
    }
    if prevalent is not None:
        macros["prevalent_full"] = _macro_over(prevalent["per_node"],
                                               sorted(prev_nodes))
        macros["prevalent_shared"] = _macro_over(prevalent["per_node"], shared)
    return {
        "naming": INCIDENT_NAMING,
        "tags": {
            "arm": "incident",
            # Every macro line names its own node set; the block-level tag says
            # both are present, which is R2.2's whole requirement.
            "node_set": "both (shared + full, reported separately)",
            "cell_type": "marginal",
            "claim_type": "discrimination",
        },
        "eligibility": {
            "definition": "incident-eligible(d, c) := c NOT IN R_d (spec D2)",
            "source": "bundle column (E1 pre-index closure) — a CORPUS property, "
                      "never a run property (spec R2.3); no fit output enters it",
            "positives": "eligible AND label==1 (D3)",
            "negatives": "eligible AND observed AND label==0 (D4)",
            "n_eligible_cells": n_eligible_cells,
            "n_scored_cells": n_scored_cells,
        },
        "arm_label": arm_label,
        "min_count": int(min_count),
        "macros": macros,
        "node_sets": {
            "n_incident_scoreable": len(inc_nodes),
            "n_prevalent_scoreable": len(prev_nodes),
            "n_shared": len(shared),
        },
        # Three reasons, three counts, NEVER summed (spec §8.5).
        "skipped_by_reason": readout["ranking"].get("skipped_by_reason", {}),
        "readout": readout,
    }


def format_incident_readout(block) -> str:
    """The incident block as driver log lines, with the D7 rule on the table."""
    if not block:
        return ("[driver]   incident arm SKIPPED: this corpus carries no E1 "
                "pre-index closure column")
    m, ns = block["macros"], block["node_sets"]
    sk = block.get("skipped_by_reason") or {}

    def _row(name, d):
        if d is None:
            return f"    {name:<24} n/a"
        return (f"    {name:<24} AUC={_f(d['auc']).strip():<7} "
                f"AP={_f(d['ap']).strip():<7} (over {d['n_nodes']} nodes)")

    lines = [
        f"[incident readout: {block['arm_label']}]  {block['naming']}",
        "  arm=incident · cell=marginal · claim=DISCRIMINATION (never prospective)",
        "  eligibility: c NOT IN R_d (spec D2), a CORPUS property — prior carriers "
        "leave BOTH classes",
        "  macro AUC/AP by ARM x NODE SET (R2.2 — a delta across different node "
        "sets is not a comparison):",
        _row("prevalent / full", m.get("prevalent_full")),
        _row("prevalent / shared", m.get("prevalent_shared")),
        _row("incident / full", m.get("incident_full")),
        _row("incident / shared", m.get("incident_shared")),
        f"  node sets: prevalent-scoreable={ns['n_prevalent_scoreable']}  "
        f"incident-scoreable={ns['n_incident_scoreable']}  "
        f"shared={ns['n_shared']}  (min_count={block['min_count']})",
        "  skipped columns, THREE reasons counted separately (never summed): "
        f"degenerate={sk.get('degenerate_test_column', 0)}  "
        f"small={sk.get('small_test_column', 0)}  "
        f"CONSTANT={sk.get('constant_prediction_column', 0)}",
        "  (the constant count is R2.1's guard: a train-degenerate node's constant "
        "column acquires negatives under the incident mask and would otherwise "
        "score a hard 0.5 INSIDE the macro)",
    ]
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


# The All-of-Us disclosure floor (`analysis/pc/evaluate.py:76-78`). It is an EGRESS
# rule, not a statistical dial: a cell with fewer than 20 of either class does not
# leave the workspace, in a printed table or in a JSON. Kept structurally separate
# from `min_count` (which decides what is COMPUTED) and from `min_positives` (a
# model-internal powering dial) — conflating publishing floors with model dials is
# the mistake the 0110 plan calls out by name.
EGRESS_MIN_COUNT = 20

# The two D6 strata, by whether the document already carried the PARENT before its
# index. Reported, never gating (spec D6, corrected explicitly by Shawn): requiring
# a pre-index P starves the cells, and "no P / gains c" is a legitimate positive —
# it tests DE NOVO specific prediction, the harder and more interesting half.
STRATUM_P_KNOWN = "P known pre-index (P in R_d)"
STRATUM_P_UNKNOWN = "P not known pre-index (P not in R_d)"


def _suppressed(n_pos, n_neg, floor=EGRESS_MIN_COUNT):
    """The egress record for a cell too small to disclose — counts REMOVED.

    R3.5: stratified cell tables are a disclosure surface. A suppressed cell says
    that it was suppressed and which floor it failed; it does not say by how much,
    because "n_pos=3" is exactly the number that may not leave."""
    return {"suppressed": f"<{int(floor)}",
            "reason": f"either class below the egress floor of {int(floor)}"}


def _p_strata(rows, yc, sc, p, elig, egress_min_count):
    """One edge's cell, split by the D6 stratum, with the egress floor applied.

    `elig[d, p] is False` means the document ALREADY CARRIED the parent before its
    index — "P known pre-index". Its complement is the *de novo* half: the document
    reaches P and c in the same label window. Both are legitimate; the first is the
    subtyping question ("has a <parent>, which <child>?") asked of a patient whose
    parent diagnosis is on the record, the second asks the model to find the
    specific node with no parent context at all, and they do not have the same
    difficulty.

    A stratum that fails `egress_min_count` on EITHER class reports as suppressed
    with no counts (R3.5) — stratifying halves each cell, so most edges are expected
    to land here, and that is a fact about the corpus, not a failure of the code."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    known = ~elig[rows, p]
    out = {}
    for name, sel in ((STRATUM_P_KNOWN, known), (STRATUM_P_UNKNOWN, ~known)):
        ys, ps = yc[sel], sc[sel]
        n_pos, n_neg = int(ys.sum()), int(len(ys) - ys.sum())
        if n_pos < egress_min_count or n_neg < egress_min_count:
            out[name] = _suppressed(n_pos, n_neg, egress_min_count)
            continue
        out[name] = {
            "n_pos": n_pos, "n_neg": n_neg, "prev": float(ys.mean()),
            "cond_auc": float(roc_auc_score(ys, ps)),
            "cond_ap": float(average_precision_score(ys, ps)),
            "ece": _ece(ys, ps, n_bins=5),
        }
    return out


def conditional_readout(proba, y, mask, parent_int, C, *, min_count=10,
                        eligibility=None, egress_min_count=EGRESS_MIN_COUNT):
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
    A pooled ECE gauges whether P(child|parent) is calibrated (needed for VOI); a
    PER-NODE ECE summary (mean/max/worst) accompanies it so a single miscalibrated node
    can't hide inside the pool (the unified-head calibration check, insight 0069).

    MASK-INDEPENDENCE: pass the FULL-closure observation mask (all-ones) as ``mask``
    regardless of the run's training label_mask_mode. The `label` array is already
    mask-mode-independent (closure membership), so an all-ones eval mask fixes the
    cohort/negative sets identically across full- and closure-mask runs — otherwise
    the closure mask silently makes the conditional eval an easier sibling-only
    contrast and cross-run numbers are not comparable (exp 0079, Trap 3). Pure numpy.

    INCIDENT-LOCAL CELLS (spec E3 / plan WP5), via ``eligibility``
    -------------------------------------------------------------
    ``eligibility`` is E1's `(D, C)` array `elig[d,c] = c ∉ R_d` (spec D2). Passing
    it turns each cell into its D5 form — the SAME local-negative construction,
    intersected with incident eligibility for the CHILD:

      * positives: in P's cohort, eligible for c, gains `closure(c)`;
      * negatives: in P's cohort, eligible for c, gains a SIBLING under P (or P and
        nothing more specific), does not gain `closure(c)`.

    and additionally reports every edge SPLIT BY THE D6 STRATUM — whether the
    document already carried the PARENT pre-index (`P ∈ R_d`) or not.

    **The stratum is reported, never gating, and neither is the parent's own
    eligibility.** The plan's phrasing ("intersect eligibility with the cohort
    construction") is deliberately NOT read as `elig[:, p]` here: restricting the
    parent cohort to documents incident-eligible for P would BE the pre-index-P gate
    D6 forbids in as many words, and would delete the "no P / gains c" positives
    that are the harder and more interesting half of the question. So eligibility
    enters at the CHILD row selection (which is exactly D5's clause (i)) and the
    parent's eligibility enters only as the stratum key.

    ``eligibility=None`` (the default) reproduces this function's pre-E3 output
    NUMERICALLY EXACTLY — no strata, no suppression, no changed floor. That is the
    regression that protects the 0104/0109/0110 conditional comparison, and it is
    asserted in `tests/scripts/test_incident_conditional.py`.

    ``egress_min_count`` (R3.5) is the EGRESS floor, applied only on the incident
    path: any pooled cell or stratum with either class below it is emitted as
    `{"suppressed": "<20"}` with its counts REMOVED, in the printed table and in the
    results dict alike. It is deliberately a separate argument from ``min_count``
    (what gets computed) — publishing floors and model-internal dials stay
    structurally separate."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    children, depth = _dag_children_and_depth(parent_int, C)
    incident = eligibility is not None
    elig = (np.asarray(eligibility).astype(bool) if incident
            else np.ones_like(np.asarray(mask), dtype=bool))
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
            sel = mask[cohort, c] == 1
            if incident:
                sel = sel & elig[cohort, c]              # D5 clause (i)
            rows = cohort[sel]
            yc = y[rows, c]
            n_pos, n_neg = int(yc.sum()), int(len(yc) - yc.sum())
            if n_pos < max(min_count, 1) or n_neg < max(min_count, 1):
                continue
            sc = proba[rows, c]
            # marginal (de-novo) AP: child vs ALL observed docs — the sharpening bar.
            mrows = np.where(mask[:, c] == 1)[0]
            if incident:
                mrows = mrows[elig[mrows, c]]
            my = y[mrows, c]
            marg_ap = (float(average_precision_score(my, proba[mrows, c]))
                       if 0 < my.sum() < len(my) else None)
            edge = {
                "parent": p, "child": c, "depth": depth[p], "cohort": int(len(cohort)),
                "n_pos": n_pos, "prev": float(yc.mean()),
                "cond_auc": float(roc_auc_score(yc, sc)),
                "cond_ap": float(average_precision_score(yc, sc)),
                "marg_ap": marg_ap,
                # PER-NODE reliability: this node's OWN ECE on child-vs-siblings, so a
                # per-node miscalibration can't hide inside the pooled ECE (which
                # averages an over- against an under-confident node). Fewer bins (5)
                # because per-node cohorts are small (n_pos,n_neg >= min_count each).
                "ece": _ece(yc, sc, n_bins=5)}
            if incident:
                # D6: the SAME cell, split by whether the document already carried
                # the PARENT. Reported, never gating — and the two strata routinely
                # have materially different AUCs, so pooling them into one unlabeled
                # number is the thing the spec forbids.
                edge["strata"] = _p_strata(
                    rows, yc, sc, p, elig, egress_min_count)
                # R3.5: the POOLED cell is a disclosure surface too.
                if n_pos < egress_min_count or n_neg < egress_min_count:
                    edge = {k: edge[k] for k in ("parent", "child", "depth")}
                    edge.update(_suppressed(n_pos, n_neg, egress_min_count))
                    edges.append(edge)
                    continue
                edge["n_neg"] = n_neg
            edges.append(edge)
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
    # Per-node reliability summary: mean/max/worst over the per-edge ECEs. A max that
    # dwarfs the pooled ECE is the signal that pooling is flattering the calibration.
    node_eces = [(e["ece"], e["parent"], e["child"]) for e in edges
                 if e.get("ece") is not None]
    node_ece = None
    if node_eces:
        vals = [v for v, _, _ in node_eces]
        worst = max(node_eces, key=lambda t: t[0])
        node_ece = {"mean": float(np.mean(vals)), "max": float(worst[0]),
                    "worst_parent": int(worst[1]), "worst_child": int(worst[2]),
                    "n_nodes": len(node_eces)}
    out = {"edges": edges, "parents": parents, "ece": ece, "node_ece": node_ece}
    if incident:
        # Only the incident variant gains keys, so the prevalent block's shape (and
        # every reader of it) is exactly what it was.
        out["incident"] = _incident_conditional_summary(edges, egress_min_count,
                                                        min_count)
    return out


def _incident_conditional_summary(edges, egress_min_count, min_count):
    """The E3 block's own header: R2.2's edge-set discipline, R3.2's ordering, the
    D7 naming rule and the §8 tags — carried in the JSON, not just in a log line.

    Per R3.3 the surviving-EDGE SETS are reported, not only the averages: pooled,
    P-known and P-unknown survive different edges, and a pooled-vs-stratum delta
    computed across different edge sets is the same non-comparison C2.2 names on the
    marginal axis."""
    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    scored = [e for e in edges if e.get("cond_auc") is not None]
    out = {
        "naming": INCIDENT_NAMING,
        "tags": {"arm": "incident", "node_set": "surviving edges (reported below)",
                 "cell_type": "conditional-pooled + conditional-stratum(P known / "
                              "P unknown)",
                 "claim_type": "discrimination"},
        "cell_definition": ("D5 local negative for P->c: eligible for c (D2), gains "
                            "a sibling under P or P-but-nothing-more-specific, does "
                            "not gain closure(c)"),
        "stratum_definition": ("D6: P in R_d vs P not in R_d — REPORTED, NEVER "
                               "GATING; 'no P / gains c' is a legitimate positive "
                               "(de novo specific prediction)"),
        "egress_min_count": int(egress_min_count),
        "min_count": int(min_count),
        "egress_note": (f"cells with either class < {int(egress_min_count)} are "
                        "SUPPRESSED with their counts removed, here and in every "
                        "printed table (All-of-Us disclosure floor)"),
        "n_edges_emitted": len(edges),
        "n_edges_suppressed": sum(1 for e in edges if e.get("suppressed")),
        "pooled": {"n_edges": len(scored),
                   "cond_auc": _mean([e["cond_auc"] for e in scored]),
                   "cond_ap": _mean([e["cond_ap"] for e in scored])},
    }
    for name in (STRATUM_P_KNOWN, STRATUM_P_UNKNOWN):
        surviving = [e for e in scored
                     if (e.get("strata", {}).get(name, {}).get("cond_auc")
                         is not None)]
        out.setdefault("strata", {})[name] = {
            "n_edges": len(surviving),
            "edge_set": [[int(e["parent"]), int(e["child"])] for e in surviving],
            "cond_auc": _mean([e["strata"][name]["cond_auc"] for e in surviving]),
            "cond_ap": _mean([e["strata"][name]["cond_ap"] for e in surviving]),
        }
    # The both-strata edge set: the only set on which the two strata's AUCs are a
    # comparison rather than two different averages (R2.2 discipline, R3.3).
    both = [e for e in scored
            if all(e.get("strata", {}).get(n, {}).get("cond_auc") is not None
                   for n in (STRATUM_P_KNOWN, STRATUM_P_UNKNOWN))]
    out["strata_shared_edge_set"] = {
        "n_edges": len(both),
        "edge_set": [[int(e["parent"]), int(e["child"])] for e in both],
        STRATUM_P_KNOWN: _mean([e["strata"][STRATUM_P_KNOWN]["cond_auc"]
                                for e in both]),
        STRATUM_P_UNKNOWN: _mean([e["strata"][STRATUM_P_UNKNOWN]["cond_auc"]
                                  for e in both]),
    }
    return out


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


def _person_keyed_cal_split(doc_keys, seed, cal_frac=0.25):
    """PERSON-keyed calibration/fit split for the driver-path calibration block
    (R5.6). Returns `(cal_sel, fit_sel)` boolean arrays over `doc_keys`'s own row
    order, partitioning by PERSON (`_dr.person_of`) rather than by row, so a
    multi-doc person's several documents in `doc_keys` all land on the SAME side.

    Replaces a per-ROW `rng.random(n_rows) < cal_frac` draw, which let one
    person's several documents straddle the boundary: some FIT the isotonic
    calibrator while another was held out to GRADE it — grading a calibrator
    partly on the same person's own (correlated) covariates it was just fit on,
    an in-sample leak dressed as an out-of-sample ECE improvement. That is the
    exp 0079 run-2 failure this split exists to prevent, reintroduced by
    multi-doc corpora; pinned by
    `test_multidoc_seams.py::test_person_keyed_cal_split_no_person_straddles`.

    Mirrors the DISTRIBUTED twin's `F.pmod(F.hash(person_id, seed), 4) == 0`
    split (`main`'s `readout_mode == "distributed"` branch) in KIND —
    person-keyed, deterministic given `seed` — not in exact bucket arithmetic:
    the two run in different runtimes (numpy here, Spark there) and neither
    promises the other's literal RNG stream, only the invariant both exist to
    guarantee (no person split across cal/fit).

    `np.unique` sorts by VALUE, so which persons land in `cal` is a function of
    person identity + seed alone — independent of row/collect order, the same
    order-independence property the twin's hash gets from hashing instead of
    drawing (and `_doc_key_sample` gets for the A/B gate's sample_frac, R5.5)."""
    person = _dr.person_of(np.asarray(doc_keys, dtype=np.int64))
    uniq_persons, inverse = np.unique(person, return_inverse=True)
    rng = np.random.default_rng(int(seed))
    person_cal = rng.random(uniq_persons.shape[0]) < float(cal_frac)
    cal_sel = person_cal[inverse]           # broadcast back to doc_keys's row order
    return cal_sel, ~cal_sel


def calibrate_per_node(proba_cal, y_cal, m_cal, proba_te, C, *, min_pos=20):
    """Per-node ISOTONIC calibration fit on a HELD-OUT calibration set, applied to TEST.

    The head-independent (two-stage) path to CALIBRATED P(node) — a VOI prerequisite —
    fit on theta alone, independent of the co-fit head. Isotonic is monotone so it
    preserves the ranking (AUC/AP unchanged) while fixing the reliability curve.
    (At 41-anchor scale the ridge-bounded co-fit head is itself well-calibrated —
    exp 0082 — so this two-stage calibration is a fallback, not the only route.)

    ``proba_cal`` MUST be OUT-OF-SAMPLE predictions (a held-out slice the per-node LR
    did NOT train on) — fitting the calibrator on in-sample train predictions learns
    the wrong (over-confident) correction and can worsen ECE (exp 0079 run 2). A node
    with too few observed calibration positives (or a single class) passes through
    UNCALIBRATED. Pure numpy + sklearn."""
    from sklearn.isotonic import IsotonicRegression
    out = np.asarray(proba_te, dtype=np.float64).copy()
    for c in range(C):
        cal = np.asarray(m_cal[:, c], bool)
        yc = y_cal[cal, c]
        if int(yc.sum()) < min_pos or len(np.unique(yc)) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(proba_cal[cal, c], yc)
        out[:, c] = iso.transform(proba_te[:, c])
    return out


def format_conditional_readout(cond, int2cid, name_by_id):
    """Render conditional_readout HONESTLY (VOI/metrics report §2): cond_AUC is the
    headline discrimination number (prevalence-independent); cond_AP/lift are marked
    'context' (mostly the base-rate rise). Per-parent top-1 is shown WITH its
    majority-class baseline and balanced accuracy, sorted by depth then top-1-minus-
    majority (the honest lift). A pooled ECE reports calibration. Eval is
    mask-independent (full-closure cohorts) — see conditional_readout."""
    # SUPPRESSED edges (incident variant, R3.5) carry no metrics — only the fact
    # that they were suppressed — so every consumer below filters on `cond_auc`.
    edges = [e for e in cond["edges"] if e.get("cond_auc") is not None]
    parents = cond["parents"]
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
    ne = cond.get("node_ece")
    if ne:
        wp, wc = _name(ne["worst_parent"]), _name(ne["worst_child"])
        lines.append(
            f"  per-node reliability (ECE over {ne['n_nodes']} nodes): "
            f"mean={ne['mean']:.4f}  max={ne['max']:.4f} (worst {wp}->{wc})  "
            f"vs pooled={_f(ece).strip()}   "
            f"[max>>pooled => pooling flatters calibration]")
    lines.append("  per-parent multiclass top-1 vs majority-class baseline "
                 "(which child, given the parent):")
    for p in sorted(parents, key=lambda p: (p["depth"], -(p["top1"] - p["majority"]))):
        ba = "" if p.get("bal_acc") is None else f" bal_acc={p['bal_acc']:.3f}"
        lines.append(
            f"    d{p['depth']} {_name(p['parent']):<22} "
            f"top1={p['top1']:.3f} (majority={p['majority']:.3f}){ba}  "
            f"(n={p['n']}, {p['n_children']} children)")
    return "\n".join(lines)


def format_incident_conditional(cond) -> str:
    """Render the INCIDENT conditional block: pooled first, strata second (R3.2).

    Every printed number here is a count of EDGES or an average over them — never a
    cell count — so the table is disclosable as printed; the suppressed cells are
    counted, not shown (R3.5)."""
    inc = (cond or {}).get("incident")
    if not inc:
        return "[incident conditional]  no incident block (eligibility not threaded)"
    s = inc["strata"]
    sh = inc["strata_shared_edge_set"]

    def _row(name, d, key="cond_auc"):
        v = d.get(key) if d else None
        n = d.get("n_edges") if d else 0
        return f"    {name:<34} cond_AUC={_f(v)}  (over {n} edges)"

    return "\n".join([
        f"[incident conditional]  {inc['naming']}",
        "  arm=incident · cell=conditional · claim=DISCRIMINATION",
        f"  cell: {inc['cell_definition']}",
        f"  stratum: {inc['stratum_definition']}",
        "  POOLED FIRST (primary), strata second:",
        _row("pooled (both strata)", inc["pooled"]),
        _row(STRATUM_P_KNOWN, s[STRATUM_P_KNOWN]),
        _row(STRATUM_P_UNKNOWN, s[STRATUM_P_UNKNOWN]),
        "  ...and on the SHARED edge set, where the two strata ARE a comparison "
        f"({sh['n_edges']} edges): "
        f"P-known={_f(sh[STRATUM_P_KNOWN]).strip()}  "
        f"P-unknown={_f(sh[STRATUM_P_UNKNOWN]).strip()}",
        f"  edges emitted={inc['n_edges_emitted']}, of which SUPPRESSED "
        f"(<{inc['egress_min_count']} on a class, counts withheld)="
        f"{inc['n_edges_suppressed']}",
        f"  EGRESS: {inc['egress_note']}",
    ])


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

    # RARITY SPLIT (insight 0066 direct test): does supervision help the LOW-POSITIVE
    # nodes, even if the macro over all nodes is flat? Split the per-node AUC/AP by test
    # positive count at the median (nodes scored in BOTH arms), macro each half.
    gpn, upn = g.get("per_node") or {}, u.get("per_node") or {}
    both = sorted(set(gpn) & set(upn), key=lambda c: gpn[c]["n_pos"])

    def _mac(nodes, arm_pn, key):
        v = [arm_pn[c][key] for c in nodes if arm_pn[c].get(key) is not None]
        return float(np.mean(v)) if v else None
    if len(both) >= 8:
        # RARITY SPLIT by test-positive-count QUARTILE. The single-median split hides the
        # extreme tail; the rarest quartile (Q1) is where the gate is most starved and where
        # insight 0066 predicts PC's ONLY headroom. A positive Q1 delta = PC rescues the
        # extreme low-mass tail even if the macro is negative.
        npos = np.array([gpn[c]["n_pos"] for c in both], dtype=float)
        edges = np.quantile(npos, [0.25, 0.5, 0.75])

        def _qbin(c):
            n = gpn[c]["n_pos"]
            return int(n >= edges[0]) + int(n >= edges[1]) + int(n >= edges[2])
        bins = {q: [] for q in range(4)}
        for c in both:
            bins[_qbin(c)].append(c)
        print(f"[driver]   RARITY SPLIT by test-+ct QUARTILE ({len(both)} shared nodes, "
              f"+ct edges {[int(e) for e in edges]}):", flush=True)
        for q, lbl in enumerate(("Q1 rarest", "Q2       ", "Q3       ", "Q4 common")):
            nodes = bins[q]
            ns = sorted(gpn[c]["n_pos"] for c in nodes)
            rng = f"+ct {ns[0]}-{ns[-1]}" if ns else "-"
            print(f"[driver]     {lbl} n={len(nodes):<3} ({rng:<12}) "
                  f"AUC {_d(_mac(nodes, gpn, 'auc'), _mac(nodes, upn, 'auc'))}"
                  f"   AP {_d(_mac(nodes, gpn, 'ap'), _mac(nodes, upn, 'ap'))}", flush=True)
        print("[driver]     (a POSITIVE Q1 delta = PC rescues the extreme low-mass tail; "
              "flat/negative across ALL quartiles = the gate already serves even the rarest.)",
              flush=True)

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
    np.savez(out / "gated_pc_result.npz",
             **lam_arrays, alpha=gp["alpha"], w_CK=gp["w_CK"],
             b_CK=np.asarray(gp.get("b_CK", np.zeros(C)), dtype=np.float64))
    manifest = dict(manifest_fields)
    manifest["per_node_domain_mass"] = (
        {str(k): v for k, v in domain_mass.items()} if domain_mass else None)
    manifest["results"] = results
    manifest["partial"] = partial
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
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


def _doc_key_sample(df, sample_frac, seed, *, n_buckets=10000):
    """Keep the `sample_frac` share of DOCUMENTS whose int64 doc key (`_doc_key_
    column`, spec R5.4) hashes into the low `n_buckets * sample_frac` buckets —
    the DOC-KEY-grained twin of `readout_ab_report`'s old `DataFrame.sample()`
    call (R5.5, seam 6).

    `DataFrame.sample()` draws a Bernoulli trial per ROW INDEX within a
    partition: a function of the query PLAN (partitioning, upstream shuffles),
    not of document identity. `readout_ab_report` got away with that because ONE
    physical read of ONE cached DataFrame feeds both the driver and the
    distributed collect, but the promise its own log line makes —
    "restricted to the SAME `sample_frac` sample" for a given `seed` — is a
    promise about DOCUMENT identity, not about today's happening-to-match
    physical plan. Hashing the doc key makes the kept set a pure function of
    WHICH DOCUMENTS exist plus `seed`, independent of partitioning or row order,
    exactly the property `case_finding_assembly.split_train_test` and the
    distributed calibration split (`main`'s `readout_mode == "distributed"`
    branch) already get from hashing instead of drawing. `n_buckets=10000`
    mirrors `split_train_test`'s own bucket width.

    Rows are the unit of `_doc_key_column`, i.e. DOCUMENTS, not persons — a
    multi-doc person's several rows hash independently, on purpose: A2's job is
    "the A and B collects compare the SAME documents" (both call sites hash the
    SAME frame with the SAME seed), not "a person's documents move as a block"
    (that grouping is A3's calibration split, which needs cal/fit to never share
    a person; the A/B gate has no such constraint — it only needs A and B to
    agree on what they scored)."""
    from pyspark.sql import functions as F
    cut = int(round(float(sample_frac) * n_buckets))
    bucket = F.pmod(F.hash(_doc_key_column(df), F.lit(int(seed))), F.lit(int(n_buckets)))
    return df.filter(bucket < F.lit(cut))


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


def _collect_head_proba(df, C, *, prob_col="probability", label_col="label",
                        mask_col="labelMask", sample_frac=1.0, seed=0,
                        with_doc_keys=False):
    """Collect the co-fit head's per-node P(node)=sigmoid(w_CK.theta) + label/mask
    to (proba (D,C), y (D,C), mask (D,C)) — or, with `with_doc_keys=True`, a
    fourth `doc_key_order` mirroring `_collect_theta_labels`'s int64 DOC KEY
    order (spec R5.4), so the co-fit head's own readout can dedup its detection
    pool to persons (R5.7) the same way the pc_topics_lr readout does. Only
    meaningful for a supervised (weightY>0) transform, which appends `prob_col`.
    Cluster-covered. `sample_frac` row-subsamples before the collect (bounds the
    driver footprint; see _collect_theta_labels).

    `with_doc_keys` defaults OFF and is opt-in, not a `doc_keys=None`-style
    always-fourth-return: `gated_pc_readout.run_readout` (untouched by this WP)
    calls this with a bare `hp, hy, hm = _collect_head_proba(test_scored, C)`
    three-tuple unpack, and changing the return ARITY unconditionally breaks it
    at import time with no type error to catch it until the tuple unpacks
    wrong. `gated_pc_cloud.main`'s own co-fit head call site passes
    `with_doc_keys=True` explicitly."""
    if sample_frac < 1.0:
        df = df.sample(withReplacement=False, fraction=float(sample_frac),
                       seed=int(seed))
    cols = [prob_col, label_col, mask_col]
    if with_doc_keys:
        cols = [_doc_key_column(df)] + cols
    rows = df.select(*cols).collect()
    if not rows:
        z = np.zeros((0, C), dtype=np.float64)
        return (z, z, z.copy(), []) if with_doc_keys else (z, z, z.copy())
    if with_doc_keys:
        doc_key_order = [int(r["doc_key"]) for r in rows]
        _assert_unique_doc_keys(doc_key_order, where="_collect_head_proba")
    proba = np.asarray([r[prob_col].toArray() for r in rows], dtype=np.float64)
    y_DC = np.asarray([[float(v) for v in r[label_col]] for r in rows],
                      dtype=np.float64)
    mask_DC = np.asarray([[float(v) for v in r[mask_col]] for r in rows],
                         dtype=np.float64)
    if with_doc_keys:
        return proba, y_DC, mask_DC, doc_key_order
    return proba, y_DC, mask_DC


# --------------------------------------------------------------------------- #
# Distributed readout (plan 2026-08-20-distributed-readout-plan.md, v2.1).     #
# --------------------------------------------------------------------------- #
# Above this C the driver readout stops being safe: it collects θ (D,K float64)
# plus label+mask (D,C float64 each) for BOTH splits and then builds (D,C) float64
# proba arrays on top — 8*D*(K + 3C) bytes, ~24 GB at K=C=3,300 over 300k docs.
# 500 is the plan's own `default driver at C<=500` line: it keeps every historical
# cardiovascular run (C=444) on the byte-identical driver path while whole-Mondo
# (C~3,300) routes to the distributed fit automatically.
_DRIVER_READOUT_MAX_C = 500
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


def resolve_readout_mode(mode, C, max_driver_c=_DRIVER_READOUT_MAX_C):
    """Resolve `--readout-mode {driver,distributed,auto}` against this run's C.

    `auto` is the only interesting case: `driver` while the θ/label/mask collect
    still fits (C <= 500), `distributed` beyond it. Explicit modes pass through, so
    an A/B run can force either path on the same corpus."""
    if mode not in ("driver", "distributed", "auto"):
        raise ValueError(f"unknown readout mode {mode!r}")
    if mode != "auto":
        return mode
    return "driver" if int(C) <= int(max_driver_c) else "distributed"


def resolve_readout_calibration(flag):
    """`--readout-calibration {on,off}` -> bool, the single gate `main` consults.

    A helper rather than an inline comparison because "is the calibration block
    running" decides whether a run has an ECE record at all, and that decision is
    written into the manifest, printed to the driver log and read by the dev
    profile — three places that must not disagree about what the string meant.
    """
    if flag in (None, True, False):              # tolerate a pre-flag namespace
        return True if flag is None else bool(flag)
    if flag not in ("on", "off"):
        raise ValueError(f"unknown readout calibration mode {flag!r}")
    return flag == "on"


def resolve_eval_path(flag, readout_mode):
    """`--eval-path {driver,distributed}` -> the eval path this run scores on (WP-B).

    ORTHOGONAL to `--readout-mode`, which chose the FIT path (driver θ-collect vs
    executor batched L-BFGS). This chooses the EVAL path: `driver` runs the shipped
    `_densify_lean_blocks` collect + `readout_from_proba` (the O(N·C) driver collect,
    audit §5f); `distributed` scores the per-node ranking arms via
    `distributed_readout.score_cells_arms_df` / `per_node_metric_arms_rows` with
    nothing (D_te,C) reaching the driver — the path the exp 0111 episode corpus needs
    (×2.66 the docs).

    It only has meaning when the FIT was distributed (the executor-side scored frame
    and the fitted raw-θ params are what the cell explode reads); under
    `readout_mode=driver` there is no such frame, so `distributed` degrades to
    `driver` with a caller-printed note. Default `driver` until the parity gate is
    green on the current cache — this WP changes NO existing run until the default is
    flipped in a later step."""
    if flag in (None,):
        return "driver"
    if flag not in ("driver", "distributed"):
        raise ValueError(f"unknown eval path {flag!r}")
    if flag == "distributed" and readout_mode != "distributed":
        return "driver"
    return flag


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
        print(f"[driver]   readout checkpoint write FAILED ({exc}); the solve "
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
                print(f"[driver]   readout checkpoint {path.name} IGNORED: "
                      f"fingerprint {got[:12]} != {fingerprint[:12]} — it belongs "
                      "to a different arm/corpus/basis. Starting cold.", flush=True)
                return None
            return (np.asarray(z["W_std"], dtype=np.float64),
                    np.asarray(z["b_std"], dtype=np.float64),
                    int(z["iter"]))
    except Exception as exc:
        print(f"[driver]   readout checkpoint {path.name} UNREADABLE ({exc}); "
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
                         theta_topm):
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
    readout is re-run under this code — `make gated-pc-readout ID=<n>` once."""
    path = _readout_heads_path(run_dir, label)
    tmp = path.with_name(path.name + ".tmp")
    fp = _readout_heads_fingerprint(C, K, theta_topm, label, degenerate)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            # File OBJECT, not a name: np.savez appends `.npz` to a str path,
            # which would make the tmp name (and thus the rename) a guess.
            np.savez(fh,
                     version=np.str_(_HEADS_VERSION),
                     label=np.str_(label or "arm"),
                     V=np.ascontiguousarray(V, dtype=np.float64),
                     b_raw=np.ascontiguousarray(b_raw, dtype=np.float64),
                     const=np.ascontiguousarray(const, dtype=np.float64),
                     degenerate=np.ascontiguousarray(degenerate, dtype=bool),
                     C=np.int64(int(C)), K=np.int64(int(K)),
                     theta_topm=np.int64(int(theta_topm)),
                     fingerprint=np.str_(fp))
        tmp.replace(path)
        print(f"[driver]   wrote readout heads {path.name} "
              f"(C={int(C)} K={int(K)} topm={int(theta_topm)}) — "
              "conversion_analysis --deciles on scores from this, no re-fit",
              flush=True)
        return True
    except Exception as exc:                         # pragma: no cover - I/O failure
        print(f"[driver]   readout heads write FAILED ({exc}); the fit is "
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
                print(f"[driver]   readout heads {path.name} IGNORED: version "
                      f"{ver!r} != {_HEADS_VERSION!r}", flush=True)
                return None
            gC, gK, gm = int(z["C"]), int(z["K"]), int(z["theta_topm"])
            V = np.asarray(z["V"], dtype=np.float64)
            b_raw = np.asarray(z["b_raw"], dtype=np.float64)
            const = np.asarray(z["const"], dtype=np.float64)
            degenerate = np.asarray(z["degenerate"], dtype=bool)
    except Exception as exc:
        print(f"[driver]   readout heads {path.name} UNREADABLE ({exc})",
              flush=True)
        return None
    for name, want, got in (("C", C, gC), ("K", K, gK),
                            ("theta_topm", theta_topm, gm)):
        if want is not None and int(want) != got:
            print(f"[driver]   readout heads {path.name} IGNORED: {name}={got} "
                  f"!= manifest {name}={int(want)} — a different arm/run",
                  flush=True)
            return None
    if V.shape != (gC, gK):
        print(f"[driver]   readout heads {path.name} IGNORED: V shape {V.shape} "
              f"!= (C={gC}, K={gK})", flush=True)
        return None
    return V, b_raw, const, degenerate, gC, gK, gm


def _fit_readout_heads(train_scored, C, K, *, l2=1.0, gtol=_READOUT_GTOL,
                       max_iter=_READOUT_MAX_ITER, history=6, label="", depth=None,
                       warm_start=None, theta_topm=0, checkpoint_path=None,
                       checkpoint_every=10,
                       topic_col="topicDistribution",
                       label_col="label", mask_col="labelMask"):
    """Fit all C per-node readout heads with ONE batched distributed L-BFGS.

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
        print(f"[driver]   {tag}theta top-m mass: "
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
    x0 = None
    if warm_start is not None:
        W0, b0 = unfold_standardization(warm_start[0], warm_start[1], mu, sd)
        x0 = (np.where(keep[:, None], W0, 0.0), np.where(keep, b0, 0.0))
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = None
    if ckpt_path is not None:
        # Derived AFTER the moments pass because the counts come from it —
        # exact-integer sums, the reproducible identity of this arm's problem
        # (see _readout_ckpt_fingerprint for why mu/sd bytes are NOT hashed).
        fingerprint = _readout_ckpt_fingerprint(C, K, n_obs, n_pos, theta_topm)
        resumed = _read_readout_ckpt(ckpt_path, fingerprint)
        if resumed is not None:
            W_ck, b_ck, ck_iter = resumed
            # Same zeroing rule as the warm-start path above, for the same reason:
            # a degenerate node's data term is zeroed, so its objective is the bare
            # ridge and any nonzero row would be walked back to 0 over real
            # distributed passes. (A checkpoint written by THIS problem already has
            # them at zero — the fingerprint pins the degenerate mask — so this is
            # a belt-and-braces restatement of the contract, not a fixup.)
            x0 = (np.where(keep[:, None], W_ck, 0.0), np.where(keep, b_ck, 0.0))
            print(f"[driver]   {tag}resuming batched solve from checkpoint "
                  f"(iter {ck_iter} recorded); curvature history is not carried, "
                  "early iterations re-learn it", flush=True)
    print(f"[driver]   {tag}distributed readout fit: C={C} K={K}, "
          f"{int(keep.sum())} fittable nodes, {int(degenerate.sum())} degenerate "
          f"(constant fallback), observed train cells={int(n_obs.sum())}"
          f"{' (warm start)' if x0 is not None else ''}"
          f"{f' (theta top-m={theta_topm})' if theta_topm > 0 else ''}", flush=True)

    with _dr.make_spark_stats_fn(
            train_scored, C, K, mu, sd,
            fold_standardization=fold_standardization,
            standardized_grad_from_raw=standardized_grad_from_raw,
            topic_col=topic_col, label_col=label_col, mask_col=mask_col,
            depth=depth, topm=theta_topm) as stats_fn:

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
            print(f"[driver]   {_tag}batched L-BFGS iter {p['iter']}: "
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
            print(f"[driver]   {tag}could not remove readout checkpoint "
                  f"{ckpt_path.name} ({exc})", flush=True)
    V, b_raw = fold_standardization(W_std, b_std, mu, sd)
    gmax = float(info["grad_inf_norm"][keep].max()) if keep.any() else 0.0
    # `converged` = gtol OR the principled numerical stall; at gtol=1e-4 (sklearn's
    # own tol) every node should stop on the gradient, so a nonzero stalled count is
    # the diagnostic that the summed-loss roundoff floor was hit first.
    print(f"[driver]   {tag}batched L-BFGS: {int(info['n_stats_calls'])} data passes "
          f"({int(info['n_node_evals'])} node-passes, "
          f"{info['n_node_evals'] / max(int(info['n_stats_calls']), 1):.0f} avg), "
          f"{int(info['converged'][keep].sum())}/{int(keep.sum())} converged "
          f"({int(info['converged_gtol'][keep].sum())} gtol, "
          f"{int(info['stalled'][keep].sum())} stalled), max|grad|={gmax:.3g}, "
          f"max iters={int(info['n_iter'].max())}, "
          f"line-search failures={int(info['line_search_failures'])}", flush=True)
    return V, b_raw, const, degenerate, info


def distributed_score_arm(train_scored, test_scored, C, K, *, recall_targets,
                          fdr_targets, min_count=0, label="", l2=1.0,
                          gtol=_READOUT_GTOL, max_iter=_READOUT_MAX_ITER,
                          history=6, depth=None, warm_start=None, theta_topm=0,
                          checkpoint_dir=None, checkpoint_every=10,
                          topic_col="topicDistribution", label_col="label",
                          mask_col="labelMask", id_col="person_id",
                          elig_col=None):
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
        topic_col=topic_col, label_col=label_col, mask_col=mask_col)
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
                             C, K, theta_topm)
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


# The pr/detection axes score_cells_arms_df does NOT carry: PR@recall / recall@FDR
# are per-node threshold sweeps and detection is a per-doc max over ALL C nodes —
# neither is a per-(node) reduction of the exploded cells, so both are marked
# skipped on the eval_path=distributed path rather than approximated. They are
# co-fit-head / diagnostic axes, not the ranking headline the parity gate proves;
# run `--eval-path driver` on a corpus the collect fits to read them. (A distributed
# per-doc-max detection is a natural later addition; it is out of WP-B's scope.)
_EVAL_DIST_PR_SKIP = {"par": {}, "raf": {}, "n_labels_scored": 0,
                      "skipped": "eval_path=distributed (per-node PR sweep not "
                                 "carried by the cell explode; run --eval-path "
                                 "driver)"}
_EVAL_DIST_DET_SKIP = {"skipped": "eval_path=distributed (detection is a per-doc "
                                  "max over all C; not carried by the per-node "
                                  "cell explode — run --eval-path driver)"}


def _readout_from_per_node(per_node_metrics, C, *, recall_targets, fdr_targets):
    """A `readout_from_proba`-shaped dict from a distributed per-node metric table.

    `per_node_metrics` is `per_node_metric_rows` / `per_node_metric_arms_rows`'
    output — the `_bundle_masked["per_label"]` shape, one record per node. This
    reproduces `readout_from_proba`'s ranking + per_node fields from it EXACTLY
    (`analysis.pc.evaluate._macro` on the per_label, then the same scored-node
    per_node filter), and marks pr/detection skipped-distributed. It is the bridge
    that lets the eval_path=distributed ranking feed every consumer the driver
    readout feeds (`format_arm_readout`, the headline, the incident block's shared
    node set) without a (D,C) collect."""
    from analysis.pc.evaluate import _macro
    ranking_macro = _macro(per_node_metrics)
    per_node = {}
    for c in range(int(C)):
        rl = per_node_metrics.get(c, {})
        if rl.get("skipped") is None and rl.get("auc") is not None:
            per_node[c] = {"auc": float(rl["auc"]),
                           "ap": (None if rl.get("ap") is None
                                  else float(rl["ap"])),
                           "n_pos": int(rl.get("n_pos", 0))}
    return {"ranking": ranking_macro, "pr": dict(_EVAL_DIST_PR_SKIP),
            "detection": dict(_EVAL_DIST_DET_SKIP), "per_node": per_node}


def distributed_ranking_readout(test_scored, C, V, b_raw, *, recall_targets,
                                fdr_targets, min_count=0, id_col="person_id",
                                elig_col=None, theta_topm=0,
                                arm_label="gated_pc (pc_topics_lr)"):
    """Collect-free PREVALENT (+ INCIDENT) ranking readout — eval_path=distributed (WP-B).

    The eval_path=distributed replacement for `_collect_lean_proba` +
    `_densify_lean_blocks` + `readout_from_proba`'s ranking axis (+ `incident_readout`):
    `distributed_readout.score_cells_arms_df` explodes the observed-or-incident test
    cells and `per_node_metric_arms_rows` groups them by node, so nothing (D_te,C)
    reaches the driver (audit §5f — the O(N·C) collect that breaks a 16 GB driver at
    ×2.66). The incident arm rides the SAME explode via E1's `R_d` fourth CSR run
    (`elig_col`), scored on the corpus eligibility the census gated it with.

    Returns `(prevalent_readout, incident_block_or_None)`:

      * `prevalent_readout` — `readout_from_proba` shape (ranking + per_node; pr /
        detection skipped-distributed, see the module constants above);
      * `incident_block` — the `_assemble_incident_block` output built from the
        distributed incident per-node table, structurally identical to
        `incident_readout`'s so the parity gate compares two numbers not two shapes;
        `None` when `elig_col` is absent (no incident arm on this corpus).

    `theta_topm > 0` is REFUSED here: `score_cells_arms_df` is dense-θ only (spec
    R5.4 defers the top-m + doc-key + arms combination), and a silent full-θ eval of
    a top-m fit would score a model on features it was not fit on — the same trap
    `_collect_lean_proba`'s `theta_topm` guard prevents on the driver path."""
    if int(theta_topm) > 0:
        raise ValueError(
            "distributed_ranking_readout: eval_path=distributed does not yet carry "
            "top-m truncation (score_cells_arms_df is dense-θ only). Run "
            "--eval-path driver at this --readout-theta-topm, or --readout-theta-topm "
            "0 with --eval-path distributed.")
    C = int(C)
    scored = test_scored.withColumn("doc_key", _doc_key_column(test_scored, id_col))
    if elig_col is not None:
        cells = _dr.score_cells_arms_df(scored, V, b_raw, C, elig_col=elig_col)
        prev_pn, inc_pn = _dr.per_node_metric_arms_rows(cells, C, min_count=min_count)
    else:
        cells = _dr.score_cells_df(scored, V, b_raw, C)
        prev_pn = _dr.per_node_metric_rows(cells, C, min_count=min_count)
        inc_pn = None
    prevalent = _readout_from_per_node(prev_pn, C, recall_targets=recall_targets,
                                       fdr_targets=fdr_targets)
    if inc_pn is None:
        return prevalent, None
    from analysis.pc.evaluate import _macro
    inc_readout = _readout_from_per_node(inc_pn, C, recall_targets=recall_targets,
                                         fdr_targets=fdr_targets)
    # R2.1's guard is ON for the incident macro's skip accounting, exactly as the
    # driver path sets skip_constant=True — the per-node table already carries the
    # constant-column skips (per_node_metric_arms_rows scores incident with
    # skip_constant=True), so its _macro reproduces the three-reason counts.
    inc_readout["ranking"] = {"skipped_by_reason":
                              _macro(inc_pn)["skipped_by_reason"]}
    # n_scored_cells is exact and free from the per-node records (n_pos+n_neg over
    # every node, scored or skipped). n_eligible_cells is a (D,C) diagnostic the
    # distributed path does not materialize (eligible-but-unobserved cells are never
    # emitted); the census is its authoritative source, so it is left None here.
    n_scored = int(sum(int(r.get("n_pos", 0)) + int(r.get("n_neg", 0))
                       for r in inc_pn.values()))
    block = _assemble_incident_block(
        inc_readout, prevalent, min_count, arm_label,
        n_eligible_cells=None, n_scored_cells=n_scored)
    return prevalent, block


def readout_ab_report(train_scored, test_scored, C, K, *, recall_targets,
                      fdr_targets, min_count=0, label="", seed=0, n_rows=2000,
                      sample_frac=1.0, distributed=None,
                      max_iter=_READOUT_MAX_ITER, theta_topm=0):
    """A/B the distributed readout against the driver readout on the SAME θ.

    The plan's correctness gate (step 2: "cardiovascular A/B equality run vs the
    driver readout, same frozen fit, same seed"). It REPORTS — no asserts — because
    the two paths are not expected to agree to machine precision and the size of the
    disagreement is the finding: sklearn stops at `tol=1e-4` on its own gradient,
    which leaves it ~5e-4 from the optimum in predicted probability, and it is the
    LESS converged of the two parties. Per-node AUC deltas of ~1e-4 and macro deltas
    below 1e-4 are the expected outcome; anything at 1e-2 is a formulation bug.

    `sample_frac < 1.0` subsamples the frames ONCE, up front, and runs BOTH paths on
    those exact cached rows — the gate has to compare two solvers on one dataset, so
    letting only the driver side sample (as the production driver path does, to fit
    its collect in 8g) would report the sampling, not the solver. That also makes the
    gate affordable at cardiovascular scale on the standard 8g driver: exp 0102 could
    only afford `readout_sample_frac=0.3` there. The subsample itself is doc-key
    hashed (R5.5, seam 6), not `DataFrame.sample()` — whole documents are kept or
    dropped as units, so a person's several docs are never split between "sampled
    in" and "sampled out" by the row-position accident `.sample()` would allow.

    `distributed` optionally passes in an already-computed
    `distributed_score_arm` result so the gate costs one extra readout, not two —
    but only at `sample_frac=1.0`, since a passed-in result was fit on all the
    rows. At `sample_frac < 1.0` its FIT PARAMS still travel: the sampled problem
    is the same C convex problems on a subset of their rows, so the full-data
    `(V, b_raw)` is a legitimate warm start for it (mapped through the sampled
    fit's own moments inside `_fit_readout_heads`), and this gate is exactly the
    capped-budget regime where that is worth passes.

    `theta_topm > 0` is passed to the distributed side so the gate reports the path
    the run actually took — but it then measures TRUNCATION + solver against the
    full-θ driver oracle, not the solver alone, and the report says so. Run the gate
    at `theta_topm=0` to isolate the solver; run it at the production `theta_topm` to
    price what the truncation costs in per-node AUC."""
    from analysis.pc.evaluate import _lr_proba_per_label_masked

    tag = f"{label} " if label else ""
    sampled = []
    warm = distributed[5] if (distributed is not None
                              and len(distributed) > 5) else None
    if sample_frac < 1.0:
        # DOC-KEY-keyed filter (R5.5/seam 6), not Spark's `.sample()`. `.sample()`
        # draws a Bernoulli trial per ROW INDEX within a partition — a function of
        # the query PLAN (partitioning, upstream shuffles), not of document
        # identity. That is fine as long as exactly one physical read of ONE
        # DataFrame feeds both paths (true below, since `train_scored`/
        # `test_scored` are reassigned and cached before either collect runs), but
        # it silently stops being "the same sample" the moment a caller re-derives
        # these frames a different way (a different upstream filter, a differently
        # partitioned bundle) and expects `seed`+`sample_frac` alone to reproduce
        # it — which is exactly what this gate's OWN log line below promises.
        # Hashing the int64 doc key (`_doc_key_column`, spec R5.4) makes the kept
        # set a pure function of WHICH DOCUMENTS exist plus the seed — reproducible
        # under a repartition or a differently-ordered upstream, which a
        # two-docs-per-person fixture is what tells apart from `.sample()`'s
        # position luck (`_doc_key_sample`'s own docstring; see
        # test_multidoc_seams.py::test_doc_key_sample_is_order_and_partition_independent).
        # This samples DOCUMENTS independently, same grain as the alignment dict
        # below (doc key, not person) — a multi-doc person's several documents can
        # land on either side, same as any other document; keeping a person's
        # documents TOGETHER is a different property (A3's calibration split,
        # where straddling cal/fit is the correctness bug being fixed).
        #
        # cache(): both paths must read the SAME rows, and an uncached filter is
        # re-evaluated on every action (same seed, but also the same recompute cost).
        train_scored = _doc_key_sample(train_scored, sample_frac, seed).cache()
        test_scored = _doc_key_sample(test_scored, sample_frac, seed).cache()
        sampled = [train_scored, test_scored]
        print(f"[driver]   A/B gate: both paths restricted to the SAME "
              f"{sample_frac:g} doc-key sample (seed={seed})", flush=True)
        distributed = None                    # the passed-in fit saw all the rows
    if distributed is None:
        distributed = distributed_score_arm(
            train_scored, test_scored, C, K, recall_targets=recall_targets,
            fdr_targets=fdr_targets, min_count=min_count, label=label,
            max_iter=max_iter, warm_start=warm, theta_topm=theta_topm)
    r_dist, p_dist, _, _, ids_dist = distributed[:5]

    Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(train_scored, C)
    Pi_te, y_te, m_te, ids_drv = _collect_theta_labels(test_scored, C)
    p_drv = _lr_proba_per_label_masked(Pi_tr, y_tr, m_tr, Pi_te, C)
    # R5.7: both sides of the gate dedup their detection pool to persons, or an
    # A/B "agreement" on that one field would really be two different multi-doc
    # weightings coincidentally landing close.
    r_drv = readout_from_proba(p_drv, y_te, m_te, C, recall_targets=recall_targets,
                               fdr_targets=fdr_targets, min_count=min_count,
                               doc_keys=ids_drv)

    def _fmt(v):
        return "n/a" if v is None else f"{v:.6f}"

    def _delta(a, b):
        return "n/a" if (a is None or b is None) else f"{a - b:+.2e}"

    lines = [f"A/B readout equality gate ({tag}C={C}, K={K}): distributed vs driver"]
    for key in ("auc", "ap"):
        a, b = r_dist["ranking"][key], r_drv["ranking"][key]
        lines.append(f"  macro {key.upper():3s} dist={_fmt(a)} driver={_fmt(b)} "
                     f"(Δ{_delta(a, b)})")
    pd_, pv = r_dist["per_node"], r_drv["per_node"]
    both = sorted(set(pd_) & set(pv))
    if both:
        d_auc = np.array([abs(pd_[c]["auc"] - pv[c]["auc"]) for c in both])
        worst = both[int(np.argmax(d_auc))]
        lines.append(f"  per-node |ΔAUC| over {len(both)} shared nodes: "
                     f"max={d_auc.max():.2e} (node {worst}), mean={d_auc.mean():.2e}, "
                     f"n>1e-3={int((d_auc > 1e-3).sum())}")
    lines.append(f"  nodes scored: dist={r_dist['ranking']['n_labels_scored']} "
                 f"driver={r_drv['ranking']['n_labels_scored']}")
    # Row-wise probability agreement on a sampled subset, aligned by int64 DOC KEY
    # (the two collects walk the same partitions but neither promises an order).
    # The key is per-DOCUMENT (`_doc_key_column`, spec R5.4), so under multi-doc a
    # person's several documents align one-to-one instead of colliding in this
    # dict — the seam that made person-keyed alignment silently overwrite rows.
    pos = {int(k): i for i, k in enumerate(ids_dist)}
    rows = [(i, pos[int(k)]) for i, k in enumerate(ids_drv) if int(k) in pos]
    if rows:
        rng = np.random.default_rng(int(seed))
        take = rng.choice(len(rows), size=min(int(n_rows), len(rows)), replace=False)
        i_drv = np.array([rows[t][0] for t in take])
        i_dst = np.array([rows[t][1] for t in take])
        dp = np.abs(p_dist[i_dst].astype(np.float64) - p_drv[i_drv])
        lines.append(f"  max |Δp| over {len(take)} sampled rows x {C} nodes: "
                     f"{float(dp.max()):.2e} (mean {float(dp.mean()):.2e})")
    lines.append("  (sklearn's tol=1e-4 stopping rule is the less-converged party; "
                 "~5e-4 in p / ~1e-4 in per-node AUC is the expected disagreement.)")
    if int(theta_topm) > 0:
        lines.append(f"  NOTE: the distributed side ran on top-m truncated theta "
                     f"(m={int(theta_topm)}) and the driver side on full theta, so "
                     "these deltas price the TRUNCATION, not the solver.")
    print("\n".join("[driver]   " + ln for ln in lines), flush=True)
    for df in sampled:
        df.unpersist()
    return {"distributed": r_dist, "driver": r_drv}


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
            Pi_te, y_te, mte, ids_te = _collect_theta_labels(m.transform(bundle.test_df), C)
            arm = score_arm(Pi_tr, y_tr, mtr, Pi_te, y_te, mte, C,
                            recall_targets=rt, fdr_targets=ft,
                            min_count=args.min_label_count, doc_keys=ids_te)
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
        headIntercept=bool(getattr(args, "head_intercept", False)),
        headStandardize=bool(getattr(args, "head_standardize", False)),
        optimizeDocConcentration=args.optimize_doc_concentration,
        frontierCol="frontier", gateNBg=args.n_bg, gateTpn=args.tpn,
        localizeHead=bool(getattr(args, "localize_head", False)),
        headSupport=str(getattr(args, "head_support", "siblings")),
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
    if closure_parents is not None:
        est.setClosureParents(closure_parents)
    return est


# --------------------------------------------------------------------------- #
# Multi-domain / Mondo corpus seam (shared by the fit and gated_pc_readout).   #
# --------------------------------------------------------------------------- #
# The `dag_source` values that route through the Mondo branch of the multi-domain
# assembler: a POPULATION index, `min_n=0` (the DAG arrives already powered) and
# the Mondo build inputs folded into the key. `mondo` is exp 0088/0104's powered
# ANCHOR hierarchy (`mondo_dag`, OMOP-concept-id nodes); `mondo_native` is exp
# 0110's native Mondo label space (`mondo_native_dag`, Mondo-term-id nodes).
# `gated_pc_readout` imports this rather than re-listing it, so the fit's routing
# and the re-readout's key routing cannot drift apart.
_MONDO_DAG_SOURCES = ("mondo", "mondo_native")

# exp 0111 WP-D2. `EpisodeDocSpec().name` read once, so the driver's episode wiring
# and the cache key agree on the doc-unit token without importing the class at
# module import time (the assembler modules stay lazy-imported). The literal
# tagged onto every episode/random document's `source_cohort` — the two arms are
# whole-population single-cohort corpora, so one tag (not the population index's
# cancer/general split) is the honest label, applied IDENTICALLY to both arms so
# they differ only in index location (D12). The doc_id becomes
# "episode:{person}:{index_date}" on both arms; the arm identity rides the cache
# key (episode_sampling.arm), not the doc_id.
_EPISODE_DOC_SPEC_NAME = "episode"
_EPISODE_SOURCE_COHORT = "episode"

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
    """The corpus_manifest block for a multi-domain / Mondo fit.

    Everything the bundle's cache key needs AND everything re-assembling it from
    scratch needs, in one dict — including `billing` and the Mondo build inputs,
    which a post-hoc rebuild cannot invent. `min_n` is the EFFECTIVE value handed
    to the assembler (0 on the Mondo path: that DAG is already powered), not the
    CLI default, because the key folds what was used.

    `dag_source` has two Mondo flavours and they share every build input
    (`mondo_version` / `mondo_branch` / `min_positives` / `mondo_cache_dir`) and
    every assembler setting (population index, min_n=0 because the DAG arrives
    already powered):
      `mondo`         exp 0088/0104's powered ANCHOR hierarchy (mondo_dag), keyed
                      on OMOP concept ids, optionally spliced by `dag_collapse`;
      `mondo_native`  exp 0110's native label space (mondo_native_dag), keyed on
                      Mondo term ids, with the splice intrinsic to the build.
    `dag_collapse` is pinned False on the native path: its reduction is not an
    option there, it is part of the construction, so exposing the flag would only
    let a run ask for it twice."""
    mondo = args.dag_source in _MONDO_DAG_SOURCES
    # exp 0111 WP-D2: when an --index-arm is chosen the Mondo corpus is assembled
    # on a driver-built EXTERNAL episode/random index (see mondo_assemble_fn), not
    # the population/disease index. index_mode="external" is what routes the
    # assembler onto the injection seam, and the doc unit becomes EpisodeDocSpec
    # (doc_id = cohort:person:index) so a person's several documents stay distinct.
    episode_arm = str(getattr(args, "index_arm", "") or "")
    if episode_arm and not mondo:
        raise ValueError(
            f"--index-arm {episode_arm!r} is the exp-0111 episode/random index and "
            "is defined only on the Mondo path (--dag-source mondo/mondo_native); "
            f"got --dag-source {args.dag_source}.")
    if episode_arm:
        base_index_mode = "external"
    else:
        base_index_mode = "population" if mondo else "disease"
    return {
        "dag_source": args.dag_source,
        "disease": args.disease, "cdr": args.cdr, "billing": args.billing,
        "source_table": args.source_table,
        "extra_domains": list(extra_domains),
        "index_mode": base_index_mode,
        "person_mod": args.person_mod, "vocab_size": args.vocab_size,
        "min_df": args.min_df, "min_patient_count": args.min_patient_count,
        "doc_min_length": args.doc_min_length,
        "min_n": 0 if mondo else args.min_n,
        "holdout_frac": args.holdout_frac, "n_bg": args.n_bg, "tpn": args.tpn,
        "strip_mode": args.strip_mode, "lookback_days": args.lookback_days,
        "label_window_days": args.label_window_days,
        "label_mask_mode": args.label_mask_mode, "emit_labels": True,
        "window_mode": args.window_mode,
        "prior_obs_days": args.prior_obs_days, "window_days": args.window_days,
        "mondo_version": args.mondo_version if mondo else "",
        "mondo_branch": (args.mondo_branch or "") if mondo else "",
        "min_positives": args.min_positives if mondo else 0,
        "mondo_cache_dir": args.mondo_cache_dir if mondo else "",
        # exp 0109's splice-to-fixpoint DAG reduction. Mondo-only (it names class
        # nodes of the Mondo hierarchy) and OFF by default, so an existing
        # experiment's spec — and therefore its bundle key — is unchanged.
        "dag_collapse": bool(args.dag_collapse)
                        if args.dag_source == "mondo" else False,
        # The doc unit this corpus is assembled under — a cache-key input as of
        # R5.3, recorded in the spec (and hence the manifest) so a re-readout
        # recomputes the fit's own key. On the episode/random arms it is
        # EpisodeDocSpec (doc_id = cohort:person:index), which moves the key
        # naturally; otherwise it is today's constant (PatientCohortDocSpec).
        "doc_spec": (_EPISODE_DOC_SPEC_NAME if episode_arm
                     else doc_spec_identity()),
        # exp 0111 WP-D2: the external-index identity. Present ONLY on an episode/
        # random arm, so every population/disease/mondo spec keys byte-identically
        # to before. `arm` is what distinguishes the two arms' bundles; the rest are
        # the gate/cap/salt parameters that determine which index rows exist.
        # `sidecar_uri` is deliberately OUTSIDE this dict — the sidecar is keyed
        # independently and is not corpus identity for the bundle (it never enters
        # the key), it only tells the MISS closure where to load/build it.
        **({"episode_sampling": {
                "arm": episode_arm,
                "gap_days": int(args.episode_gap_days),
                "cap": int(args.episode_cap),
                "salt": str(args.episode_salt),
                "prior_obs_days": int(args.episode_prior_obs_days),
                "window_days": int(args.episode_window_days)},
            "episode_sidecar_uri": str(getattr(args, "episode_sidecar_uri", "")
                                       or "")}
           if episode_arm else {}),
        # exp 0111 WP-D3 (D13): the gate/label SEPARATION identity. Present ONLY on
        # an episode/random arm WITH a positive gate width (`--gate-frontier-days`),
        # so an un-gated episode bundle and every non-episode bundle key byte-
        # identically to before. A D13-gated bundle carries the SAME 365-day
        # `label`/`labelMask` but a 90-day `frontier`, so it is a DISTINCT artifact —
        # the mode token (`gate90d`) is what splits its cache key from the un-gated
        # one. Kept as a SIBLING of `episode_sampling` (not folded into that dict) so
        # WP-D2's pinned sampling-dict shape is unchanged.
        **({"gate_frontier_mode": _gate_frontier_mode(
                int(getattr(args, "gate_frontier_days", 0) or 0))}
           if (episode_arm
               and int(getattr(args, "gate_frontier_days", 0) or 0) > 0) else {}),
        # E1's pre-index closure column. Mondo-only — it is built by re-running
        # the SAME attestation provider on the feature window, and only the Mondo
        # paths construct one driver-side — and OFF by default, so an existing
        # experiment's spec (and therefore its bundle key) is unchanged.
        "preindex_closure": (bool(getattr(args, "preindex_closure", False))
                             if mondo else False),
    }


def _multidomain_params(spec):
    """`(assembly_params, key_extra)` for one corpus spec.

    `key_extra` carries the identity markers that are NOT assembler kwargs — the
    `multidomain` / `mondo` flags and the Mondo build inputs — which is what folds
    them into the cache key without changing any SNOMED key.

    Both Mondo flavours fold the same build inputs; `mondo_native` additionally
    folds its own marker + version, and (in `compute_bundle_cache_key`) the source
    hashes of the two NEW modules it is built from. Everything native is folded
    ONLY when it is selected, so a `dag_source=mondo` key — exp 0104's and 0109's
    — is byte-identical to what it was before exp 0110 existed."""
    dag_source = str(spec.get("dag_source", "snomed"))
    mondo = dag_source in _MONDO_DAG_SOURCES
    assembly = {k: spec[k] for k in _SPEC_ASSEMBLY_KEYS}
    assembly["extra_domains"] = tuple(spec.get("extra_domains") or ())
    assembly["emit_labels"] = True
    # `doc_spec` is a key-only identity (the assembler builds its own spec and
    # takes no such kwarg), so it rides key_extra. A spec written before the field
    # existed defaults to today's constant and therefore keys byte-identically.
    key_extra = {"multidomain": True,
                 "doc_spec": str(spec.get("doc_spec") or doc_spec_identity())}
    # exp 0111 WP-D2: fold the external-index identity ONLY on the external path,
    # so every population/disease/mondo key stays byte-identical (the `dag_collapse`
    # discipline). `index_mode` itself already rides `assembly` (and hence the key)
    # via _SPEC_ASSEMBLY_KEYS; this adds the arm + gate/cap/salt parameters that
    # `index_mode="external"` alone does not name.
    if str(spec.get("index_mode")) == "external" and spec.get("episode_sampling"):
        key_extra["episode_sampling"] = dict(spec["episode_sampling"])
    # exp 0111 WP-D3 (D13): fold the gate/label-separation identity ONLY on the
    # external path and ONLY when a gate is requested, so an un-gated episode bundle
    # (and every non-episode bundle) keys byte-identically. A gated bundle carries a
    # 90-day `frontier` under the same 365-day label, so it is a distinct artifact.
    if (str(spec.get("index_mode")) == "external"
            and spec.get("gate_frontier_mode")):
        key_extra["gate_frontier_mode"] = str(spec["gate_frontier_mode"])
    if mondo:
        key_extra.update(mondo=True, mondo_version=spec["mondo_version"],
                         mondo_branch=spec.get("mondo_branch") or "",
                         min_positives=spec["min_positives"])
        # Folded ONLY when on: a collapse-OFF Mondo spec must key byte-identically
        # to one written before exp 0109 existed (0104's cached bundle).
        if spec.get("dag_collapse"):
            from mondo_collapse import DAG_COLLAPSE_VERSION
            key_extra.update(dag_collapse=True,
                             dag_collapse_version=DAG_COLLAPSE_VERSION)
        if dag_source == "mondo_native":
            from mondo_native_dag import MONDO_NATIVE_VERSION
            key_extra.update(mondo_native=True,
                             mondo_native_version=MONDO_NATIVE_VERSION)
        # E1. Folded ONLY when on (R1.2): a bundle carrying the pre-index column
        # is a different artifact, but a spec that does not ask for it must key
        # byte-identically to one written before this existed.
        if spec.get("preindex_closure"):
            from preindex_closure import PREINDEX_CLOSURE_VERSION
            key_extra.update(preindex_closure=True,
                             preindex_closure_version=PREINDEX_CLOSURE_VERSION)
    return assembly, key_extra


def multidomain_cache_key(spec) -> str:
    """The bundle cache key for a multi-domain / Mondo corpus spec."""
    from _case_finding_cache import _KEY_PARAM_NAMES, compute_bundle_cache_key
    assembly, key_extra = _multidomain_params(spec)
    key_params = {k: assembly[k] for k in _KEY_PARAM_NAMES if k in assembly}
    key_params.update(key_extra)
    return compute_bundle_cache_key(**key_params)


# --------------------------------------------------------------------------- #
# exp 0111 WP-D2: the episode / matched-random EXTERNAL index seam.            #
# --------------------------------------------------------------------------- #
# Everything here runs ONLY inside the Mondo assemble closure, i.e. ONLY on a
# bundle cache MISS — a HIT reloads the baked bundle and pays for none of the
# sidecar load or the index build. The index frame is handed VERBATIM to
# multi_domain's `index_mode="external"` seam (WP-C), so no episode logic lands in
# any source-hashed module; this driver owns all of it.


def _resolve_episode_sidecar_uri(spec, cache_uri):
    """Where the first-attestation sidecar the episode index reads lives.

    Explicit `--episode-sidecar-uri` wins; otherwise the same default
    `build-conversion-sidecar` uses — a `conversion_sidecar` sibling of the bundle
    cache. The sidecar is keyed INDEPENDENTLY of the bundle (its own
    `conversion_sidecar_key`), so a bundle-key move never orphans it and it belongs
    beside, never inside, a bundle key's dir."""
    uri = str(spec.get("episode_sidecar_uri") or "").rstrip("/")
    if uri:
        return uri
    if cache_uri:
        return f"{str(cache_uri).rstrip('/')}/conversion_sidecar"
    raise ValueError(
        "exp 0111 episode index needs a sidecar location on a cache MISS: pass "
        "--episode-sidecar-uri (a persistent in-boundary bucket) or --cache-uri "
        "(the sidecar defaults to <cache-uri>/conversion_sidecar).")


def _load_or_build_first_attestation(spark, spec, *, sidecar_uri, code_map_norm,
                                     code_map_identity, dag_source):
    """The E4 first-attestation frame `(person_id, node_cid, first_attested_date)`.

    Load the sidecar at its own key; on a MISS build just the first-attestation
    half from the code map already in hand (the DAG build is happening in this same
    closure) and save it. The horizon half (`build_and_save`'s second parquet) is
    index-dependent and unused here, so it is deliberately NOT built — a later
    conversion analysis rebuilds it under the same key if it needs it. Runs only on
    a bundle cache MISS, and the sidecar itself is usually a HIT (the probe or a
    prior fit built it), so the full-history scan is paid at most once per cluster."""
    from conversion_sidecar import (build_conversion_sidecar,
                                     conversion_sidecar_key, save_sidecar,
                                     try_load_sidecar)
    key = conversion_sidecar_key(
        cdr=spec["cdr"], person_mod=int(spec["person_mod"]),
        dag_source=dag_source, mondo_version=spec.get("mondo_version") or "",
        mondo_branch=spec.get("mondo_branch") or "",
        min_positives=int(spec.get("min_positives") or 0),
        code_map_identity=code_map_identity)
    first = try_load_sidecar(spark, sidecar_uri, key)
    if first is not None:
        print(f"[episode]   sidecar HIT ({sidecar_uri})", flush=True)
        return first
    print(f"[episode]   sidecar MISS — building first-attestation once "
          f"({sidecar_uri})", flush=True)
    from charmpheno.omop import load_omop_bigquery
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=spec["cdr"], billing_project=spec["billing"],
        person_sample_mod=int(spec["person_mod"]), source_table="condition_era")
    first = build_conversion_sidecar(cond, code_map_norm)
    try:
        save_sidecar(first, sidecar_uri, key)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[episode]   WARNING: sidecar write to {sidecar_uri} failed "
              f"({type(exc).__name__}: {exc}); proceeding with the in-memory "
              "frame (next MISS rebuilds it).", flush=True)
    return first


def _build_external_index_df(spark, spec, *, code_map_norm, code_map_identity,
                             dag_source, cache_uri):
    """`(index_df, ordinal_df)` for the episode / matched-random arm (WP-D2).

    `index_df` = `(person_id, index_date, source_cohort)` — exactly the shape
    multi_domain's external seam consumes; `source_cohort` is a single literal (both
    arms are whole-population single-cohort corpora). `ordinal_df` =
    `(person_id, index_date, episode_ordinal)` for the episode arm — WP-D1's
    UNBOUNDED chronological ordinal, carried for the R7.5 drop-rate diagnostic and
    joined onto the bundle by `_attach_bounded_doc_index`; `None` for the random arm
    (a uniform draw has no episode ordinal). `first_attestation` (the E4
    `(person_id, node_cid, first_attested_date)` sidecar frame) is returned so the
    WP-D3 gate-frontier post-pass reuses it WITHOUT reloading the sidecar — it is
    already loaded here, and the load is the expensive part.

    D12 MATCHED: the random arm is drawn over the EPISODE arm's surviving persons and
    shares every gate/cap/salt, so the two arms differ only in index location."""
    from pyspark.sql import functions as F

    from episode_index import (INDEX_COL, PERSON_COL, EPISODE_COL,
                               episode_index_frame, random_index_frame)
    samp = spec["episode_sampling"]
    arm = str(samp["arm"])
    sidecar_uri = _resolve_episode_sidecar_uri(spec, cache_uri)
    first = _load_or_build_first_attestation(
        spark, spec, sidecar_uri=sidecar_uri, code_map_norm=code_map_norm,
        code_map_identity=code_map_identity, dag_source=dag_source)
    from conversion_sidecar import load_observation_period
    obs = load_observation_period(spark, cdr=spec["cdr"], billing=spec["billing"])

    common = dict(prior_obs_days=int(samp["prior_obs_days"]),
                  window_days=int(samp["window_days"]),
                  cap=int(samp["cap"]), salt=str(samp["salt"]))
    episode = episode_index_frame(first, obs, gap_days=int(samp["gap_days"]),
                                  **common)
    if arm == "episode":
        index = episode.select(PERSON_COL, INDEX_COL)
        ordinal_df = episode.select(
            PERSON_COL, INDEX_COL,
            F.col(EPISODE_COL).alias("episode_ordinal"))
    elif arm == "random":
        # The episode arm's SURVIVING persons (post-gate) define the matched
        # population; the random draw shares the gates and cap.
        persons = episode.select(PERSON_COL).distinct()
        index = random_index_frame(obs, persons=persons, **common)
        ordinal_df = None
    else:
        raise ValueError(f"episode_sampling.arm must be 'episode' or 'random', "
                         f"got {arm!r}")
    index_df = index.withColumn(
        "source_cohort", F.lit(_EPISODE_SOURCE_COHORT))
    return index_df, ordinal_df, first


def _attach_bounded_doc_index(bundle, *, ordinal_df=None):
    """Bake the BOUNDED within-corpus document index onto the bundle's frames (WP-D2).

    `_doc_key_column` (WP-A1) reads a column literally named `episode_no` and REQUIRES
    it in `[0, RADIX=64)` — it is the low bits of `doc_key = person_id*64 + episode_no`.
    WP-D1's `episode_no` is the UNBOUNDED chronological ordinal (a chronic patient's
    70th episode carries 70) and must NEVER reach the doc key. So here, at the one
    point the whole corpus is in hand, we synthesize a DENSE per-person
    `row_number()-1` (0-based) over each person's KEPT documents ordered by index_date,
    and write THAT as `episode_no`. cap <= 3 << 64, so the bound holds; the
    `episode_no < 64` guard in `_doc_key_column` and `_assert_unique_doc_keys` are the
    tripwires if the raw ordinal ever leaked through instead.

    WHY DERIVE FROM THE BUNDLE, NOT CARRY THE INDEX FRAME'S ORDINAL
    ---------------------------------------------------------------
    The bounded index must be reconstructible AT READOUT from what the bundle carries.
    The bundle carries `doc_id = "episode:{person}:{index_date}"` (EpisodeDocSpec), so
    a dense rank over the parsed `index_date` within `person_id` reconstructs the exact
    same 0-based index the fit used — no fit-only state, no drift. Baking it as a real
    column means every readout collect (which transforms these frames) sees it with no
    readout edit, and it survives the parquet save/reload byte-for-byte.

    Person-keyed split (all of a person's documents land on one side), so the dense
    rank computed independently within train and within test equals the within-person
    rank globally.

    `ordinal_df` (episode arm only): WP-D1's UNBOUNDED ordinal `(person_id, index_date,
    episode_ordinal)`, left-joined so R7.5's drop-rate-by-ordinal diagnostic can read
    it. When absent (random arm), `episode_ordinal` mirrors the bounded index — a
    uniform draw has no chronological episode number."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    def _with_index(df):
        # doc_id = "episode:{person}:{index_date}"; the index_date is the LAST
        # component (EpisodeDocSpec APPENDS it), recovered by splitting on ':'.
        idx_str = F.element_at(F.split(F.col("doc_id"), ":"), -1)
        df = df.withColumn("_idx_str", idx_str)
        w = Window.partitionBy("person_id").orderBy(
            F.col("_idx_str").asc(), F.col("doc_id").asc())
        df = df.withColumn("episode_no",
                           (F.row_number().over(w) - F.lit(1)).cast("long"))
        if ordinal_df is not None:
            ordj = ordinal_df.select(
                F.col("person_id"),
                F.col("index_date").cast("string").alias("_idx_str"),
                F.col("episode_ordinal").cast("long").alias("episode_ordinal"))
            df = df.join(F.broadcast(ordj), on=["person_id", "_idx_str"],
                         how="left")
        else:
            df = df.withColumn("episode_ordinal",
                               F.col("episode_no") + F.lit(1))
        return df.drop("_idx_str")

    bundle.train_df = _with_index(bundle.train_df)
    bundle.test_df = _with_index(bundle.test_df)
    return bundle


# --------------------------------------------------------------------------- #
# exp 0111 WP-D3 (D13): separate the estimator's GATE from the outcome LABEL.  #
# --------------------------------------------------------------------------- #
# The gate an episode document reads (`frontierCol="frontier"`) becomes its
# 90-day PRESENTATION window [index, index+90d); the outcome the PC head reads
# (`labelCol="label"`) stays the full 365-day forward frame. `none` is the
# byte-identical default everywhere (no separation — the 365-day frontier the
# assembler baked stands); a positive width W is `gateWd`, the token that both
# moves the bundle cache key (a gated bundle is a DISTINCT artifact) and drives
# the MISS-only post-pass window.

_GATE_FRONTIER_NONE = "none"


def _gate_frontier_mode(gate_days) -> str:
    """The gate-frontier identity token for a forward-window width (days).

    0 (or off) -> 'none'; a positive W -> 'gateWd' (e.g. 'gate90d'). The token is
    the cache-key identity — it is what makes a D13-gated episode bundle a distinct
    cached artifact from the un-gated (365-day-frontier) one — and, parsed back by
    `_gate_frontier_days`, the driver of the post-pass gate window."""
    d = int(gate_days or 0)
    return _GATE_FRONTIER_NONE if d <= 0 else f"gate{d}d"


def _gate_frontier_days(mode) -> int:
    """Inverse of `_gate_frontier_mode`: the width (days) a mode token names, or 0
    for 'none'/absent. A malformed token RAISES rather than silently disabling the
    gate — a wrong-results hazard, never a quiet no-op."""
    import re
    m = str(mode or "").strip()
    if not m or m == _GATE_FRONTIER_NONE:
        return 0
    match = re.fullmatch(r"gate(\d+)d", m)
    if not match:
        raise ValueError(f"unrecognized gate_frontier_mode {mode!r}; expected "
                         "'none' or 'gate<days>d' (e.g. 'gate90d').")
    return int(match.group(1))


def _attach_gate_frontier(spark, bundle, *, first_attestation, before_dag,
                          gate_days, n_bg, tpn):
    """exp 0111 WP-D3 (D13): overwrite each document's GATE with its 90-day
    presentation window, leaving the 365-day outcome LABEL untouched.

    WHY A MISS-ONLY DRIVER POST-PASS, NOT A HIT-TIME TRANSFORM
    ---------------------------------------------------------
    The gate frontier is the roll-up of a document's in-window attested nodes onto
    the label DAG's SURVIVORS, and that roll-up walks the PRE-PRUNE DAG
    (`before_dag`) exactly as `attach_frontiers` does at assembly. `before_dag` is a
    build by-product that lives ONLY inside the assemble closure — it is never
    serialized into a loaded HIT bundle — so the swap MUST run here, before the
    bundle is cached; a later HIT then reloads a bundle whose `frontier` is ALREADY
    the gate. `assemble(emit_labels=True)` baked `label`/`labelMask` from the
    365-day frontier BEFORE it returned (case_finding_assembly.py:461-466), so those
    columns are already frozen at 365 days; this pass overwrites ONLY `frontier` —
    that is the whole of the D13 separation.

    THE GATE = [index, index + gate_days). For each document (doc_id =
    "episode:{person}:{index_date}") the gate's attested node set is the sidecar
    `node_cid`s whose FIRST attestation lands in the half-open forward window
    [index, index+gate_days) — the SAME join `diag_episode_probe.gate_occupancy`
    (WP-B2) measures. Half-open matches the label window's own convention (a code
    exactly at index+gate_days is OUTSIDE the gate). An empty gate yields `[]`, a
    valid background-only frontier the gated LDA accepts — the document is NEVER
    dropped.

    ROLL-UP BY CALLING `attach_frontiers` (never editing it). Concept-id
    attestations climb to engine-id survivors through the identical
    `attach_frontiers(attested_df, before_dag, keep, cid2int, lay)` the ordinary
    frontier path uses; only the INPUT differs (the 90-day attested set rather than
    the 365-day one), so the gate frontier is the ordinary frontier over a narrower
    window.

    KEEP == the survivor set (proven, not assumed). At assembly `keep =
    after_dag.nodes()` and `(parent_int, int2cid, cid2int) = after_dag.to_engine()`;
    `to_engine` maps the anchor to 0 and EVERY other node of `after_dag.nodes()` to
    1..N, so `set(cid2int) == after_dag.nodes() == keep`. We therefore read the
    survivor set straight off the bundle as `set(bundle.cid2int)` and reconstruct
    `lay = DagLayout(bundle.parent_int, n_bg, tpn)` identically to the fit
    (gated_pc_cloud.py's fit path), so the gate frontier lives in the SAME engine-id
    node space as the label — the head's C rows and the gate's blocks stay aligned.

    JOIN INTEGRITY, LOUD. Every bundle document must receive EXACTLY ONE new
    frontier. The per-document gate frame is asserted unique on (person_id, doc_id);
    the overwrite is a LEFT join (so a document with no gate row surfaces as a NULL
    we catch, not a vanished row) that must not change the document count and must
    leave NO document without a gate. A missing / duplicate / mismatched join RAISES
    — it never silently falls back to an empty or stale frontier. (An EMPTY gate is
    a present `[]`, distinct from an ABSENT document's null, so the null check
    separates the two.)
    """
    from spark_vi.models.topic.dag_placement import DagLayout
    from charmpheno.omop.case_finding_assembly import attach_frontiers

    keep = set(bundle.cid2int)               # == after_dag.nodes() (proven above)
    cid2int = bundle.cid2int
    lay = DagLayout(bundle.parent_int, n_bg=n_bg, tpn=tpn)

    def _gate(df):
        attested = _gate_attested_frame(df, first_attestation, gate_days=gate_days)
        gate_fr = attach_frontiers(attested, before_dag, keep, cid2int, lay)
        return _overwrite_frontier(df, gate_fr)

    bundle.train_df = _gate(bundle.train_df)
    bundle.test_df = _gate(bundle.test_df)
    return bundle


def _gate_attested_frame(df, first_attestation, *, gate_days):
    """Per document, the CONCEPT-id node set attested inside its 90-day gate.

    Returns `(person_id, doc_id, attested_cids: array<bigint>)` — the shape
    `attach_frontiers` consumes. For each document (doc_id =
    "episode:{person}:{index_date}") the set is the `first_attestation` `node_cid`s
    whose FIRST attestation lands in the half-open forward window
    [index, index+gate_days) — the SAME join `diag_episode_probe.gate_occupancy`
    (WP-B2) measures. A LEFT join keyed on `person_id` plus a `when(in_win, node_cid)`
    keeps a document with NO in-window attestation as a present row with an EMPTY
    `attested_cids` (collect_set drops the off-window nulls), never dropping it —
    an empty gate is a valid background-only frontier."""
    from pyspark.sql import functions as F

    fa = first_attestation.select(
        F.col("person_id").cast("long").alias("person_id"),
        F.col("node_cid").cast("long").alias("node_cid"),
        F.col("first_attested_date").cast("date").alias("first_attested_date"))
    # index_date is the LAST ':' component (EpisodeDocSpec appends it) — the same
    # parse `_attach_bounded_doc_index` uses; cast to a real date for the window.
    docs = (df.select("person_id", "doc_id").distinct()
            .withColumn("_idx",
                        F.to_date(F.element_at(F.split(F.col("doc_id"), ":"), -1))))
    in_win = ((F.col("first_attested_date") >= F.col("_idx"))
              & (F.col("first_attested_date")
                 < F.date_add(F.col("_idx"), int(gate_days))))
    return (docs.join(fa, on="person_id", how="left")
            .groupBy("person_id", "doc_id")
            .agg(F.collect_set(F.when(in_win, F.col("node_cid")))
                 .alias("attested_cids")))


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
    """A MISS-ONLY assembler for the Mondo corpus: build the DAG + climb, THEN assemble.

    The Mondo hierarchy costs a whole-Mondo -> OMOP mapping, a power-count over
    every anchor and a concept_ancestor climb — minutes of BigQuery — and the fit
    consumes NONE of it directly: the DagLayout comes from `bundle.parent_int`, and
    the frontier the climb produced is already baked into the cached
    `frontier`/`label` columns. So the build lives inside the assemble seam, where
    `load_or_build_case_finding_bundle` reaches it only after `try_load` has
    missed. A HIT pays for none of it.

    `on_inputs(count_of=..., terminal_cids=..., reduced=...)` hands back the
    by-products (the diag-only probe's per-terminal +counts) — it is NOT called on a
    HIT, which is exactly why `per_node_head_report` tolerates an empty count_of.

    `spec["dag_collapse"]` (exp 0109, default False) additionally puts the built
    engine DAG through the splice-to-fixpoint reduction before assembly — a
    DIFFERENT label DAG, hence a different cache key, hence its own bundle.

    `spec["dag_source"] == "mondo_native"` (exp 0110) swaps the whole front end:
    `mondo_native_dag` builds the label DAG out of Mondo's own is-a graph
    (closure-support powering, induced Hasse, splice) and hands back a per-code
    attestation map instead of a SNOMED climb frame. Same seam, same caching, same
    `min_n=0` contract — only the DAG and the provider differ, which is exactly
    what the `before_dag` / `attested_provider` override pair exists for.

    `spec["index_mode"] == "external"` (exp 0111 WP-D2) routes the corpus onto the
    driver-built episode / matched-random index: `_external_seam` builds the
    `(person_id, index_date, source_cohort)` frame from the E4 sidecar (MISS-only —
    a HIT never loads it), the assembler and the attestation provider both key on
    EpisodeDocSpec, and `_attach_bounded_doc_index` bakes the BOUNDED within-corpus
    `episode_no` the readout doc key needs onto the bundle. All of it is
    driver-owned; no episode logic enters a source-hashed module.

    `spec["preindex_closure"]` (E1, default False) adds a POST-PASS on the
    assembled bundle: `preindex_closure.attach_preindex_closure_to_bundle` re-runs
    the same provider over the same DAG on the FEATURE window and writes the
    per-document sparse `R_d` column plus its witness. Placed here — between the
    assembler's return and the cache write — rather than inside `multi_domain`,
    because THAT module's source hash is folded into every multi-domain bundle key
    and editing it would orphan every cached bundle including 0104's record. Same
    reasoning, same seam, same discipline as `dag_collapse` above.
    """
    native = str(spec.get("dag_source", "snomed")) == "mondo_native"

    def _with_preindex(spark, bundle, *, before_dag, provider, assembly_params):
        """The E1 post-pass, or the bundle untouched when the flag is off.

        Everything it needs about the corpus comes from `assembly_params` (the
        exact kwargs the assembler was just called with, so the window and the
        sample cannot drift from the corpus) and from the BUNDLE (`parent_int` /
        `cid2int` / `int2cid` — the POST-PRUNE internals, per R1.3, never from
        assembler internals the driver does not hold)."""
        if not spec.get("preindex_closure"):
            return bundle
        from preindex_closure import (attach_preindex_closure_to_bundle,
                                      format_preindex_report)
        with _phase("pre-index closure (E1): re-derive the feature window"):
            bundle = attach_preindex_closure_to_bundle(
                spark, bundle, before_dag=before_dag, attested_provider=provider,
                cdr=spec["cdr"], billing=spec["billing"],
                person_mod=assembly_params["person_mod"],
                lookback_days=assembly_params["lookback_days"],
                label_window_days=assembly_params["label_window_days"],
                n_bg=assembly_params["n_bg"], tpn=assembly_params["tpn"],
                index_mode=assembly_params.get("index_mode", "population"),
                disease=assembly_params.get("disease", "rare6"))
            print(format_preindex_report(bundle), flush=True)
        return bundle

    def _with_gate_frontier(spark, bundle, *, first_attestation, before_dag,
                            assembly_params):
        """The exp 0111 WP-D3 (D13) gate-frontier swap, or the bundle untouched when
        `gate_frontier_mode` is 'none'/absent.

        Runs ONLY on the external path (an --index-arm) and ONLY when a positive gate
        width is requested, so an un-gated episode/random bundle — and every non-
        episode bundle — is byte-for-byte what it was. `n_bg`/`tpn` come from
        `assembly_params` (the exact kwargs the assembler was called with), so the
        gate's DagLayout cannot drift from the label's; `before_dag` and the reused
        `first_attestation` frame both come from the closure, never from a HIT
        bundle (which carries neither)."""
        gate_days = _gate_frontier_days(spec.get("gate_frontier_mode"))
        if gate_days <= 0:
            return bundle
        with _phase(f"exp-0111 D13 gate-frontier swap "
                    f"([index, index+{gate_days}d), MISS-only)"):
            bundle = _attach_gate_frontier(
                spark, bundle, first_attestation=first_attestation,
                before_dag=before_dag, gate_days=gate_days,
                n_bg=int(assembly_params["n_bg"]), tpn=int(assembly_params["tpn"]))
        return bundle

    def _external_seam(spark, assembly_params, *, code_map_sdf, concept_col,
                       node_col, code_map_identity, dag_source):
        """exp 0111: `(index_df, assemble_doc_spec, provider_doc_spec, ordinal_df,
        first_attestation)`.

        On the external path (an --index-arm was chosen) this normalizes the label
        front end's code map, builds the driver-owned episode/random index frame from
        the sidecar (MISS-only) and selects EpisodeDocSpec for BOTH the assembler's BOW
        and the attestation provider, so the doc_id the provider derives matches the
        one the corpus is keyed on. It also hands back the E4 first-attestation frame
        so the WP-D3 gate post-pass reuses it (no second sidecar load). Off the
        external path it is a NO-OP that touches NONE of its code-map inputs: no index
        frame, PatientCohortDocSpec on the provider, the assembler builds its own
        default doc spec (doc_spec=None), and no first-attestation frame — byte-for-
        byte the prior behavior (the code map is normalized only when the episode
        index actually needs it)."""
        from charmpheno.omop.doc_spec import EpisodeDocSpec, PatientCohortDocSpec
        if str(assembly_params.get("index_mode")) != "external":
            return None, None, PatientCohortDocSpec(), None, None
        from conversion_sidecar import normalize_code_map
        code_map_norm = normalize_code_map(
            code_map_sdf, concept_col=concept_col, node_col=node_col)
        arm = str(spec["episode_sampling"]["arm"])
        with _phase(f"build exp-0111 {arm} index (external, MISS-only)"):
            index_df, ordinal_df, first = _build_external_index_df(
                spark, spec, code_map_norm=code_map_norm,
                code_map_identity=code_map_identity, dag_source=dag_source,
                cache_uri=cache_uri)
        dml = int(assembly_params.get("doc_min_length") or 0)
        return (index_df, EpisodeDocSpec(min_doc_length=dml), EpisodeDocSpec(),
                ordinal_df, first)

    def _assemble_mondo_native(spark, **assembly_params):
        from charmpheno.omop.multi_domain import (
            assemble_multidomain_case_finding_corpus)
        from mondo_native_dag import (
            MONDO_NATIVE_VERSION, build_mondo_native_fit_inputs,
            format_native_build_report, format_native_powering_report,
            make_mondo_native_attested_provider)
        build = _build_inputs or build_mondo_native_fit_inputs
        assemble = _assemble or assemble_multidomain_case_finding_corpus
        branch = spec.get("mondo_branch") or ""
        with _phase(f"build native Mondo label DAG (branch={branch or 'ALL'})"):
            before_dag, code_map_sdf, kept_cids, support_of, stats = build(
                spark, cdr=spec["cdr"], billing=spec["billing"],
                mondo_version=spec["mondo_version"],
                mondo_cache_dir=spec.get("mondo_cache_dir") or "data/mondo",
                min_positives=spec["min_positives"],
                branch_root=(branch or None))
            # exp 0111 WP-D2: build the external episode/random index (if any) and
            # pick the doc spec BOTH the provider and the assembler use. The code map
            # is normalized inside the seam ONLY on the external path, so the ordinary
            # Mondo path touches none of it.
            index_df, assemble_doc_spec, provider_doc_spec, ordinal_df, first_att = (
                _external_seam(spark, assembly_params, code_map_sdf=code_map_sdf,
                               concept_col="std_cid", node_col="node_cid",
                               code_map_identity=(
                                   f"mondo_native:{MONDO_NATIVE_VERSION}:"
                                   f"{len(kept_cids)}"),
                               dag_source="mondo_native"))
            provider = make_mondo_native_attested_provider(
                code_map_sdf, doc_spec=provider_doc_spec)
            # BOTH lines before any fit: C and K are expected to GROW here
            # (closure support >= direct support), and the plan says measure,
            # do not guess.
            print(format_native_powering_report(stats), flush=True)
            print(format_native_build_report(stats), flush=True)
            if on_inputs is not None:
                # Same by-products contract as the anchor path, in the native id
                # space: `count_of` is the per-node CLOSURE support (what
                # per_node_head_report annotates each head with) and
                # `terminal_cids` the final node set.
                on_inputs(count_of=support_of, terminal_cids=kept_cids,
                          reduced={"n_classes": stats["n_final_nodes"]
                                   - stats["n_coded_kept"], "native": stats})
        bundle = assemble(spark, before_dag=before_dag,
                          attested_provider=provider, index_df=index_df,
                          doc_spec=assemble_doc_spec, **assembly_params)
        bundle = _with_preindex(spark, bundle, before_dag=before_dag,
                                provider=provider, assembly_params=assembly_params)
        if index_df is not None:
            bundle = _attach_bounded_doc_index(bundle, ordinal_df=ordinal_df)
            bundle = _with_gate_frontier(
                spark, bundle, first_attestation=first_att,
                before_dag=before_dag, assembly_params=assembly_params)
        return bundle

    def _assemble_mondo(spark, **assembly_params):
        from charmpheno.omop.multi_domain import (
            assemble_multidomain_case_finding_corpus)
        from mondo_dag import build_mondo_fit_inputs, make_mondo_attested_provider
        build = _build_inputs or build_mondo_fit_inputs
        assemble = _assemble or assemble_multidomain_case_finding_corpus
        branch = spec.get("mondo_branch") or ""
        with _phase(f"build Mondo DAG + climb (branch={branch or 'ALL'})"):
            before_dag, climb_sdf, terminal_cids, count_of, reduced = build(
                spark, cdr=spec["cdr"], billing=spec["billing"],
                mondo_version=spec["mondo_version"],
                mondo_cache_dir=spec.get("mondo_cache_dir") or "data/mondo",
                min_positives=spec["min_positives"],
                branch_root=(branch or None))
            # exp 0111 WP-D2: build the external episode/random index (if any) and
            # pick the doc spec BOTH the provider and the assembler use. The sidecar
            # code map is the climb frame (normalized inside the seam only on the
            # external path); its identity is the terminal count, exactly what
            # conversion_sidecar.code_map_from_manifest folds, so the fit and the probe
            # compute the SAME sidecar key.
            index_df, assemble_doc_spec, provider_doc_spec, ordinal_df, first_att = (
                _external_seam(spark, assembly_params, code_map_sdf=climb_sdf,
                               concept_col="descendant_concept_id",
                               node_col="ancestor_concept_id",
                               code_map_identity=f"mondo:{len(terminal_cids)}",
                               dag_source="mondo"))
            provider = make_mondo_attested_provider(
                climb_sdf, doc_spec=provider_doc_spec)
            print(f"[mondo]   powered terminals={len(terminal_cids)}, "
                  f"class nodes={reduced['n_classes']}, "
                  f"branch={branch or 'ALL'}", flush=True)
            if spec.get("dag_collapse"):
                # exp 0109. Applied HERE — between the hierarchy build and the
                # assembler — rather than inside `mondo_dag`, because that module's
                # source hash is folded into every Mondo bundle key and editing it
                # would orphan every cached bundle including 0104's (see
                # mondo_collapse's module docstring). Terminals are untouched, so
                # `climb_sdf` / `terminal_cids` stay exactly right.
                from mondo_collapse import (collapse_engine_dag,
                                            format_collapse_report)
                before_dag, collapse_stats = collapse_engine_dag(before_dag)
                print(format_collapse_report(collapse_stats), flush=True)
            if on_inputs is not None:
                on_inputs(count_of=count_of, terminal_cids=terminal_cids,
                          reduced=reduced)
        bundle = assemble(spark, before_dag=before_dag,
                          attested_provider=provider, index_df=index_df,
                          doc_spec=assemble_doc_spec, **assembly_params)
        bundle = _with_preindex(spark, bundle, before_dag=before_dag,
                                provider=provider, assembly_params=assembly_params)
        if index_df is not None:
            bundle = _attach_bounded_doc_index(bundle, ordinal_df=ordinal_df)
            bundle = _with_gate_frontier(
                spark, bundle, first_attestation=first_att,
                before_dag=before_dag, assembly_params=assembly_params)
        return bundle

    return _assemble_mondo_native if native else _assemble_mondo


def multidomain_load_or_build(spark, spec, *, cache_uri=None, on_inputs=None,
                              _build_inputs=None, _assemble=None):
    """Cached multi-domain / Mondo corpus: HIT reloads, MISS assembles + writes through.

    The single entry point for both the fit driver and the standalone re-readout —
    same key, same assembler, same write-through — so a fit's bundle is findable by
    a later readout and a readout's rebuild is reusable by a later fit.
    """
    from _case_finding_cache import load_or_build_case_finding_bundle
    from charmpheno.omop.multi_domain import (
        assemble_multidomain_case_finding_corpus)

    assembly, key_extra = _multidomain_params(spec)
    if key_extra.get("mondo"):
        assemble_fn = mondo_assemble_fn(spec, on_inputs=on_inputs,
                                        _build_inputs=_build_inputs,
                                        _assemble=_assemble, cache_uri=cache_uri)
    else:
        assemble_fn = _assemble or assemble_multidomain_case_finding_corpus
    return load_or_build_case_finding_bundle(
        spark, cache_uri=cache_uri, _assemble_fn=assemble_fn,
        _key_extra=key_extra, **assembly)


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
    p.add_argument("--dag-source", choices=["snomed", "mondo", "mondo_native"],
                   default="snomed",
                   help="snomed (default): the disease's SNOMED anchor forest via "
                        "concept_ancestor. mondo: the whole-Mondo powered hierarchy "
                        "(exp 0088) with a POPULATION index + SNOMED-climb attestation "
                        "(routes through the multi-domain assembler). mondo_native: "
                        "exp 0110's NATIVE Mondo label space — labels are Mondo terms "
                        "powered by is-a CLOSURE support, the DAG is Mondo's own "
                        "hierarchy transitively reduced over the kept set, and a doc "
                        "attests the most-specific terms its codes map/climb to. "
                        "--dag-collapse does not apply (the splice is intrinsic).")
    p.add_argument("--mondo-branch", default="",
                   help="mondo: restrict to one body-system Mondo subtree (e.g. "
                        "MONDO:0004995 = cardiovascular disorder) — the Step-A template. "
                        "'' = whole Mondo.")
    p.add_argument("--min-positives", type=int, default=100,
                   help="mondo: keep anchors with >= this many whole-pop patients "
                        "(the K dial; exp 0088 used 100).")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    p.add_argument("--dag-collapse", action="store_true",
                   help="mondo: SPLICE-TO-FIXPOINT the label DAG before assembly "
                        "(exp 0109) — repeatedly remove every class node with "
                        "exactly one kept child (wiring its parents to that child) "
                        "and every class node left with none, until no such node "
                        "remains. Terminals (the powered anchors) and the root are "
                        "never removed. This is what kills the 763 degenerate "
                        "'constant fallback' readout cells at whole-Mondo, which are "
                        "exactly {root} u {only-children} under closure masking. OFF "
                        "by default: it changes the corpus (and its cache key), so "
                        "existing experiments reproduce byte-identically.")
    p.add_argument("--preindex-closure", action="store_true",
                   help="mondo: also compute E1's PRE-INDEX CLOSURE column — per "
                        "document, the sparse engine-id closure of what the "
                        "patient already carried BEFORE the index, i.e. the label "
                        "definition evaluated on the FEATURE window instead of the "
                        "label window (preindex_closure.py). It is what makes "
                        "incident eligibility (c NOT IN R_d) computable at eval "
                        "time, and it is a CORPUS property: computed once, stored "
                        "with the corpus, reused byte-identically by every run "
                        "being compared, with nothing a fit produces entering it. "
                        "Costs one extra full-history condition scan + one "
                        "attestation pass at BUILD time only (a cache HIT pays "
                        "nothing). OFF by default: it changes the bundle (and its "
                        "cache key), so existing experiments reproduce "
                        "byte-identically.")
    # exp 0111 (WP-D2): episode-anchored / matched-random index arms. When
    # --index-arm is set the Mondo corpus is assembled on a DRIVER-built external
    # index (index_mode="external") of at most `--episode-cap` documents per
    # person keyed on their gap-and-islands first-attestation EPISODES (episode
    # arm) or on uniform-random in-observation dates (random arm), instead of the
    # one-doc-per-person population/disease index. Both arms are multi-doc and
    # share every gate/cap/salt — the random arm is drawn over the episode arm's
    # SURVIVING persons so the two compare on an identical person set (D12).
    p.add_argument("--index-arm", choices=["episode", "random"], default="",
                   help="exp 0111: assemble on a driver-built EXTERNAL episode index "
                        "(index_mode=external) instead of the population/disease "
                        "index. 'episode' anchors documents on gap-and-islands "
                        "first-attestation episodes; 'random' draws matched uniform "
                        "in-observation dates over the episode arm's surviving "
                        "persons. Empty (default) keeps the current single-doc "
                        "index. Mondo path only (dag_source mondo/mondo_native).")
    p.add_argument("--episode-gap-days", type=int, default=90,
                   help="exp 0111: gap (days) that splits one first-attestation "
                        "episode from the next (gap-and-islands). 90 is the probe's "
                        "settled value.")
    p.add_argument("--episode-cap", type=int, default=3,
                   help="exp 0111: per-person document cap for BOTH arms (deterministic "
                        "salted sample). Must be < 64 (the doc-key radix); 3 is the "
                        "probe's settled value.")
    p.add_argument("--episode-salt", default="0111",
                   help="exp 0111: salt for the deterministic per-person cap sample "
                        "(episode arm) and the uniform index draw (random arm). Same "
                        "salt => identical draw on any rerun; never F.rand(). It is a "
                        "corpus-identity input and folds into the bundle cache key.")
    p.add_argument("--episode-prior-obs-days", type=int, default=365,
                   help="exp 0111: prior-observation gate (days) an index must clear "
                        "for BOTH arms. 365 matches the assembler's lookback floor "
                        "(_LOOKBACK_PRIOR_OBS_DAYS) — the 0111 primary arm.")
    p.add_argument("--episode-window-days", type=int, default=365,
                   help="exp 0111: forward-observation gate (days) an index must clear "
                        "for BOTH arms. 365 matches the assembler's label window.")
    p.add_argument("--episode-sidecar-uri", default="",
                   help="exp 0111: root URI of the E4 first-attestation sidecar the "
                        "episode index is built from. The sidecar has its OWN "
                        "(bundle-key-independent) key, so it is NOT part of the bundle "
                        "cache key. Empty (default) derives <cache-uri>/"
                        "conversion_sidecar, matching build-conversion-sidecar. Point "
                        "it at a persistent in-boundary bucket so it survives a "
                        "cluster (AGENTS.md). Loaded ONLY on a bundle cache MISS; a "
                        "MISS with no sidecar present builds it once and saves it.")
    # exp 0111 (WP-D3 / D13): separate the estimator's GATE from the outcome LABEL.
    # When > 0 (and an --index-arm is set) each document's `frontier` gate is swapped
    # to its [index, index+N-day) PRESENTATION window while `label`/`labelMask` stay
    # the full 365-day frame the assembler baked — the D13 separation. A MISS-only
    # driver post-pass; the gate mode (`gateNd`) folds into the bundle cache key, so
    # a gated bundle is a distinct artifact from the un-gated one. 0 (default) keeps
    # the 365-day frontier as the gate, so every existing key stays byte-identical.
    p.add_argument("--gate-frontier-days", type=int, default=0,
                   help="exp 0111 (D13): forward-window width (days) of the GATE the "
                        "gated estimator reads, separated from the 365-day outcome "
                        "label. > 0 swaps each episode/random document's `frontier` "
                        "to its [index, index+N-day) presentation window (90 is the "
                        "spec's gate); the outcome `label`/`labelMask` stay the "
                        "365-day frame. Folds into the bundle cache key as `gateNd`. "
                        "0 (default) leaves the frontier at 365 days — no separation, "
                        "byte-identical key. Requires --index-arm (external path).")
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
    p.add_argument("--readout-sample-frac", type=float, default=1.0,
                   help="row-subsample fraction for the driver-side readout collects "
                        "(theta + per-node LR + (N,C) proba arrays are all O(N)). <1.0 "
                        "bounds driver memory at whole-Mondo K/C; the per-node LR needs "
                        "only enough rows to fit. 1.0 (default) = full collect.")
    p.add_argument("--readout-mode", choices=["driver", "distributed", "auto"],
                   default="auto",
                   help="where the pc_topics_lr readout is fit. 'driver' collects "
                        "per-doc theta (D,K float64) AND dense label/mask (D,C "
                        "float64) for BOTH splits, then fits C sklearn LRs on the "
                        "driver — 8*D*(K+3C) bytes, ~24 GB at whole-Mondo K=C=3,300 "
                        "over 300k docs, which is why it breaks there. 'distributed' "
                        "fits all C heads in ONE batched L-BFGS on the executors "
                        "(the same treeAggregate seam the co-fit head uses) and "
                        "collects only a LEAN test-split eval bundle — float32 "
                        "proba + uint8 label/mask, 6 bytes/cell (plan v2.1) — so the "
                        "whole driver eval stack runs unchanged. 'auto' (default) = "
                        f"driver at C<={_DRIVER_READOUT_MAX_C}, else distributed.")
    p.add_argument("--readout-ab-check", action="store_true",
                   help="run BOTH readout paths on the same scored frames and print "
                        "the deltas (macro AUC/AP, per-node |dAUC|, sampled max "
                        "|dproba|). The plan's cardiovascular (C=444) correctness "
                        "gate. Report only — never asserts. Ignored unless the mode "
                        f"resolves to distributed AND C<={_DRIVER_READOUT_MAX_C} "
                        "(the driver path must still be affordable to compare to).")
    p.add_argument("--readout-max-iter", type=int, default=_READOUT_MAX_ITER,
                   help="iteration cap for EACH batched readout solve (distributed "
                        "mode). Every iteration is a full pass over the train split "
                        "— at C=437 a cold solve spends all 200 (~30 min), so this "
                        "is the readout's wall-clock knob. Lower it for the dev "
                        "RANKING loop (CHARM_DEV caps it at 60), never for a run of "
                        "record: the capped point is whatever the budget bought.")
    p.add_argument("--readout-theta-topm", type=int, default=0,
                   help="fit and score the distributed readout on each doc's top-M "
                        "theta entries (truncated, NOT renormalized), 0 (default) = "
                        "off / full K. The readout's cost is memory traffic — "
                        "observed cells x K-wide dense dot products, ~1.7 TB per "
                        "pass at C=K=3,827 over 56M cells (~65s) — and truncation "
                        "cuts it by K/M. Legitimate because per-doc theta is a "
                        "Dirichlet posterior mean over thousands of topics and is "
                        "concentrated; the fit logs the MEASURED mass coverage "
                        "(mean/p10) at m=64..512 first, so set this from that line "
                        "rather than from faith. Truncation is applied once at "
                        "ingest, so moments, fit, scoring and eval all see the same "
                        "narrower design matrix — a different, exact model, not an "
                        "approximation of the full-K one.")
    p.add_argument("--readout-calibration", choices=["on", "off"], default="on",
                   help="run the post-hoc ISOTONIC calibration block (a SECOND "
                        "batched readout solve on a 75% split, plus two lean "
                        "collects). Its output is the conditional ECE reliability "
                        "diagnostic — not a ranking signal — so the dev loop turns "
                        "it off (CHARM_DEV does this automatically) and halves the "
                        "supervised arm's readout wall-clock. Never off for a run of "
                        "record: VOI readiness is exactly the calibrated posterior.")
    p.add_argument("--eval-path", choices=["driver", "distributed"], default="driver",
                   help="EVAL path for the gated_pc ranking arms (exp 0111 WP-B), "
                        "ORTHOGONAL to --readout-mode (the FIT path). 'driver' (the "
                        "default): the shipped _densify_lean_blocks collect + "
                        "readout_from_proba (the O(N.C) driver collect, audit S5f). "
                        "'distributed': score the prevalent + incident per-node "
                        "ranking via score_cells_arms_df/per_node_metric_arms_rows "
                        "with nothing (D_te,C) reaching the driver, and fit the "
                        "isotonic calibrator on BINNED sufficient stats (no cal-slice "
                        "collect) — the path the episode corpus needs (x2.66 docs). "
                        "Only meaningful under --readout-mode distributed; the "
                        "conditional/detection/PR axes need the collect and are "
                        "skipped on this path. Default 'driver' until the WP-B parity "
                        "gate is green on the current cache.")
    p.add_argument("--head-converge-iters", type=int, default=25,
                   help="Newton steps for the post-fit HEAD-FORMULATION LADDER "
                        "(localize-head only): converge each localized head variant on "
                        "the FROZEN final θ for this many full steps to isolate which "
                        "formulation difference costs the co-fit head its AUC.")
    p.add_argument("--head-fixed-ridge", type=float, default=1.0,
                   help="the ABSOLUTE L2 ridge for the ladder's FIXED-ridge head "
                        "variants (sklearn-style; does not vanish at separation, unlike "
                        "the shipped relative ridge).")
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
    p.add_argument("--diag-only", action="store_true",
                   help="FAST head-starvation probe: fit the gated_pc arm ONLY (skip "
                        "the θ-collect readouts, baselines, conditional + ladder — the "
                        "slow part), then print the per-node head-magnitude histogram "
                        "(|w_c| on each node's support vs its positive count) and exit. "
                        "Pair with a small --max-iter (e.g. 8): starvation (|w_c|≈0 for "
                        "the low-positive nodes) is visible in a few iters, in minutes.")
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
            log=lambda msg: print(f"[driver] {msg}", flush=True))
        if args.dag_source in _MONDO_DAG_SOURCES or extra_domains:
            # MULTI-DOMAIN corpus (per-domain vocabularies, features_0..N-1), in
            # either of its two flavours, both CACHED through the same seam:
            #   dag_source=mondo — the label DAG is the whole-Mondo powered
            #     hierarchy (exp 0088), patients are placed by SNOMED-climb and the
            #     index is population-wide (no single disease); min_n=0 because that
            #     DAG is already powered. The DAG build + climb now live INSIDE the
            #     assemble seam, so a cache HIT skips them entirely (~5 min of
            #     BigQuery per run that used to be unconditional).
            #   otherwise      — the disease-anchored SNOMED forest with extra
            #     domains bolted on (the one-off comparison run).
            # Both require lookback windowing (the forward per-patient window is
            # condition-defined and does not window a second domain).
            if args.window_mode != "lookback":
                raise ValueError(
                    f"--dag-source {args.dag_source} / --extra-domains requires "
                    f"--window-mode lookback (got {args.window_mode})")
            corpus_spec = multidomain_corpus_spec(args, extra_domains)

            def _stash_mondo_inputs(*, count_of, terminal_cids, reduced):
                # Only reached on a cache MISS; the diag-only probe degrades
                # gracefully without these (per_node_head_report).
                args._count_of = count_of

            label = (args.dag_source.upper()
                     if args.dag_source in _MONDO_DAG_SOURCES else "MULTI-DOMAIN")
            with _phase(f"assemble {label} corpus (cached, cond + "
                        f"{list(extra_domains)})"):
                bundle = multidomain_load_or_build(
                    spark, corpus_spec, cache_uri=args.cache_uri,
                    on_inputs=_stash_mondo_inputs)
                vocab_maps = bundle.vocab_maps
                args._domain_cols = [f"features_{i}" for i in range(len(vocab_maps))]
                args._domain_names = ["condition", *extra_domains]
            print(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)
        else:
            with _phase("assemble corpus (cached, emit_labels)"):
                bundle = load_or_build_case_finding_bundle(
                    spark, cache_uri=args.cache_uri,
                    # key-ONLY input (the assembler builds its own doc spec):
                    # closes the doc-spec cache-key hole on the single-domain
                    # path too. Today's value is the fold's default, so every
                    # SNOMED key is byte-identical.
                    _key_extra={"doc_spec": doc_spec_identity()},
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
        # PRE-FLIGHT cost profile at the data-build boundary: fan-out / support sizes /
        # head matrix memory+compute (dense vs localized) so a big fit's cost is visible
        # BEFORE compute is committed (esp. the whole-Mondo run — high-fan-out parents
        # inflate the localized support). Logged for every gated fit.
        _v_total = sum(len(vm) for vm in vocab_maps)
        _, _prof = lay.cost_report(C, vocab_size=_v_total,
                                   localized=bool(getattr(args, "localize_head", False)))
        print("\n".join("[cost] " + ln for ln in _prof.splitlines()), flush=True)
        # Readout routing, decided once for every arm (see resolve_readout_mode).
        readout_mode = resolve_readout_mode(args.readout_mode, C)
        print(f"[driver]   readout_mode={args.readout_mode} -> {readout_mode} "
              f"(C={C}, driver-collect ceiling C<={_DRIVER_READOUT_MAX_C})", flush=True)
        if readout_mode == "distributed" and args.readout_sample_frac < 1.0:
            # The flag exists only to bound the driver collect; the distributed path
            # has no collect to bound, and uniform row sampling would still gut the
            # rare tail the quartile split reports on. Say so rather than silently
            # honouring a knob that would now change the FIT.
            print(f"[driver]   readout_sample_frac={args.readout_sample_frac} IGNORED "
                  "under readout_mode=distributed (it bounded a driver collect that "
                  "no longer happens; the fit uses every row). It still applies "
                  "inside --readout-ab-check, where the driver path is the thing "
                  "being compared and has to fit.", flush=True)
        # θ-width lever + calibration switch, both resolved once for every arm. A
        # top-m readout is a DIFFERENT (narrower, exactly-fit) model, so it belongs
        # in the run's log next to the mode, not buried in a fit banner.
        theta_topm = int(getattr(args, "readout_theta_topm", 0) or 0)
        if theta_topm > 0 and readout_mode != "distributed":
            print(f"[driver]   readout_theta_topm={theta_topm} IGNORED under "
                  "readout_mode=driver (the truncation lives in the distributed "
                  "path's ingest adapters; the driver collect fits full theta)",
                  flush=True)
            theta_topm = 0
        elif theta_topm > 0:
            print(f"[driver]   readout_theta_topm={theta_topm}: the readout fits and "
                  f"scores each doc's top-{theta_topm} theta entries (truncated, not "
                  "renormalized) — see the per-fit 'theta top-m mass' line for the "
                  "measured coverage this is buying against", flush=True)
        # WP-B: the EVAL path, orthogonal to the FIT path resolved above. Only
        # meaningful once the fit is distributed (the executor-side scored frame is
        # what the cell explode reads); resolve_eval_path degrades it to driver
        # otherwise. Default driver until the parity gate is green.
        eval_path = resolve_eval_path(getattr(args, "eval_path", "driver"),
                                      readout_mode)
        if eval_path == "distributed" and theta_topm > 0:
            # score_cells_arms_df is dense-θ only (spec R5.4 defers top-m + doc-key +
            # arms); scoring a top-m fit on full θ would evaluate a model on features
            # it was never fit on. Keep the truncation and fall back to the driver
            # collect for the eval rather than silently mixing feature maps.
            print(f"[driver]   eval_path=distributed IGNORED because "
                  f"readout_theta_topm={theta_topm}>0 (the cell explode is dense-θ "
                  "only); using the driver collect for the eval.", flush=True)
            eval_path = "driver"
        if getattr(args, "eval_path", "driver") == "distributed":
            if eval_path == "distributed":
                print("[driver]   eval_path=distributed (WP-B): the gated_pc ranking "
                      "arms score via score_cells_arms_df/per_node_metric_arms_rows "
                      "(no O(N.C) driver collect) and the calibrator fits on BINNED "
                      "stats. The conditional/detection/PR axes need the collect and "
                      "are SKIPPED on this path.", flush=True)
            else:
                print("[driver]   eval_path=distributed IGNORED under "
                      f"readout_mode={readout_mode!r} (the cell explode reads the "
                      "distributed fit's scored frame; there is none on the driver "
                      "path). Re-run with --readout-mode distributed.", flush=True)
        run_calibration = resolve_readout_calibration(
            getattr(args, "readout_calibration", "on"))
        if not run_calibration:
            print("[driver]   readout_calibration=off: skipping the post-hoc "
                  "isotonic calibration block (a second batched solve + two lean "
                  "collects). Its output is the conditional ECE diagnostic, not a "
                  "ranking signal — final numbers come from a full run.", flush=True)
        ab_check = bool(args.readout_ab_check) and readout_mode == "distributed"
        if bool(args.readout_ab_check) and not ab_check:
            print("[driver]   readout_ab_check ignored (readout_mode resolved to "
                  "driver — there is nothing to compare against)", flush=True)
        elif ab_check and C > _DRIVER_READOUT_MAX_C:
            print(f"[driver]   readout_ab_check SKIPPED: C={C} exceeds the driver "
                  f"path's own ceiling ({_DRIVER_READOUT_MAX_C}); the gate is meant "
                  "to run at cardiovascular scale (C=444)", flush=True)
            ab_check = False

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

        # ---- E2/WP4: is the INCIDENT arm available on this corpus? -------------
        # The witness decides (R1.4), never a `try: select`. Asking Spark for a
        # column a pre-E1 bundle does not carry is an AnalysisException landing in
        # the middle of the readout — the mixed-vintage failure the witness exists
        # to turn into a sentence. On a miss the incident arm is SKIPPED with a
        # printed line and the prevalent arm is untouched.
        from preindex_closure import bundle_preindex_witness
        _pw = bundle_preindex_witness(bundle)
        elig_col = str(_pw.get("col_name")) if _pw else None
        if elig_col is None:
            print("[driver]   INCIDENT arm (E2) SKIPPED: this corpus carries no "
                  "pre-index closure witness. Rebuild the corpus with "
                  "--preindex-closure (a different bundle cache key — nothing "
                  "already cached is invalidated) to get the incident block. The "
                  "prevalent arm is unaffected.", flush=True)
        elif readout_mode != "distributed":
            # The eligibility column rides the LEAN collect, which only the
            # distributed path performs. At C <= 500 (the driver path's own
            # ceiling) the incident arm is not wired; whole-Mondo, where this
            # program lives, resolves to distributed.
            print(f"[driver]   INCIDENT arm (E2) SKIPPED: readout_mode resolved to "
                  f"{readout_mode!r} and the eligibility column rides the LEAN "
                  "(distributed) collect. Re-run with --readout-mode distributed "
                  "for the incident block.", flush=True)
            elig_col = None
        else:
            print(f"[driver]   INCIDENT arm (E2) ON: eligibility from bundle column "
                  f"{elig_col!r} ({_pw.get('version')}) — a CORPUS property (spec "
                  "R2.3); {INCIDENT}".replace("{INCIDENT}", INCIDENT_NAMING),
                  flush=True)

        def _score(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, doc_keys=None):
            return score_arm(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C,
                             recall_targets=rt, fdr_targets=ft,
                             min_count=args.min_label_count, doc_keys=doc_keys)

        def _score_full(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, doc_keys=None):
            """(readout, per-node test proba) — proba reused for the conditional
            'sharpening' readout so the per-node LR is fit once per arm. `doc_keys`
            (R5.7) is the TEST split's collect order, threaded to
            `readout_from_proba` so the detection pool dedups to persons."""
            from analysis.pc.evaluate import _lr_proba_per_label_masked
            proba = _lr_proba_per_label_masked(Pi_tr, y_tr, m_tr, Pi_te, C)
            readout = readout_from_proba(
                proba, y_te, m_te, C, recall_targets=rt, fdr_targets=ft,
                min_count=args.min_label_count, doc_keys=doc_keys)
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

        def _incident_conditional(proba_te, y_te, elig_te, label):
            """The E3 twin of `_conditional`: the same cells, incident-local.

            THE `_ones` CONVENTION IS BROKEN HERE, DELIBERATELY, AND ONLY HERE.

            The rule it protects (`conditional_readout`'s docstring, exp 0079 Trap
            3): pass the FULL-closure observation mask — all-ones — so the
            cohort/negative sets are fixed identically across full- and
            closure-mask runs; a run-dependent eval mask silently turns the
            conditional eval into an easier sibling-only contrast and cross-run
            numbers stop being comparable. That rule is UNTOUCHED above: the
            prevalent conditional readout still passes `np.ones_like(y_te)`, and
            this call passes it too.

            What is new is a THIRD array — `eligibility` — which restricts the
            cells. It is a genuine departure from "the eval sees everything", so it
            is stated rather than slipped in, and here is why cross-run
            comparability survives it (spec R2.3): incident eligibility is a pure
            function of `(bundle, R_d)` — spec D2 — computed once per CORPUS by E1,
            stored with the corpus, and reused byte-identically by every run being
            compared. No prediction, threshold or degeneracy set of THIS run enters
            it; structurally, the readout has no code path that could construct
            one. Trap 3's failure was an eval mask that moved with the RUN. This one
            moves with the corpus, which is the same thing the labels do.
            """
            cond = conditional_readout(proba_te, y_te, np.ones_like(y_te),
                                       bundle.parent_int, C,
                                       min_count=args.min_label_count,
                                       eligibility=elig_te)
            print(format_incident_conditional(cond).replace(
                "[incident conditional]", f"[incident conditional: {label}]"),
                flush=True)
            return cond

        # Every manifest field the FIT determines, built once and reused by both
        # saves below — the early fit-only write and the final authoritative one.
        # Keeping it in one dict is what makes "the same manifest, plus results"
        # true by construction rather than by two lists staying in sync.
        manifest_fields = {
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
            "readout_mode": readout_mode,
            "eval_path": eval_path,
            "readout_sample_frac": (1.0 if readout_mode == "distributed"
                                    else args.readout_sample_frac),
            # Both change WHAT was fit / what was reported, so they belong in the
            # manifest next to the mode: a top-m readout is a narrower model, and
            # a calibration-skipped run has no ECE record at all.
            "readout_theta_topm": theta_topm,
            "readout_calibration": "on" if run_calibration else "off",
            # The solver iteration cap the fit ACTUALLY used — i.e. post-CHARM_DEV
            # capping, since the dev profile rewrites args before we get here. A
            # recovery re-readout (gated_pc_readout) must reproduce the run it is
            # rescuing, and without this field it cannot know whether the lost
            # readout was a 60-iter dev smoke or a 200-iter record run; it defaulted
            # to 200 and the operator had to remember to pass --readout-max-iter 60.
            "readout_max_iter": int(args.readout_max_iter),
            "recall_targets": args._recall_targets,
            "fdr_targets": args._fdr_targets,
            "with_dag_head": args.with_dag_head,
            "skip_unsup_gated": args.skip_unsup_gated,
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
                "min_n": (corpus_spec or {}).get("min_n", args.min_n),
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

        with _phase(f"gated_pc fit (weightY={args.weight_y}, K={lay.K})"):
            pc_est = _build_pc_estimator(args, weight_y=args.weight_y, gated=True)
            if args.eval_every > 0:
                pc_est.setOnIteration(_make_eval_logger(bundle, C, args))
            pc_model = pc_est.fit(bundle.train_df)
            # EARLY SAVE, before any readout work touches the cluster: the fit is
            # the hours-long unrepeatable half and the readout is where runs die,
            # so the model reaches durable storage the moment it exists. The final
            # save overwrites these same two paths with the full record; until it
            # does, `partial="fit-only"` says so and `gated_pc_readout` can score
            # this run from the npz alone.
            _save_fit(out, pc_model.result.global_params, C, manifest_fields,
                      partial="fit-only")
            print(f"[driver]   saved FIT-ONLY result to {out} (readout pending; "
                  "re-scoreable with gated_pc_readout)", flush=True)
            if args.diag_only:
                # FAST head-starvation probe: skip every θ-collect / readout / baseline
                # (the slow part) and just read the fitted head. The per-iter ||grad_y||
                # / |w_CK|max trajectory already printed during the fit; this adds the
                # per-node DISTRIBUTION that the aggregate hides.
                print(per_node_head_report(
                    pc_model.headWeights(), lay, C, bundle.int2cid,
                    getattr(args, "_count_of", {})), flush=True)
                return 0
            # Transform each split ONCE (each transform re-runs CAVI over the split);
            # the supervised transform appends BOTH topicDistribution and probability.
            train_scored = pc_model.transform(bundle.train_df).cache()
            test_scored = pc_model.transform(bundle.test_df).cache()
            _sf = args.readout_sample_frac
            _sd = args.seed if args.seed is not None else 0
            if readout_mode == "distributed" and eval_path == "distributed":
                # WP-B: fit the heads on the executors (as the collect path does),
                # then score COLLECT-FREE via the cell explode — no (D_te,C) proba
                # ever reaches the driver (audit §5f). proba_gp/y_te/m_te/elig_te
                # therefore do not exist; the conditional/detection axes that need
                # them are skipped below on `proba_gp is None`.
                _ck = Path(out) / "readout_ckpt_gated_pc.npz" if out else None
                _V_gp, _b_gp, _const_gp, _deg_gp, _ = _fit_readout_heads(
                    train_scored, C, lay.K, label="gated_pc",
                    max_iter=args.readout_max_iter, theta_topm=theta_topm,
                    checkpoint_path=_ck, checkpoint_every=10)
                if out:
                    _write_readout_heads(out, "gated_pc", _V_gp, _b_gp, _const_gp,
                                         _deg_gp, C, lay.K, theta_topm)
                results["gated_pc"], _inc_dist = distributed_ranking_readout(
                    test_scored, C, _V_gp, _b_gp, recall_targets=rt, fdr_targets=ft,
                    min_count=args.min_label_count, elig_col=elig_col,
                    arm_label="gated_pc (pc_topics_lr)")
                _gp_fit = (_V_gp, _b_gp)
                proba_gp = y_te = m_te = ids_gp = elig_te = None
                _dist_gp = None
            elif readout_mode == "distributed":
                # No θ collect at all: the per-node LRs are fit on the executors and
                # only the lean test-split eval bundle comes back. Pi_tr/y_tr/m_tr
                # therefore do not exist on this path — everything below that needs
                # them (localized oracle, formulation ladder) is a DRIVER-path
                # diagnostic and is gated accordingly.
                _dist_gp = distributed_score_arm(
                    train_scored, test_scored, C, lay.K, recall_targets=rt,
                    fdr_targets=ft, min_count=args.min_label_count, label="gated_pc",
                    max_iter=args.readout_max_iter, theta_topm=theta_topm,
                    checkpoint_dir=out, elig_col=elig_col)
                results["gated_pc"], proba_gp, y_te, m_te, ids_gp = _dist_gp[:5]
                _gp_fit = _dist_gp[5]         # raw-θ params: the calibration warm start
                elig_te = _dist_gp[6]         # E1's (D,C) eligibility, or None
                _inc_dist = None
            else:
                Pi_tr, y_tr, m_tr, ids_tr = _collect_theta_labels(
                    train_scored, C, sample_frac=_sf, seed=_sd)
                Pi_te, y_te, m_te, ids_gp = _collect_theta_labels(
                    test_scored, C, sample_frac=_sf, seed=_sd)
                results["gated_pc"], proba_gp = _score_full(
                    Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, doc_keys=ids_gp)
                elig_te = None
                _inc_dist = None
            _dump_partial_results(out, results)
            print(format_arm_readout("gated_pc (pc_topics_lr)", results["gated_pc"]),
                  flush=True)
            # E2/WP4: the SECOND readout_from_proba call, on the incident cohort.
            # Same proba, same labels, a different eval mask — no re-fit, no second
            # collect. Dumped immediately so a death below still leaves it on disk.
            # Under eval_path=distributed the incident block was already built
            # collect-free by `distributed_ranking_readout` (from the arms explode),
            # so use it rather than re-scoring a (D,C) proba that does not exist.
            if eval_path == "distributed":
                _inc = _inc_dist
            else:
                _inc = incident_readout(
                    proba_gp, y_te, m_te, elig_te, C, recall_targets=rt,
                    fdr_targets=ft, min_count=args.min_label_count,
                    prevalent=results["gated_pc"],
                    arm_label="gated_pc (pc_topics_lr)", doc_keys=ids_gp)
            if _inc is not None:
                results["gated_pc_incident"] = _inc
                _dump_partial_results(out, results)
                print(format_incident_readout(_inc), flush=True)
            if ab_check and _dist_gp is not None:
                # The plan's step-2 correctness gate, on the arm whose readout is the
                # headline. Reuses the distributed result already computed, so the
                # extra cost is one driver-path readout.
                readout_ab_report(
                    train_scored, test_scored, C, lay.K, recall_targets=rt,
                    fdr_targets=ft, min_count=args.min_label_count, label="gated_pc",
                    seed=_sd, sample_frac=_sf, distributed=_dist_gp,
                    max_iter=args.readout_max_iter, theta_topm=theta_topm)
            # Conditional 'sharpening' readout: P(child | parent-cohort) by DAG depth.
            # It scores the FULL (D,C) proba against an all-ones mask, so it needs
            # the collect — under eval_path=distributed (`proba_gp is None`) it is
            # skipped with a note. It is a co-fit / VOI diagnostic, not the ranking
            # headline the WP-B parity gate proves; run --eval-path driver for it.
            if proba_gp is None:
                print("[driver]   conditional 'sharpening' + incident-conditional "
                      "readouts SKIPPED under eval_path=distributed (they score the "
                      "full (D,C) proba against an all-ones mask — the collect this "
                      "path avoids; run --eval-path driver on a corpus the collect "
                      "fits)", flush=True)
            else:
                results["gated_pc_conditional"] = _conditional(
                    proba_gp, y_te, m_te, "gated_pc")
                # E3/WP5: the same cells, incident-local, with the D6 P-stratum. An
                # ADDITION beside the prevalent block above, never a replacement —
                # the prevalent conditional numbers are what 0104/0109 are compared on.
                if elig_te is not None:
                    results["gated_pc_incident_conditional"] = _incident_conditional(
                        proba_gp, y_te, elig_te, "gated_pc")
                    _dump_partial_results(out, results)
            # ORACLE LOCALIZED readout (A-vs-B diagnostic for the co-fit head): the
            # BEST-POSSIBLE per-node logistic fit on EXACTLY the co-fit head's topic
            # support (allowed_with_siblings) — same hypothesis class as the localized
            # head, but fit optimally (sklearn) instead of by the co-fit Newton step.
            #   oracle ≈ full-K readout  => the signal IS in the support; the co-fit
            #     head is merely UNDER-FIT on it (recoverable: tune the head fit).
            #   oracle ≈ co-fit head     => the discriminative signal is OUTSIDE the
            #     local support; localization is fundamentally lossy (widen support or
            #     concede two-stage).
            # DRIVER-PATH ONLY, deliberately: this (and the formulation ladder it
            # feeds) is a per-node LADDER over feature SUBSETS of θ — a different
            # hypothesis class per node, which the batched multi-head solver does not
            # express (it fits one shared full-K design). It is a co-fit-head
            # diagnostic, not part of the headline readout, so it stays on the driver
            # and is skipped when the θ collect it needs is the thing we are avoiding.
            if getattr(args, "localize_head", False) and readout_mode != "driver":
                print("[driver]   oracle-localized readout + head-formulation ladder "
                      "SKIPPED under readout_mode=distributed (they fit per-node "
                      "SUPPORT-restricted LRs on a driver-side θ collect; re-run with "
                      "--readout-mode driver at a C the collect fits)", flush=True)
            elif getattr(args, "localize_head", False):
                from analysis.pc.evaluate import _lr_proba_per_label_masked as _lrm
                support_cols = [np.asarray(sorted(lay.allowed_with_siblings(c)), dtype=int)
                                for c in range(C)]
                proba_oracle = _lrm(Pi_tr, y_tr, m_tr, Pi_te, C,
                                    feature_cols=support_cols)
                results["gated_pc_oracle_conditional"] = _conditional(
                    proba_oracle, y_te, m_te,
                    "gated_pc oracle-localized (support-only LR)")
                args._support_cols = support_cols     # reused by the formulation ladder
            # Post-hoc ISOTONIC calibration of the head-independent proba → calibrated
            # conditional posteriors for VOI (the two-stage route, independent of the
            # co-fit head). Isotonic is monotone so
            # AUC/top-1 are unchanged; only the reliability (ECE) moves. The calibrator
            # is fit on a HELD-OUT 75/25 train split (OUT-OF-SAMPLE) — in-sample fitting
            # worsened ECE (exp 0079 run 2). Raw vs calibrated are compared within the
            # SAME 75%-fit LR so the delta is the calibration effect alone.
            #
            # The whole block is a DIAGNOSTIC and costs a second batched solve plus two
            # more lean collects — at whole-Mondo C that is the same order as the arm's
            # main readout. `--readout-calibration off` (what CHARM_DEV sets) drops it:
            # the isotonic ECE is a reliability report, never a ranking signal, so the
            # dev loop's comparisons survive without it and the numbers of record come
            # from the full run that keeps it on.
            if run_calibration and eval_path == "distributed":
                # WP-B COLLECT-FREE calibration. The FIT no longer collects the
                # (D_cal,C) calibration cells: the cal slice is scored into cells and
                # reduced to C×n_bins BINNED sufficient stats (n, sum_y), the driver
                # fits ONE weighted isotonic per node on the bins (min_pos=20 and the
                # single-class skip preserved, spec R5.6), and an honest ECE-on-test
                # is read from the TEST slice's own binned stats (raw vs calibrated-
                # through-the-breakpoints). The conditional-EDGE ECE the collect path
                # reports needs the full (D,C) proba, so it is skipped here — this is
                # the reliability ECE a collect-free path can honestly produce.
                from pyspark.sql import functions as _F
                _h = _F.pmod(_F.hash(_F.col("person_id"), _F.lit(int(_sd))),
                             _F.lit(4))
                _cal_df = train_scored.filter(_h == 0)
                _fit_df = train_scored.filter(_h != 0)
                _Vc, _bc, _constc, _degc, _ = _fit_readout_heads(
                    _fit_df, C, lay.K, label="gated_pc calibration-fit",
                    max_iter=args.readout_max_iter, warm_start=_gp_fit,
                    theta_topm=theta_topm)
                _cal_scored = _cal_df.withColumn(
                    "doc_key", _doc_key_column(_cal_df))
                _cal_cells = _dr.score_cells_df(_cal_scored, _Vc, _bc, C)
                _cnt, _sy = _dr.binned_calibration_stats(_cal_cells, C)
                _bp = _dr.fit_binned_isotonic(_cnt, _sy, C, min_pos=20)
                _te_scored = test_scored.withColumn(
                    "doc_key", _doc_key_column(test_scored))
                _te_cells = _dr.score_cells_df(_te_scored, _Vc, _bc, C)
                _tcnt, _tsy = _dr.binned_calibration_stats(_te_cells, C)
                _ece_raw = _dr.pooled_reliability_ece_from_bins(_tcnt, _tsy)
                _ece_cal = _dr.pooled_reliability_ece_from_bins(
                    _tcnt, _tsy, breakpoints=_bp)
                _n_cal = sum(1 for b in _bp if b is not None)
                results["gated_pc_calibration_binned"] = {
                    "method": "binned isotonic (WP-B, collect-free)",
                    "n_bins": _dr._CALIB_BINS, "min_pos": 20,
                    "n_nodes_calibrated": int(_n_cal),
                    "reliability_ece_raw": _ece_raw,
                    "reliability_ece_cal": _ece_cal,
                    "note": "pooled equal-width reliability ECE on the TEST slice "
                            "(collect-free); the conditional-edge ECE needs the "
                            "driver collect and is on --eval-path driver"}
                print(f"[driver]   BINNED isotonic calibration (WP-B, collect-free): "
                      f"{_n_cal} nodes calibrated (min_pos=20); pooled reliability "
                      f"ECE-on-test raw={_f(_ece_raw).strip()} -> "
                      f"calibrated={_f(_ece_cal).strip()}", flush=True)
            elif run_calibration:
                if readout_mode == "distributed":
                    # Distributed twin of the 75/25 calibration split. The split is a
                    # HASH of person_id (deterministic, complementary, and no driver-side
                    # row index to sample from), and the second batched fit on the 75%
                    # is a real extra cluster pass — the price of an out-of-sample
                    # calibrator, unchanged in kind from the driver path's second LR fit.
                    # Memory is NOT the binding constraint here even though the train
                    # split is ~4x the test split: only the 25% CALIBRATION slice and the
                    # test split are ever collected (both lean), never the 75% fit slice.
                    from pyspark.sql import functions as _F
                    _h = _F.pmod(_F.hash(_F.col("person_id"), _F.lit(int(_sd))),
                                 _F.lit(4))
                    _cal_df = train_scored.filter(_h == 0)
                    _fit_df = train_scored.filter(_h != 0)
                    # Warm-started from the arm's OWN main fit: the 75% problem is the
                    # same C heads on three quarters of the same rows, so the main
                    # fit's raw-θ params are a near-solution for it — the point of
                    # paying for the second solve is the OUT-OF-SAMPLE calibrator, not
                    # a rediscovery of the same coefficients from zero.
                    # theta_topm rides along: the calibrator has to be fit and applied
                    # on the SAME features as the arm it calibrates, or the reliability
                    # curve describes a model nobody scored with.
                    _Vc, _bc, _constc, _degc, _ = _fit_readout_heads(
                        _fit_df, C, lay.K, label="gated_pc calibration-fit",
                        max_iter=args.readout_max_iter, warm_start=_gp_fit,
                        theta_topm=theta_topm)
                    proba_cal, y_cal, m_cal, _, _ = _collect_lean_proba(
                        _cal_df, C, _Vc, _bc, degenerate=_degc, const=_constc,
                        theta_topm=theta_topm)
                    proba_te_fit, _, _, _, _ = _collect_lean_proba(
                        test_scored, C, _Vc, _bc, degenerate=_degc, const=_constc,
                        theta_topm=theta_topm)
                    # NOTE: calibrate_per_node returns a float64 copy of proba_te_fit, so
                    # this one diagnostic doubles (8 vs 4 bytes/cell) the test-split
                    # probability array for the length of the conditional readouts below.
                    proba_te_cal = calibrate_per_node(
                        proba_cal, y_cal, m_cal, proba_te_fit, C)
                    del proba_cal, y_cal, m_cal
                else:
                    from analysis.pc.evaluate import _lr_proba_per_label_masked
                    # Driver twin of the distributed split above, PERSON-keyed (R5.6)
                    # — see `_person_keyed_cal_split`'s docstring for why (the exp
                    # 0079 run-2 failure, reintroduced by multi-doc; pinned in
                    # tests/scripts/test_multidoc_seams.py).
                    cal_sel, fit_sel = _person_keyed_cal_split(
                        ids_tr, args.seed if args.seed is not None else 0)
                    proba_cal = _lr_proba_per_label_masked(
                        Pi_tr[fit_sel], y_tr[fit_sel], m_tr[fit_sel], Pi_tr[cal_sel], C)
                    proba_te_fit = _lr_proba_per_label_masked(
                        Pi_tr[fit_sel], y_tr[fit_sel], m_tr[fit_sel], Pi_te, C)
                    proba_te_cal = calibrate_per_node(
                        proba_cal, y_tr[cal_sel], m_tr[cal_sel], proba_te_fit, C)
                _ones = np.ones_like(y_te)
                cond_raw = conditional_readout(proba_te_fit, y_te, _ones,
                                               bundle.parent_int, C,
                                               min_count=args.min_label_count)
                cond_cal = conditional_readout(proba_te_cal, y_te, _ones,
                                               bundle.parent_int, C,
                                               min_count=args.min_label_count)
                results["gated_pc_conditional_cal"] = cond_cal
                print(f"[driver]   conditional ECE (VOI readiness, held-out isotonic): raw="
                      f"{_f(cond_raw.get('ece')).strip()} -> "
                      f"calibrated={_f(cond_cal.get('ece')).strip()}", flush=True)
            # co-fit head's own per-node P(node) readout (secondary), from the SAME
            # scored test frame (no second CAVI pass). No LR is involved — the
            # `probability` column is already the per-doc (C,) P(node) — so the
            # distributed variant is just the LEAN collector over that column,
            # float32/uint8 instead of three float64 (D,C) arrays.
            #
            # THERE IS NO CO-FIT HEAD AT weight_y=0, and that is the scaled-back
            # mainline (exps 0104/0109/0110): `OnlinePCLDAModel._transform` appends
            # `probability` only when weightY != 0 (spark_vi/mllib/topic/pc.py), and
            # `predictProbability` refuses outright there — the head sits at its zero
            # seed, so sigmoid(0)=0.5 for every doc and every node. Asking for the
            # column anyway raised an AnalysisException that killed exp 0110 AFTER the
            # expensive pc_topics_lr arm had already printed. Same guard, same two
            # witnesses (the config fact and the column itself) as the re-readout tool
            # `gated_pc_readout.run_readout`, which fixed this on its own path in
            # a7f724d while this one kept the unconditional call.
            _head_wy = float(getattr(args, "weight_y", 0.0) or 0.0)
            _head_col = "probability" in test_scored.columns
            if _head_wy == 0.0 or not _head_col:
                print(f"[driver]   co-fit head readout + conditional SKIPPED "
                      f"(weight_y={_head_wy:g}, probability column "
                      f"{'present' if _head_col else 'absent'}): an unsupervised fit "
                      "has no co-fit head to read out — its transform appends no "
                      "probability column and the head is at its zero seed (P=0.5 "
                      "everywhere)", flush=True)
            else:
                if readout_mode == "distributed":
                    hp, hy, hm, ids_hd, _ = _collect_lean_proba(
                        test_scored, C, score_col="probability")
                else:
                    hp, hy, hm, ids_hd = _collect_head_proba(
                        test_scored, C, sample_frac=_sf, seed=_sd,
                        with_doc_keys=True)
                results["gated_pc_head"] = readout_from_proba(
                    hp, hy, hm, C, recall_targets=rt, fdr_targets=ft,
                    min_count=args.min_label_count, doc_keys=ids_hd)
                print(format_arm_readout("gated_pc (co-fit head)",
                                         results["gated_pc_head"]), flush=True)
                # Conditional readout on the CO-FIT HEAD proba too — the UNIFIED-model
                # P(child|parent): a single model emitting calibrated conditional
                # posteriors with no post-hoc fit. At 41-anchor scale the ridge
                # (head_l2) bounds the head AND it is well-calibrated (exp 0082: co-fit
                # ECE ~0.010, competitive with the two-stage readout LR above), so this
                # is the primary VOI-ready readout; the head-independent pc_topics_lr
                # is the reference.
                results["gated_pc_head_conditional"] = _conditional(
                    hp, hy, hm, "gated_pc co-fit head")
                print(f"[driver]   co-fit head |w_CK|max="
                      f"{float(np.abs(pc_model.headWeights()).max()):.4g} "
                      f"(head_l2={args.head_l2})", flush=True)
            # HEAD-FORMULATION LADDER: step the co-fit head's EXACT formulation toward
            # the sklearn oracle ONE factor at a time, on the SAME frozen gated θ +
            # closure mask, so we read off WHICH difference (convergence / ridge type /
            # intercept / standardization) costs the co-fit head its conditional AUC.
            # Means only (the per-depth blocks above cover full-K / oracle / co-fit).
            def _mean_cauc(cond):
                v = [r["cond_auc"] for r in (cond or {}).get("edges", [])
                     if r.get("cond_auc") is not None]
                return float(np.mean(v)) if v else float("nan")
            if getattr(args, "localize_head", False) and hasattr(args, "_support_cols"):
                from analysis.pc.evaluate import _lr_proba_per_label_masked as _lrm

                def _cmean(proba):
                    return _mean_cauc(conditional_readout(
                        proba, y_te, np.ones_like(y_te), bundle.parent_int, C,
                        min_count=args.min_label_count))

                sc = args._support_cols
                _ni = max(args.head_converge_iters, 1)
                _hl2, _hnr, _fr = args.head_l2, args.head_newton_ridge, args.head_fixed_ridge
                w_rel, b_rel = _converge_localized_head(
                    Pi_tr, y_tr, m_tr, sc, C, head_l2=_hl2, head_newton_ridge=_hnr,
                    n_iters=_ni, ridge_mode="relative")
                w_fix, b_fix = _converge_localized_head(
                    Pi_tr, y_tr, m_tr, sc, C, head_l2=_hl2, head_newton_ridge=_hnr,
                    n_iters=_ni, ridge_mode="fixed", fixed_ridge=_fr)
                w_fxi, b_fxi = _converge_localized_head(
                    Pi_tr, y_tr, m_tr, sc, C, head_l2=_hl2, head_newton_ridge=_hnr,
                    n_iters=_ni, ridge_mode="fixed", fixed_ridge=_fr, intercept=True)
                ladder = [
                    ("co-fit head (as TRAINED)",
                     _mean_cauc(results.get("gated_pc_head_conditional"))),
                    ("engine Newton [rel-ridge, no-icpt, CONVERGED]",
                     _cmean(_localized_head_proba(Pi_te, w_rel, b_rel))),
                    ("  + FIXED ridge (λ=%g)" % _fr,
                     _cmean(_localized_head_proba(Pi_te, w_fix, b_fix))),
                    ("  + FIXED ridge + INTERCEPT",
                     _cmean(_localized_head_proba(Pi_te, w_fxi, b_fxi))),
                    ("sklearn [no-intercept, standardized]",
                     _cmean(_lrm(Pi_tr, y_tr, m_tr, Pi_te, C, feature_cols=sc,
                                 fit_intercept=False, standardize=True))),
                    ("sklearn [intercept, NOT standardized]",
                     _cmean(_lrm(Pi_tr, y_tr, m_tr, Pi_te, C, feature_cols=sc,
                                 fit_intercept=True, standardize=False))),
                    ("sklearn ORACLE [intercept, standardized]",
                     _mean_cauc(results["gated_pc_oracle_conditional"])),
                    ("full-K readout (all K, sklearn)",
                     _mean_cauc(results["gated_pc_conditional"])),
                ]
                print("[driver]   HEAD-FORMULATION LADDER  cond_AUC (frozen θ, "
                      "localized support):", flush=True)
                for _name, _v in ladder:
                    print(f"[driver]     {_name:46s} {_v:.3f}", flush=True)
                print("[driver]     |w|max — engine rel=%.4g  fixed=%.4g  fixed+icpt=%.4g"
                      % (float(np.abs(w_rel).max()), float(np.abs(w_fix).max()),
                         float(np.abs(w_fxi).max())), flush=True)
            train_scored.unpersist(); test_scored.unpersist()

        if not args.skip_unsup_gated:
            n_it = (args.baseline_max_iter if args.baseline_max_iter and
                    args.baseline_max_iter > 0 else args.max_iter)
            with _phase(f"unsup_gated fit (weightY=0, {n_it} iters)"):
                us_args = argparse.Namespace(**vars(args))
                us_args.max_iter = int(n_it)
                us_est = _build_pc_estimator(us_args, weight_y=0.0, gated=True)
                us_model = us_est.fit(bundle.train_df)
                us_train_scored = us_model.transform(bundle.train_df)
                us_test_scored = us_model.transform(bundle.test_df)
                if readout_mode == "distributed":
                    # The unsup twin is the CONTROLLED incumbent, so its readout must
                    # come off the same path as the PC arm's or the headline delta
                    # would compare two solvers instead of two representations.
                    # Cache first: unlike the PC arm's frames these are not persisted
                    # by the caller, and the distributed path reads the train split
                    # twice (moments pass, then the L-BFGS projection) — each read of
                    # an uncached transform re-runs CAVI over the whole split.
                    us_train_scored = us_train_scored.cache()
                    us_test_scored = us_test_scored.cache()
                    _dist_us = distributed_score_arm(
                        us_train_scored, us_test_scored, C, lay.K, recall_targets=rt,
                        fdr_targets=ft, min_count=args.min_label_count,
                        label="unsup_gated", max_iter=args.readout_max_iter,
                        theta_topm=theta_topm, checkpoint_dir=out)
                    results["unsup_gated"], proba_us, y_te, m_te, _ = _dist_us[:5]
                else:
                    Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(
                        us_train_scored, C,
                        sample_frac=args.readout_sample_frac,
                        seed=(args.seed if args.seed is not None else 0))
                    Pi_te, y_te, m_te, ids_us = _collect_theta_labels(
                        us_test_scored, C,
                        sample_frac=args.readout_sample_frac,
                        seed=(args.seed if args.seed is not None else 0))
                    results["unsup_gated"], proba_us = _score_full(
                        Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, doc_keys=ids_us)
                _dump_partial_results(out, results)
                print(format_arm_readout("unsup_gated (pc_topics_lr)",
                                         results["unsup_gated"]), flush=True)
                if ab_check:
                    readout_ab_report(
                        us_train_scored, us_test_scored, C, lay.K, recall_targets=rt,
                        fdr_targets=ft, min_count=args.min_label_count,
                        label="unsup_gated",
                        seed=(args.seed if args.seed is not None else 0),
                        sample_frac=args.readout_sample_frac, distributed=_dist_us,
                        max_iter=args.readout_max_iter, theta_topm=theta_topm)
                # Conditional A/B: does supervision sharpen P(child|parent) vs the
                # unsupervised twin? (The metric the clinician workflow cares about.)
                results["unsup_gated_conditional"] = _conditional(
                    proba_us, y_te, m_te, "unsup_gated")
                # Per-node domain mass on the UNSUPERVISED λ too, so the A/B tells us
                # whether the hierarchy-aligned specialization is a PC effect or a
                # property of the gated multi-domain representation itself (0078).
                if readout_mode == "distributed":
                    us_train_scored.unpersist(); us_test_scored.unpersist()
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
                if readout_mode == "distributed":
                    # UNGATED arm: its θ is --k wide, not the gate layout's lay.K.
                    dh_train = dh_model.transform(bundle.train_df).cache()
                    dh_test = dh_model.transform(bundle.test_df).cache()
                    results["dag_head"] = distributed_score_arm(
                        dh_train, dh_test, C, int(args.k), recall_targets=rt,
                        fdr_targets=ft, min_count=args.min_label_count,
                        label="dag_head", max_iter=args.readout_max_iter,
                        theta_topm=theta_topm, checkpoint_dir=out)[0]
                    dh_train.unpersist(); dh_test.unpersist()
                else:
                    Pi_tr, y_tr, m_tr, _ = _collect_theta_labels(
                        dh_model.transform(bundle.train_df), C)
                    Pi_te, y_te, m_te, ids_dh = _collect_theta_labels(
                        dh_model.transform(bundle.test_df), C)
                    results["dag_head"] = _score(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te,
                                                 doc_keys=ids_dh)
                _dump_partial_results(out, results)
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
            # The authoritative write: the SAME two paths the fit-only save
            # already landed, now carrying every arm's readout (and the per-node
            # domain mass, which needed no readout but is reported next to it).
            _save_fit(out, gp, C, manifest_fields, results=results,
                      domain_mass=domain_mass)
            print(f"[driver]   saved gated_pc result to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
