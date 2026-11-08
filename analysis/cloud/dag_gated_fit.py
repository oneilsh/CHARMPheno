"""Cloud fit driver for the DAG-gated LDA on the Mondo label subgraph (model_class gated_pc).

Assembles (or reloads from cache) the multi-domain native-Mondo corpus — per
document its features and the Mondo terms it attests (`labels`; with
`--gate-attestation ever` also `gate_labels`), plus the label subgraph the code
ladder reaches — and fits `DagGatedLDAEstimator` on it (spec 2026-10-30): the DAG
(`setDag`), the gate's labels column, `minDocsPerNode` (default 100, counted on the
training split), the per-domain feature columns and vocabularies, and the balanced
seed's node -> attesting codes, all in Mondo curies and concept ids. The fit
knobs start from the estimator's recommended preset; a driver flag overrides one.

The fitted model is the run's node set of record: it is saved MLlib-style to
`<run>/model`, with `<run>/label_dag.json` (the DAG given, the node set learned),
`<run>/bundle_meta.json` and the manifest's `corpus_manifest.int2cid` (the integer
bridge the run dir's readers key by, DERIVED from the model) and `<run>/code_map.tsv`
(the code ladder landed on the kept nodes). The λ / α also go to
`<run>/gated_pc_result.npz` for the readers that take the arrays. The fit ends at
that save; every readout and evaluation is `dag_gated_readout.py` against it.
Experiment front matter says `model_class: dag_gated` (or the old `gated_pc`).

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
from spark_vi.dag import children_and_min_depth
from spark_vi.readout import (LocalReadoutData, SparkReadoutData, closure_matrix,
                              collect_lean_proba, fit_heads, write_heads)
from spark_vi.readout import evaluate as _ev
from spark_vi.readout.heads import READOUT_GTOL, READOUT_MAX_ITER


def _cprint(msg, **kw) -> None:
    """print() through term_colors.highlight_line (ERROR/WARN/HIT/MISS/
    rebuild/macro-AUC/eta-boost -- see term_colors.py). Identity when color
    is disabled or nothing matches, so this is safe on every "[driver]"/
    "[episode]"/"[mondo]" status line in this module, not just the ones that
    happen to light up.

    LOCAL import, not module-level: this module is reachable transitively
    from `dag_gated_readout.py`'s top-level import (itself imported at module
    level by `hpoa_stage2_probe.py`, which rides Spark's `--py-files` for its
    own mapPartitions kernel) -- term_colors must never become a top-level
    dependency an executor could be asked to resolve. See term_colors.py's
    module docstring."""
    from term_colors import highlight_line
    print(highlight_line(msg), **kw)


def _array_blocks(proba, y, mask=None, doc_keys=None):
    """Dense test-split arrays as one in-memory score block (`readout.evaluate`):
    the local backend of every readout metric, for tests and the sklearn oracle arm.
    `doc_keys=None` numbers the rows."""
    proba = np.asarray(proba, dtype=np.float32)
    y = np.asarray(y).astype(np.uint8)
    mask = np.ones_like(y) if mask is None else np.asarray(mask).astype(np.uint8)
    ids = (np.arange(proba.shape[0], dtype=np.int64) if doc_keys is None
           else np.asarray(doc_keys, dtype=np.int64))
    return _ev.Blocks([{"ids": ids, "proba": proba, "y": y, "mask": mask}])


def detection_readout(proba_DC, y_DC, recall_targets, *, doc_keys=None):
    """Case-vs-background detection on dense arrays (`readout.evaluate.detection`):
    the per-unit max over the informative disease nodes against the root's label.

    `doc_keys` pools documents onto PERSONS (max over every (document, node) cell a
    person owns, the foreground indicator OR'd the same way), so a person with three
    documents is one case, not three votes (spec R5.7). `None` keeps the
    per-document pool. Constant columns are excluded before the pooling: a constant
    column in the max can pin it and the AUC at exactly 0.5 (exp 0104)."""
    return _ev.detection(
        _array_blocks(proba_DC, y_DC, doc_keys=doc_keys), recall_targets,
        unit_divisor=None if doc_keys is None else _dk.DOC_KEY_RADIX,
        unit_name="person")


def readout_from_blocks(blocks, C, *, recall_targets, fdr_targets, min_count=0,
                        skip_constant=False, person_pool=True, score="proba",
                        curve_columns=()):
    """The readout block from score blocks (Spark or local; `readout.evaluate`):
    ranking (AUC/AP + top-k screening over each node's masked rows), per-node
    precision@recall / recall@FDR, case-vs-background detection (person grain by
    default: the blocks' ids are doc keys) and the per-node (auc, n_pos) the
    headline splits by rarity.

    `skip_constant` (R2.1) drops a node whose scores are constant over its scored
    rows from the ranking macro (its AUC is exactly 0.5 by construction). Off for
    the prevalent arms, on for the incident arm. `curve_columns` adds `curves`, the
    thinned ROC/PR points of those nodes (ADR 0051: curves for a node sample)."""
    return _ev.readout_metrics(
        blocks, C, recall_targets=recall_targets, fdr_targets=fdr_targets,
        min_count=min_count, skip_constant=skip_constant, score=score,
        unit_divisor=_dk.DOC_KEY_RADIX if person_pool else None, unit_name="person",
        curve_columns=curve_columns)


def readout_from_proba(proba, y_te, m_te, C, *, recall_targets, fdr_targets,
                       min_count=0, skip_constant=False, doc_keys=None):
    """`readout_from_blocks` on dense arrays. `doc_keys` pools detection onto
    persons; `None` keeps the per-document pool."""
    return readout_from_blocks(
        _array_blocks(proba, y_te, m_te, doc_keys), C, recall_targets=recall_targets,
        fdr_targets=fdr_targets, min_count=min_count, skip_constant=skip_constant,
        person_pool=doc_keys is not None)


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
# `dag_gated_readout.observe_root_everywhere` is what makes the root head a real
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


def _marginal_ece_by_depth(blocks, scored_nodes, depth, n_bins=10):
    """Pooled MARGINAL calibration (ECE over ALL test docs, not the parent cohort)
    per depth, for both arms. The flat head's σ(z_c) is a P(c | parent) and is
    expected to be badly calibrated as a marginal (it never saw the background);
    the product is the arm that claims to be a marginal P(c | d). Depth 0 is the
    root head itself — the detection head's own calibration. `blocks` carry the
    flat "proba" and the "stacked" product."""
    groups = {}
    for c in scored_nodes:
        groups.setdefault(int(depth.get(c) if depth.get(c) is not None else -1),
                          []).append(int(c))
    ece = _ev.calibration_by_group(blocks, groups, scores=("proba", "stacked"),
                                   n_bins=n_bins)
    return {str(d): {"n_nodes": len(cs), "flat": ece.get(("proba", d)),
                     "stacked": ece.get(("stacked", d))}
            for d, cs in sorted(groups.items())}


def stacked_readout(blocks, parent_int, C, *, recall_targets, fdr_targets,
                    min_count=0, prevalent=None, arm_label="gated_pc",
                    person_pool=True, curve_columns=()):
    """The `gated_pc_stacked` results block: the readout of the closure product of
    the SAME per-doc probabilities the prevalent arm scored, plus the reads that
    make the product interpretable next to it:

      * `readout` — ranking / PR / detection / per_node on `P_stack`, the same
        mask and the same person-grain detection pool as the prevalent arm;
      * `root_only_detection` — detection on the root head alone (column 0, which
        under the stacked arm is a case-vs-background logistic): if the stacked
        detection does not beat this, the product adds nothing a single detection
        head would not;
      * `marginal_ranking` — HSLDA's own read, per-node AUC over ALL test docs;
      * `paired_vs_prevalent` — per-node AUC delta (stacked − flat) over the
        shared scored nodes, pooled and by depth (counts of nodes only);
      * `marginal_ece_by_depth` — pooled marginal ECE of both arms by depth.

    `blocks` are the flat arm's score blocks (Spark or local); the product is
    formed where each block lives, so nothing (D, C)-shaped reaches the driver.
    `parent_int` is the bundle's engine-id parent map."""
    C = int(C)
    M = closure_matrix(parent_int, C)
    sb = _ev.with_stacked(blocks, M).persist()
    unit = _dk.DOC_KEY_RADIX if person_pool else None
    try:
        readout = _ev.readout_metrics(
            sb, C, recall_targets=recall_targets, fdr_targets=fdr_targets,
            min_count=min_count, score="stacked", unit_divisor=unit, unit_name="person",
            curve_columns=curve_columns)
        root_det = _ev.detection(sb, recall_targets, columns="root",
                                 unit_divisor=unit, unit_name="person")
        _, depth = children_and_min_depth(parent_int, C)
        n_terms = M.sum(axis=1)
        # HSLDA's own read — per-node AUC over ALL test docs (y_c vs everyone, no
        # cohort): the de-novo case-finding question the within-cohort ranking never
        # asks. The flat head is a P(c | parent) and is expected to collapse here (it
        # never saw the background); the product is the arm that claims a marginal.
        # Same scorer and min_count, no mask, both arms, paired.

        def _marg(score):
            per = _ev.column_metrics(sb, score=score, mask=None)
            rk, _pr = _ev.ranking_and_pr(per, C, recall_targets=(), fdr_targets=(),
                                         min_count=min_count)
            return {"macro": rk["macro"],
                    "per_node": {c: r for c, r in rk["per_label"].items()
                                 if r.get("skipped") is None
                                 and r.get("auc") is not None}}
        marginal = {
            "definition": "per-node AUC of P(c|d) over ALL test docs (no mask: "
                          "y_c vs everyone), the de-novo read; same scorer and "
                          "min_count as the within-cohort ranking",
            "flat": _marg("proba"), "stacked": _marg("stacked")}
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
                sb, sorted(set(readout["per_node"]) | {0}), depth),
        }
    finally:
        sb.unpersist()
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
            lines.append(f"    min depth {d}: n={s['n']} median={_med(s['median'])} "
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
            lines.append(f"      min depth {d}: n={st['n']} median={_med(st['median'])} "
                         f"wins/losses={st['wins']}/{st['losses']}")
    ece = block.get("marginal_ece_by_depth") or {}
    if ece:
        lines.append("  marginal ECE over ALL test docs (flat sigma(z_c) vs "
                     "stacked), by min depth [0 = root head]:")
        for d, s in ece.items():
            lines.append(f"    min depth {d}: n_nodes={s['n_nodes']} flat="
                         f"{_f(s['flat']).strip()} stacked={_f(s['stacked']).strip()}")
    return "\n".join("[driver] " + ln if not ln.startswith("[") else ln
                     for ln in lines)


def score_arm(Pi_tr, y_tr, m_tr, Pi_te, y_te, m_te, C, *, recall_targets,
              fdr_targets, min_count=0, doc_keys=None):
    """Full readout for one arm's theta via the pc_topics_lr proba (a fresh per-node
    LR on the shaped theta). Pure; used inline by the driver and by dag_gated_readout
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

    `name` exists so the standalone re-readout (dag_gated_readout, which writes
    into a FINISHED run's dir) lands in its own `results_readout.json` instead of
    clobbering the record the fit itself left behind."""
    tmp = out / (name + ".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out / name)


# --------------------------------------------------------------------------- #
# Fit resume: the engine's own checkpoint under <run>/fit_ckpt.                #
# --------------------------------------------------------------------------- #
FIT_CKPT_DIR = "fit_ckpt"
FIT_IDENTITY_FILE = "fit_identity.json"
# Estimator Params that may differ between the interrupted run and its resume:
# the iteration budget (a resume finishes or extends it) and the persistence plumbing.
_RESUME_FREE_PARAMS = frozenset({"maxIter", "saveDir", "saveInterval", "resumeFrom"})


def fit_identity(est, bundle_key):
    """What must match for a checkpoint to be resumed: the corpus (the bundle cache
    key, which folds the spec AND the assembler's source) and every estimator Param
    except the budget and the persistence plumbing. A long value (the gate DAG, a
    seed profile) is kept as its hash, so the file stays small and readable."""
    import hashlib
    params = {}
    for p in est.params:
        if p.name in _RESUME_FREE_PARAMS or not (est.isSet(p) or est.hasDefault(p)):
            continue
        v = json.dumps(est.getOrDefault(p), sort_keys=True, default=str)
        params[p.name] = (v if len(v) <= 200 else
                          "sha256:" + hashlib.sha256(v.encode()).hexdigest()[:16])
    return {"bundle_key": str(bundle_key), "params": params}


def plan_fit_resume(ckpt_dir, identity, max_iter, *, fresh=False):
    """`("fresh", 0)`, `("resume", n_done)` or `("done", n_done)` for this fit.

    A checkpoint is resumed only when its recorded identity equals this run's; a
    mismatch, or a checkpoint without an identity, raises SystemExit(2) naming the
    difference, because warm-starting a different corpus or configuration would
    train silently on the wrong thing. `max_iter` is the TOTAL budget: a resume runs
    the remaining `max_iter - n_done` iterations, and a checkpoint already at the
    budget is "done" (load it, do not refit). `fresh=True` discards the checkpoint."""
    import shutil
    ckpt_dir = Path(ckpt_dir)
    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if not (ckpt_dir / "manifest.json").exists():
        return "fresh", 0
    idp = ckpt_dir / FIT_IDENTITY_FILE
    if not idp.exists():
        raise SystemExit(f"[driver] ERROR: {ckpt_dir} holds a fit checkpoint with no "
                         f"{FIT_IDENTITY_FILE}, so it cannot be checked against this "
                         "run; pass --fresh-fit to discard it.")
    saved = json.loads(idp.read_text())
    if saved != identity:
        diffs = []
        if saved.get("bundle_key") != identity["bundle_key"]:
            diffs.append(f"bundle_key {saved.get('bundle_key')} != "
                         f"{identity['bundle_key']}")
        sp, ip = saved.get("params", {}), identity["params"]
        diffs += [f"{k}: {sp.get(k)} != {ip.get(k)}"
                  for k in sorted(set(sp) | set(ip)) if sp.get(k) != ip.get(k)]
        raise SystemExit("[driver] ERROR: refusing to resume the fit checkpoint in "
                         f"{ckpt_dir}; it was made for a different corpus or "
                         "configuration:\n  " + "\n  ".join(diffs) +
                         "\nRevert the change, or pass --fresh-fit to start over.")
    n_done = int(json.loads((ckpt_dir / "manifest.json").read_text())["n_iterations"])
    return ("done" if n_done >= int(max_iter) else "resume"), n_done


def _save_fit(out, gp, C, manifest_fields, *, results=None, domain_mass=None,
              partial=None):
    """Write `<out>/gated_pc_result.npz` + `manifest.json`. Called TWICE per run.

    **Why twice.** The fit is the expensive, unrepeatable half — hours of CAVI at
    whole-Mondo K — and the readout that follows it is the FRAGILE half: it is
    where the driver collects, the second batched solve and the ECE diagnostic
    live, and where the 0104 smokes died repeatedly. Writing the model only at the
    very end meant every readout death also threw away the fit, so recovery meant
    re-running the fit instead of re-running `dag_gated_readout` against it. So the
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
    # Atomic writes (tmp + os.replace): the final write OVERWRITES the fit-only
    # one, so a crash DURING it must not tear the npz/manifest that is the fit's
    # only durable copy — the same rule the readout checkpoint follows.
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


def write_bundle_meta(run_dir, bundle, model, name="bundle_meta.json"):
    """`<run>/bundle_meta.json`: what a reader needs to interpret the run without the
    bundle — the FITTED node set as the integer bridge (`node_bridge`: `parent_int`,
    `int2cid`, `cid2int`, `name_by_id`, from the model, never rebuilt from the
    bundle), the bundle's vocab maps and ledger, and the DAG the fit was given
    (`label_dag`, user ids). Ids, ontology/concept names and counts only, no patient
    data. The HDFS cache dies with the cluster; the run dir does not, so
    `inspect-topics` reads this first and never needs the bundle rebuilt."""
    from _case_finding_cache import _meta_dict
    meta = _meta_dict(bundle)
    br = node_bridge(model, bundle.name_by_id)
    out = {"parent_int": {str(c): list(ps) for c, ps in br["parent_int"].items()},
           "int2cid": {str(e): c for e, c in br["int2cid"].items()},
           "cid2int": {str(c): e for c, e in br["cid2int"].items()},
           "name_by_id": {str(c): n for c, n in br["name_by_id"].items()},
           "ledger": meta["ledger"], "label_dag": meta["dag"]}
    for k in ("vocab_maps", "vocab_map"):
        if k in meta:
            out[k] = meta[k]
    p = Path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return p


def write_code_map(run_dir, pairs, name="code_map.tsv"):
    """`<run>/code_map.tsv`: `std_cid\tnode_cid` rows of the attestation map on the
    FITTED node set (`landed_code_map`; WP-C'). Ids only — no patient data."""
    p = Path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        fh.write("std_cid\tnode_cid\n")
        for a, b in pairs:
            fh.write(f"{int(a)}\t{int(b)}\n")
    return p


# --------------------------------------------------------------------------- #
# The fitted node set, as the run dir's readers see it (spec 2026-10-30).      #
# --------------------------------------------------------------------------- #
# The fit learns the node set (which label-DAG nodes got a topic block) from the
# TRAINING documents; everything downstream reads it from the fitted model, never
# from the bundle. The model is saved MLlib-style to <run>/model (MLWritable, any
# Hadoop URI); the readers that predate it (inspect_topics, voi_tables, framings,
# topic_npmi, readout_dag) keep their integer bridge — engine id -> node cid,
# `parent_int` over engine ids — now DERIVED from the model (`node_bridge`) and
# written to <run>/bundle_meta.json and the manifest's corpus_manifest.int2cid.
MODEL_DIR = "model"
LABEL_DAG_FILE = "label_dag.json"
# The integer id of the synthetic root in the bridge (engine id 0), the
# `MONDO_NATIVE_ROOT_CID` / `_FOREST_ROOT_CID` convention.
BRIDGE_ROOT_CID = -1


def node_cid(node) -> int:
    """A user node id as the bridge's integer: a Mondo curie's numeric part
    (`mondo_cid`), an int concept id as itself."""
    if isinstance(node, str):
        from charmpheno.mondo.native_dag import mondo_cid
        return mondo_cid(node)
    return int(node)


def node_bridge(model, names=None) -> dict:
    """The fitted node set as the integer bridge the run dir's readers take:
    `{parent_int, int2cid, cid2int, name_by_id}`.

    Engine id 0 is the synthetic root (cid `BRIDGE_ROOT_CID`), 1..N the kept nodes in
    the model's engine order (the order of their topic blocks and of the readout's
    heads, `spark_vi.readout.labels`). `parent_int` is `{engine child: [engine
    parents]}`, the root omitted as a key (the `DagLayout` map). `names` is the
    bundle's `{node id: name}`."""
    ns = model._ns()
    names = names or {}
    int2cid = {0: BRIDGE_ROOT_CID}
    name_by_id = {BRIDGE_ROOT_CID: "root"}
    for e, u in enumerate(ns.user_ids[1:], start=1):
        c = node_cid(u)
        int2cid[e] = c
        name_by_id[c] = str(names.get(u, u))
    return {"parent_int": {int(c): [int(p) for p in ps]
                           for c, ps in ns.engine_parents.items()},
            "int2cid": int2cid, "cid2int": {c: e for e, c in int2cid.items()},
            "name_by_id": name_by_id}


def node_set_counts(model, n_given) -> dict:
    """The fitted node set's receipt: counts of nodes only (egress-safe)."""
    ns = model._ns()
    return {"n_given": int(n_given), "n_kept": len(ns.nodes),
            "n_pruned": len(ns.pruned), "n_spliced": len(ns.spliced),
            "min_docs_per_node": int(ns.min_docs_per_node),
            "n_train_docs": int(ns.n_docs)}


def landed_code_map(model, code_map, int2cid) -> list:
    """The bundle's code ladder `(std_cid, node id)` landed on the FITTED node set:
    `(std_cid, node cid)` rows, each code on the kept node its term is (or the
    nearest kept ancestors of a pruned / spliced term, the fit's own landing). A
    term the fit never saw lands nowhere and is dropped. The 'own codes' of a kept
    node, for the census, the relevance selector and the framings."""
    landing = model._ns().frontier.landing
    rows = set()
    for std, term in code_map:
        for e in landing.get(term, ()):
            rows.add((int(std), int(int2cid[int(e)])))
    return sorted(rows)


def save_fitted_model(model, run_dir) -> str:
    """Save the fitted DagGatedLDAModel to `<run>/model` (MLWritable, overwrite).
    The run dir is a local (workspace-mounted) path, so it is written as a
    `file://` URI: a bare path would resolve against fs.defaultFS (HDFS on a
    cluster)."""
    uri = (Path(run_dir) / MODEL_DIR).resolve().as_uri()
    model.write().overwrite().save(uri)
    return uri


def load_fitted_model(run_dir):
    """The fitted DagGatedLDAModel saved by the fit (`<run>/model`)."""
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    p = Path(run_dir) / MODEL_DIR
    if not (p / "metadata").exists():
        raise FileNotFoundError(
            f"{p} holds no saved DagGatedLDAModel: this run predates the node-set "
            "redesign (spec 2026-10-30) or its fit did not finish; re-fit it")
    return DagGatedLDAModel.load(p.resolve().as_uri())


def write_label_dag(run_dir, dag, model, *, labels_col, name=LABEL_DAG_FILE):
    """`<run>/label_dag.json`: the DAG the fit was given (`dag`: the bundle's,
    `[[child, parent | null], ...]` in user ids), the gate's labels column, and the
    node set the fit learned (`NodeSet.to_dict`: kept nodes in engine order, kept
    DAG, pruned -> landing, spliced, support). Node ids and counts only."""
    from charmpheno.omop.case_finding_assembly import dag_edges
    p = Path(run_dir) / name
    p.write_text(json.dumps({"dag": dag_edges(dag), "labels_col": labels_col,
                             "node_set": model._ns().to_dict()}))
    return p


def attach_heads_targets(df, model, labels_col, *, mask_mode="full",
                         label_col="label", mask_col="labelMask"):
    """Append the readout heads' dense targets, built from a labels column on the
    FITTED model's node set (spec 2026-10-30 R6, R8): `label_col` / `mask_col`,
    `array<double>` of length C = kept nodes + 1, index 0 the root and index e the
    model's engine node e (`spark_vi.readout.labels.NodeLabeler`: label = closure of
    the landed labels; `mask_mode` 'full' or 'closure'). Labels the node set does not
    know are ignored (the label year may attest a node the training documents never
    reached; it lands on its kept ancestors like any pruned one, and a node outside
    the given DAG cannot occur: the bundle's labels are DAG nodes).

    The labeler rides a broadcast (ADR 0047); the row function closes over the
    handle only and calls spark_vi code, so the executors need only the spark-vi
    zip."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType
    from spark_vi.readout.labels import NodeLabeler

    lab = NodeLabeler.from_node_set(model._ns(), mask_mode=mask_mode,
                                    unknown_labels="ignore")
    bc = df.sparkSession.sparkContext.broadcast(lab)

    def _targets(labels, _bc=bc):
        lb = _bc.value
        act, obs = lb(labels)
        C = lb.C
        y = [0.0] * C
        for i in act:
            y[i] = 1.0
        if obs is None:
            m = [1.0] * C
        else:
            m = [0.0] * C
            for i in obs:
                m[i] = 1.0
        return (y, m)

    schema = StructType([StructField("y", ArrayType(DoubleType(), False), False),
                         StructField("m", ArrayType(DoubleType(), False), False)])
    tmp = "__heads_targets"
    return (df.withColumn(tmp, F.udf(_targets, schema)(F.col(labels_col)))
            .withColumn(label_col, F.col(f"{tmp}.y"))
            .withColumn(mask_col, F.col(f"{tmp}.m"))
            .drop(tmp))


def attach_engine_frontier(df, model, labels_col, *, out_col="frontier"):
    """Append the gate's view of a labels column on the fitted node set: `out_col`,
    the engine-id frontier (`array<bigint>`, the model's own `FrontierFn`: landing,
    then the most-specific landed nodes) — what the fit's gate admitted for these
    documents. For the gated NPMI, which scores a node block on the documents its
    gate admits. Broadcast, as `attach_heads_targets`."""
    import dataclasses

    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, LongType

    fn = dataclasses.replace(model._frontier_fn(), unknown_labels="ignore")
    bc = df.sparkSession.sparkContext.broadcast(fn)

    def _frontier(labels, _bc=bc):
        return sorted(int(e) for e in _bc.value(labels))

    return df.withColumn(out_col, F.udf(_frontier, ArrayType(LongType()))(
        F.col(labels_col)))


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
                mask_col="labelMask", id_col="person_id",
                engine="spark", curve_columns=()):
    """Fit the per-node heads on the train split, score the test split, macro the
    metrics: `(readout, test score blocks, (V, b_raw))`. The blocks
    (`readout.evaluate.Blocks`, persisted on Spark) feed further readouts of the same
    scores, such as the stacked arm; the caller `unpersist()`s them.

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
        checkpoint_every=checkpoint_every)
    if checkpoint_dir is not None:
        write_heads(checkpoint_dir, label, V, b_raw, const, degenerate, C, K,
                    theta_topm, W_std=info.get("W_std"))
    # ADR 0051 step 4: the test split is scored where it lives (score blocks) and
    # the metrics merge per-node tallies; no (D, C) matrix reaches the driver.
    blocks = data_te.score_blocks(V, b_raw, degenerate=degenerate, const=const)
    readout = readout_from_blocks(blocks, C, recall_targets=recall_targets,
                                  fdr_targets=fdr_targets, min_count=min_count,
                                  curve_columns=curve_columns)
    return readout, blocks, (V, b_raw)


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


# Driver flag -> estimator Param, for the fit knobs the recommended preset sets
# (`DagGatedLDAEstimator.recommended`, RECOMMENDED_PARAMS). A flag left unset (None)
# keeps the preset's value; a flag given overrides it.
_FLAG_TO_PARAM = (
    ("n_bg", "gateNBg", int), ("tpn", "gateTpn", int),
    ("max_iter", "maxIter", int), ("tau0", "learningOffset", float),
    ("kappa", "learningDecay", float), ("subsampling_rate", "subsamplingRate", float),
    ("gamma_shape", "gammaShape", float), ("cavi_max_iter", "caviMaxIter", int),
    ("cavi_tol", "caviTol", float),
    ("optimize_doc_concentration", "optimizeDocConcentration", bool),
    ("gate_warm", "gateWarm", int), ("gate_seed_mode", "gateSeedMode", str),
    ("gate_seed_leaves", "gateSeedLeaves", bool),
    ("gate_open_seed", "gateOpenSeed", bool),
)
# The clinical driver's node-set threshold: a node gets a topic block when at least
# this many TRAINING documents' label closures reach it (>= the egress floor, so a
# per-node table built on a kept node clears it). Spec 2026-10-30.
DEFAULT_MIN_DOCS_PER_NODE = 100


def _build_estimator(args):
    """The DagGatedLDAEstimator for the fit: the recommended preset
    (`DagGatedLDAEstimator.recommended`) with the driver's flags on top, the label DAG
    and labels column, the node-set threshold (`minDocsPerNode`), the per-domain
    feature columns and vocabularies, and the balanced seed's node -> attesting
    terms (`nodeTerms`) and the profile seed, both keyed by node id and term id.

    Reads, from `args`: the CLI flags (`_FLAG_TO_PARAM`, `doc_concentration`,
    `min_docs_per_node`, `seed`) and the corpus-derived `_dag`, `_labels_col`,
    `_domain_cols`, `_vocabularies`, `_node_terms`, `_gate_seed_profile` (each
    optional)."""
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator

    over = {}
    for flag, param, cast in _FLAG_TO_PARAM:
        v = getattr(args, flag, None)
        if v is not None:
            over[param] = cast(v)
    dc = getattr(args, "doc_concentration", None)
    if dc is not None:
        if not dc > 0:
            raise ValueError(f"--doc-concentration must be > 0; got {dc}")
        over["docConcentration"] = [float(dc)]
    if getattr(args, "seed", None) is not None:
        over["seed"] = int(args.seed)
    mdn = getattr(args, "min_docs_per_node", None)
    over["minDocsPerNode"] = int(DEFAULT_MIN_DOCS_PER_NODE if mdn is None else mdn)
    over["labelsCol"] = str(getattr(args, "_labels_col", None) or "labels")
    dom_cols = getattr(args, "_domain_cols", None)
    if dom_cols:
        over["featuresCols"] = list(dom_cols)
    est = DagGatedLDAEstimator.recommended(**over)
    dag = getattr(args, "_dag", None)
    if dag is not None:
        est.setDag(dag)
    vocabs = getattr(args, "_vocabularies", None)
    if vocabs:
        est.setVocabularies(vocabs)
    nt = getattr(args, "_node_terms", None)
    if nt:
        est.setNodeTerms(nt)
    gsp = getattr(args, "_gate_seed_profile", None)
    if gsp:
        est.setGateSeedProfile(gsp)
    return est


def vocabularies_of(vocab_maps) -> list:
    """The per-domain vocab maps `{term: index}` as the estimator's `vocabularies`:
    one list of term ids per feature column, in index order."""
    out = []
    for vm in vocab_maps:
        terms = [None] * len(vm)
        for t, i in vm.items():
            terms[int(i)] = t
        out.append(terms)
    return out


def node_terms_of(code_map, vocab_map0, column="features_0") -> dict:
    """The code ladder `(std_cid, node id)` as the estimator's `nodeTerms`:
    `{node: [(column, std_cid)]}` over the codes in the condition vocabulary.
    Keyed by the most-specific term each code resolves to; the estimator drops the
    entries of nodes it does not keep. Terms are qualified with the feature column,
    so a condition concept id that is also a token of another domain stays
    unambiguous."""
    out: dict = {}
    for std, node in code_map:
        if int(std) in vocab_map0:
            out.setdefault(node, set()).add((column, int(std)))
    return {n: sorted(ts) for n, ts in out.items()}


# --------------------------------------------------------------------------- #
# Multi-domain / Mondo corpus seam (shared by the fit and dag_gated_readout).   #
# --------------------------------------------------------------------------- #
# The `dag_source` values a manifest may carry. Only `mondo_native` is built since
# the 2026-10 cleanup (`_refuse_removed_spec_options`); `dag_gated_readout` imports
# this to recognize a Mondo run's manifest.
_MONDO_DAG_SOURCES = ("mondo", "mondo_native")

# The multi-domain assembler's kwargs, derived from a corpus SPEC — the dict the
# fit writes into `manifest["corpus_manifest"]` and the re-readout reads back out.
# One derivation, two callers, so a re-readout's cache key cannot drift from the
# key the fit stored the bundle under.
#
# Not here since spec 2026-10-30 (not bundle inputs any more): the node-set
# threshold (`min_n`, `min_positives`), the topic layout (`n_bg`, `tpn`) and the
# readout's `label_mask_mode` / `emit_labels`.
#
# `doc_spec` is a corpus-identity field that is NOT in this tuple, on purpose:
# every name here is forwarded to the assembler, and neither assembler accepts a
# doc-spec argument (both hard-code `PatientCohortDocSpec` internally). It is a
# key-only input and travels in `key_extra` instead — see `doc_spec_identity`.
_SPEC_ASSEMBLY_KEYS = (
    "disease", "cdr", "billing", "person_mod", "holdout_frac",
    "vocab_size", "min_df", "min_patient_count", "doc_min_length",
    "strip_mode", "lookback_days", "label_window_days", "index_mode",
)


def doc_spec_identity() -> str:
    """The DOC-UNIT identity token this driver's corpora are assembled under.

    Read off the class rather than written down, so that swapping the driver's
    doc spec moves the token — and therefore the bundle cache key — without anyone
    having to remember to (audit seam 4; spec R5.3). `min_doc_length`, the spec's
    only other identity-bearing parameter, is already folded into the key as
    `doc_min_length`. `compute_bundle_cache_key` folds this only when it differs
    from `DEFAULT_DOC_SPEC`."""
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    return PatientCohortDocSpec().name


def multidomain_corpus_spec(args, extra_domains) -> dict:
    """The corpus_manifest block for a native-Mondo multi-domain fit.

    Everything the bundle's cache key needs AND everything re-assembling it from
    scratch needs, in one dict — including `billing` and the Mondo build inputs,
    which a post-hoc rebuild cannot invent. `index_mode` is `population` (there is
    no single disease). The removed options keep their OFF values (`dag_collapse`,
    `label_set`, `preindex_closure`) so `_refuse_removed_spec_options` reads one
    shape."""
    return {
        "dag_source": args.dag_source,
        "disease": args.disease, "cdr": args.cdr, "billing": args.billing,
        "source_table": args.source_table,
        "extra_domains": list(extra_domains),
        "index_mode": "population",
        "person_mod": args.person_mod, "vocab_size": args.vocab_size,
        "min_df": args.min_df, "min_patient_count": args.min_patient_count,
        "doc_min_length": args.doc_min_length,
        "holdout_frac": args.holdout_frac,
        "strip_mode": args.strip_mode, "lookback_days": args.lookback_days,
        "label_window_days": args.label_window_days,
        "window_mode": args.window_mode,
        "prior_obs_days": args.prior_obs_days, "window_days": args.window_days,
        "mondo_version": args.mondo_version,
        "mondo_branch": args.mondo_branch or "",
        "mondo_cache_dir": args.mondo_cache_dir,
        "label_set": "", "label_set_path": "",
        "dag_collapse": False,
        # The doc unit (R5.3), a cache-key input.
        "doc_spec": doc_spec_identity(),
        # Label redesign R2 (exp 0146): the bundle's TRAIN split also carries the
        # all-ever attestation (`gate_labels`). Present ONLY when asked.
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
    key_extra = {"multidomain": True,
                 "doc_spec": str(spec.get("doc_spec") or doc_spec_identity())}
    if spec.get("gate_attestation") and str(spec["gate_attestation"]) != "label":
        key_extra["gate_attestation"] = str(spec["gate_attestation"])
    from charmpheno.mondo.native_dag import MONDO_NATIVE_VERSION
    key_extra.update(mondo=True, mondo_version=spec["mondo_version"],
                     mondo_branch=spec.get("mondo_branch") or "")
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
    doc_id (`PatientCohortDocSpec`) — and goes through the SAME `provider` the labels
    used (same code ladder), so the two columns differ only in WHEN, never in HOW a
    code becomes a node. Returns `(person_id, doc_id, gate_labels)`; a doc whose
    person attests nothing keeps a present, empty array."""
    from pyspark.sql import functions as F
    from charmpheno.omop.case_finding_assembly import GATE_LABELS_COL, LABELS_COL
    roster = docs.select("person_id", "doc_id").distinct().withColumn(
        "source_cohort", F.element_at(F.split(F.col("doc_id"), ":"), 1))
    ev = events.select("person_id", "concept_id").join(
        roster.select("person_id", "source_cohort").distinct(), "person_id")
    att = provider(ev)
    dt = att.schema[LABELS_COL].dataType.simpleString()
    att = att.select("doc_id", F.col(LABELS_COL).alias(GATE_LABELS_COL))
    return (roster.select("person_id", "doc_id").join(att, "doc_id", "left")
            .withColumn(GATE_LABELS_COL, F.coalesce(
                F.col(GATE_LABELS_COL), F.array().cast(dt))))


def _gate_count_receipt(df, label_col, gate_col):
    """Pooled R2.3 receipt: attested (doc, node) cells and attested docs, label year
    vs all-ever, and the median per-node ratio of attesting train docs. Counts only."""
    from pyspark.sql import functions as F

    def per_node(col):
        return {r["n"]: int(r["c"]) for r in
                df.select(F.explode(col).alias("n")).groupBy("n")
                  .agg(F.count("*").alias("c")).collect()}
    b, a = per_node(label_col), per_node(gate_col)
    ratios = sorted(a.get(n, 0) / c for n, c in b.items() if c > 0)
    return {"cells_label": sum(b.values()), "cells_ever": sum(a.values()),
            "docs_gated_label": df.where(F.size(label_col) > 0).count(),
            "docs_gated_ever": df.where(F.size(gate_col) > 0).count(),
            "median_node_ratio": (ratios[len(ratios) // 2] if ratios else float("nan")),
            "n_nodes": len(b)}


def _attach_gate_ever(spark, bundle, *, provider, cdr, billing, person_mod,
                      _events=None):
    """Label redesign R2 (spec 2026-10-13; R8 of spec 2026-10-30): add `gate_labels`
    to each TRAIN document — its person's ALL-EVER attestations through the same
    code ladder — beside the label-year `labels`, which stay untouched (they are
    the readout's); the test split is untouched (the test transform is ungated).
    The fit's gate reads `gate_labels` (`labelsCol`). MISS-only, inside the
    assemble closure, where the provider exists. One full-history `condition_era`
    read (the sidecar's load; ~70 s on 0137's corpus).

    Why the future may gate a TRAIN document: the gate shapes topics on the train
    split only; test documents are folded in ungated, so no test outcome reaches
    them (spec 2026-10-13, Why)."""
    from charmpheno.omop.case_finding_assembly import GATE_LABELS_COL, LABELS_COL
    events = _events
    if events is None:
        from charmpheno.omop import load_omop_bigquery
        events = load_omop_bigquery(spark=spark, cdr_dataset=cdr,
                                    billing_project=billing,
                                    person_sample_mod=person_mod,
                                    source_table="condition_era")
    before = bundle.train_df
    gate = _gate_ever_attested(before, events, provider)
    after = _attach_gate_column(before, gate, GATE_LABELS_COL).cache()
    rec = _gate_count_receipt(after, LABELS_COL, GATE_LABELS_COL)
    print(f"[gate-ever] train gate, label year -> ever: attested (doc, node) cells "
          f"{rec['cells_label']:,} -> {rec['cells_ever']:,}; docs with any label "
          f"{rec['docs_gated_label']:,} -> {rec['docs_gated_ever']:,}; median "
          f"per-node ratio of attesting docs {rec['median_node_ratio']:.2f} over "
          f"{rec['n_nodes']} nodes", flush=True)
    bundle.train_df = after
    bundle.gate_ever_receipt = rec
    return bundle


def _attach_gate_column(df, gate, col):
    """`df` with `col` joined from `gate` on the doc key, refusing a missing /
    duplicate / mismatched join LOUDLY (WP-D3 join integrity).

    `gate` carries `(person_id, doc_id, col)`, one row per document. The guards, in
    order: (1) the gate frame must be UNIQUE on (person_id, doc_id) — an ambiguous
    gate is a raise, never a silent pick; (2) a LEFT join must not change the
    document count; (3) NO document may come back without a gate — a missing join
    surfaces as a NULL (distinct from an EMPTY gate's present `[]`) and is a hard
    error, never an empty-gate fallback. Every other column rides through."""
    from pyspark.sql import functions as F

    tmp = f"_gate_{col}"
    g = gate.select("person_id", "doc_id", F.col(col).alias(tmp)).cache()
    try:
        n_gate = g.count()
        n_keys = g.select("person_id", "doc_id").distinct().count()
        if n_gate != n_keys:
            raise ValueError(
                f"gate join: {n_gate - n_keys} duplicate (person_id, doc_id) rows "
                f"in the {col!r} frame — refusing an ambiguous gate (one per "
                "document is required).")
        n_before = df.count()
        base = df.drop(col) if col in df.columns else df
        joined = base.join(g, on=["person_id", "doc_id"], how="left")
        n_after = joined.count()
        if n_after != n_before:
            raise ValueError(
                f"gate join: the {col!r} join changed the document count "
                f"({n_before} -> {n_after}) — a duplicate gate key inflated the "
                "bundle; refusing.")
        n_missing = joined.where(F.col(tmp).isNull()).count()
        if n_missing:
            raise ValueError(
                f"gate join: {n_missing} bundle document(s) received no {col!r} — "
                "a missing join is a hard error, never an empty-gate fallback.")
        return joined.withColumn(col, F.col(tmp)).drop(tmp)
    finally:
        g.unpersist()


def mondo_assemble_fn(spec, *, on_inputs=None, _build_inputs=None, _assemble=None,
                      cache_uri=None):
    """A MISS-ONLY assembler for the native Mondo corpus: build the code ladder and
    the label subgraph, THEN assemble.

    The build costs a whole-Mondo -> OMOP mapping and a code climb — minutes of
    BigQuery — and the fit consumes it only through the bundle (its `dag`, its
    `labels`, its `native_code_map`). So the build lives inside the assemble seam,
    where `load_or_build_case_finding_bundle` reaches it only after `try_load` has
    missed. A HIT pays for none of it.

    `spec["gate_attestation"] == "ever"` (exp 0146) then adds each TRAIN
    document's all-ever attestations as `gate_labels`.

    `on_inputs(stats=...)` hands back the build receipt; it is NOT called on a HIT.

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
            format_native_mapping_report, make_mondo_native_attested_provider)
        build = _build_inputs or build_mondo_native_fit_inputs
        assemble = _assemble or assemble_multidomain_case_finding_corpus
        branch = spec.get("mondo_branch") or ""
        with _phase(f"build the Mondo code ladder + label subgraph "
                    f"(branch={branch or 'ALL'})"):
            inputs = build(
                spark, cdr=spec["cdr"], billing=spec["billing"],
                mondo_version=spec["mondo_version"],
                mondo_cache_dir=spec.get("mondo_cache_dir") or "data/mondo",
                branch_root=(branch or None))
            provider = make_mondo_native_attested_provider(
                inputs.code_map_sdf, doc_spec=PatientCohortDocSpec())
            # BOTH lines before any fit: the subgraph's size is measured here.
            print(format_native_mapping_report(inputs.stats), flush=True)
            print(format_native_build_report(inputs.stats), flush=True)
            if on_inputs is not None:
                on_inputs(stats=inputs.stats)
        bundle = assemble(spark, label_dag=inputs.dag, names=inputs.names,
                          attested_provider=provider, index_df=None,
                          doc_spec=None, **assembly_params)
        if spec.get("gate_attestation") == "ever":
            with _phase("label redesign R2: all-ever gate labels (train, MISS-only)"):
                bundle = _attach_gate_ever(
                    spark, bundle, provider=provider,
                    cdr=spec["cdr"], billing=spec["billing"],
                    person_mod=assembly_params["person_mod"])
        # WP-C' (exp 0136): the `(std_cid, Mondo term)` code ladder rides the bundle
        # as an ATTRIBUTE, is persisted in the cache meta, and is landed on the
        # fitted node set by the driver (`landed_code_map` -> <run>/code_map.tsv).
        _collect = getattr(inputs.code_map_sdf, "collect", None)  # a stub has none
        if _collect is not None:
            bundle.native_code_map = sorted(
                (int(r["std_cid"]), str(r["node_id"])) for r in _collect())
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
        description="DAG-gated LDA fit on the Mondo label subgraph; ends at the saved "
                    "fit (every readout is dag_gated_readout.py). The fit knobs "
                    "default to the estimator's recommended preset "
                    "(DagGatedLDAEstimator.recommended); a flag given overrides it.",
        allow_abbrev=False)
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
    p.add_argument("--disease", default="rare6")
    p.add_argument("--extra-domains", default="",
                   help="comma-separated non-condition domains (drug, procedure, "
                        "measurement) for a MULTI-DOMAIN fit (MixEHR-style per-domain "
                        "vocab; domain 0 is always conditions). Requires "
                        "--window-mode lookback.")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--strip-mode", choices=["test_only", "both"], default="test_only")
    p.add_argument("--label-mask-mode", choices=["full", "closure"], default="full",
                   help="the readout's per-node observation policy, recorded in the "
                        "manifest for dag_gated_readout (the heads' labelMaskMode): "
                        "'full' (background is a true negative everywhere) or "
                        "'closure' (observe the active closure + DAG siblings only). "
                        "Not a bundle input.")
    p.add_argument("--dag-source", choices=["mondo_native"], default="mondo_native",
                   help="the label space: Mondo terms reached by the code ladder and "
                        "their Mondo ancestors (the only source since the 2026-10 "
                        "cleanup).")
    p.add_argument("--mondo-branch", default="",
                   help="restrict the label subgraph to one body-system Mondo subtree "
                        "(e.g. MONDO:0004995 = cardiovascular disorder); '' = whole "
                        "Mondo.")
    p.add_argument("--min-docs-per-node", type=int, default=DEFAULT_MIN_DOCS_PER_NODE,
                   help="the estimator's minDocsPerNode: a node gets a topic block "
                        "only when at least this many TRAINING documents' label "
                        "closures reach it (>= the egress floor). Default "
                        f"{DEFAULT_MIN_DOCS_PER_NODE}. Fit-only: the bundle does not "
                        "depend on it.")
    p.add_argument("--mondo-version", default="2026-06-02")
    p.add_argument("--mondo-cache-dir", default="data/mondo")
    # The fit knobs. Default None = the recommended preset's value
    # (RECOMMENDED_PARAMS); a value given overrides it. Fit-only: no cache key moves.
    p.add_argument("--n-bg", type=int, default=None,
                   help="gateNBg: shared background topics (preset 200).")
    p.add_argument("--tpn", type=int, default=None,
                   help="gateTpn: topics per kept node (preset 1); K = n_bg + "
                        "kept nodes * tpn.")
    p.add_argument("--gate-warm", type=int, default=None, metavar="N",
                   help="gateWarm: leaves-first gate schedule, N iterations per "
                        "level (leaves first), then the full gate; 0 = single round "
                        "(preset 2). Report 2026-10-19-leaves-first-gate-sim.")
    p.add_argument("--gate-seed-mode", choices=("subtree", "balanced"), default=None,
                   help="gateSeedMode, the open-time seed: 'subtree' = the word counts "
                        "of the documents a block admits; 'balanced' = each child's "
                        "subtree share equally plus the parent's own coded documents, "
                        "descendants' own codes removed (exp 0160; preset).")
    p.add_argument("--gate-seed-leaves", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="gateSeedLeaves: seed every leaf block at iteration 0 from its "
                        "own documents' word counts (exp 0162; preset on).")
    p.add_argument("--gate-open-seed", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="gateOpenSeed: seed each block when it opens (preset on).")
    p.add_argument("--gate-seed-profile", default="", metavar="TSV",
                   help="With gateWarm > 0: an external phenotype profile table "
                        "(mondo_id, concept_id, neg; e.g. hpo_profile_marks.py output) "
                        "that REPLACES a profiled node's condition-domain seed (leaves "
                        "at iteration 0, later blocks at opening); unprofiled nodes "
                        "keep the count seed. Fit-only.")
    p.add_argument("--gate-attestation", choices=("label", "ever"), default="label",
                   help="Label redesign R2 (spec 2026-10-13): which attestations GATE a "
                        "TRAIN document's topic blocks. 'label' (default): the label "
                        "year's (`labels`). 'ever': anywhere in the person's record "
                        "(`gate_labels`, full-history condition_era through the same "
                        "code ladder). The readout always reads the label year. A "
                        "distinct bundle key.")
    p.add_argument("--optimize-doc-concentration",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="optimizeDocConcentration (preset off).")
    p.add_argument("--max-iter", type=int, default=None, help="maxIter (preset 60).")
    p.add_argument("--subsampling-rate", type=float, default=None,
                   help="subsamplingRate (preset 0.1).")
    p.add_argument("--tau0", type=float, default=None,
                   help="learningOffset, the Robbins-Monro offset (preset 64).")
    p.add_argument("--kappa", type=float, default=None,
                   help="learningDecay (preset 0.51).")
    p.add_argument("--gamma-shape", type=float, default=None,
                   help="gammaShape (preset 100).")
    p.add_argument("--cavi-max-iter", type=int, default=None,
                   help="caviMaxIter (preset 100).")
    p.add_argument("--cavi-tol", type=float, default=None, help="caviTol (preset 1e-3).")
    p.add_argument("--doc-concentration", type=float, default=None,
                   help="docConcentration, a scalar Dirichlet alpha (preset 0.5).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fit-save-interval", type=int, default=0,
                   help="checkpoint the fit every N SVI iters (0 = off) to "
                        "<out>/fit_ckpt; re-running the same command resumes from it, "
                        "finishing the iteration budget in total. Writes are atomic; "
                        "each costs a lambda dump, so keep N modest (e.g. 10). "
                        "fit_ckpt is removed once the finished fit is saved.")
    p.add_argument("--fresh-fit", action="store_true",
                   help="discard an existing <out>/fit_ckpt and fit from scratch "
                        "(otherwise a matching checkpoint is resumed, and a "
                        "mismatched one is refused).")
    p.add_argument("--num-partitions", type=int, default=0,
                   help="repartition the corpus to this many partitions before "
                        "fitting (0 = leave as-is). Few parquet part-files pin the "
                        "job to ~2 executors under dynamic allocation; set this ≈ "
                        "total cluster executor cores. Pair with "
                        "CHARM_SPARK_CONF='spark.locality.wait=0s'.")
    p.add_argument("--cache-uri", default=None)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def _profile_seed(path, dag, vocab_map0, column="features_0"):
    """`--gate-seed-profile` -> the estimator's `gateSeedProfile` `{node: {(column,
    concept_id): weight}}` (`profile_eta.build_profile_seed`, keyed by Mondo curie and
    condition concept id; nodes outside the label subgraph are not seeded; the
    estimator drops the unkept ones). Returns `(profile, stats)`; stats are counts."""
    import pandas as pd
    from profile_eta import build_profile_seed
    gs_df = pd.read_csv(path, sep="\t")
    by_node = {str(m): str(m) for m in gs_df["mondo_id"].astype(str).unique()
               if str(m) in dag}
    seed, stats = build_profile_seed(
        gs_df, eid_by_mondo=by_node,
        vocab_index_by_concept={int(c): int(c) for c in vocab_map0})
    return ({n: {(column, int(c)): w for c, w in tw.items()}
             for n, tw in seed.items()}, stats)


def main() -> int:
    args = parse_args()
    configure_logging()
    # The run dir exists from minute zero and every [driver] line is teed to a
    # durable, sanitized driver_log.md inside it — open-append-close per line,
    # so results survive wrapper death, cluster timeout, and summary.md
    # truncation (exp 0103's smoke lost 4h of readout output to exactly that).
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from _driver_common import install_stdout_tee
    install_stdout_tee(out / "driver_log.md")
    extra_domains = tuple(d for d in args.extra_domains.split(",") if d)
    with make_spark_session(app_name="dag-gated-fit") as spark:
        # Driver-disk telemetry, IN-BAND (see disk_telemetry's docstring).
        start_disk_telemetry(
            extra_dirs=[d for d in spark.sparkContext.getConf()
                        .get("spark.local.dir", "").split(",") if d],
            log=lambda msg: _cprint(f"[driver] {msg}", flush=True))
        if args.dag_source != "mondo_native":
            raise ValueError(
                f"--dag-source {args.dag_source!r}: the gated fit runs on the native "
                "Mondo label space only (the SNOMED and anchor-Mondo paths were "
                "removed in the 2026-10 cleanup; last at commit fff7bf4067)")
        if args.window_mode != "lookback":
            raise ValueError(
                f"--dag-source {args.dag_source} / --extra-domains requires "
                f"--window-mode lookback (got {args.window_mode})")
        corpus_spec = multidomain_corpus_spec(args, extra_domains)

        with _phase(f"assemble MONDO_NATIVE corpus (cached, cond + "
                    f"{list(extra_domains)})"):
            bundle = multidomain_load_or_build(spark, corpus_spec,
                                               cache_uri=args.cache_uri)
            vocab_maps = bundle.vocab_maps
            args._domain_cols = [f"features_{i}" for i in range(len(vocab_maps))]
            args._domain_names = ["condition", *extra_domains]
        _cprint(f"[driver]   ledger: {json.dumps(bundle.ledger)}", flush=True)
        # The estimator's inputs, all from the bundle, in user ids (spec 2026-10-30):
        # the label subgraph, the labels column (the gate's: label year, or all-ever
        # with --gate-attestation ever), the per-domain vocabularies, and the code
        # ladder as the balanced seed's node -> attesting terms.
        from charmpheno.omop.case_finding_assembly import GATE_LABELS_COL, LABELS_COL
        args._dag = bundle.dag
        args._labels_col = (GATE_LABELS_COL if args.gate_attestation == "ever"
                            else LABELS_COL)
        if args._labels_col not in bundle.train_df.columns:
            raise ValueError(f"the bundle's train split has no {args._labels_col!r} "
                             f"column (columns: {bundle.train_df.columns})")
        args._vocabularies = vocabularies_of(vocab_maps)
        code_map = getattr(bundle, "native_code_map", None) or []
        args._node_terms = node_terms_of(code_map, vocab_maps[0])
        _cprint(f"[driver]   label DAG: {len(bundle.dag)} node(s) given; gate labels "
                f"{args._labels_col!r}; nodeTerms for {len(args._node_terms)} node(s) "
                f"({sum(len(v) for v in args._node_terms.values())} condition codes)",
                flush=True)
        args._gate_seed_profile = None
        args._gate_seed_profile_stats = {}
        if args.gate_seed_profile:
            with _phase("build the profile seed (driver-side)"):
                gs, gs_stats = _profile_seed(args.gate_seed_profile, bundle.dag,
                                             vocab_maps[0])
                if not gs:
                    raise ValueError("--gate-seed-profile seeded NO node (none of the "
                                     "profiled nodes has a specific profile concept "
                                     "in this run's vocabulary) — check the TSV")
                args._gate_seed_profile = gs
                args._gate_seed_profile_stats = gs_stats
                _cprint(f"[driver]   profile seed: {json.dumps(gs_stats)}", flush=True)
        v_desc = " + ".join(f"{n}:{len(vm)}"
                            for n, vm in zip(args._domain_names, vocab_maps))
        _cprint(f"[driver]   corpus: V=({v_desc}) vocab", flush=True)
        if args.num_partitions and args.num_partitions > 0:
            # Repartition UP so the fit demands (and spreads across) more executors
            # than the cached parquet's few part-files would.
            before = bundle.train_df.rdd.getNumPartitions()
            bundle.train_df = bundle.train_df.repartition(args.num_partitions).cache()
            bundle.test_df = bundle.test_df.repartition(args.num_partitions).cache()
            bundle.train_df.count(); bundle.test_df.count()   # materialize the spread
            _cprint(f"[driver]   repartitioned corpus {before} -> "
                  f"{args.num_partitions} partitions (train+test cached)", flush=True)

        pc_est = _build_estimator(args)
        g = pc_est.getOrDefault
        if args._gate_seed_profile and int(g("gateWarm")) <= 0:
            raise ValueError("--gate-seed-profile requires gateWarm > 0")
        n_bg, tpn, max_iter = int(g("gateNBg")), int(g("gateTpn")), int(g("maxIter"))
        with _phase(f"gated_pc fit (DagGatedLDAEstimator: the node set is learned "
                    f"from the train split, minDocsPerNode={g('minDocsPerNode')})"):
            # Resume (the engine's own checkpoint, <out>/fit_ckpt): a re-run of the
            # same command continues an interrupted fit when the checkpoint's
            # identity (bundle key + every estimator Param but the budget) matches.
            ckpt = out / FIT_CKPT_DIR
            identity = fit_identity(pc_est, multidomain_cache_key(corpus_spec))
            action, n_done = plan_fit_resume(ckpt, identity, max_iter,
                                             fresh=bool(args.fresh_fit))
            if action == "resume":
                pc_est.setResumeFrom(str(ckpt))
                pc_est.setMaxIter(max_iter - n_done)
                _cprint(f"[driver]   RESUMING the fit from {ckpt} at iteration "
                        f"{n_done}; {max_iter - n_done} to go", flush=True)
            if args.fit_save_interval > 0:
                ckpt.mkdir(parents=True, exist_ok=True)
                (ckpt / FIT_IDENTITY_FILE).write_text(json.dumps(identity, indent=1))
                pc_est.setSaveDir(str(ckpt))
                pc_est.setSaveInterval(int(args.fit_save_interval))
            if action == "done":
                # The checkpoint already holds the whole budget (the driver died
                # after the last checkpoint but before the save below). Its node set
                # rides the VIResult; the Params are the estimator's.
                from spark_vi.io.export import load_result
                from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
                pc_model = pc_est._copyValues(DagGatedLDAModel(load_result(ckpt)))
                _cprint(f"[driver]   fit checkpoint {ckpt} is already at "
                        f"{n_done}/{max_iter} iterations; not refitting", flush=True)
            else:
                pc_model = pc_est.fit(bundle.train_df)
                if action == "resume":
                    pc_model.setMaxIter(max_iter)
                # Park the FINISHED fit as a resumable checkpoint before any post-fit
                # write: if one of them fails (a save, a disk, a preempted master), a
                # re-run of the same command finds the checkpoint at its full budget
                # ("done" above) and does not refit. Removed once every save succeeded.
                from spark_vi.io.export import save_result
                save_result(pc_model.result, str(ckpt))
                (ckpt / FIT_IDENTITY_FILE).write_text(json.dumps(identity, indent=1))
                _cprint(f"[driver]   parked the finished fit in {ckpt} (resumable "
                        "until the saves below succeed)", flush=True)

            lay = pc_model._layout()
            br = node_bridge(pc_model, bundle.name_by_id)
            C = len(br["int2cid"])           # readout heads = kept nodes + root
            counts = node_set_counts(pc_model, len(bundle.dag))
            _cprint(f"[driver]   node set: {counts['n_given']} given / "
                    f"{counts['n_kept']} kept / {counts['n_pruned']} pruned / "
                    f"{counts['n_spliced']} spliced (minDocsPerNode="
                    f"{counts['min_docs_per_node']}, counted on "
                    f"{counts['n_train_docs']} train docs); K={lay.K} ({n_bg} bg + "
                    f"{len(lay.nodes)} nodes x {tpn}), C={C} heads", flush=True)
            _v_total = sum(len(vm) for vm in vocab_maps)
            _, _prof = lay.cost_report(vocab_size=_v_total)
            print("\n".join("[cost] " + ln for ln in _prof.splitlines()), flush=True)

            # The fitted model is the run's node set of record: everything after the
            # fit (readout, framings, inspect-topics) reads it from here.
            muri = save_fitted_model(pc_model, out)
            _cprint(f"[driver]   saved the fitted model -> {muri}", flush=True)
            write_label_dag(out, bundle.dag, pc_model, labels_col=args._labels_col)
            write_bundle_meta(out, bundle, pc_model)
            if code_map:
                # WP-C': <run>/code_map.tsv, the code ladder landed on the FITTED
                # node set (std concept id -> kept node cid): the census's own codes.
                _lcm = landed_code_map(pc_model, code_map, br["int2cid"])
                write_code_map(out, _lcm)
                _cprint(f"[driver]   wrote the code map on the fitted node set: "
                        f"{len(_lcm)} (code, node) pairs over "
                        f"{len({n for _, n in _lcm})} nodes", flush=True)

            # Every manifest field the FIT determines.
            manifest_fields = {
                "model_class": "gated_pc",
                "model_path": MODEL_DIR,
                "disease": args.disease,
                "strip_mode": args.strip_mode, "label_mask_mode": args.label_mask_mode,
                "window_mode": args.window_mode, "lookback_days": args.lookback_days,
                "label_window_days": args.label_window_days,
                "n_bg": n_bg, "tpn": tpn, "K": lay.K, "C": C,
                "subsampling_rate": float(g("subsamplingRate")),
                "tau0": float(g("learningOffset")),
                "kappa": float(g("learningDecay")), "max_iter": max_iter,
                "estimator_params": {
                    p.name: g(p) for p in pc_est.params
                    if p.name not in ("dagEdges", "vocabularies", "nodeTerms",
                                      "gateSeedProfile")
                    and (pc_est.isSet(p) or pc_est.hasDefault(p))},
                "labels_col": args._labels_col,
                "node_set": counts,
                "dag_source": args.dag_source,
                "extra_domains": list(extra_domains),
                "domain_names": args._domain_names,
                "domain_vocab_sizes": [len(vm) for vm in vocab_maps],
                "ledger": bundle.ledger,
                # The corpus SPEC verbatim (every bundle cache-key input and every
                # rebuild input), so a post-hoc dag_gated_readout recomputes the exact
                # key and can REBUILD the bundle when the cache is cold; plus the
                # fitted node set's bridge (int2cid / name_by_id), which the readers
                # of a run dir key their per-node tables by.
                "corpus_manifest": {
                    **corpus_spec, "cache_uri": args.cache_uri,
                    "domain_vocab_sizes": [len(vm) for vm in vocab_maps],
                    "int2cid": {str(e): c for e, c in br["int2cid"].items()},
                    "name_by_id": {str(c): n for c, n in br["name_by_id"].items()}},
            }
            if int(g("gateWarm")) > 0:
                manifest_fields["gate_schedule"] = {
                    "warm": int(g("gateWarm")), "open_seed": bool(g("gateOpenSeed")),
                    "seed_mode": str(g("gateSeedMode")),
                    "seed_leaves": bool(g("gateSeedLeaves"))}
                if args._gate_seed_profile:
                    manifest_fields["gate_schedule"]["seed_profile"] = {
                        "path": args.gate_seed_profile,
                        **args._gate_seed_profile_stats}
            _save_fit(out, pc_model.result.global_params, C, manifest_fields,
                      partial="fit-only")
            _cprint(f"[driver]   saved FIT-ONLY result to {out} (readout pending; "
                    "re-scoreable with dag_gated_readout)", flush=True)
            if ckpt.exists():
                # Every post-fit save succeeded: the finished fit is durable in
                # <run>/model and the npz, so the resume checkpoint has done its job
                # (and is another lambda-sized copy on disk).
                import shutil
                shutil.rmtree(ckpt, ignore_errors=True)
            # The fit ends here (cleanup 2026-10): every readout and evaluation is
            # `dag_gated_readout.py` against this saved fit. Per-node per-domain λ
            # mass needs no readout, so it is still reported with the fit.
            gp = pc_model.result.global_params
            if extra_domains and isinstance(gp["lambda"], dict):
                print(format_per_node_domain_mass(
                    per_node_domain_mass(gp["lambda"], lay, args._domain_names),
                    args._domain_names, br["int2cid"], br["name_by_id"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
