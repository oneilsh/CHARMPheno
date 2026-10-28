"""From per-key metrics to the readout's per-label records and macro averages: the
skip rules and the macro (moved from `analysis/pc/evaluate.py` with ADR 0051 step 4;
the metrics themselves now come from `core`, not scikit-learn).

A column is dropped from the macro for exactly one of three reasons, counted
separately and never summed (incident-eval spec §8): no positives or no negatives
(`SKIP_DEGENERATE`), either class under `min_count` (`SKIP_SMALL`; on All of Us a
count under 20 is not disclosable), or, when asked, a constant prediction over the
scored rows (`SKIP_CONSTANT`: its AUC is exactly 0.5 by construction, a number that
looks like a measurement and is not one).
"""
from __future__ import annotations

SKIP_DEGENERATE = "degenerate_test_column"
SKIP_SMALL = "small_test_column"
SKIP_CONSTANT = "constant_prediction_column"
SKIP_CODES = (SKIP_DEGENERATE, SKIP_SMALL, SKIP_CONSTANT)

_TOPK = ("topk_frac", "prec_at_k", "recall_at_k", "lift_at_k")


def score_record(m, *, min_count=0, skip_constant=False):
    """The ranking record of one column from its `core.node_metrics` dict:
    `{auc, ap, n_pos, n_neg, skipped, skip_code}` plus the top-k fields when scored.
    The checks run in this order so a column has one reason."""
    n_pos, n_neg = int(m["n_pos"]), int(m["n_neg"])
    base = {"auc": None, "ap": None, "n_pos": n_pos, "n_neg": n_neg}
    if n_pos == 0 or n_neg == 0:
        which = "all-positive" if n_neg == 0 else "all-negative"
        return {**base, "skipped": f"degenerate test column ({which}); AUC undefined",
                "skip_code": SKIP_DEGENERATE}
    if min_count > 0 and (n_pos < min_count or n_neg < min_count):
        return {**base, "skipped": (f"small test column (min class < {min_count}); "
                                    "AUC unreliable / count not disclosable — masked"),
                "skip_code": SKIP_SMALL}
    if skip_constant and int(m["n_distinct"]) == 1:
        return {**base, "skipped": ("constant prediction column (no per-document "
                                    "information; its AUC would be exactly 0.5)"),
                "skip_code": SKIP_CONSTANT}
    return {"auc": m["auc"], "ap": m["ap"], "n_pos": n_pos, "n_neg": n_neg,
            "skipped": None, "skip_code": None, **{k: m.get(k) for k in _TOPK}}


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]          # drop None / NaN
    return sum(vals) / len(vals) if vals else None


def macro(per_label):
    """Macro AUC / AP / top-k over the scored records, with the skip counts by
    reason (`skipped_by_reason`) beside the lumped total."""
    scored = [d for d in per_label.values() if d.get("skipped") is None]
    by_reason = {code: 0 for code in SKIP_CODES}
    for d in per_label.values():
        if d.get("skip_code") in by_reason:
            by_reason[d["skip_code"]] += 1
    return {
        "auc": _mean([d["auc"] for d in scored]),
        "ap": _mean([d["ap"] for d in scored]),
        "n_labels_scored": len(scored),
        "n_labels_skipped": len(per_label) - len(scored),
        "skipped_by_reason": by_reason,
        "topk_frac": next((d.get("topk_frac") for d in scored
                           if d.get("topk_frac") is not None), None),
        "lift_at_k": _mean([d.get("lift_at_k") for d in scored]),
        "prec_at_k": _mean([d.get("prec_at_k") for d in scored]),
        "recall_at_k": _mean([d.get("recall_at_k") for d in scored]),
    }


def pr_record(m, *, min_count=0):
    """The precision@recall / recall@FDR record of one column: skipped below
    `max(min_count, 1)` of either class (the constant guard does not apply here)."""
    n_pos, n_neg = int(m["n_pos"]), int(m["n_neg"])
    floor = max(int(min_count), 1)
    if n_pos < floor or n_neg < floor:
        return {"skipped": True, "n_pos": n_pos, "n_neg": n_neg}
    return {"skipped": False, "n_pos": n_pos, "n_neg": n_neg,
            "par": dict(m["par"]), "raf": dict(m["raf"])}


def pr_macro(per_node, recall_targets, fdr_targets):
    """Macro precision@recall and recall@FDR over the scored PR records."""
    scored = [d for d in per_node.values() if not d["skipped"]]
    return {"n_scored": len(scored),
            "par": {float(t): _mean([d["par"][float(t)] for d in scored])
                    for t in recall_targets},
            "raf": {float(q): _mean([d["raf"][float(q)] for d in scored])
                    for q in fdr_targets}}
