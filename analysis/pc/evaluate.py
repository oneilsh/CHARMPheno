"""Per-label metric primitives for the post-hoc readout (id-agnostic, numpy in).

What survives of the in-memory PC evaluation harness (ADR 0038, removed by ADR
0050): the per-label scorer (`_score_label`: heldout ROC AUC, AP and top-k
screening, with the SKIP_* reasons for a degenerate label), the macro-average
(`_macro`), the per-label masked logistic baseline (`_lr_proba_per_label[_masked]`,
the oracle `spark_vi.readout.solver` reproduces) and the masked bundle
scorer (`_bundle_masked`). The readout drivers (`analysis/cloud/gated_pc_cloud.py`,
`gated_pc_readout.py`) import them. This layer is allowed scikit-learn: it is
measurement, not a model.

A test-set label column that is constant (all-0 or all-1) makes AUC undefined;
such labels are skipped with a recorded reason and left out of the macro-average
rather than crashing the run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler



# The three — and only three — reasons a label column is dropped from the macro.
# They are COUNTED SEPARATELY and never summed (incident-eval spec §8, standing
# rule 5): "how many nodes had no negatives", "how many were too small to
# disclose" and "how many carry a constant prediction column" are three different
# facts about a run, and a single `n_labels_skipped` hides all three.
SKIP_DEGENERATE = "degenerate_test_column"
SKIP_SMALL = "small_test_column"
SKIP_CONSTANT = "constant_prediction_column"
SKIP_CODES = (SKIP_DEGENERATE, SKIP_SMALL, SKIP_CONSTANT)


def _topk_screening_metrics(
    y_true: np.ndarray, proba: np.ndarray, n_pos: int, n_neg: int,
    topk_frac: float,
) -> dict[str, Any]:
    """Top-fraction screening metrics — the rare-disease-honest complement to AUC.

    Rank the scored rows by predicted score and take the top ``topk_frac`` (e.g.
    0.01 = top 1%). Then:

      * ``prec_at_k``  — of that top slice, the fraction who are true positives;
      * ``recall_at_k``— of ALL positives, the fraction captured in the slice;
      * ``lift_at_k``  — ``prec_at_k`` / prevalence, i.e. how many times better
        than the do-nothing baseline of flagging a random ``topk_frac`` (which,
        like predicting the majority class, catches positives only at the base
        rate). lift 1.0 == no better than chance; lift >> 1 == real enrichment.

    AUC integrates over all thresholds and can look respectable while the actionable
    top slice is barely above prevalence — for a screening / value-of-information
    use only the top slice is ever acted on, so this is the operating point that
    matters. ``k = ceil(topk_frac * n)`` (>=1), ties broken by a stable argsort.
    Returns None fields where undefined (no rows, no positives). Never raises — a
    computation failure degrades to None so the AUC path is never at risk.
    """
    out = {"topk_frac": float(topk_frac),
           "prec_at_k": None, "recall_at_k": None, "lift_at_k": None}
    try:
        n = int(n_pos) + int(n_neg)
        if n == 0 or n_pos == 0:
            return out
        k = min(n, max(1, int(np.ceil(float(topk_frac) * n))))
        order = np.argsort(-proba, kind="stable")[:k]
        tp = float((y_true[order] == 1).sum())
        prevalence = float(n_pos) / float(n)
        prec = tp / float(k)
        out["prec_at_k"] = prec
        out["recall_at_k"] = tp / float(n_pos)
        out["lift_at_k"] = (prec / prevalence) if prevalence > 0 else None
    except Exception:
        return {"topk_frac": float(topk_frac),
                "prec_at_k": None, "recall_at_k": None, "lift_at_k": None}
    return out


def _score_label(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_count: int = 0,
    *,
    skip_constant: bool = False,
    topk_frac: float = 0.01,
) -> dict[str, Any]:
    """ROC AUC + AP for one label column, or a skip record if it is unscoreable.

    AUC (and a meaningful AP) require both classes present in ``y_true``. A
    constant heldout column (all-0 => no positives, all-1 => no negatives) is
    reported as ``{"skipped": <reason>, "auc": None, "ap": None}`` so the caller
    can drop it from the macro-average instead of raising.

    ``min_count`` (default 0 = off) additionally masks *small* heldout columns:
    when either class has fewer than ``min_count`` test cells, the AUC is too
    noisy to trust (and, on All-of-Us, a count < 20 is not disclosable), so the
    column is reported skipped and dropped from the macro — exactly as a
    degenerate column is. The ``n_pos``/``n_neg`` are still recorded in the
    result (callers suppress them at display time via ``format_results_table``'s
    ``min_label_count`` handling); scoring simply refuses to compute an AUC.

    ``skip_constant`` (incident-eval spec R2.1) adds the THIRD skip reason: a
    column whose predictions are CONSTANT over the scored rows
    (``np.ptp(proba) == 0``) carries no per-document information by construction,
    and ``roc_auc_score`` returns exactly 0.5 for it — a number that looks like a
    measurement and is not one. This is the ranking-axis twin of the guard
    ``detection_readout`` has applied to its pooled max since exp 0104, and the
    incident cohort is where it becomes load-bearing: a node whose TRAIN cell was
    degenerate gets a constant fallback column, is all-positive under the
    prevalent mask (hence already skipped as degenerate), but ACQUIRES negatives
    under the incident mask when the prior carriers that made it all-positive
    leave both classes. Without this guard those columns enter the macro at 0.5.

    It defaults OFF because turning it on unconditionally would move every
    recorded prevalent macro in the repo (exps 0104/0109/0110 are compared on
    them); the incident arm passes ``skip_constant=True`` and reports the count
    separately. Every skip record also carries a machine-readable ``skip_code``
    from :data:`SKIP_CODES` beside the human sentence, which is what lets a caller
    report three counts instead of one.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    proba = np.asarray(proba, dtype=np.float64).ravel()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        which = "all-positive" if n_neg == 0 else "all-negative"
        return {
            "auc": None,
            "ap": None,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "skipped": f"degenerate test column ({which}); AUC undefined",
            "skip_code": SKIP_DEGENERATE,
        }
    if min_count > 0 and (n_pos < min_count or n_neg < min_count):
        return {
            "auc": None,
            "ap": None,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "skipped": (
                f"small test column (min class < {min_count}); "
                "AUC unreliable / count not disclosable — masked"
            ),
            "skip_code": SKIP_SMALL,
        }
    # Checked AFTER the two count-based skips so the reasons stay disjoint and a
    # single column is attributed to exactly one of them.
    if skip_constant and proba.size and float(np.ptp(proba)) == 0.0:
        return {
            "auc": None,
            "ap": None,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "skipped": (
                "constant prediction column (no per-document information; "
                "roc_auc_score would return exactly 0.5)"
            ),
            "skip_code": SKIP_CONSTANT,
        }
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "ap": float(average_precision_score(y_true, proba)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "skipped": None,
        "skip_code": None,
        # Top-fraction screening metrics (additive; a scored column always carries
        # them). Skip records omit them and readers use .get(...) -> None.
        **_topk_screening_metrics(y_true, proba, n_pos, n_neg, topk_frac),
    }


def _macro(per_label: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Macro-average AUC/AP over the labels that were scored (not skipped).

    ``skipped_by_reason`` breaks the skip total down by :data:`SKIP_CODES`. It is
    additive (the existing keys are untouched, so every recorded number and every
    reader of them is unaffected) and it exists because a lumped
    ``n_labels_skipped`` cannot answer the question R2.1 raises: how many columns
    did the constant guard catch?"""
    scored = [d for d in per_label.values() if d.get("skipped") is None]
    aucs = [d["auc"] for d in scored]
    aps = [d["ap"] for d in scored]
    by_reason = {code: 0 for code in SKIP_CODES}
    for d in per_label.values():
        code = d.get("skip_code")
        if code in by_reason:
            by_reason[code] += 1
    # Macro-average the top-fraction screening metrics over the same scored labels
    # (a label with lift_at_k None — e.g. zero prevalence — drops out, like the
    # AUC skips). topk_frac is uniform across labels; read it off any scorer.
    lifts = [d["lift_at_k"] for d in scored if d.get("lift_at_k") is not None]
    precs = [d["prec_at_k"] for d in scored if d.get("prec_at_k") is not None]
    recs = [d["recall_at_k"] for d in scored if d.get("recall_at_k") is not None]
    topk_frac = next((d.get("topk_frac") for d in scored
                      if d.get("topk_frac") is not None), None)
    return {
        "auc": float(np.mean(aucs)) if aucs else None,
        "ap": float(np.mean(aps)) if aps else None,
        "n_labels_scored": len(aucs),
        "n_labels_skipped": len(per_label) - len(aucs),
        "skipped_by_reason": by_reason,
        "topk_frac": topk_frac,
        "lift_at_k": float(np.mean(lifts)) if lifts else None,
        "prec_at_k": float(np.mean(precs)) if precs else None,
        "recall_at_k": float(np.mean(recs)) if recs else None,
    }


def _lr_proba_per_label(
    X_fit: np.ndarray,
    y_fit_DC: np.ndarray,
    X_te: np.ndarray,
    C: int,
) -> np.ndarray:
    """Fit one :class:`LogisticRegression` per label and return ``(N_te, C)`` probs.

    ``X_fit`` / ``y_fit_DC`` are already restricted to the usable (labeled) rows.
    A training column with a single class cannot fit a logistic regression, so we
    fall back to a constant prediction equal to that lone class value (yielding a
    chance-level heldout AUC rather than an exception).
    """
    N_te = X_te.shape[0]
    proba = np.zeros((N_te, C), dtype=np.float64)
    # Standardize on the training features (leak-free: fit on X_fit only) so
    # L-BFGS converges — unscaled high-dimensional counts otherwise hit the
    # iteration limit, leaving an under-fit, unfairly weak baseline. X_fit is
    # shared across labels here, so one scaler suffices.
    scaler = StandardScaler().fit(X_fit)
    X_fit_s, X_te_s = scaler.transform(X_fit), scaler.transform(X_te)
    for c in range(C):
        yc = y_fit_DC[:, c]
        classes = np.unique(yc)
        if classes.size < 2:
            proba[:, c] = float(classes[0]) if classes.size else 0.0
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_fit_s, yc)
        # Column of P(y=1): the positive class is 1.0 given binary {0,1} labels.
        pos_col = int(np.where(lr.classes_ == 1)[0][0])
        proba[:, c] = lr.predict_proba(X_te_s)[:, pos_col]
    return proba


def _lr_proba_per_label_masked(
    X_fit_full: np.ndarray,
    y_DC: np.ndarray,
    obs_DC: np.ndarray,
    X_te: np.ndarray,
    C: int,
    feature_cols=None,
    fit_intercept: bool = True,
    standardize: bool = True,
) -> np.ndarray:
    """Per-label :class:`LogisticRegression` trained on each label's OWN observed
    train rows, returning ``(N_te, C)`` probabilities.

    The multi-task counterpart of :func:`_lr_proba_per_label`: instead of one
    shared set of labeled rows, column ``c`` is fit only on the rows where
    ``obs_DC[:, c]`` is True (the cells observed for outcome ``c``). Same
    single-class fallback as :func:`_lr_proba_per_label` (a constant prediction
    equal to the lone class), so an all-one-class or empty observed set yields a
    chance-level heldout AUC rather than an exception.

    ``feature_cols`` (``None`` = all K features): a length-``C`` list where
    ``feature_cols[c]`` is the array of FEATURE columns node ``c`` may read. This
    is the "oracle localized readout" — fitting the best-possible per-node logistic
    on EXACTLY the co-fit head's topic support (``allowed_with_siblings``) — the
    A-vs-B diagnostic: if it matches the full-K readout the co-fit head is merely
    under-fit on its support (recoverable); if it collapses to the co-fit head's
    level the discriminative signal is genuinely outside the local support
    (localization is fundamentally lossy). An empty column set falls back to the
    class-prior constant (chance).
    """
    N_te = X_te.shape[0]
    proba = np.zeros((N_te, C), dtype=np.float64)
    for c in range(C):
        rows = np.where(obs_DC[:, c].astype(bool))[0]
        yc = y_DC[rows, c]
        classes = np.unique(yc)
        cols = None if feature_cols is None else np.asarray(feature_cols[c], dtype=int)
        if classes.size < 2 or (cols is not None and cols.size == 0):
            proba[:, c] = float(classes[0]) if classes.size else 0.0
            continue
        Xf = X_fit_full[rows] if cols is None else X_fit_full[np.ix_(rows, cols)]
        Xt = X_te if cols is None else X_te[:, cols]
        # Standardize on this label's observed TRAIN rows (leak-free), apply to
        # test, so L-BFGS converges on high-dimensional raw counts — unscaled
        # 5000-dim counts otherwise hit the iteration limit (ConvergenceWarning),
        # leaving an under-fit, unfairly weak baseline that flatters PC. The
        # `standardize` / `fit_intercept` toggles exist for the head-formulation
        # LADDER (isolating which sklearn-vs-engine-head difference costs the co-fit
        # head its AUC); both default True = the standard readout.
        if standardize:
            scaler = StandardScaler().fit(Xf)
            Xf, Xt = scaler.transform(Xf), scaler.transform(Xt)
        lr = LogisticRegression(max_iter=1000, fit_intercept=fit_intercept)
        lr.fit(Xf, yc)
        pos_col = int(np.where(lr.classes_ == 1)[0][0])
        proba[:, c] = lr.predict_proba(Xt)[:, pos_col]
    return proba


def _bundle_masked(
    proba_DC: np.ndarray,
    y_te_DC: np.ndarray,
    mask_te_DC: np.ndarray,
    C: int,
    min_count: int = 0,
    *,
    skip_constant: bool = False,
) -> dict[str, Any]:
    """Score each label column ONLY over its observed test cells, then macro it.

    Reuses :func:`_score_label` (degenerate- and, when ``min_count > 0``,
    small-column skips included) and :func:`_macro`. Column ``c`` is scored over
    the rows where ``mask_te_DC[:, c]`` is True; a column with no observed test
    cell, a single class among them, or (for ``min_count > 0``) fewer than
    ``min_count`` cells of either class is reported skipped and dropped from the
    macro-average.

    ``skip_constant`` forwards R2.1's constant-column guard to :func:`_score_label`
    — the guard is evaluated on the column RESTRICTED TO ITS SCORED ROWS, which is
    the axis that matters: a column can be constant on one node's cohort and
    informative on another's.
    """
    per_label: dict[int, dict[str, Any]] = {}
    for c in range(C):
        rows = np.where(mask_te_DC[:, c].astype(bool))[0]
        per_label[c] = _score_label(
            y_te_DC[rows, c], proba_DC[rows, c], min_count=min_count,
            skip_constant=skip_constant,
        )
    return {"per_label": per_label, "macro": _macro(per_label)}

