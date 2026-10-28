"""Per-label metric primitives for the post-hoc readout (id-agnostic, numpy in).

What survives of the in-memory PC evaluation harness (ADR 0038, removed by ADR
0050): the per-label scorer (`_score_label`), the macro (`_macro`) and the masked
bundle scorer (`_bundle_masked`), now thin wrappers over `spark_vi.eval.metrics`
(ADR 0051 step 4: one exact definition shared with the Spark readout), and the
per-label masked logistic baseline (`_lr_proba_per_label[_masked]`, the oracle
`spark_vi.readout.solver` reproduces). This layer is allowed scikit-learn for the
baselines: it is measurement, not a model.

A test-set label column that is constant (all-0 or all-1) makes AUC undefined;
such labels are skipped with a recorded reason and left out of the macro-average
rather than crashing the run.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from spark_vi.eval.metrics import core as _mcore
from spark_vi.eval.metrics import report as _mreport

# The three — and only three — reasons a label column is dropped from the macro
# (`spark_vi.eval.metrics.report`); counted separately, never summed.
SKIP_DEGENERATE = _mreport.SKIP_DEGENERATE
SKIP_SMALL = _mreport.SKIP_SMALL
SKIP_CONSTANT = _mreport.SKIP_CONSTANT
SKIP_CODES = _mreport.SKIP_CODES


def _score_label(
    y_true: np.ndarray,
    proba: np.ndarray,
    min_count: int = 0,
    *,
    skip_constant: bool = False,
    topk_frac: float = 0.01,
) -> dict[str, Any]:
    """ROC AUC, AP and top-k screening for one label column, or a skip record
    (`spark_vi.eval.metrics.report.score_record`): degenerate (one class), small
    (either class under ``min_count``; not disclosable on All of Us under 20) or,
    with ``skip_constant``, a constant prediction (AUC exactly 0.5 by
    construction). Top-k: k = ceil(topk_frac n), lift = precision@k / prevalence,
    a tie at the cut counted at its mean positive rate."""
    m = _mcore.node_metrics(_mcore.tally(proba, y_true), topk_frac=topk_frac)
    return _mreport.score_record(m, min_count=min_count, skip_constant=skip_constant)


def _macro(per_label: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Macro AUC / AP / top-k over the scored labels, with the skip counts by reason
    (`spark_vi.eval.metrics.report.macro`)."""
    return _mreport.macro(per_label)


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

