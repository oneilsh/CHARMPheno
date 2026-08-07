"""Reusable, id-agnostic evaluation harness: faithful PC vs. the Hughes baselines.

This is the eval/baseline layer that Phase C's All-of-Us OMOP driver calls. It
takes plain numpy arrays in and returns a structured metrics dict out — it knows
nothing about person ids, OMOP concepts, or clinical meaning. Unlike the model
*core* (numpy/scipy/autograd only), this layer is allowed ``scikit-learn`` for
the baselines and metrics: it is measurement, not the constrained objective.

The comparison set is the one from Hughes, Hope, Weiner, McCoy, Perlis, Sudderth
& Doshi-Velez 2017/2018 — for each of ``C`` binary labels/outcomes we fit three
models and score heldout per-label ``P(y = 1)`` on a held-out split:

  1. **PC** — the faithful :class:`~analysis.pc.model.PCTopicModel` with
     ``weight_y > 0``. Topics are reshaped by the label; ``predict_proba`` is the
     model's own logistic head on the label-free MAP ``pi``.
  2. **Two-stage (unsupervised -> LR)** — the SAME class with ``weight_y = 0``
     (an unsupervised LDA-MAP fit), then a plain
     :class:`~sklearn.linear_model.LogisticRegression` per label on the frozen
     train ``Pi`` -> ``y``, scored on the test ``Pi``. This is the representation
     that ignores the (often rare) label direction — the baseline PC should beat.
  3. **LR-on-codes** — a :class:`~sklearn.linear_model.LogisticRegression` per
     label straight on the raw count matrix ``X`` (no topic bottleneck).

Metrics per label are heldout **ROC AUC** and **average precision (AP)**, plus a
**macro-average** across the non-degenerate labels. A test-set label column that
is constant (all-0 or all-1) makes AUC undefined; such labels are *skipped* with
a recorded reason and left out of the macro-average rather than crashing the run.

Semi-supervised (``labeled_mask``): PC trains on ALL of ``X_tr`` but only the
masked rows carry their label (the faithful model's ``pi`` is label-free anyway);
the two LR baselines cannot use unlabeled rows, so they train on the labeled rows
only. Both the two-stage's unsupervised topic fit and PC still see every row.

Everything is deterministic given ``seed`` (the model init) — the LR baselines
are convex, so they add no randomness.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.pc.model import PCTopicModel


def _as_y_DC(y: np.ndarray) -> np.ndarray:
    """Coerce labels to a ``(D, C)`` float array; a 1D vector becomes one column."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    return y


def _score_label(
    y_true: np.ndarray, proba: np.ndarray
) -> dict[str, Any]:
    """ROC AUC + AP for one label column, or a skip record if AUC is undefined.

    AUC (and a meaningful AP) require both classes present in ``y_true``. A
    constant heldout column (all-0 => no positives, all-1 => no negatives) is
    reported as ``{"skipped": <reason>, "auc": None, "ap": None}`` so the caller
    can drop it from the macro-average instead of raising.
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
        }
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "ap": float(average_precision_score(y_true, proba)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "skipped": None,
    }


def _macro(per_label: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Macro-average AUC/AP over the labels that were scored (not skipped)."""
    aucs = [d["auc"] for d in per_label.values() if d.get("skipped") is None]
    aps = [d["ap"] for d in per_label.values() if d.get("skipped") is None]
    return {
        "auc": float(np.mean(aucs)) if aucs else None,
        "ap": float(np.mean(aps)) if aps else None,
        "n_labels_scored": len(aucs),
        "n_labels_skipped": len(per_label) - len(aucs),
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
    for c in range(C):
        yc = y_fit_DC[:, c]
        classes = np.unique(yc)
        if classes.size < 2:
            proba[:, c] = float(classes[0]) if classes.size else 0.0
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_fit, yc)
        # Column of P(y=1): the positive class is 1.0 given binary {0,1} labels.
        pos_col = int(np.where(lr.classes_ == 1)[0][0])
        proba[:, c] = lr.predict_proba(X_te)[:, pos_col]
    return proba


def evaluate_pc_vs_baselines(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    *,
    K: int,
    weight_y: float,
    alpha: float = 1.1,
    tau: float = 1.1,
    pi_iters: int = 100,
    max_iter: int = 500,
    seed: int = 0,
    labeled_mask: np.ndarray | None = None,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Fit + heldout-score PC and the two Hughes baselines on ``C`` binary labels.

    Parameters
    ----------
    X_tr, X_te : (D_tr, V), (D_te, V) nonnegative count matrices (dense).
    y_tr, y_te : (D, C) binary label arrays in {0, 1}. A 1D vector is treated as a
        single label (``C == 1``). ``y_te`` supplies the heldout ground truth.
    K : int
        Number of topics for both the PC and unsupervised (two-stage) fits.
    weight_y : float
        PC prediction weight (> 0 for the supervised PC model). The two-stage
        model always refits the SAME class with ``weight_y = 0``.
    alpha, tau, pi_iters, max_iter, seed :
        Passed through to :class:`~analysis.pc.model.PCTopicModel` (the harness
        default ``pi_iters=100`` matches the model; tests may lower it for speed).
    labeled_mask : (D_tr,) or None
        Semi-supervised row mask. ``None`` => all train rows labeled. PC trains on
        every row (label-free ``pi``) but only masked rows contribute their label;
        the LR baselines train on the masked rows only.
    **model_kwargs :
        Extra keyword args forwarded to every ``PCTopicModel`` constructor (e.g.
        ``lambda_w``, ``pi_step_size``) — kept shared between the PC and two-stage
        fits so the only difference is ``weight_y``.

    Returns
    -------
    dict
        ``{"PC": ..., "two_stage": ..., "lr_codes": ...}`` where each model maps to
        ``{"per_label": {c: {"auc", "ap", "n_pos", "n_neg", "skipped"}},
        "macro": {"auc", "ap", "n_labels_scored", "n_labels_skipped"}}``, plus a
        ``"meta"`` block (``C``, ``K``, ``weight_y``, ``n_train``, ``n_test``,
        ``n_labeled``, model display names). Enough to build a per-label
        PC-vs-baselines table and a macro summary.
    """
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    y_tr_DC = _as_y_DC(y_tr)
    y_te_DC = _as_y_DC(y_te)
    C = y_tr_DC.shape[1]
    if y_te_DC.shape[1] != C:
        raise ValueError(
            f"y_tr has C={C} labels but y_te has {y_te_DC.shape[1]}"
        )

    D_tr = X_tr.shape[0]
    if labeled_mask is None:
        lab_idx = np.arange(D_tr)
    else:
        labeled_mask = np.asarray(labeled_mask)
        lab_idx = np.where(labeled_mask.astype(bool))[0]

    shared = dict(
        K=K, C=C, alpha=alpha, tau=tau, pi_iters=pi_iters,
        max_iter=max_iter, seed=seed, **model_kwargs,
    )

    # --- Model 1: faithful PC (supervised, label reshapes the topics) ---------
    pc = PCTopicModel(weight_y=weight_y, **shared).fit(X_tr, y_tr_DC, labeled_mask)
    pc_proba = pc.predict_proba(X_te)                      # (N_te, C)

    # --- Model 2: two-stage (unsupervised weight_y=0 -> LR per label) ---------
    # SAME class, weight_y=0 == LDA-MAP. Topics see every train row; the LR head
    # is fit only on labeled rows' frozen Pi (unlabeled rows have no target).
    unsup = PCTopicModel(weight_y=0.0, **shared).fit(X_tr, y_tr_DC)
    Pi_tr = unsup.Pi_                                       # (D_tr, K)
    Pi_te = unsup.transform(X_te)                          # (N_te, K)
    ts_proba = _lr_proba_per_label(
        Pi_tr[lab_idx], y_tr_DC[lab_idx], Pi_te, C
    )

    # --- Model 3: LR straight on the raw codes (labeled rows only) ------------
    lrc_proba = _lr_proba_per_label(
        X_tr[lab_idx], y_tr_DC[lab_idx], X_te, C
    )

    def _bundle(proba: np.ndarray) -> dict[str, Any]:
        per_label = {c: _score_label(y_te_DC[:, c], proba[:, c]) for c in range(C)}
        return {"per_label": per_label, "macro": _macro(per_label)}

    return {
        "PC": _bundle(pc_proba),
        "two_stage": _bundle(ts_proba),
        "lr_codes": _bundle(lrc_proba),
        "meta": {
            "C": C,
            "K": int(K),
            "weight_y": float(weight_y),
            "n_train": int(D_tr),
            "n_test": int(X_te.shape[0]),
            "n_labeled": int(lab_idx.size),
            "model_names": {
                "PC": "PC (faithful)",
                "two_stage": "two-stage (unsup+LR)",
                "lr_codes": "LR-on-codes",
            },
        },
    }


def format_results_table(results: dict[str, Any]) -> str:
    """Pretty-print a per-label PC-vs-baselines table (AUC/AP) + macro summary.

    Mirrors the reporting style of ``analysis/pc/tests/test_reference_oracle.py``:
    a compact fixed-width table, one row per (model, label), a per-model macro
    line, and a footer that flags any skipped (degenerate) heldout label columns.
    ``results`` is the dict returned by :func:`evaluate_pc_vs_baselines`.
    """
    meta = results["meta"]
    names = meta["model_names"]
    C = meta["C"]
    order = ["PC", "two_stage", "lr_codes"]

    lines: list[str] = []
    lines.append(
        f"[PC vs baselines] K={meta['K']} weight_y={meta['weight_y']:g} "
        f"C={C} labels  n_train={meta['n_train']} "
        f"(labeled={meta['n_labeled']}) n_test={meta['n_test']}"
    )
    header = f"  {'model':<22} {'label':>6} {'AUC':>8} {'AP':>8}  {'n_pos/n_neg (te)':>18}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    skipped: list[str] = []
    for key in order:
        block = results[key]
        disp = names.get(key, key)
        for c in range(C):
            d = block["per_label"][c]
            if d.get("skipped"):
                auc_s, ap_s = "  --  ", "  --  "
                skipped.append(f"{disp} label {c}: {d['skipped']}")
            else:
                auc_s = f"{d['auc']:.4f}"
                ap_s = f"{d['ap']:.4f}"
            npn = f"{d['n_pos']}/{d['n_neg']}"
            lines.append(
                f"  {disp:<22} {c:>6} {auc_s:>8} {ap_s:>8}  {npn:>18}"
            )
        m = block["macro"]
        macro_auc = "  --  " if m["auc"] is None else f"{m['auc']:.4f}"
        macro_ap = "  --  " if m["ap"] is None else f"{m['ap']:.4f}"
        lines.append(
            f"  {disp:<22} {'MACRO':>6} {macro_auc:>8} {macro_ap:>8}  "
            f"({m['n_labels_scored']} scored, {m['n_labels_skipped']} skipped)"
        )
        lines.append("  " + "-" * (len(header) - 2))

    if skipped:
        lines.append("  skipped (degenerate heldout label columns):")
        for s in skipped:
            lines.append(f"    - {s}")

    return "\n".join(lines)
