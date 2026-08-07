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

Multi-task / index-drug (:func:`evaluate_pc_multitask`): the same three-model
comparison under a per-CELL ``(D, C)`` observed-mask, where each document is
labeled for only some of the ``C`` outcomes (the Hughes index-drug pattern is
exactly one observed cell per row). ONE shared PC is fit across all heads with
``label_mask``; each outcome is scored only over its observed test cells, and the
baselines fit each outcome's logistic head on that outcome's own observed rows —
so all three models see the identical supervision. Any per-cell mask is accepted.

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


def _lr_proba_per_label_masked(
    X_fit_full: np.ndarray,
    y_DC: np.ndarray,
    obs_DC: np.ndarray,
    X_te: np.ndarray,
    C: int,
) -> np.ndarray:
    """Per-label :class:`LogisticRegression` trained on each label's OWN observed
    train rows, returning ``(N_te, C)`` probabilities.

    The multi-task counterpart of :func:`_lr_proba_per_label`: instead of one
    shared set of labeled rows, column ``c`` is fit only on the rows where
    ``obs_DC[:, c]`` is True (the cells observed for outcome ``c``). Same
    single-class fallback as :func:`_lr_proba_per_label` (a constant prediction
    equal to the lone class), so an all-one-class or empty observed set yields a
    chance-level heldout AUC rather than an exception.
    """
    N_te = X_te.shape[0]
    proba = np.zeros((N_te, C), dtype=np.float64)
    for c in range(C):
        rows = np.where(obs_DC[:, c].astype(bool))[0]
        yc = y_DC[rows, c]
        classes = np.unique(yc)
        if classes.size < 2:
            proba[:, c] = float(classes[0]) if classes.size else 0.0
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_fit_full[rows], yc)
        pos_col = int(np.where(lr.classes_ == 1)[0][0])
        proba[:, c] = lr.predict_proba(X_te)[:, pos_col]
    return proba


def _bundle_masked(
    proba_DC: np.ndarray,
    y_te_DC: np.ndarray,
    mask_te_DC: np.ndarray,
    C: int,
) -> dict[str, Any]:
    """Score each label column ONLY over its observed test cells, then macro it.

    Reuses :func:`_score_label` (degenerate-column skip included) and
    :func:`_macro`. Column ``c`` is scored over the rows where ``mask_te_DC[:, c]``
    is True; a column with no observed test cell, or with a single class among
    them, is reported as skipped and dropped from the macro-average.
    """
    per_label: dict[int, dict[str, Any]] = {}
    for c in range(C):
        rows = np.where(mask_te_DC[:, c].astype(bool))[0]
        per_label[c] = _score_label(y_te_DC[rows, c], proba_DC[rows, c])
    return {"per_label": per_label, "macro": _macro(per_label)}


def evaluate_pc_multitask(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    mask_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    mask_te: np.ndarray,
    *,
    K: int,
    weight_y: float,
    alpha: float = 1.1,
    tau: float = 1.1,
    pi_iters: int = 100,
    max_iter: int = 500,
    doc_batch_size: int = 2048,
    seed: int = 0,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Joint multi-task PC vs. the Hughes baselines under per-cell missing labels.

    The multi-task / **index-drug** counterpart of :func:`evaluate_pc_vs_baselines`.
    A single shared :class:`~analysis.pc.model.PCTopicModel` carries ``C`` outcome
    heads while each document is labeled for only *some* of the ``C`` outcomes: the
    ``(D, C)`` masks ``mask_tr`` / ``mask_te`` mark the observed cells. The primary
    use is the index-drug pattern — exactly ONE observed cell per row (a patient
    labeled only for the drug they initiated) — but any per-cell mask is accepted.

    Three models, each scored per outcome ONLY over that outcome's observed test
    cells (``mask_te[:, c]``):

      1. **PC** — ONE shared PC fit on ``(X_tr, y_tr, label_mask=mask_tr)`` with
         ``weight_y > 0``. Every head is trained off the same label-free ``pi``
         representation, each on its own observed cells; ``predict_proba`` gives
         all ``C`` heads for every test doc.
      2. **Two-stage** — the SAME class with ``weight_y = 0`` (unsupervised
         LDA-MAP; sees every train row's words), then one
         :class:`~sklearn.linear_model.LogisticRegression` per outcome on the
         frozen train ``Pi`` restricted to that outcome's observed train rows.
      3. **LR-on-codes** — one logistic regression per outcome on the raw counts
         ``X``, again restricted to that outcome's observed train rows.

    Both baselines fit each column only where it is observed, so all three models
    see the *same* supervision. Metrics per outcome are heldout ROC AUC and AP; a
    degenerate observed test column (no positives, no negatives, or empty) is
    skipped and dropped from the macro-average via the shared
    :func:`_score_label`/:func:`_macro` helpers.

    Parameters
    ----------
    X_tr, X_te : (D_tr, V), (D_te, V) nonnegative count matrices (dense).
    y_tr, y_te : (D, C) binary labels in {0, 1}. Values at unobserved cells are
        ignored, so any placeholder is fine there; ``y_te`` supplies the heldout
        ground truth over the observed test cells.
    mask_tr, mask_te : (D, C) observed-cell masks (True/1 == observed). Coerced to
        ``(D, C)`` — a 1D ``(D,)`` mask is promoted for the single-outcome case.
    K : int
        Number of topics for both the PC and unsupervised (two-stage) fits.
    weight_y : float
        PC prediction weight (> 0). The two-stage model refits the SAME class with
        ``weight_y = 0``.
    alpha, tau, pi_iters, max_iter, seed :
        Passed through to :class:`~analysis.pc.model.PCTopicModel`.
    doc_batch_size : int
        Document-minibatch size for the PC / two-stage fits' full-batch gradient
        assembly (see :class:`~analysis.pc.model.PCTopicModel`). Bounds driver
        memory at real-corpus scale; the objective/optimizer are unchanged.
    **model_kwargs :
        Extra constructor kwargs shared by the PC and two-stage fits.

    Returns
    -------
    dict
        Same shape as :func:`evaluate_pc_vs_baselines`
        (``{"PC", "two_stage", "lr_codes", "meta"}``, each model bundle carrying
        ``per_label`` + ``macro``), so :func:`format_results_table` renders it. The
        ``"meta"`` block additionally carries per-column observed counts
        ``n_obs_train`` / ``n_obs_test`` (length-``C`` lists) and ``n_obs_train``'s
        total in ``n_labeled``.
    """
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    y_tr_DC = _as_y_DC(y_tr)
    y_te_DC = _as_y_DC(y_te)
    C = y_tr_DC.shape[1]
    if y_te_DC.shape[1] != C:
        raise ValueError(f"y_tr has C={C} labels but y_te has {y_te_DC.shape[1]}")

    def _as_mask(m, D):
        m = np.asarray(m, dtype=np.float64)
        if m.ndim == 1:
            m = m[:, None]
        if m.shape != (D, C):
            raise ValueError(f"mask has shape {m.shape}, expected (D, C)=({D}, {C})")
        return m

    mask_tr_DC = _as_mask(mask_tr, X_tr.shape[0])
    mask_te_DC = _as_mask(mask_te, X_te.shape[0])

    shared = dict(
        K=K, C=C, alpha=alpha, tau=tau, pi_iters=pi_iters,
        max_iter=max_iter, doc_batch_size=doc_batch_size, seed=seed, **model_kwargs,
    )

    # --- Model 1: ONE shared faithful PC over all C heads (per-cell mask) ------
    pc = PCTopicModel(weight_y=weight_y, **shared).fit(
        X_tr, y_tr_DC, label_mask=mask_tr_DC
    )
    pc_proba = pc.predict_proba(X_te)                      # (N_te, C)

    # --- Model 2: two-stage (unsupervised weight_y=0 -> per-column masked LR) --
    unsup = PCTopicModel(weight_y=0.0, **shared).fit(X_tr, y_tr_DC)
    Pi_tr = unsup.Pi_                                       # (D_tr, K)
    Pi_te = unsup.transform(X_te)                          # (N_te, K)
    ts_proba = _lr_proba_per_label_masked(Pi_tr, y_tr_DC, mask_tr_DC, Pi_te, C)

    # --- Model 3: LR straight on the raw codes (per-column masked) ------------
    lrc_proba = _lr_proba_per_label_masked(X_tr, y_tr_DC, mask_tr_DC, X_te, C)

    n_obs_train = [int(mask_tr_DC[:, c].sum()) for c in range(C)]
    n_obs_test = [int(mask_te_DC[:, c].sum()) for c in range(C)]

    return {
        "PC": _bundle_masked(pc_proba, y_te_DC, mask_te_DC, C),
        "two_stage": _bundle_masked(ts_proba, y_te_DC, mask_te_DC, C),
        "lr_codes": _bundle_masked(lrc_proba, y_te_DC, mask_te_DC, C),
        "meta": {
            "C": C,
            "K": int(K),
            "weight_y": float(weight_y),
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "n_labeled": int(sum(n_obs_train)),
            "n_obs_train": n_obs_train,
            "n_obs_test": n_obs_test,
            "model_names": {
                "PC": "PC (faithful, joint)",
                "two_stage": "two-stage (unsup+LR)",
                "lr_codes": "LR-on-codes",
            },
        },
    }


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
    doc_batch_size: int = 2048,
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
    doc_batch_size : int
        Document-minibatch size for the PC / two-stage fits' full-batch gradient
        assembly (see :class:`~analysis.pc.model.PCTopicModel`); memory-bounding
        only, the objective/optimizer are unchanged.
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
        max_iter=max_iter, doc_batch_size=doc_batch_size, seed=seed, **model_kwargs,
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
    ``results`` is the dict returned by :func:`evaluate_pc_vs_baselines` or
    :func:`evaluate_pc_multitask`; the multi-task result additionally carries
    per-column observed counts (``meta["n_obs_train"]``/``["n_obs_test"]``), which
    are appended as a footer when present.
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

    if "n_obs_train" in meta:  # multi-task: report per-column observed counts
        lines.append("  observed cells per label (train / test):")
        for c in range(C):
            lines.append(
                f"    - label {c}: {meta['n_obs_train'][c]} / {meta['n_obs_test'][c]}"
            )

    return "\n".join(lines)
