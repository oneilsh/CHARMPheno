"""How much of the case-finding AUC is prevalent recall? (insight 0100; exp 0143)

On the native-Mondo ribbon the index date is random and the label is "X coded in the
year after it", while the 5-year pre-index features still carry the patient's own
attestation codes (the leakage strip is a no-op there). So a patient diagnosed with X
before the index is scored on features that contain X's own code, and the label rewards
predicting X's RE-coding.

This module re-scores the saved heads per readout node X twice over the same held-out
patients:
  * ALL — the current evaluation;
  * INCIDENT — patients with prior X evidence set aside (neither positive nor negative):
    "prior X" = any mapped attestation code of X or of a label descendant of X in the
    patient's pre-index condition features (`prior_matrix`).
The gap ALL − INCIDENT per node is the part of the AUC that rewards recognizing a past
diagnosis. Prior evidence is read from the 5-year lookback only (what the features
hold), so diagnoses older than that count as incident — the gap is a lower bound.

Pure numpy; the driver (`gated_pc_readout --prevalence-check`) scores the test split and
collects one sparse code list per patient."""
from __future__ import annotations

import numpy as np


__all__ = ["auc", "framing3_sources", "vocab_to_nodes", "prior_matrix", "sidecar_nodes_to_readout",
           "prior_from_sets", "containment", "node_aucs", "summarize", "EGRESS_FLOOR",
           "prior_rows", "rows_from_node_lists", "node_aucs_from_metrics"]

EGRESS_FLOOR = 20


def auc(score, y):
    """Mann-Whitney AUC of `score` for positives `y` (ties at half credit)."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def framing3_sources(run_dir):
    """`(theta, raw)`: the run's new-diagnosis check for θ heads and for the frequency
    raw-code heads (parsed JSON, or None). θ: the whole-history check
    (`prevalence_check_history.json`, exps 0145-0151), else the record stacked check
    (`prevalence_check_stacked[_nodes]_mass99.json`, 0152 on); convergence re-solves
    (`_warm`, `_it<N>`) and raw arms never count as θ. Raw: `..._raw2000` on either
    naming. The survey and `survey_analysis` both read through this, so they cannot
    disagree about which file is framing 3."""
    import json
    from pathlib import Path
    run_dir = Path(run_dir)

    def _load(*names):
        for n in names:
            f = run_dir / n
            if f.exists():
                return json.loads(f.read_text())
        return None
    theta = _load("prevalence_check_history.json",
                  "prevalence_check_stacked_nodes_mass99.json",
                  "prevalence_check_stacked_mass99.json")
    if theta is None:
        for f in sorted(run_dir.glob("prevalence_check_stacked*.json")):
            if not any(t in f.name for t in ("raw", "_warm", "_it")):
                theta = json.loads(f.read_text())
                break
    raw = _load("prevalence_check_stacked_nodes_raw2000.json",
                "prevalence_check_stacked_raw2000.json")
    return theta, raw


def vocab_to_nodes(own_idx, pos_sources):
    """{condition vocab idx: [readout nodes]}: a code attests label node e (own_idx) and
    therefore every readout node whose positive sources include e."""
    by_label = {}
    for n, srcs in enumerate(pos_sources):
        for e in srcs:
            by_label.setdefault(int(e), []).append(int(n))
    out = {}
    for e, vs in own_idx.items():
        for v in vs:
            out.setdefault(int(v), set()).update(by_label.get(int(e), []))
    return {v: sorted(ns) for v, ns in out.items()}


def prior_matrix(keys, codes_by_key, v2n, C):
    """(D, C) bool: patient d has prior evidence for node c. `codes_by_key` maps a doc
    key to its condition vocab indices."""
    M = np.zeros((len(keys), int(C)), dtype=bool)
    for i, k in enumerate(keys):
        for v in codes_by_key.get(int(k), ()):
            ns = v2n.get(int(v))
            if ns:
                M[i, ns] = True
    return M


def _curie(cid):
    return f"MONDO:{int(cid):07d}"


def sidecar_nodes_to_readout(node_cids, readout_int2cid, parent_adj, *, root_cid=-1):
    """{sidecar Mondo node cid: [readout nodes]} — WHOLE-HISTORY prior rule (exp 0145).

    The E4 sidecar's node ids come from the code map rebuilt WITHOUT the label-set
    filter (`conversion_sidecar.code_map_from_manifest`), so they can be finer Mondo
    terms than the fit's labels. A first attestation of term t counts as prior
    evidence for every readout node whose Mondo term is t or an ancestor of t — the
    is-a reading of "a prior diagnosis of something under X". `parent_adj` is Mondo's
    `{curie: [parent curies]}`; the readout root is never a target."""
    cid2r = {int(c): int(r) for r, c in readout_int2cid.items()
             if int(r) != 0 and int(c) != int(root_cid)}
    out = {}
    for n in node_cids:
        seen, stack = set(), [_curie(n)]
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            stack.extend(parent_adj.get(t, ()))
        rs = sorted({cid2r[int(t.split(":", 1)[1])] for t in seen
                     if int(t.split(":", 1)[1]) in cid2r})
        if rs:
            out[int(n)] = rs
    return out


def prior_from_sets(keys, nodes_by_person, C, person_of):
    """(D, C) bool from `{person_id: iterable of readout nodes}`; `person_of` maps a
    doc key to its person."""
    M = np.zeros((len(keys), int(C)), dtype=bool)
    for i, k in enumerate(keys):
        ns = nodes_by_person.get(int(person_of(k)))
        if ns:
            M[i, sorted(int(n) for n in ns)] = True
    return M


def containment(a, b):
    """Share of `a`'s True cells that are also True in `b` (nan if `a` is empty). The
    lookback rule's prior cells should sit inside the whole-history rule's — a low
    share means the two disagree about the index date or the node ids."""
    n = int(a.sum())
    return float((a & b).sum() / n) if n else float("nan")


def prior_rows(code_lists, v2n, C):
    """(n, C) bool for one block of rows: row i has prior evidence for node c when one
    of its condition vocab indices `code_lists[i]` maps to c (`vocab_to_nodes`)."""
    M = np.zeros((len(code_lists), int(C)), dtype=bool)
    for i, codes in enumerate(code_lists):
        for v in codes:
            ns = v2n.get(int(v))
            if ns:
                M[i, ns] = True
    return M


def rows_from_node_lists(node_lists, C):
    """(n, C) bool from per-row lists of readout nodes (None = no prior)."""
    M = np.zeros((len(node_lists), int(C)), dtype=bool)
    for i, ns in enumerate(node_lists):
        if ns:
            M[i, sorted(int(n) for n in ns)] = True
    return M


def node_aucs_from_metrics(m_all, m_inc, nodes, *, floor=EGRESS_FLOOR):
    """Per node, from its column metrics over all patients (`m_all[c]`) and with the
    prior-evidence patients set aside (`m_inc[c]`; `spark_vi.eval.metrics`): both
    AUCs, the positives in each, and the prior share of positives / negatives (the
    rows the second set drops). Nodes under `floor` positives in either setting, or
    under `floor` incident negatives, are skipped."""
    out = {}
    for c in nodes:
        a, b = m_all.get(c), m_inc.get(c)
        if a is None or b is None:
            continue
        n_all, n_inc = int(a["n_pos"]), int(b["n_pos"])
        if n_all < floor or n_inc < floor or int(b["n_neg"]) < floor:
            continue
        nn = int(a["n_neg"])
        out[int(c)] = {
            "auc_all": a["auc"], "auc_incident": b["auc"],
            "n_pos_all": n_all, "n_pos_incident": n_inc,
            "prior_share_pos": (n_all - n_inc) / n_all,
            "prior_share_neg": (nn - int(b["n_neg"])) / nn if nn else float("nan")}
    return out


def node_aucs(score, y, prior, nodes, *, floor=EGRESS_FLOOR):
    """`node_aucs_from_metrics` on dense (D, C) arrays (the in-memory backend)."""
    from spark_vi.eval.metrics import local as ml
    cols = [int(c) for c in nodes]
    m_all = ml.per_key_metrics(ml.column_pieces(score, y, columns=cols))
    m_inc = ml.per_key_metrics(ml.column_pieces(score, y, ~np.asarray(prior, bool),
                                                columns=cols))
    return node_aucs_from_metrics(m_all, m_inc, cols, floor=floor)


def summarize(per_node):
    """Pooled: macro AUC all vs incident, the per-node gap distribution, and the gap by
    the prior share of positives (where the prevalent cases are)."""
    if not per_node:
        return {"n_nodes": 0}
    a = np.array([v["auc_all"] for v in per_node.values()])
    b = np.array([v["auc_incident"] for v in per_node.values()])
    s = np.array([v["prior_share_pos"] for v in per_node.values()])
    gap = a - b
    bands = {}
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
        sel = (s >= lo) & (s < hi)
        if sel.sum():
            bands[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {
                "n_nodes": int(sel.sum()), "auc_all": float(a[sel].mean()),
                "auc_incident": float(b[sel].mean()),
                "median_gap": float(np.median(gap[sel]))}
    return {"n_nodes": int(len(a)), "macro_auc_all": float(a.mean()),
            "macro_auc_incident": float(b.mean()),
            "gap_median": float(np.median(gap)), "gap_p90": float(np.quantile(gap, 0.9)),
            "n_gap_over_0p05": int((gap > 0.05).sum()),
            "prior_share_pos_median": float(np.median(s)), "by_prior_share": bands}
