"""Framing 3 — undiagnosed look-alikes (spec 2026-10-13 clinician framings, R3).

"Who looks like disease X but carries no X code, and why?" The ranking is the record's
stacked readout head P_stack(X) — the marginal case-finding score already measured on
0137 (stacked de-novo macro AUC 0.822) — over held-out patients WITHOUT an X label. The
"why" is a pooled feature profile: codes over-represented among the top look-alikes
relative to a random sample of the test split, kept only when X's own patients also
carry them more than background (so the profile is X-like, not merely "sick").

WHO THE LOOK-ALIKES ARE (0142 first run). In lookback mode the features are the
pre-index history and the label is "coded in the year after the index date", so "no X
label" also covers patients whose X diagnosis sits in their history and was not re-coded
in the label year. `split_lookalikes` separates (i) X attestation code in the history,
(ii) only an X-named concept the code map missed, (iii) neither — the undiagnosed-by-code
look-alikes, profiled on their own.

One validation the data hand us for free: concepts whose NAME is X's name but which the
code map missed (0141: a second "Hypertrophic cardiomyopathy" concept) are true X
diagnoses that the label never saw. If the ranking is finding real undiagnosed-by-label
cases, they should be over-represented among the top look-alikes.

Pure numpy; the driver (`gated_pc_readout --lookalikes`) scores the test split and
collects the few hundred records it needs. Pooled numbers only; every printed rate rests
on >= 20 patients (egress floor)."""
from __future__ import annotations

import numpy as np

__all__ = ["auc", "top_lookalikes", "code_counts", "feature_profile", "share_with",
           "split_lookalikes", "EGRESS_FLOOR"]

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


def top_lookalikes(score, y, keys, top_n):
    """Keys of the `top_n` highest-scoring patients without the label, and the score
    at the cut."""
    score = np.asarray(score, dtype=np.float64)
    neg = np.flatnonzero(~np.asarray(y).astype(bool))
    o = neg[np.argsort(-score[neg], kind="stable")][:int(top_n)]
    return [keys[i] for i in o], (float(score[o[-1]]) if len(o) else float("nan"))


def code_counts(bags_list, excl_by_dom=None):
    """{(domain, vocab idx): number of patients carrying it}, excluded ids dropped."""
    excl_by_dom = excl_by_dom or {}
    c = {}
    for bags in bags_list:
        for d, b in bags.items():
            ex = excl_by_dom.get(d, ())
            for v in b:
                if int(v) in ex:
                    continue
                k = (int(d), int(v))
                c[k] = c.get(k, 0) + 1
    return c


def feature_profile(top_bags, pos_bags, bg_bags, *, excl_by_dom=None, top=10,
                    floor=EGRESS_FLOOR, smooth=0.5):
    """Codes over-represented in the look-alikes vs background AND more common in X's
    own patients than in background. Rows `(key, rate_top, rate_pos, rate_bg, lift)`,
    lift = smoothed rate_top / rate_bg, ranked by lift; a code is shown only when at
    least `floor` look-alikes carry it."""
    ct, cp, cb = (code_counts(b, excl_by_dom) for b in (top_bags, pos_bags, bg_bags))
    nt, np_, nb = max(len(top_bags), 1), max(len(pos_bags), 1), max(len(bg_bags), 1)
    rows = []
    for k, n in ct.items():
        if n < int(floor):
            continue
        rt = n / nt
        rb = (cb.get(k, 0) + smooth) / (nb + 2 * smooth)
        rp = cp.get(k, 0) / np_
        if rp <= rb:
            continue
        rows.append((k, rt, rp, cb.get(k, 0) / nb, rt / rb))
    rows.sort(key=lambda r: -r[4])
    return rows[:int(top)]


def share_with(bags_list, ids_by_dom):
    """(n patients carrying any of `ids_by_dom`, n patients)."""
    n = 0
    for bags in bags_list:
        if any(int(v) in ids_by_dom.get(d, ()) for d, b in bags.items() for v in b):
            n += 1
    return n, len(bags_list)


def split_lookalikes(bags_list, own_ids, unmapped_ids, *, dom=0):
    """Partition look-alike records: `prior_dx` (carries one of X's attestation codes),
    `unmapped_only` (no attestation code, but an X-named concept the map missed),
    `undiagnosed` (neither)."""
    out = {"prior_dx": [], "unmapped_only": [], "undiagnosed": []}
    for b in bags_list:
        codes = set(int(v) for v in b.get(dom, {}))
        if codes & own_ids:
            out["prior_dx"].append(b)
        elif codes & unmapped_ids:
            out["unmapped_only"].append(b)
        else:
            out["undiagnosed"].append(b)
    return out
