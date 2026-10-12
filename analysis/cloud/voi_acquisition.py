"""Simulated code acquisition on held-out patients (spec 2026-10-07 §D4, WP-D part 3;
the 2026-08-14 VOI metrics report's evaluation of VOI).

The question part 1's tables cannot answer: when you actually ASK about codes in the
order VOI recommends, does the posterior concentrate on the patient's true disease
faster than asking in a fixed or random order, and are the posteriors it reaches
calibrated? Part 1 ranks codes by expected information gain under a model; this
replays that ranking against real records and scores it.

THE SIMULATION (why each piece has the shape it has)
----------------------------------------------------
* Patients: held-out (test split) persons positive on at least one member of a
  candidate set S — "known to be somewhere in G". Their truth is the set T of
  members they are positive on (R1a makes nested members near-disjoint).
* Query pool: per acquisition domain, the `pool` codes with the highest population
  EIG (part 1's ranking), minus attestation codes (members' and the set node's own
  codes). A query is "is code v anywhere in this patient's record?", answered from
  the record itself: x = 1 iff v is in the patient's domain bag.
* Likelihood: part 1's presence model, p_m(v) = 1 - exp(-L_i * (rho * beta_m(v) +
  (1 - rho) * beta_bg(v))), with L_i the patient's OWN token count in that domain
  (record length is observable without revealing any code, and it is what makes a
  long record's absence stronger evidence than a short one's). A patient with no
  drug record has p = 0 for every drug and every drug answer is uninformative —
  the right behaviour, not a special case.
* Posterior: the prior times the product of answered likelihoods (naive Bayes over
  the asked codes). The independence assumption is the model's, so posteriors are
  expected to be overconfident; that is why calibration is reported next to
  accuracy rather than assumed.
* Strategies, all over the same pool: `greedy` asks, at each step, the code with the
  highest EIG under the CURRENT posterior (adaptive VOI); `static` asks in part 1's
  fixed population-EIG order (the table, read top-down); `random` asks in a random
  order (the floor). `full` answers the whole pool at once — the ceiling the
  strategies approach.
* Metrics per step k, pooled over patients: mean posterior on the truth P(T), top-1
  (argmax in T), balanced top-1 (mean over single-positive classes, so rare members
  count), entropy in bits, and hits (how many asked codes were present). Calibration:
  ECE of the max posterior (10 bins) at the last step and for `full`.

Pure numpy; the driver (`gated_pc_readout --acquisition`) only collects the test
patients' code sets. Pooled numbers only — a set with fewer than 20 patients is
suppressed (egress floor)."""
from __future__ import annotations

import numpy as np

import voi_tables as vt

__all__ = ["build_pool", "simulate_patient", "simulate_set", "format_acquisition",
           "STRATEGIES", "EGRESS_FLOOR"]

STRATEGIES = ("greedy", "static", "random")
EGRESS_FLOOR = 20


def build_pool(prior, profiles_by_dom, bg_by_dom, excl_by_dom, *, pool=150, L=40.0,
               rho=0.3):
    """The query pool, `[(dom, vocab_idx)]` in population-EIG order (the `static`
    order), plus the per-domain mixture rows `{d: (M, Q_d) mix, (Q_d,) idx}` the
    per-patient likelihood is built from. `profiles_by_dom[d]` is (M, V_d)."""
    items, mix = [], {}
    for d, prof in profiles_by_dom.items():
        bg = bg_by_dom[d]
        P = vt.presence_prob(prof, bg[None, :], L=L, rho=rho)
        eig, _ = vt.eig_table(prior, P)
        eig = eig.copy()
        ex = np.fromiter(excl_by_dom.get(d, ()), dtype=np.int64)
        if ex.size:
            eig[ex[ex < eig.size]] = -np.inf
        top = np.argsort(eig)[::-1][:int(pool)]
        top = top[np.isfinite(eig[top])]
        mix[d] = (float(rho) * prof[:, top] + (1.0 - float(rho)) * bg[None, top], top)
        items.extend((d, int(v), float(eig[v])) for v in top)
    items.sort(key=lambda t: -t[2])
    return [(d, v) for d, v, _ in items], mix


def _patient_matrix(pool_items, mix, bags, lengths):
    """(M, Q) presence probabilities and (Q,) answers for one patient."""
    col = {}
    for d, (m, idx) in mix.items():
        p = 1.0 - np.exp(-float(lengths.get(d, 0.0)) * m)
        for j, v in enumerate(idx):
            col[(d, int(v))] = p[:, j]
    P = np.stack([col[k] for k in pool_items], axis=1)
    x = np.array([v in bags.get(d, ()) for d, v in pool_items], dtype=bool)
    return P, x


def _update(post, p_col, present):
    lik = p_col if present else 1.0 - p_col
    new = post * np.clip(lik, 1e-12, 1.0)
    s = new.sum()
    return new / s if s > 0 else post


def simulate_patient(prior, P, x, *, strategy, steps, rng=None):
    """Posterior trajectory `(steps + 1, M)` and the asked answers for one patient.
    `P` (M, Q) presence probabilities, `x` (Q,) answers; the pool's column order is
    the `static` order."""
    M, Q = P.shape
    post = np.asarray(prior, dtype=np.float64).copy()
    traj = [post.copy()]
    asked = np.zeros(Q, dtype=bool)
    order = (rng.permutation(Q) if strategy == "random" else np.arange(Q))
    hits = []
    for k in range(int(steps)):
        if k >= Q:
            traj.append(post.copy())
            hits.append(False)
            continue
        if strategy == "greedy":
            avail = np.flatnonzero(~asked)
            eig, _ = vt.eig_table(post, P[:, avail])
            q = int(avail[int(np.argmax(eig))])
        else:
            q = int(order[k])
        asked[q] = True
        post = _update(post, P[:, q], bool(x[q]))
        traj.append(post.copy())
        hits.append(bool(x[q]))
    return np.stack(traj), np.array(hits, dtype=bool)


def _full_posterior(prior, P, x):
    logp = np.log(np.clip(np.asarray(prior, dtype=np.float64), 1e-300, 1.0))
    lik = np.where(x[None, :], P, 1.0 - P)
    logp = logp + np.log(np.clip(lik, 1e-12, 1.0)).sum(axis=1)
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def _ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def _score(posts, truths):
    """Pooled metrics for a list of posteriors (M,) against truth index lists."""
    ptrue = np.array([p[t].sum() for p, t in zip(posts, truths)])
    top = np.array([int(np.argmax(p)) in t for p, t in zip(posts, truths)])
    ent = np.array([float(vt._entropy(p)) for p in posts])
    single = [(t[0], c) for t, c in zip(truths, top) if len(t) == 1]
    by = {}
    for cls, c in single:
        by.setdefault(cls, []).append(c)
    bal = float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")
    return {"p_true": float(ptrue.mean()), "top1": float(top.mean()),
            "bal_top1": bal, "entropy": float(ent.mean())}


def simulate_set(prior, pool_items, mix, patients, *, steps=20, seed=0,
                 report_steps=(0, 1, 2, 3, 5, 10, 20)):
    """Run every strategy over `patients` = `[(truth idx list, bags {d: set},
    lengths {d: tokens})]`. Returns the pooled metrics dict (no per-patient data)."""
    rng = np.random.default_rng(int(seed))
    steps = int(steps)
    ks = sorted({k for k in report_steps if k <= steps} | {steps})
    out = {"n": len(patients), "pool": len(pool_items), "steps": steps,
           "report_steps": ks, "strategies": {}}
    truths = [t for t, _, _ in patients]
    mats = [_patient_matrix(pool_items, mix, b, l) for _, b, l in patients]
    for strat in STRATEGIES:
        trajs, hits = [], []
        for P, x in mats:
            tr, h = simulate_patient(prior, P, x, strategy=strat, steps=steps, rng=rng)
            trajs.append(tr)
            hits.append(h)
        rows = {}
        for k in ks:
            r = _score([tr[k] for tr in trajs], truths)
            r["hits"] = float(np.mean([h[:k].sum() for h in hits])) if k else 0.0
            rows[k] = r
        last = [tr[steps] for tr in trajs]
        out["strategies"][strat] = {
            "by_step": rows,
            "ece_last": _ece([p.max() for p in last],
                             [int(np.argmax(p)) in t for p, t in zip(last, truths)])}
    full = [_full_posterior(prior, P, x) for P, x in mats]
    out["full"] = _score(full, truths)
    out["full"]["ece"] = _ece([p.max() for p in full],
                              [int(np.argmax(p)) in t for p, t in zip(full, truths)])
    return out


def format_acquisition(name, members_named, prior, res):
    """Markdown block for one set; pooled numbers only, suppressed under the floor."""
    L = [f"## {name}"]
    if res["n"] < EGRESS_FLOOR:
        L.append(f"(fewer than {EGRESS_FLOOR} held-out patients; suppressed)")
        return "\n".join(L)
    L.append(f"{res['n']} held-out patients · pool {res['pool']} codes · prior: "
             + " · ".join(f"{nm} {100 * p:.0f}%" for nm, p in zip(members_named, prior)))
    L.append("")
    L.append("| strategy | step | P(true) | top-1 | bal top-1 | H (bits) | hits |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for strat in STRATEGIES:
        for k, r in res["strategies"][strat]["by_step"].items():
            if k == 0 and strat != STRATEGIES[0]:
                continue
            L.append(f"| {strat if k else 'prior'} | {k} | {r['p_true']:.3f} | "
                     f"{r['top1']:.3f} | {r['bal_top1']:.3f} | {r['entropy']:.2f} | "
                     f"{r['hits']:.1f} |")
    f = res["full"]
    L.append(f"| full pool | {res['pool']} | {f['p_true']:.3f} | {f['top1']:.3f} | "
             f"{f['bal_top1']:.3f} | {f['entropy']:.2f} | — |")
    L.append("")
    L.append("calibration (ECE of the max posterior): " + " · ".join(
        f"{s} @{res['steps']} {res['strategies'][s]['ece_last']:.3f}"
        for s in STRATEGIES) + f" · full {f['ece']:.3f}")
    return "\n".join(L)
