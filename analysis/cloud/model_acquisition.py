"""Model-based acquisition on held-out patients (WP-D part 3b; 0138's read: "the order
is right, the engine is weak").

Part 3 (`voi_acquisition`) scored question orders with a per-code presence model built
from the topic profiles, and that engine needed a temper of 0.05-0.1 to be calibrated
and was a weak classifier for broad sets. This module keeps the questions and swaps the
ENGINE for the model the readout already validated (0138 part 2): after each revealed
code, the patient's θ is re-inferred from the codes revealed so far, and the saved
readout heads score it.

THE ENGINE (why each piece has the shape it has)
------------------------------------------------
* θ from a partial record: the deployment fold-in, exactly — UNGATED full-K CAVI over
  the per-domain-normalized concatenated E[log β], with the fitted α
  (`DagGatedLDAModel._transform`; `_cavi_doc_inference` itself). The empty record
  folds in to θ = α / sum(α): the model's own "nothing known yet" state.
* Heads: the record's saved readout heads (`readout_heads_gated_pc_<tag>.npz`):
  p_c = σ(clip(V_c·θ + b_c, ±50)), degenerate nodes at their constant. Only the heads
  in the closure of the candidate set's members are evaluated.
* Which member: q_m ∝ P_stack(m) = Π_{a ∈ closure(m)} p_a, normalized over the set.
  For a set under a common ancestor G every member's closure contains G's, so the
  shared factor cancels and q is the stacked conditional P(m | G) renormalized — the
  quantity 0138 part 2 found calibrated on full records.
* Answers: "is code v anywhere in the record?" A YES adds v (with its record count) to
  the revealed bag and θ is re-inferred. A NO changes nothing: a bag-of-codes model has
  no likelihood for an absent code, so this engine can learn only from presences.
  That is a real difference from part 3's engine (whose posterior was driven by
  absences) and it is reported, not hidden: `hits` is the number of YES answers.
* Greedy choice: the code maximizing p(v) · KL(q'_v || q), the expected change in the
  member posterior. p(v) = 1 - exp(-L_d · (θ·E[β])_v) is the model's predictive
  presence for a record of L_d distinct codes in v's domain; q'_v is the posterior
  after adding v once. q'_v comes from a short warm-started CAVI (`cand_iters`, 5)
  from the current γ, batched over every unasked candidate — a ranking device; the
  chosen code then gets the full fold-in.
* Ceilings: `pool` folds in every pool code the patient has; `record` folds in the
  whole record, the diagnosis codes included — the setting of 0138 part 2, so it is a
  check that this engine reproduces the readout, not a fair acquisition target.

Pure numpy over saved parameters; the driver only collects the patients. Pooled numbers
only, suppressed under the egress floor (`voi_acquisition.EGRESS_FLOOR`)."""
from __future__ import annotations

import numpy as np
from scipy.special import digamma

import voi_acquisition as va

__all__ = ["ModelEngine", "simulate_set_model", "format_model_acquisition"]

_SCORE_CLIP = 50.0


def _expElogtheta(G):
    return np.exp(digamma(G) - digamma(G.sum(axis=-1, keepdims=True)))


class ModelEngine:
    """The saved fit + heads, restricted to one candidate set's members.

    `lams` is the per-domain λ list (K, V_d) in domain order; `alpha` (K,); heads
    `(V (C,K), b (C,), degenerate (C,) bool, const (C,), topm)`; `closure` a
    `{member: [readout nodes in its closure]}` map; `members` the set's label nodes."""

    def __init__(self, lams, alpha, heads, closure, members, *, cavi_max_iter=100,
                 cavi_tol=1e-3, cand_iters=5):
        sizes = [int(l.shape[1]) for l in lams]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        from spark_vi.models.topic.dag_gated_lda import expElogbeta_from_lambda
        self.eb = expElogbeta_from_lambda(dict(enumerate(lams)))
        # E[β] is needed only at pool columns (the predictive presence); keeping the
        # λ blocks and their row sums avoids a second dense (K, V) float64 copy.
        self._lams = [np.asarray(l) for l in lams]
        self._rowsum = [l.sum(axis=1) for l in self._lams]
        self.alpha = np.asarray(alpha, dtype=np.float64)
        V, b, deg, const, topm = heads
        nodes = sorted({a for m in members for a in closure[m]})
        self.Vs = np.asarray(V, dtype=np.float64)[nodes]
        self.bs = np.asarray(b, dtype=np.float64)[nodes]
        self.deg = np.asarray(deg, dtype=bool)[nodes]
        self.const = np.asarray(const, dtype=np.float64)[nodes]
        self.topm = int(topm)
        pos = {a: j for j, a in enumerate(nodes)}
        self.Msub = np.zeros((len(members), len(nodes)))
        for i, m in enumerate(members):
            for a in closure[m]:
                self.Msub[i, pos[a]] = 1.0
        self.max_iter, self.tol, self.cand_iters = int(cavi_max_iter), float(cavi_tol), int(cand_iters)

    def gidx(self, d, v):
        return int(self.offsets[d] + v)

    def ebeta_cols(self, items):
        """(K, Q) E[β] at `(domain, vocab idx)` items."""
        return np.stack([self._lams[d][:, v] / self._rowsum[d] for d, v in items],
                        axis=1)

    def fold_in(self, idx, cnt, gamma0=None):
        """γ for a revealed bag (global ids, counts); the deployment CAVI."""
        from spark_vi.models.topic.lda import _cavi_doc_inference
        if len(idx) == 0:
            return self.alpha.copy()
        g0 = self.alpha + 1.0 if gamma0 is None else gamma0
        gamma, _, _, _ = _cavi_doc_inference(
            indices=np.asarray(idx, dtype=np.int64), counts=np.asarray(cnt, np.float64),
            expElogbeta=self.eb, alpha=self.alpha, gamma_init=g0,
            max_iter=self.max_iter, tol=self.tol)
        return gamma

    def posterior(self, theta):
        """q over members for θ (K,) or a batch (Q, K)."""
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if self.topm and self.topm < th.shape[1]:
            cut = np.partition(th, th.shape[1] - self.topm, axis=1)[:, [-self.topm]]
            th = np.where(th >= cut, th, 0.0)
        z = np.clip(th @ self.Vs.T + self.bs, -_SCORE_CLIP, _SCORE_CLIP)
        p = 1.0 / (1.0 + np.exp(-z))
        if self.deg.any():
            p[:, self.deg] = self.const[self.deg]
        logq = np.log(np.clip(p, 1e-12, 1.0)) @ self.Msub.T
        logq -= logq.max(axis=1, keepdims=True)
        q = np.exp(logq)
        q /= q.sum(axis=1, keepdims=True)
        return q[0] if np.ndim(theta) == 1 else q

    def candidate_thetas(self, gamma, idx, cnt, cand):
        """θ after adding each candidate once, warm-started from `gamma`, batched."""
        Q = len(cand)
        G = np.tile(gamma, (Q, 1))
        ebC = self.eb[:, cand].T                                  # (Q, K)
        ebR = self.eb[:, idx] if len(idx) else None               # (K, nR)
        cnt = np.asarray(cnt, dtype=np.float64)
        for _ in range(self.cand_iters):
            Et = _expElogtheta(G)
            contrib = ebC / ((Et * ebC).sum(axis=1, keepdims=True) + 1e-100)
            if ebR is not None:
                pn = Et @ ebR + 1e-100                            # (Q, nR)
                contrib = contrib + (cnt / pn) @ ebR.T
            G = self.alpha + Et * contrib
        return G / G.sum(axis=1, keepdims=True)


def _kl(p, q):
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(q, 1e-300, 1.0)
    return (p * (np.log2(p) - np.log2(q))).sum(axis=-1)


def _patient_tokens(eng, pool_items, bags):
    """Per pool item: present?, count; and the whole record as global (idx, cnt)."""
    present = np.array([v in bags.get(d, {}) for d, v in pool_items], dtype=bool)
    counts = np.array([float(bags.get(d, {}).get(v, 0.0)) for d, v in pool_items])
    rec_i, rec_c = [], []
    for d, b in bags.items():
        for v, c in b.items():
            rec_i.append(eng.gidx(d, v))
            rec_c.append(float(c))
    return present, counts, np.array(rec_i, dtype=np.int64), np.array(rec_c)


def simulate_patient_model(eng, pool_g, pool_dom, present, counts, lengths, *,
                           strategy, steps, rng=None, Eb_pool=None):
    """Member-posterior trajectory `(steps + 1, M)` and the YES answers."""
    Q = len(pool_g)
    idx, cnt = [], []
    gamma = eng.fold_in(idx, cnt)
    q = eng.posterior(gamma / gamma.sum())
    traj, hits = [q], []
    asked = np.zeros(Q, dtype=bool)
    order = rng.permutation(Q) if strategy == "random" else np.arange(Q)
    L = np.array([float(lengths.get(int(d), 0.0)) for d in pool_dom])
    for k in range(int(steps)):
        if k >= Q:
            traj.append(q); hits.append(False); continue
        if strategy == "greedy":
            avail = np.flatnonzero(~asked)
            theta = gamma / gamma.sum()
            pv = 1.0 - np.exp(-L[avail] * (theta @ Eb_pool[:, avail]))
            qc = eng.posterior(eng.candidate_thetas(gamma, idx, cnt, pool_g[avail]))
            j = int(avail[int(np.argmax(pv * _kl(qc, q[None, :])))])
        else:
            j = int(order[k])
        asked[j] = True
        if present[j]:
            idx.append(int(pool_g[j])); cnt.append(float(counts[j]))
            gamma = eng.fold_in(idx, cnt, gamma0=gamma)
            q = eng.posterior(gamma / gamma.sum())
        traj.append(q)
        hits.append(bool(present[j]))
    return np.stack(traj), np.array(hits, dtype=bool)


def simulate_set_model(eng, pool_items, patients, *, steps=20, seed=0,
                       report_steps=(0, 1, 2, 3, 5, 10, 20)):
    """`patients` = `[(truth idx list, bags {d: {v: count}}, lengths {d: L})]`.
    Same result shape as `voi_acquisition.simulate_set`, plus the `record` ceiling."""
    rng = np.random.default_rng(int(seed))
    steps = int(steps)
    ks = sorted({k for k in report_steps if k <= steps} | {steps})
    pool_g = np.array([eng.gidx(d, v) for d, v in pool_items], dtype=np.int64)
    pool_dom = np.array([d for d, _ in pool_items], dtype=np.int64)
    Eb_pool = eng.ebeta_cols(pool_items)
    truths = [t for t, _, _ in patients]
    toks = [_patient_tokens(eng, pool_items, b) for _, b, _ in patients]
    out = {"n": len(patients), "pool": len(pool_items), "steps": steps, "tau": 1.0,
           "tau_fit": None, "engine": "model", "report_steps": ks, "strategies": {}}
    for strat in va.STRATEGIES:
        trajs, hits = [], []
        for (present, counts, _, _), (_, _, lengths) in zip(toks, patients):
            tr, h = simulate_patient_model(eng, pool_g, pool_dom, present, counts,
                                           lengths, strategy=strat, steps=steps, rng=rng,
                                           Eb_pool=Eb_pool)
            trajs.append(tr); hits.append(h)
        rows = {}
        for k in ks:
            r = va._score([tr[k] for tr in trajs], truths)
            r["hits"] = float(np.mean([h[:k].sum() for h in hits])) if k else 0.0
            rows[k] = r
        last = [tr[steps] for tr in trajs]
        out["strategies"][strat] = {
            "by_step": rows,
            "ece_last": va._ece([p.max() for p in last],
                                [int(np.argmax(p)) in t for p, t in zip(last, truths)])}
    for key, pick in (("full", lambda pr, c, ri, rc: (pool_g[pr], c[pr])),
                      ("record", lambda pr, c, ri, rc: (ri, rc))):
        posts = []
        for present, counts, ri, rc in toks:
            i, c = pick(present, counts, ri, rc)
            g = eng.fold_in(list(i), list(c))
            posts.append(eng.posterior(g / g.sum()))
        out[key] = va._score(posts, truths)
        out[key]["ece"] = va._ece([p.max() for p in posts],
                                  [int(np.argmax(p)) in t for p, t in zip(posts, truths)])
    return out


def format_model_acquisition(name, members_named, res):
    """Markdown block for one set (pooled numbers only, suppressed under the floor).
    The prior row is the engine's empty-record posterior, the same for every patient."""
    L = [f"## {name}"]
    if res["n"] < va.EGRESS_FLOOR:
        L.append(f"(fewer than {va.EGRESS_FLOOR} held-out patients; suppressed)")
        return "\n".join(L)
    L.append(f"{res['n']} held-out patients · engine: θ fold-in + saved heads · pool "
             f"{res['pool']} codes · members: " + " · ".join(members_named))
    L.append("")
    L.append("| strategy | step | P(true) | top-1 | bal top-1 | H (bits) | hits |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for strat in va.STRATEGIES:
        for k, r in res["strategies"][strat]["by_step"].items():
            if k == 0 and strat != va.STRATEGIES[0]:
                continue
            L.append(f"| {strat if k else 'empty record'} | {k} | {r['p_true']:.3f} | "
                     f"{r['top1']:.3f} | {r['bal_top1']:.3f} | {r['entropy']:.2f} | "
                     f"{r['hits']:.1f} |")
    for key, lab in (("full", f"full pool ({res['pool']})"),
                     ("record", "whole record (incl. diagnosis codes)")):
        r = res[key]
        L.append(f"| {lab} | — | {r['p_true']:.3f} | {r['top1']:.3f} | "
                 f"{r['bal_top1']:.3f} | {r['entropy']:.2f} | — |")
    L.append("")
    L.append("calibration (ECE of the max posterior): " + " · ".join(
        f"{s} @{res['steps']} {res['strategies'][s]['ece_last']:.3f}"
        for s in va.STRATEGIES) + f" · full pool {res['full']['ece']:.3f} · whole "
        f"record {res['record']['ece']:.3f}")
    return "\n".join(L)
