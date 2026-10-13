"""Framing 1 — differential with reasons (spec 2026-10-13 clinician framings, R0/R1).

"For this patient, which disease in group G, and what in their record points there?"
Four engines are scored on the same held-out patients (D4: the engine is chosen by
evaluation, not assumed):

E1  presence model with SUBTYPE profiles (D5). Each signature topic of a member is a
    subtype; the record's likelihood under a subtype uses every code of the vocabulary,
    present AND absent, through p(v) = 1 - exp(-L_d · (ρ β_t(v) + (1-ρ) β_bg(v))) with
    L_d the record's distinct codes in domain d. Because each domain's mixture sums to
    one, the absent-code term collapses to -L_d · (1 - mass of excluded codes), so a
    record costs O(its own codes) per subtype. Member score = log Σ_t w_t exp(ll_t),
    w_t ∝ the subtype's evidence — a patient looks like ONE subtype, not their average —
    and the temper τ multiplies the member score (the whole member likelihood).
E2  the record's stacked readout heads renormalized over the members (0139's
    baseline), on the whole-record θ (the deployment fold-in).
E3  a multiclass (softmax) head over the members on log θ, fit on TRAIN-split patients
    positive on exactly one member; L2 chosen on an internal split. Features: the
    background topics plus the members' blocks.
E4  gated per-disease likelihood: for each member m, the per-document ELBO of the
    record under the training gate {background ∪ block(m)} (the CAVI bound with β fixed:
    Σ_n c_n log φ-normalizer − KL(q(θ) || Dir(α))). Every gate has the same number of
    topics and the same α, so the bounds are comparable across members. Bayes over
    members with a temper τ.

Tempers (E1, E4) are fit on one half of the evaluation patients (log loss of the truth)
and every engine is scored on the OTHER half. E2/E3 are untempered.

Records (R0.1): attestation codes (members' and the group's own codes) and order-only
`[measured]` tokens are removed before any engine sees the record (D1, D2).

Reasons (R1.1): E1 — per present code, the member's responsibility-weighted log-odds of
presence minus the other members' mean; E4 — per code, c_n · (log φ-normalizer under
the member's gate − the mean over the other gates). Pooled per member over correctly
called patients, names only, suppressed under the egress floor.

Pure numpy over saved parameters; the driver (`gated_pc_readout --differential`) only
collects the patients."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import digamma, logsumexp

__all__ = ["filter_record", "PresenceSubtypeEngine", "GatedLikelihoodEngine",
           "MulticlassHead", "fit_tau_scores", "posterior_from_scores", "metrics",
           "pool_reasons", "EGRESS_FLOOR"]

EGRESS_FLOOR = 20


# ----------------------------------------------------------------------------- #
# records                                                                        #
# ----------------------------------------------------------------------------- #

def filter_record(bags, excl_by_dom):
    """Drop excluded vocab ids per domain. Returns `(bags, lengths)` with lengths = the
    number of distinct remaining codes per domain."""
    out, lens = {}, {}
    for d, b in bags.items():
        ex = excl_by_dom.get(d, ())
        kept = {int(v): float(c) for v, c in b.items() if int(v) not in ex}
        out[d] = kept
        lens[d] = float(len(kept))
    return out, lens


# ----------------------------------------------------------------------------- #
# E1 — presence model with subtype profiles                                      #
# ----------------------------------------------------------------------------- #

class PresenceSubtypeEngine:
    """`lams` per-domain λ (K, V_d); `subtypes` {member: [(topic, weight)]}; `n_bg`
    background topics (their evidence-weighted mean is β_bg); `evidence` (K,)."""

    def __init__(self, lams, subtypes, n_bg, evidence, members, excl_by_dom, *, rho=0.3):
        self.members = list(members)
        self.rho = float(rho)
        self.doms = list(range(len(lams)))
        ev = np.asarray(evidence, dtype=np.float64)
        bgw = ev[:n_bg] / ev[:n_bg].sum()
        self.sub = {}
        for m in self.members:
            ts = subtypes.get(m) or []
            if not ts:
                continue
            w = np.array([max(float(x), 0.0) for _, x in ts])
            w = w / w.sum() if w.sum() > 0 else np.full(len(ts), 1 / len(ts))
            self.sub[m] = ([int(t) for t, _ in ts], w)
        # E[β] rows only for the subtype topics (a full (K, V) copy is ~10^8 floats);
        # the background profile is the evidence-weighted mean of the background rows.
        used = sorted({t for ts, _ in self.sub.values() for t in ts})
        self.beta, self.bg = [], []
        for lam in lams:
            lam = np.asarray(lam)
            rs = lam.sum(axis=1)
            self.beta.append({t: np.asarray(lam[t], np.float64) / rs[t] for t in used})
            self.bg.append(bgw @ (np.asarray(lam[:n_bg], np.float64)
                                  / rs[:n_bg, None]))
        # per (subtype topic, domain): mixture mass on the excluded codes, so the
        # absent-code term is -L_d * (1 - excluded mass)
        self.excl_mass = {}
        for m, (ts, _) in self.sub.items():
            for t in ts:
                for d in self.doms:
                    ex = np.fromiter(excl_by_dom.get(d, ()), dtype=np.int64)
                    mix_ex = (self._mix(t, d, ex).sum() if ex.size else 0.0)
                    self.excl_mass[(t, d)] = float(mix_ex)

    def _mix(self, t, d, v):
        return self.rho * self.beta[d][t][v] + (1.0 - self.rho) * self.bg[d][v]

    def subtype_ll(self, bags, lengths):
        """{member: (ll per subtype array, per-subtype per-code log-odds dict)}."""
        out = {}
        for m, (ts, w) in self.sub.items():
            lls, odds = [], []
            for t in ts:
                ll = 0.0
                od = {}
                for d in self.doms:
                    L = float(lengths.get(d, 0.0))
                    if L <= 0:
                        continue
                    ll -= L * (1.0 - self.excl_mass[(t, d)])
                    vs = np.fromiter(bags.get(d, {}).keys(), dtype=np.int64)
                    if vs.size == 0:
                        continue
                    mx = self._mix(t, d, vs)
                    lo = np.log(np.clip(-np.expm1(-L * mx), 1e-300, 1.0)) + L * mx
                    ll += float(lo.sum())
                    for v, x in zip(vs.tolist(), lo.tolist()):
                        od[(d, v)] = x
                lls.append(ll)
                odds.append(od)
            out[m] = (np.array(lls), odds)
        return out

    def candidate_scores(self, bags, lengths, cand_items):
        """Member log-likelihoods after adding each candidate `(d, v)` once, and the
        engine's predictive P(yes) for it — vectorized per (subtype, domain).

        Adding v to domain d changes that domain's length L_d -> L_d + 1, which moves
        the absent-code term and every present code's log-odds in d; the other domains'
        terms are untouched. Under this engine a NO is already in the record (every
        absent code enters the likelihood), so only a YES changes the differential.
        Returns `(S (C, M) untempered member scores, p_yes (C,) per member (C, M))`."""
        cand_items = list(cand_items)
        C, M = len(cand_items), len(self.members)
        cd = np.array([d for d, _ in cand_items], dtype=np.int64)
        cv = np.array([v for _, v in cand_items], dtype=np.int64)
        S = np.zeros((C, M))
        P = np.zeros((C, M))
        mean_fill = []
        cur_all = self.subtype_ll(bags, lengths)
        for j, m in enumerate(self.members):
            if m not in self.sub:
                mean_fill.append(j)
                continue
            ts, w = self.sub[m]
            lls = np.zeros((len(ts), C))
            pys = np.zeros((len(ts), C))
            for a, t in enumerate(ts):
                base = 0.0
                dom_term = {}
                for d in self.doms:
                    L = float(lengths.get(d, 0.0))
                    vs = np.fromiter(bags.get(d, {}).keys(), dtype=np.int64)
                    term = lambda LL: (-LL * (1.0 - self.excl_mass[(t, d)])
                                       + (float((np.log(np.clip(-np.expm1(-LL * self._mix(t, d, vs)),
                                                                1e-300, 1.0))
                                                 + LL * self._mix(t, d, vs)).sum())
                                          if vs.size else 0.0))
                    cur = term(L) if L > 0 else 0.0
                    base += cur
                    dom_term[d] = (cur, term(L + 1.0), L + 1.0)
                for d in set(cd.tolist()):
                    sel = cd == d
                    cur, new, L1 = dom_term[d]
                    mx = self._mix(t, d, cv[sel])
                    lo = np.log(np.clip(-np.expm1(-L1 * mx), 1e-300, 1.0)) + L1 * mx
                    lls[a, sel] = base - cur + new + lo
                    pys[a, sel] = -np.expm1(-L1 * mx)
            S[:, j] = logsumexp(np.log(w)[:, None] + lls, axis=0)
            # predictive presence under the member: subtype responsibilities now
            cur_ll = cur_all[m][0]
            r = np.exp(np.log(w) + cur_ll - logsumexp(np.log(w) + cur_ll))
            P[:, j] = r @ pys
        if mean_fill:
            keep = [j for j in range(M) if j not in mean_fill]
            for j in mean_fill:
                S[:, j] = S[:, keep].mean(axis=1) if keep else 0.0
                P[:, j] = P[:, keep].mean(axis=1) if keep else 0.0
        return S, P

    def scores(self, sll):
        """Member log-likelihoods log Σ_t w_t exp(ll_t); members without a profile get
        the mean of the others (uninformative). τ is applied by the caller."""
        s = {}
        for m, (ll, _) in sll.items():
            _, w = self.sub[m]
            s[m] = float(logsumexp(np.log(w) + ll))
        mean = float(np.mean(list(s.values()))) if s else 0.0
        return np.array([s.get(m, mean) for m in self.members])

    def reasons(self, sll, member):
        """Per present code: the member's subtype-responsibility-weighted log-odds
        minus the other members' mean."""
        def member_odds(m):
            ll, odds = sll[m]
            _, w = self.sub[m]
            r = np.exp(np.log(w) + ll - logsumexp(np.log(w) + ll))
            acc = {}
            for rj, od in zip(r, odds):
                for k, x in od.items():
                    acc[k] = acc.get(k, 0.0) + rj * x
            return acc
        if member not in sll:
            return {}
        mine = member_odds(member)
        others = [member_odds(m) for m in sll if m != member]
        out = {}
        for k, x in mine.items():
            base = np.mean([o.get(k, 0.0) for o in others]) if others else 0.0
            out[k] = x - base
        return out


# ----------------------------------------------------------------------------- #
# E4 — gated per-disease likelihood                                              #
# ----------------------------------------------------------------------------- #

class GatedLikelihoodEngine:
    """`eb` the concatenated (K, V) exp(E[log β]); `alpha` (K,); `gates` {member:
    topic index array} = background ∪ block(member)."""

    def __init__(self, eb, alpha, gates, members, *, max_iter=100, tol=1e-3):
        self.eb, self.alpha = eb, np.asarray(alpha, dtype=np.float64)
        self.gates = {m: np.asarray(g, dtype=np.int64) for m, g in gates.items()}
        self.members = list(members)
        self.max_iter, self.tol = int(max_iter), float(tol)

    def elbos(self, idx, cnt):
        """(per-member ELBO array, {member: log φ-normalizer per code})."""
        from spark_vi.models.topic.lda import _cavi_doc_inference, _dirichlet_kl
        idx = np.asarray(idx, dtype=np.int64)
        cnt = np.asarray(cnt, dtype=np.float64)
        cols = self.eb[:, idx]                                     # (K, n)
        ar = np.arange(len(idx))
        out, lphi = [], {}
        for m in self.members:
            g = self.gates[m]
            a = self.alpha[g]
            if len(idx) == 0:
                out.append(0.0)
                lphi[m] = np.zeros(0)
                continue
            gamma, _, phi_norm, _ = _cavi_doc_inference(
                indices=ar, counts=cnt, expElogbeta=cols[g], alpha=a,
                gamma_init=a + 1.0, max_iter=self.max_iter, tol=self.tol)
            lp = np.log(phi_norm)
            out.append(float(cnt @ lp) - _dirichlet_kl(gamma, a))
            lphi[m] = lp
        return np.array(out), lphi

    def reasons(self, idx, cnt, lphi, member):
        others = [lphi[m] for m in self.members if m != member and len(lphi[m])]
        if not others or not len(lphi[member]):
            return {}
        base = np.mean(np.stack(others), axis=0)
        contrib = np.asarray(cnt) * (lphi[member] - base)
        return {int(i): float(c) for i, c in zip(idx, contrib)}


# ----------------------------------------------------------------------------- #
# E3 — multiclass head on log θ                                                   #
# ----------------------------------------------------------------------------- #

class MulticlassHead:
    """Softmax regression over the members on standardized log θ restricted to
    `feature_topics`, L2-penalized; λ chosen on an internal 80/20 split by log loss."""

    def __init__(self, feature_topics, n_classes, *, lambdas=(1e-3, 1e-2, 1e-1, 1.0),
                 seed=0):
        self.ft = np.asarray(feature_topics, dtype=np.int64)
        self.M = int(n_classes)
        self.lambdas = tuple(lambdas)
        self.seed = int(seed)

    def _feat(self, theta):
        X = np.log(np.clip(np.asarray(theta, dtype=np.float64)[:, self.ft], 1e-12, 1.0))
        return (X - self.mu) / self.sd

    @staticmethod
    def _fit(X, y, M, lam):
        n, p = X.shape
        Y = np.zeros((n, M)); Y[np.arange(n), y] = 1.0

        def f(wb):
            W = wb[:p * M].reshape(p, M); b = wb[p * M:]
            Z = X @ W + b
            lse = logsumexp(Z, axis=1, keepdims=True)
            P = np.exp(Z - lse)
            loss = -(Y * (Z - lse)).sum() / n + 0.5 * lam * (W * W).sum()
            G = (P - Y) / n
            gW = X.T @ G + lam * W
            gb = G.sum(axis=0)
            return loss, np.concatenate([gW.ravel(), gb])
        x0 = np.zeros(p * M + M)
        prior = np.bincount(y, minlength=M) + 1.0
        x0[p * M:] = np.log(prior / prior.sum())
        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        return res.x[:p * M].reshape(p, M), res.x[p * M:]

    def fit(self, theta, y):
        y = np.asarray(y, dtype=np.int64)
        Xr = np.log(np.clip(np.asarray(theta, dtype=np.float64)[:, self.ft], 1e-12, 1.0))
        self.mu = Xr.mean(axis=0)
        self.sd = Xr.std(axis=0) + 1e-6
        X = (Xr - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(y))
        cut = max(1, int(0.8 * len(y)))
        tr, va = perm[:cut], perm[cut:]
        best, best_ll = self.lambdas[0], np.inf
        if len(va) >= 5:
            for lam in self.lambdas:
                W, b = self._fit(X[tr], y[tr], self.M, lam)
                Z = X[va] @ W + b
                ll = -np.mean(Z[np.arange(len(va)), y[va]] - logsumexp(Z, axis=1))
                if ll < best_ll:
                    best, best_ll = lam, ll
        self.lam = best
        self.W, self.b = self._fit(X, y, self.M, best)
        return self

    def predict(self, theta):
        Z = self._feat(np.atleast_2d(theta)) @ self.W + self.b
        return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))


# ----------------------------------------------------------------------------- #
# tempering, posteriors, metrics                                                 #
# ----------------------------------------------------------------------------- #

def posterior_from_scores(log_prior, scores, tau):
    z = np.asarray(log_prior) + float(tau) * np.asarray(scores)
    return np.exp(z - logsumexp(z))


def fit_tau_scores(log_prior, score_list, truths, *, lo=-5.0, hi=0.5):
    """τ minimizing the mean log loss of the truth, searched on log10 τ in [lo, hi]."""
    def nll(lt):
        tau = 10.0 ** lt
        return -np.mean([np.log(max(posterior_from_scores(log_prior, s, tau)[t].sum(),
                                    1e-300))
                         for s, t in zip(score_list, truths)])
    r = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                        options={"xatol": 0.02})
    return float(10.0 ** r.x), float(r.fun)


def _ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def metrics(posts, truths):
    """Pooled: top-1, top-3, balanced top-1 (single-positive classes), log loss, ECE."""
    posts = [np.asarray(p, dtype=np.float64) for p in posts]
    top1 = np.array([int(np.argmax(p)) in t for p, t in zip(posts, truths)])
    top3 = np.array([bool(set(np.argsort(p)[::-1][:3].tolist()) & set(t))
                     for p, t in zip(posts, truths)])
    ll = np.array([-np.log(max(p[t].sum(), 1e-300)) for p, t in zip(posts, truths)])
    by = {}
    for t, c in zip(truths, top1):
        if len(t) == 1:
            by.setdefault(t[0], []).append(c)
    bal = float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")
    return {"n": len(posts), "top1": float(top1.mean()), "top3": float(top3.mean()),
            "bal_top1": bal, "logloss": float(ll.mean()),
            "ece": _ece([p.max() for p in posts], top1)}


def pool_reasons(per_patient, *, floor=EGRESS_FLOOR, top=5):
    """`per_patient` = [(member, {code: contribution})] for correctly called patients.
    Returns {member: [codes]} by mean contribution, members with >= floor patients."""
    acc, n = {}, {}
    for m, rs in per_patient:
        n[m] = n.get(m, 0) + 1
        a = acc.setdefault(m, {})
        for k, x in rs.items():
            a[k] = a.get(k, 0.0) + x
    out = {}
    for m, a in acc.items():
        if n[m] < floor:
            continue
        ranked = sorted(a.items(), key=lambda kv: -kv[1] / n[m])
        out[m] = [k for k, x in ranked[:top] if x > 0]
    return out


# ----------------------------------------------------------------------------- #
# E3 reasons — attribution through θ                                              #
# ----------------------------------------------------------------------------- #

def e3_reasons(head, theta, gamma_sum, eb_cols, cnt, member):
    """Per record code, its contribution to the member's logit relative to the other
    members' mean, through the fold-in: code v moves θ_k by c_v r_kv / sum(γ) with
    r_kv = θ_k eb_kv / Σ_j θ_j eb_jv, and the head's logit moves by W[k, m] / (sd_k θ_k)
    per unit θ_k (features are standardized log θ). θ_k cancels:

        contrib_v(m) = c_v / sum(γ) · Σ_{k in features} (W[k, m] / sd_k) · eb_kv / Σ_j θ_j eb_jv

    A first-order (gradient) attribution — cheap, and exact in the small-change limit.
    `eb_cols` is the (K, n) exp(E[log β]) at the record's codes; returns (n,)."""
    theta = np.asarray(theta, dtype=np.float64)
    norm = theta @ eb_cols + 1e-100                                # (n,)
    Wd = head.W / head.sd[:, None]                                 # (F, M)
    G = (Wd.T @ eb_cols[head.ft]) / norm                           # (M, n)
    G = G - G.mean(axis=0, keepdims=True)
    return np.asarray(cnt, dtype=np.float64) / float(gamma_sum) * G[member]


# ----------------------------------------------------------------------------- #
# Framing 2 — next question                                                      #
# ----------------------------------------------------------------------------- #

def hide_split(fb, frac, rng, *, min_codes=5):
    """Hide a random `frac` of a filtered record's codes. Returns (visible bags,
    hidden {(d, v): count}) or None for a record with fewer than `min_codes` codes."""
    items = [(d, v, c) for d, b in fb.items() for v, c in b.items()]
    if len(items) < int(min_codes):
        return None
    n_h = max(1, int(round(frac * len(items))))
    h = set(rng.choice(len(items), size=n_h, replace=False).tolist())
    vis, hid = {}, {}
    for i, (d, v, c) in enumerate(items):
        if i in h:
            hid[(d, v)] = c
        else:
            vis.setdefault(d, {})[v] = c
    return vis, hid


def kl_rows(Q, q):
    Q = np.clip(np.atleast_2d(Q), 1e-300, 1.0)
    q = np.clip(np.asarray(q), 1e-300, 1.0)
    return (Q * (np.log(Q) - np.log(q[None, :]))).sum(axis=1)


def question_values(q0, q_after, p_yes):
    """Value of asking each candidate = P(yes) · KL(q after yes || q now), nats. A NO
    is not valued: an absent code is not a negative in this data (spec D2)."""
    return np.asarray(p_yes) * kl_rows(q_after, q0)


NQ_STRATEGIES = ("VOI (E3)", "VOI (E1)", "likely present", "population table",
                 "random")


def evaluate_next_question(eng, head, patients, pool_items, flagged, rng, *,
                           ks=(1, 3, 5), frac=0.2, e1=None, e1_lp=None, e1_tau=1.0):
    """Spec R2.3 — positive-unlabeled-aware evaluation of "what to ask next".

    For each patient (`(truth idx list, filtered bags)`), hide a random `frac` of the
    record's codes; on the visible rest, rank the pool's not-yet-visible candidates —
    VOI under E3 (P(yes) from the topic model's predictive presence · KL of the E3
    differential after a yes), VOI under E1 (the presence-subtype engine's own
    predictive P(yes) · KL of its tempered differential; a NO is already in E1's
    record, so only a yes moves it), most likely present (E3's P(yes) alone), the
    population EIG table order, and random — and for the top k of each:
      * yield = the share among the hidden codes (known YES answers); absent
        candidates are unlabeled and never scored as NO;
      * the differential after revealing the ones that were hidden — P(true) and top-1
        under BOTH E3 and E1 —
    against revealing k random hidden codes (an oracle that reveals only real facts)
    and all of them (the full record). `flagged` maps pool index -> True for likely
    treatments / diagnostic consequences (R2.2). Pooled numbers only, plus the VOI (E3)
    top-3 recommendation counts {pool index: n patients}."""
    pool_g = np.array([eng.gidx(d, v) for d, v in pool_items], dtype=np.int64)
    pool_dom = np.array([d for d, _ in pool_items], dtype=np.int64)
    Eb_pool = eng.ebeta_cols(pool_items)
    ks = tuple(sorted(int(k) for k in ks))
    strategies = NQ_STRATEGIES if e1 is not None else tuple(
        x for x in NQ_STRATEGIES if x != "VOI (E1)")
    fields = ("yield", "p_true", "top1", "p_true_e1", "top1_e1")
    acc = {(s_, k): {f: [] for f in fields}
           for s_ in strategies + ("random hidden",) for k in ks}
    base = {f: [] for f in fields[1:]}
    full = {f: [] for f in fields[1:]}
    rec, n_flag, n_top = {}, 0, 0

    def post(bags):
        idx = [eng.gidx(d, v) for d, b in bags.items() for v in b]
        cnt = [c for b in bags.values() for c in b.values()]
        g = eng.fold_in(idx, cnt)
        return head.predict(g / g.sum())[0], g, idx, cnt

    def post1(bags):
        if e1 is None:
            return None
        lens = {d: float(len(b)) for d, b in bags.items()}
        return posterior_from_scores(e1_lp, e1.scores(e1.subtype_ll(bags, lens)), e1_tau)

    def score(store, q, q1, truth):
        store["p_true"].append(float(q[truth].sum()))
        store["top1"].append(int(np.argmax(q)) in truth)
        if q1 is not None:
            store["p_true_e1"].append(float(q1[truth].sum()))
            store["top1_e1"].append(int(np.argmax(q1)) in truth)

    for truth, fb in patients:
        sp = hide_split(fb, frac, rng)
        if sp is None:
            continue
        vis, hid = sp
        q0, g0, vidx, vcnt = post(vis)
        q10 = post1(vis)
        score(base, q0, q10, truth)
        allb = {d: dict(b) for d, b in vis.items()}
        for (d, v), c in hid.items():
            allb.setdefault(d, {})[v] = c
        score(full, post(allb)[0], post1(allb), truth)
        cand = np.array([j for j, (d, v) in enumerate(pool_items)
                         if v not in vis.get(d, {})], dtype=np.int64)
        if cand.size == 0:
            continue
        theta0 = g0 / g0.sum()
        qa = head.predict(eng.candidate_thetas(g0, vidx, vcnt, pool_g[cand]))
        L = np.array([float(len(vis.get(int(d), {}))) + 1.0 for d in pool_dom[cand]])
        p_yes = 1.0 - np.exp(-L * (theta0 @ Eb_pool[:, cand]))
        orders = {"VOI (E3)": cand[np.argsort(-question_values(q0, qa, p_yes),
                                              kind="stable")],
                  "likely present": cand[np.argsort(-p_yes, kind="stable")],
                  "population table": cand,
                  "random": cand[rng.permutation(cand.size)]}
        if e1 is not None:
            lens = {d: float(len(b)) for d, b in vis.items()}
            S, Pm = e1.candidate_scores(vis, lens,
                                        [pool_items[int(j)] for j in cand])
            z = e1_lp[None, :] + e1_tau * S
            q1a = np.exp(z - logsumexp(z, axis=1, keepdims=True))
            py1 = (Pm * q10[None, :]).sum(axis=1)
            orders["VOI (E1)"] = cand[np.argsort(-question_values(q10, q1a, py1),
                                                 kind="stable")]
        for j in orders["VOI (E3)"][:3]:
            rec[int(j)] = rec.get(int(j), 0) + 1
            n_flag += bool(flagged.get(int(j), False))
            n_top += 1
        hid_keys = list(hid)
        for k in ks:
            for s_ in strategies:
                picks = [pool_items[int(j)] for j in orders[s_][:k]]
                got = [it for it in picks if it in hid]
                b = {d: dict(x) for d, x in vis.items()}
                for (d, v) in got:
                    b.setdefault(d, {})[v] = hid[(d, v)]
                q, q1 = (post(b)[0], post1(b)) if got else (q0, q10)
                a = acc[(s_, k)]
                a["yield"].append(len(got) / k)
                score(a, q, q1, truth)
            sel = rng.choice(len(hid_keys), size=min(k, len(hid_keys)), replace=False)
            b = {d: dict(x) for d, x in vis.items()}
            for i in sel:
                d, v = hid_keys[int(i)]
                b.setdefault(d, {})[v] = hid[(d, v)]
            a = acc[("random hidden", k)]
            a["yield"].append(1.0)
            score(a, post(b)[0], post1(b), truth)
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"n": len(base["p_true"]), "ks": list(ks), "frac_hidden": float(frac),
            "strategies": list(strategies),
            "visible": {k: mean(v) for k, v in base.items()},
            "full": {k: mean(v) for k, v in full.items()},
            "by": {f"{s_}@{k}": {m: mean(v) for m, v in a.items()}
                   for (s_, k), a in acc.items()},
            "flag_share_top3": (n_flag / n_top) if n_top else float("nan"),
            "recommended": rec}
