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
