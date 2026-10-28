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

def label_matrix(y, M):
    """(n, M) target rows. `y` is either member indices (one-hot) or truth lists — a
    patient coded with several members spreads unit weight evenly over them (exp 0149:
    the multi-positive patients join training as soft targets instead of being set
    aside; the softmax then reads as "which member, given it is one of these")."""
    if len(y) and np.ndim(y[0]) == 0:
        y = np.asarray(y, dtype=np.int64)
        Y = np.zeros((len(y), M)); Y[np.arange(len(y)), y] = 1.0
        return Y
    Y = np.zeros((len(y), M))
    for i, t in enumerate(y):
        t = [int(x) for x in t]
        Y[i, t] = 1.0 / len(t)
    return Y


def _select_and_fit(X, Y, M, lambdas, seed):
    """λ by log loss on an internal 80/20 split, then the fit on everything."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Y))
    cut = max(1, int(0.8 * len(Y)))
    tr, va = perm[:cut], perm[cut:]
    best, best_ll = lambdas[0], np.inf
    if len(va) >= 5:
        for lam in lambdas:
            W, b = MulticlassHead._fit(X[tr], Y[tr], M, lam)
            Z = X[va] @ W + b
            ll = -np.mean((Y[va] * (Z - logsumexp(Z, axis=1, keepdims=True))).sum(axis=1))
            if ll < best_ll:
                best, best_ll = lam, ll
    W, b = MulticlassHead._fit(X, Y, M, best)
    return best, W, b


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
        Y = np.asarray(y, dtype=np.float64) if np.ndim(y) == 2 else label_matrix(y, M)

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
        prior = Y.sum(axis=0) + 1.0
        x0[p * M:] = np.log(prior / prior.sum())
        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        return res.x[:p * M].reshape(p, M), res.x[p * M:]

    def fit(self, theta, y):
        """`y`: member indices, or truth lists (soft targets; `label_matrix`)."""
        Y = label_matrix(y, self.M)
        Xr = np.log(np.clip(np.asarray(theta, dtype=np.float64)[:, self.ft], 1e-12, 1.0))
        self.mu = Xr.mean(axis=0)
        self.sd = Xr.std(axis=0) + 1e-6
        X = (Xr - self.mu) / self.sd
        self.lam, self.W, self.b = _select_and_fit(X, Y, self.M, self.lambdas,
                                                   self.seed)
        return self

    def predict(self, theta):
        Z = self._feat(np.atleast_2d(theta)) @ self.W + self.b
        return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))


class CodeHead:
    """E5 / E6 (exp 0148): softmax over the members on the RECORD'S CODES — binary
    presence of the tokens carried by >= `min_df` training patients (top `max_codes`
    by document frequency), plus log(1 + distinct codes) — and, for E6, standardized log θ
    over `theta_topics` as well. Same records (attestation and [measured] removed),
    same L2 softmax and λ selection as E3; only the input differs. Insight 0101: on new
    diagnoses 2,000 binary codes beat θ by ~0.04, most for the rarest diseases.

    Why codes make framing 2 exact: asking "is v present?" flips one input, so the
    differential after a YES is a forward pass, not a re-fold of θ (E6 still re-folds
    θ through `candidate_thetas`). P(yes) for a candidate is the training patients'
    smoothed per-member presence rate mixed by the current differential."""

    def __init__(self, n_classes, *, min_df=5, max_codes=3000, theta_topics=None,
                 lambdas=(1e-2, 1e-1, 1.0, 10.0), seed=0, select="df", rel_lambda=0.6):
        if select not in ("df", "relevance"):
            raise ValueError(f"select must be 'df' or 'relevance', not {select!r}")
        self.M = int(n_classes)
        self.min_df, self.max_codes = int(min_df), int(max_codes)
        self.select, self.rel_lambda = select, float(rel_lambda)
        self.ft = (None if theta_topics is None
                   else np.asarray(theta_topics, dtype=np.int64))
        self.lambdas, self.seed = tuple(lambdas), int(seed)

    @staticmethod
    def _items(bags):
        return [(int(d), int(v)) for d, b in bags.items() for v in b]

    def _raw(self, bags_list, thetas=None):
        X = np.zeros((len(bags_list), self.p_codes + 1 + (0 if self.ft is None
                                                         else len(self.ft))))
        for i, b in enumerate(bags_list):
            its = self._items(b)
            for it in its:
                j = self.pos.get(it)
                if j is not None:
                    X[i, j] = 1.0
            X[i, self.p_codes] = np.log1p(len(its))
        if self.ft is not None:
            X[:, self.p_codes + 1:] = np.log(np.clip(
                np.asarray(thetas, dtype=np.float64)[:, self.ft], 1e-12, 1.0))
        return X

    def _std(self, X):
        return (X - self.mu) / self.sd

    def fit(self, bags_list, y, thetas=None):
        """`y`: member indices, or truth lists (soft targets; `label_matrix`)."""
        Y = label_matrix(y, self.M)
        df, per = {}, {}
        n_m = Y.sum(axis=0)
        for b, yr in zip(bags_list, Y):
            for it in set(self._items(b)):
                df[it] = df.get(it, 0) + 1
                per.setdefault(it, np.zeros(self.M))[:] += yr
        if self.select == "relevance":
            keep = self._select_relevance(df, per, n_m, len(bags_list))
        else:
            keep = sorted((it for it, n in df.items() if n >= self.min_df),
                          key=lambda it: (-df[it], it))[:self.max_codes]
        self.codes = sorted(keep)
        self.pos = {it: j for j, it in enumerate(self.codes)}
        self.p_codes = len(self.codes)
        self.rate = {it: (v + 0.5) / (n_m + 1.0) for it, v in per.items()}
        self.rate_absent = 0.5 / (n_m + 1.0)
        Xr = self._raw(bags_list, thetas)
        self.mu, self.sd = Xr.mean(axis=0), Xr.std(axis=0) + 1e-6
        X = self._std(Xr)
        self.lam, self.W, self.b = _select_and_fit(X, Y, self.M, self.lambdas,
                                                   self.seed)
        return self

    def _select_relevance(self, df, per, n_m, n):
        """Exp 0166: codes by WITHIN-GROUP relevance instead of group document frequency.

        Why: the head separates the members, so a code common to all of them (a CBC, an
        office visit) fills a slot and separates nothing; what separates is a code's lift
        for one member over the group, P(v | m) / P(v | group). Pure lift favours codes
        too rare to estimate, so it is mixed with frequency as in the per-node readout
        (LDAvis relevance, insight 0101):

            rel_m(v) = λ log P(v | m) + (1 − λ) log [ P(v | m) / P(v | group) ]

        with add-half smoothed rates. λ does NOT carry over from the readout: within a
        group the lift for m is capped at 1 / P(m | group) (~2 for two equal members),
        where corpus-wide lift reaches 100x, so the log-frequency term dominates sooner.
        A marker on 40% of m's patients (lift 2) outranks a code on 95% of everyone only
        for λ < ~0.45; one on 10% only for λ < ~0.24. Hence the λ sweep in exp 0166.
        Support is counted WITHIN the member (>= `min_df`
        of its training patients, soft counts for multi-positive targets), so a code's
        lift for m never rests on one or two of m's patients. Members take their top
        codes in turn (round robin) up to `max_codes`, so a large member cannot crowd a
        small one out. Training rows only, as for `df` selection: fold-safe."""
        lam = self.rel_lambda
        ranked = []
        for m in range(self.M):
            if n_m[m] <= 0:
                continue
            sc = []
            for it, row in per.items():
                if row[m] < self.min_df:
                    continue
                pm = (row[m] + 0.5) / (n_m[m] + 1.0)
                pg = (df[it] + 0.5) / (n + 1.0)
                sc.append((-(lam * np.log(pm) + (1.0 - lam) * np.log(pm / pg)), it))
            sc.sort()
            ranked.append([it for _, it in sc])
        keep, seen = [], set()
        for r in range(max((len(x) for x in ranked), default=0)):
            for lst in ranked:
                if r < len(lst) and lst[r] not in seen:
                    seen.add(lst[r])
                    keep.append(lst[r])
                    if len(keep) >= self.max_codes:
                        return keep
        return keep

    def _softmax(self, Xr):
        Z = self._std(Xr) @ self.W + self.b
        return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))

    def predict(self, bags_list, thetas=None):
        return self._softmax(self._raw(bags_list, thetas))

    def candidates(self, bags, cand_items, theta_rows=None):
        """(n_cand, M): the differential if each candidate were added (present once)."""
        x0 = self._raw([bags], None if self.ft is None else
                       np.asarray(theta_rows)[:1])[0]
        X = np.tile(x0, (len(cand_items), 1))
        X[:, self.p_codes] = np.log1p(len(self._items(bags)) + 1)
        for i, it in enumerate(cand_items):
            j = self.pos.get((int(it[0]), int(it[1])))
            if j is not None:
                X[i, j] = 1.0
        if self.ft is not None:
            X[:, self.p_codes + 1:] = np.log(np.clip(
                np.asarray(theta_rows, dtype=np.float64)[:, self.ft], 1e-12, 1.0))
        return self._softmax(X)

    def p_yes(self, q, cand_items):
        """P(candidate present) = sum_m q_m P(v | m), training rates add-half smoothed."""
        R = np.stack([self.rate.get((int(d), int(v)), self.rate_absent)
                      for d, v in cand_items])
        return R @ np.asarray(q, dtype=np.float64)

    def reasons(self, bags, member):
        """{(d, v): contribution of the code's presence to the member's logit relative to
        the members' mean} — exact for the linear head (presence 0 -> 1 moves the
        standardized feature by 1/sd)."""
        out = {}
        for it in self._items(bags):
            j = self.pos.get(it)
            if j is not None:
                w = self.W[j]
                out[it] = float((w[member] - w.mean()) / self.sd[j])
        return out


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


def evaluate_next_question_codes(code_head, patients, pool_items, rng, *, ks=(1, 3, 5),
                                frac=0.2):
    """Framing 2 with the CODE engine alone (exp 0150 survey): no θ, so no fold-in per
    candidate — the differential after a YES is one forward pass (`CodeHead.candidates`)
    and P(yes) the head's per-member presence rates. Same protocol and metrics as
    `evaluate_next_question` (hide `frac`, yield, P(true), top-1), strategies VOI
    (codes), population table, random, and random hidden. A pure-codes head (E5) is
    expected: an E6 head would need θ."""
    ks = tuple(sorted(int(k) for k in ks))
    strategies = ("VOI (codes)", "population table", "random")
    acc = {(s_, k): {"yield": [], "p_true": [], "top1": []}
           for s_ in strategies + ("random hidden",) for k in ks}
    base, full = {"p_true": [], "top1": []}, {"p_true": [], "top1": []}

    def score(store, q, truth):
        store["p_true"].append(float(q[truth].sum()))
        store["top1"].append(int(np.argmax(q)) in truth)

    for truth, fb in patients:
        sp = hide_split(fb, frac, rng)
        if sp is None:
            continue
        vis, hid = sp
        q0 = code_head.predict([vis])[0]
        score(base, q0, truth)
        allb = {d: dict(b) for d, b in vis.items()}
        for (d, v), c in hid.items():
            allb.setdefault(d, {})[v] = c
        score(full, code_head.predict([allb])[0], truth)
        cand = np.array([j for j, (d, v) in enumerate(pool_items)
                         if v not in vis.get(d, {})], dtype=np.int64)
        if cand.size == 0:
            continue
        items = [pool_items[int(j)] for j in cand]
        qa = code_head.candidates(vis, items)
        py = code_head.p_yes(q0, items)
        orders = {"VOI (codes)": cand[np.argsort(-question_values(q0, qa, py),
                                                 kind="stable")],
                  "population table": cand,
                  "random": cand[rng.permutation(cand.size)]}
        hid_keys = list(hid)
        for k in ks:
            for s_ in strategies:
                picks = [pool_items[int(j)] for j in orders[s_][:k]]
                got = [it for it in picks if it in hid]
                b = {d: dict(x) for d, x in vis.items()}
                for (d, v) in got:
                    b.setdefault(d, {})[v] = hid[(d, v)]
                a = acc[(s_, k)]
                a["yield"].append(len(got) / k)
                score(a, code_head.predict([b])[0] if got else q0, truth)
            sel = rng.choice(len(hid_keys), size=min(k, len(hid_keys)), replace=False)
            b = {d: dict(x) for d, x in vis.items()}
            for i in sel:
                d, v = hid_keys[int(i)]
                b.setdefault(d, {})[v] = hid[(d, v)]
            a = acc[("random hidden", k)]
            a["yield"].append(1.0)
            score(a, code_head.predict([b])[0], truth)
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"n": len(base["p_true"]), "ks": list(ks), "frac_hidden": float(frac),
            "strategies": list(strategies),
            "visible": {k: mean(v) for k, v in base.items()},
            "full": {k: mean(v) for k, v in full.items()},
            "by": {f"{s_}@{k}": {m: mean(v) for m, v in a.items()}
                   for (s_, k), a in acc.items()}}


NQ_STRATEGIES = ("VOI (E3)", "VOI (E1)", "VOI (codes)", "likely present",
                 "population table", "random")


def evaluate_next_question(eng, head, patients, pool_items, flagged, rng, *,
                           ks=(1, 3, 5), frac=0.2, e1=None, e1_lp=None, e1_tau=1.0,
                           code_head=None):
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
    top-3 recommendation counts {pool index: n patients}.

    `code_head` (exp 0148; a `CodeHead`, E5 or E6): adds "VOI (codes)" — P(yes) from the
    head's per-member presence rates, the differential after a YES by flipping that
    input — and scores every strategy's reveals under the code differential too
    (`p_true_c`, `top1_c`); its top-3 counts are `recommended_codes`."""
    pool_g = np.array([eng.gidx(d, v) for d, v in pool_items], dtype=np.int64)
    pool_dom = np.array([d for d, _ in pool_items], dtype=np.int64)
    Eb_pool = eng.ebeta_cols(pool_items)
    ks = tuple(sorted(int(k) for k in ks))
    strategies = tuple(x for x in NQ_STRATEGIES
                       if (x != "VOI (E1)" or e1 is not None)
                       and (x != "VOI (codes)" or code_head is not None))
    c_theta = code_head is not None and code_head.ft is not None
    fields = ("yield", "p_true", "top1", "p_true_e1", "top1_e1", "p_true_c", "top1_c")
    acc = {(s_, k): {f: [] for f in fields}
           for s_ in strategies + ("random hidden",) for k in ks}
    base = {f: [] for f in fields[1:]}
    full = {f: [] for f in fields[1:]}
    rec, rec_c, n_flag, n_top = {}, {}, 0, 0

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

    def postc(bags, g):
        if code_head is None:
            return None
        th = (g / g.sum())[None, :] if c_theta else None
        return code_head.predict([bags], th)[0]

    def score(store, q, q1, truth, qc=None):
        store["p_true"].append(float(q[truth].sum()))
        store["top1"].append(int(np.argmax(q)) in truth)
        if q1 is not None:
            store["p_true_e1"].append(float(q1[truth].sum()))
            store["top1_e1"].append(int(np.argmax(q1)) in truth)
        if qc is not None:
            store["p_true_c"].append(float(qc[truth].sum()))
            store["top1_c"].append(int(np.argmax(qc)) in truth)

    def post_all(bags):
        q, g, _i, _c = post(bags)
        return q, post1(bags), postc(bags, g)

    for truth, fb in patients:
        sp = hide_split(fb, frac, rng)
        if sp is None:
            continue
        vis, hid = sp
        q0, g0, vidx, vcnt = post(vis)
        q10 = post1(vis)
        qc0 = postc(vis, g0)
        score(base, q0, q10, truth, qc0)
        allb = {d: dict(b) for d, b in vis.items()}
        for (d, v), c in hid.items():
            allb.setdefault(d, {})[v] = c
        qf, qf1, qfc = post_all(allb)
        score(full, qf, qf1, truth, qfc)
        cand = np.array([j for j, (d, v) in enumerate(pool_items)
                         if v not in vis.get(d, {})], dtype=np.int64)
        if cand.size == 0:
            continue
        theta0 = g0 / g0.sum()
        cth = eng.candidate_thetas(g0, vidx, vcnt, pool_g[cand])
        qa = head.predict(cth)
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
        if code_head is not None:
            items_c = [pool_items[int(j)] for j in cand]
            qca = code_head.candidates(vis, items_c, cth if c_theta else None)
            pyc = code_head.p_yes(qc0, items_c)
            orders["VOI (codes)"] = cand[np.argsort(-question_values(qc0, qca, pyc),
                                                    kind="stable")]
            for j in orders["VOI (codes)"][:3]:
                rec_c[int(j)] = rec_c.get(int(j), 0) + 1
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
                q, q1, qc = post_all(b) if got else (q0, q10, qc0)
                a = acc[(s_, k)]
                a["yield"].append(len(got) / k)
                score(a, q, q1, truth, qc)
            sel = rng.choice(len(hid_keys), size=min(k, len(hid_keys)), replace=False)
            b = {d: dict(x) for d, x in vis.items()}
            for i in sel:
                d, v = hid_keys[int(i)]
                b.setdefault(d, {})[v] = hid[(d, v)]
            a = acc[("random hidden", k)]
            a["yield"].append(1.0)
            q, q1, qc = post_all(b)
            score(a, q, q1, truth, qc)
    mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"n": len(base["p_true"]), "ks": list(ks), "frac_hidden": float(frac),
            "strategies": list(strategies),
            "visible": {k: mean(v) for k, v in base.items()},
            "full": {k: mean(v) for k, v in full.items()},
            "by": {f"{s_}@{k}": {m: mean(v) for m, v in a.items()}
                   for (s_, k), a in acc.items()},
            "flag_share_top3": (n_flag / n_top) if n_top else float("nan"),
            "recommended": rec, "recommended_codes": rec_c}


# --------------------------------------------------------------------------- #
# Framing 2 as WORKUP (spec 2026-10-28 workup-voi-design; exp 0167).          #
# --------------------------------------------------------------------------- #
# A candidate FACT is a condition / drug code (answer: recorded in the workup
# window or not) or a measurement concept (answer: not done, done without a result
# state, normal / negative, low, high / positive / abnormal). Every answer — the
# default ones included — is evidence through P(answer | member): "not recorded" is
# a recording outcome, informative exactly as far as recording rates differ between
# members. The belief update is naive Bayes over the answers given, tempered by tau.

WK_MAX_A = 5                          # the widest answer space (a measurement)
WK_DONE, WK_NORMAL, WK_LOW, WK_HIGH = 1, 2, 3, 4
# measurement_tokens state codes -> answer category (unlisted states count as done)
_WK_STATE = {0: WK_DONE, 2: WK_NORMAL, 10: WK_NORMAL, 12: WK_NORMAL,
             1: WK_LOW, 14: WK_LOW, 3: WK_HIGH, 11: WK_HIGH, 13: WK_HIGH,
             15: WK_HIGH}
# when a window holds several states of one test, the most abnormal wins
_WK_RANK = {WK_DONE: 0, WK_NORMAL: 1, WK_LOW: 2, WK_HIGH: 3}


def workup_answers(tokens, *, meas_dom=None, vocab_sets=None, excl=None,
                   token_base=100, drop_done=False):
    """`{fact key: answer category}` for one window's `(dom, token)` pairs; facts not
    present keep their default answer (0) implicitly.

    Keys: `("c", dom, token)` for a condition / drug code (category 1 = recorded),
    `("m", concept)` for a measurement concept (the `_WK_STATE` categories, most
    abnormal winning). `vocab_sets[dom]` restricts each domain to the corpus
    vocabulary (a measurement token's concept is kept when the token is in it);
    `excl[dom]` drops attestation codes. `drop_done` maps "done, no result" to the
    default (the spec's sensitivity run without order-only answers)."""
    out = {}
    for d, tok in tokens:
        d, tok = int(d), int(tok)
        if vocab_sets is not None and tok not in vocab_sets.get(d, ()):
            continue
        if excl is not None and tok in excl.get(d, ()):
            continue
        if meas_dom is not None and d == int(meas_dom):
            concept, state = divmod(tok, int(token_base))
            a = _WK_STATE.get(state, WK_DONE)
            if drop_done and a == WK_DONE:
                continue
            k = ("m", concept)
            if k not in out or _WK_RANK[a] > _WK_RANK[out[k]]:
                out[k] = a
        else:
            out[("c", d, tok)] = 1
    return out


def known_facts(bags, inv_maps, *, meas_dom=None, token_base=100):
    """The facts a patient's PRE-index record already carries, as workup fact keys:
    `("c", dom, token)` for a condition / drug code, `("m", concept)` for a test of
    any result. Exp 0167 run 2: only NEW facts are asked about, because an answer the
    record already holds was used to form q0 and is not independent evidence."""
    out = set()
    for d, b in bags.items():
        inv = inv_maps[d] if inv_maps and d < len(inv_maps) else {}
        for v in b:
            tok = inv.get(int(v))
            if tok is None:
                continue
            if meas_dom is not None and int(d) == int(meas_dom):
                out.add(("m", int(tok) // int(token_base)))
            else:
                out.add(("c", int(d), int(tok)))
    return out


def workup_candidates(train_answers, *, floor=EGRESS_FLOOR, known=None):
    """Facts whose non-default answer occurs for at least `floor` training patients
    for whom the fact was new (`known[i]`, the patient's pre-index facts; None = all
    new): `[(key, n_answers)]`, sorted for determinism."""
    cnt = {}
    for i, ans in enumerate(train_answers):
        kn = known[i] if known is not None else ()
        for k in ans:
            if k not in kn:
                cnt[k] = cnt.get(k, 0) + 1
    return [(k, WK_MAX_A if k[0] == "m" else 2)
            for k in sorted(cnt, key=repr) if cnt[k] >= int(floor)]


def answer_likelihoods(train_answers, Y, cands, *, smooth=0.5, known=None,
                       prior=None, prior_strength=5.0):
    """(n_cand, WK_MAX_A, M) `P(answer | member)` over the training patients for whom
    each fact was NEW (`known`), zero on the categories a fact does not have. `Y`
    (n, M) is the training patients' (possibly soft) member matrix.

    Smoothing: add-`smooth` per category, or, given `prior` (an L of the same shape),
    a Dirichlet centred on it with `prior_strength` pseudo-patients (the window-length
    bins shrink to the pooled likelihood)."""
    Y = np.asarray(Y, dtype=np.float64)
    M = Y.shape[1]
    idx = {k: i for i, (k, _) in enumerate(cands)}
    counts = np.zeros((len(cands), WK_MAX_A, M))
    n_el = np.tile(Y.sum(axis=0), (len(cands), 1))          # eligible patients
    for r, (ans, y) in enumerate(zip(train_answers, Y)):
        if known is not None:
            for k in known[r]:
                i = idx.get(k)
                if i is not None:
                    n_el[i] -= y
        kn = known[r] if known is not None else ()
        for k, a in ans.items():
            i = idx.get(k)
            if i is not None and k not in kn:
                counts[i, a] += y
    for i, (_, A) in enumerate(cands):
        counts[i, 0] = n_el[i] - counts[i, 1:A].sum(axis=0)  # the default answer
    L = np.zeros_like(counts)
    for i, (_, A) in enumerate(cands):
        c = counts[i, :A] + (smooth if prior is None
                             else prior_strength * prior[i, :A])
        L[i, :A] = c / c.sum(axis=0, keepdims=True)
    return L


def length_bin(days, edges):
    """Index of the window-length bin `days` falls in (`edges` ascending)."""
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(days),
                               side="right"))


def answer_vector(ans, cands):
    """(n_cand,) int: each candidate's answer for one patient (0 = default)."""
    return np.array([ans.get(k, 0) for k, _ in cands], dtype=np.int64)


def eligible_vector(known, cands):
    """(n_cand,) bool: the candidates that are new for one patient."""
    return np.array([k not in known for k, _ in cands], dtype=bool)


def _entropy_nats(P, axis=-1):
    P = np.clip(P, 1e-300, 1.0)
    return -(P * np.log(P)).sum(axis=axis)


def workup_posteriors(q0, L, tau=1.0):
    """(n_cand, WK_MAX_A, M) posteriors after each answer of each fact,
    `q(m | a) ∝ q0(m) P(a | m)^tau`, and (n_cand, WK_MAX_A) predictive `P(a)` under
    q0 (untempered). Categories a fact does not have get P(a) = 0."""
    q0 = np.asarray(q0, dtype=np.float64)
    Lt = np.where(L > 0, np.power(np.clip(L, 1e-300, None), float(tau)), 0.0)
    post = q0[None, None, :] * Lt
    z = post.sum(axis=-1, keepdims=True)
    post = np.divide(post, z, out=np.zeros_like(post), where=z > 0)
    return post, L @ q0


def workup_eig(q0, post, pa):
    """(n_cand,) expected information gain, nats: H(q0) − sum_a P(a) H(q(·|a)) — the
    expected KL from q0 to the belief after the answer, i.e. how far the answer is
    expected to move the differential in any direction."""
    h = _entropy_nats(np.asarray(q0, dtype=np.float64))
    return h - (pa * _entropy_nats(post)).sum(axis=1)


def _per_patient(x, n):
    """`x` given once for all patients, or once per patient: a length-n list."""
    if isinstance(x, (list, tuple)) and len(x) == n:
        return list(x)
    return [x] * n


def fit_workup_tau(q0s, truths, answers, L, *, eligible=None, lo=0.05, hi=3.0):
    """The temper maximizing the mean log posterior of the truth after one fact,
    over every eligible (patient, candidate) pair of the fitting half: it absorbs the
    naive-Bayes independence assumption (answers are not independent of the record
    that made q0). `L` one likelihood array or one per patient (length bins)."""
    from scipy.optimize import minimize_scalar
    n = len(truths)
    Ls = _per_patient(L, n)
    els = _per_patient(eligible, n)

    def nll(tau):
        tot, m = 0.0, 0
        for q0, t, a, Li, el in zip(q0s, truths, answers, Ls, els):
            post, _ = workup_posteriors(q0, Li, tau)
            lp = np.log(np.clip(post[np.arange(Li.shape[0]), a, t], 1e-300, None))
            if el is not None:
                lp = lp[el]
            tot -= lp.sum()
            m += lp.size
        return tot / max(m, 1)
    return float(minimize_scalar(nll, bounds=(lo, hi), method="bounded").x)


def _spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


WK_EIG_BINS = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, np.inf)


def evaluate_workup(q0s, truths, answers, L, pri, tau, *, ks=(1, 3), eligible=None,
                    L_pop=None):
    """R1-R4 of the spec over every eligible (patient, candidate) pair. `q0s` (n, M)
    beliefs, `truths` (n,) member index, `answers` (n, n_cand) true answers, `L` the
    likelihoods (one array, or one per patient: window-length bins), `pri` the
    training prior (the population ranking's belief, scored with `L_pop`, default the
    first L), `tau` one temper or one per patient (each cross-fit half fits its own),
    `eligible` per-patient (n_cand,) bool of the facts new to that patient.

    Realized gain of a fact, `g = log q(true | a_true) − log q0(true)`, is a proper
    score whose expectation under a calibrated model is the fact's EIG."""
    n = len(truths)
    Ls = _per_patient(L, n)
    els = _per_patient(eligible, n)
    taus = np.broadcast_to(np.asarray(tau, dtype=np.float64), (n,))
    Lp = Ls[0] if L_pop is None else L_pop
    n_c = Lp.shape[0]
    rows = np.arange(n_c)
    post_p, pa_p = workup_posteriors(pri, Lp, float(np.mean(taus)) if n else 1.0)
    pop_score = workup_eig(pri, post_p, pa_p)
    r1, top, rnd, popg, orc = [], {k: [] for k in ks}, [], [], []
    chg, toward, right, n_el = [], [], [], []
    bins = np.zeros((len(WK_EIG_BINS) - 1, 3))           # n, sum eig, sum g
    sxx = sxy = 0.0
    for q0, t, a, Li, el, tau_i in zip(q0s, truths, answers, Ls, els, taus):
        el = np.ones(n_c, bool) if el is None else np.asarray(el, bool)
        if not el.any():
            continue
        post, pa = workup_posteriors(q0, Li, tau_i)
        eig = workup_eig(q0, post, pa)
        q_after = post[rows, a]                            # (n_c, M)
        g = np.log(np.clip(q_after[:, t], 1e-300, None)) - np.log(max(q0[t], 1e-300))
        ix = np.flatnonzero(el)
        e_, g_ = eig[ix], g[ix]
        n_el.append(ix.size)
        r1.append(_spearman(e_, g_))
        order = ix[np.argsort(-e_, kind="stable")]
        for k in ks:
            top[k].append(float(g[order[:k]].mean()))
        rnd.append(float(g_.mean()))
        popg.append(float(g[ix[np.argmax(pop_score[ix])]]))
        orc.append(float(g_.max()))
        b = order[0]
        new = int(np.argmax(q_after[b]))
        chg.append(new != int(np.argmax(q0)))
        toward.append(bool(q_after[b, t] > q0[t]))
        right.append(new == int(t))
        j = np.clip(np.searchsorted(WK_EIG_BINS, e_, side="right") - 1, 0,
                    len(WK_EIG_BINS) - 2)
        np.add.at(bins, (j, 0), 1.0)
        np.add.at(bins, (j, 1), e_)
        np.add.at(bins, (j, 2), g_)
        sxx += float(e_ @ e_)
        sxy += float(e_ @ g_)
    mean = lambda x: float(np.nanmean(x)) if len(x) else float("nan")
    q = lambda x: [float(v) for v in np.nanpercentile(x, [25, 50, 75])] if len(x) else []
    chg = np.asarray(chg, bool)
    return {
        "n": len(r1), "n_candidates": int(n_c),
        "eligible_per_patient_median": float(np.median(n_el)) if n_el else 0.0,
        "tau": sorted({round(float(x), 4) for x in taus}),
        "R1_spearman_quartiles": q(r1),
        "R2": {**{f"top{k}": mean(v) for k, v in top.items()}, "random": mean(rnd),
               "population_top1": mean(popg), "oracle": mean(orc)},
        "R3": {"bins": [{"lo": float(WK_EIG_BINS[i]), "n": int(bins[i, 0]),
                         "mean_eig": float(bins[i, 1] / bins[i, 0]),
                         "mean_gain": float(bins[i, 2] / bins[i, 0])}
                        for i in range(bins.shape[0]) if bins[i, 0] > 0],
               "slope": float(sxy / sxx) if sxx > 0 else float("nan")},
        "R4": {"top1_changes_call": float(chg.mean()) if chg.size else float("nan"),
               "top1_moves_toward_truth": mean(toward),
               "changed_call_is_right": (float(np.asarray(right)[chg].mean())
                                         if chg.any() else float("nan"))},
    }


def workup_coverage(window_days, answers):
    """Coverage of the scored patients' windows: length quartiles (days), the share
    with at least one non-default answer among the candidates, the median count."""
    wd = np.asarray([w for w in window_days if w is not None], dtype=np.float64)
    nd = np.asarray([(np.asarray(a) > 0).sum() for a in answers], dtype=np.float64)
    return {"window_days_quartiles": ([float(v) for v in np.percentile(wd, [25, 50, 75])]
                                      if wd.size else []),
            "share_with_any": float((nd > 0).mean()) if nd.size else float("nan"),
            "median_non_default": float(np.median(nd)) if nd.size else float("nan")}
