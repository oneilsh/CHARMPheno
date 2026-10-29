"""Framings 1 and 2: the differential with reasons and the workup (spec 2026-10-13
clinician framings, R0/R1; spec 2026-10-28 workup-voi-design).

"For this patient, which disease in group G, and what in their record points there?"
Engines are scored on the same held-out patients (D4: the engine is chosen by
evaluation, not assumed); the driver is `framings.run_differential`:

E2  the record's stacked readout heads renormalized over the members (0139's
    baseline), on the whole-record θ (the deployment fold-in; `ModelEngine`).
E3  a multiclass (softmax) head over the members on log θ (`MulticlassHead`), fit on
    TRAIN-split patients positive on exactly one member; L2 chosen on an internal
    split. Features: the background topics plus the members' blocks.
E5  the same softmax on the record's CODES (`CodeHead`): binary presence of the
    group's frequent (or within-group relevant) codes plus log(1 + distinct codes).
E6  codes + standardized log θ (`CodeHead(theta_topics=...)`).

Records (R0.1): attestation codes (members' and the group's own codes) and order-only
`[measured]` tokens are removed before any engine sees the record (D1, D2). Reasons
(R1.1) are pooled per member over correctly called patients, names only, suppressed
under the egress floor.

Framing 2 as WORKUP (exp 0167): every fact first recorded between the index and the
first member diagnosis is a candidate question; its answers update the belief by
tempered naive Bayes and it is ranked by expected information gain
(`workup_answers` ... `evaluate_workup`, `workup_diagnostics`).

(The presence-subtype and gated-likelihood engines E1/E4 and the hidden-record
next-question protocol were removed in the 2026-10-29 cleanup; last at commit
29556f3d.)

Pure numpy over saved parameters; the driver only collects the patients."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from charmpheno.egress import EGRESS_FLOOR
from spark_vi.eval.metrics import ece

__all__ = ["filter_record", "MulticlassHead", "CodeHead", "label_matrix", "metrics",
           "pool_reasons", "e3_reasons", "EGRESS_FLOOR"]


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


# ----------------------------------------------------------------------------- #
# E4 — gated per-disease likelihood                                              #
# ----------------------------------------------------------------------------- #


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
    diagnoses 2,000 binary codes beat θ by ~0.04, most for the rarest diseases."""

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
            "ece": ece([p.max() for p in posts], top1)}


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
    f_n, f_e, f_g = np.zeros(n_c), np.zeros(n_c), np.zeros(n_c)   # per fact
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
        f_n[ix] += 1.0
        f_e[ix] += e_
        f_g[ix] += g_
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
    # Fact-level ranking accuracy (run 3): does a fact's mean EIG over the patients
    # it is new for rank with its mean REALIZED gain over the same patients? Pooling
    # over patients is what a proper score licenses (E[g] = EIG per fact under a
    # calibrated model); the per-patient R1 is not, because with sparse windows most
    # realized answers are the default, and a default that is correctly evidence
    # AGAINST the truth is commonest on the most discriminating facts.
    ok = f_n >= EGRESS_FLOOR
    fe = np.divide(f_e, f_n, out=np.zeros(n_c), where=f_n > 0)
    fg = np.divide(f_g, f_n, out=np.zeros(n_c), where=f_n > 0)
    facts = {"n_facts": int(ok.sum())}
    if ok.sum() >= 3:
        oi = np.flatnonzero(ok)
        top5 = oi[np.argsort(-fe[oi], kind="stable")[:5]]
        facts.update({
            "spearman_mean_eig_vs_mean_gain": _spearman(fe[oi], fg[oi]),
            "top5_mean_gain": float(fg[top5].mean()),
            "all_mean_gain": float(fg[oi].mean()),
            "slope": (float(fe[oi] @ fg[oi] / (fe[oi] @ fe[oi]))
                      if fe[oi] @ fe[oi] > 0 else float("nan"))})
    return {
        "n": len(r1), "n_candidates": int(n_c), "facts": facts,
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


def workup_diagnostics(q0s, truths, answers, L, tau, *, eligible=None, activity=None,
                       train=None):
    """Exp 0167 run-2 diagnostic: WHY can the per-patient ranking anti-correlate with
    realized gain? Four reads over the eligible (patient, fact) pairs:

    1. likelihood transfer — mean log P(true answer | true member) on the test pairs,
       and on the training pairs (`train` = (answers, member index, eligible lists)),
       plus predicted vs observed non-default rate under the true member: a gap is a
       train/test shift;
    2. gain by answer type — mean realized gain and mean EIG over pairs whose answer
       was the default ("not recorded" / "not done") vs a non-default one;
    3. R1 and top-1 gain by window-activity tercile (`activity`: distinct codes in the
       patient's window): absences are correlated through how much care a patient
       got, which naive Bayes treats as independent evidence, so a low-activity
       patient's high-EIG facts can all point the wrong way;
    4. per patient, total EIG vs total realized gain.
    Pooled numbers only."""
    n = len(truths)
    Ls = _per_patient(L, n)
    els = _per_patient(eligible, n)
    taus = np.broadcast_to(np.asarray(tau, dtype=np.float64), (n,))
    ll_te, pred_nd, obs_nd = [], [], []
    gd, ed, gn, en = [], [], [], []
    rows_r1, top1, sum_e, sum_g = [], [], [], []
    for q0, t, a, Li, el, tau_i in zip(q0s, truths, answers, Ls, els, taus):
        n_c = Li.shape[0]
        el = np.ones(n_c, bool) if el is None else np.asarray(el, bool)
        ix = np.flatnonzero(el)
        if ix.size == 0:
            rows_r1.append(float("nan")); top1.append(float("nan"))
            sum_e.append(0.0); sum_g.append(0.0)
            continue
        post, pa = workup_posteriors(q0, Li, tau_i)
        eig = workup_eig(q0, post, pa)
        g = (np.log(np.clip(post[np.arange(n_c), a, t], 1e-300, None))
             - np.log(max(q0[t], 1e-300)))
        lt = Li[ix, a[ix], t]
        ll_te.extend(np.log(np.clip(lt, 1e-300, None)).tolist())
        pred_nd.extend((1.0 - Li[ix, 0, t]).tolist())
        obs_nd.extend((a[ix] > 0).astype(float).tolist())
        d = a[ix] == 0
        gd.extend(g[ix][d].tolist()); ed.extend(eig[ix][d].tolist())
        gn.extend(g[ix][~d].tolist()); en.extend(eig[ix][~d].tolist())
        rows_r1.append(_spearman(eig[ix], g[ix]))
        top1.append(float(g[ix[np.argmax(eig[ix])]]))
        sum_e.append(float(eig[ix].sum())); sum_g.append(float(g[ix].sum()))
    mean = lambda x: float(np.nanmean(x)) if len(x) else float("nan")
    out = {"ll_test": mean(ll_te), "pred_nondefault": mean(pred_nd),
           "obs_nondefault": mean(obs_nd),
           "default_pairs": {"n": len(gd), "mean_gain": mean(gd), "mean_eig": mean(ed)},
           "nondefault_pairs": {"n": len(gn), "mean_gain": mean(gn),
                                "mean_eig": mean(en)},
           "sum_eig_mean": mean(sum_e), "sum_gain_mean": mean(sum_g)}
    if train is not None:
        tr_a, tr_t, tr_el = train
        ll_tr = []
        for a, t, el in zip(tr_a, tr_t, tr_el):
            ix = np.flatnonzero(np.asarray(el, bool))
            Ltr = L if not isinstance(L, (list, tuple)) else Ls[0]
            ll_tr.extend(np.log(np.clip(Ltr[ix, a[ix], t], 1e-300, None)).tolist())
        out["ll_train"] = mean(ll_tr)
    if activity is not None and n >= 3:
        act = np.asarray(activity, dtype=np.float64)
        cuts = np.nanpercentile(act, [100 / 3, 200 / 3])
        terc = np.searchsorted(cuts, act, side="right")
        out["by_activity"] = []
        for k, lab in enumerate(("low", "mid", "high")):
            m = terc == k
            if not m.any():
                continue
            r = np.asarray(rows_r1)[m]
            out["by_activity"].append({
                "tercile": lab, "n": int(m.sum()),
                "activity_median": float(np.median(act[m])),
                "R1_median": float(np.nanmedian(r)) if np.isfinite(r).any() else float("nan"),
                "top1_gain": mean(np.asarray(top1)[m])})
    return out
