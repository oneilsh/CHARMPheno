"""Pólya-Gamma variational core for the gated stick-breaking logistic-normal
topic model (design 2026-07-11-pg-stm-inference-core-design.md). Single machine,
full-batch VI + exact Gibbs cross-check. References: Polson/Scott/Windle 2013 (PG);
Linderman/Johnson/Adams 2015 (stick-breaking multinomial + PG); Blei/Lafferty 2007
(logistic-normal topic model)."""
from __future__ import annotations

import numpy as np
from polyagamma import random_polyagamma
from scipy.special import expit  # logistic sigmoid

from spark_vi.models.topic._linalg import safe_inverse


def stick_to_simplex(psi: np.ndarray) -> np.ndarray:
    """Stick-breaking map: psi (K-1,) -> theta (K,) on the simplex.
    theta_k = sigma(psi_k) * prod_{j<k}(1 - sigma(psi_j)); last topic gets the remainder."""
    psi = np.asarray(psi, dtype=np.float64)
    sig = expit(psi)                          # (K-1,)
    theta = np.empty(psi.shape[0] + 1, dtype=np.float64)
    remaining = 1.0
    for k in range(psi.shape[0]):
        theta[k] = remaining * sig[k]
        remaining *= (1.0 - sig[k])
    theta[-1] = remaining
    return theta


def simplex_to_stick(theta: np.ndarray) -> np.ndarray:
    """Inverse map: theta (K,) -> psi (K-1,). sigma(psi_k) = theta_k / (1 - sum_{j<k} theta_j)."""
    theta = np.asarray(theta, dtype=np.float64)
    psi = np.empty(theta.shape[0] - 1, dtype=np.float64)
    remaining = 1.0
    for k in range(theta.shape[0] - 1):
        frac = np.clip(theta[k] / remaining, 1e-15, 1.0 - 1e-15)
        psi[k] = np.log(frac) - np.log1p(-frac)   # logit(frac)
        remaining -= theta[k]
    return psi


def stick_trials(n: np.ndarray) -> np.ndarray:
    """Per-stick trials-at-risk b (K-1,): b[k] = sum_{j>=k} n[j]."""
    n = np.asarray(n, dtype=np.float64)
    return np.cumsum(n[::-1])[::-1][:-1].copy()


def omega_expectation(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Variational mean of the PG auxiliary: E[omega_k] = (b_k/(2 c_k)) tanh(c_k/2),
    c_k = sqrt(E[psi_k^2]). tanh(c/2)/c -> 1/2 as c->0, so the limit is b/4."""
    b = np.asarray(b, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    out = np.empty_like(b)
    small = c < 1e-6
    out[small] = b[small] / 4.0
    cc = c[~small]
    out[~small] = b[~small] / (2.0 * cc) * np.tanh(cc / 2.0)
    return out


def omega_sample(b: np.ndarray, psi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact Gibbs draw omega_k ~ PG(b_k, psi_k)."""
    return random_polyagamma(h=np.asarray(b, dtype=np.float64),
                             z=np.asarray(psi, dtype=np.float64), random_state=rng)


def psi_posterior(n, b, mu, Sigma_inv, omega):
    """Per-doc Gaussian posterior over the stick logits under PG augmentation.
    V = (Sigma_inv + diag(omega))^-1 ; m = V (Sigma_inv mu + kappa) ; kappa = a - b/2."""
    b = np.asarray(b, dtype=np.float64)
    kappa = np.asarray(n, dtype=np.float64)[:b.shape[0]] - b / 2.0
    prec = np.asarray(Sigma_inv, dtype=np.float64) + np.diag(np.asarray(omega, dtype=np.float64))
    V = np.linalg.inv(prec)
    m = V @ (np.asarray(Sigma_inv, dtype=np.float64) @ np.asarray(mu, dtype=np.float64) + kappa)
    return m, V


def sigma_iw_posterior_mean(scatter, n_docs, *, Psi0, nu0, dim):
    """Inverse-Wishart posterior mean E[Sigma] = (Psi0 + scatter)/(nu0 + n_docs - dim - 1).
    Proper prior (nu0 > dim + 1) => finite PD mean even at n_docs = 0 (the runaway cure)."""
    denom = nu0 + n_docs - dim - 1.0
    return (np.asarray(Psi0, dtype=np.float64) + np.asarray(scatter, dtype=np.float64)) / denom


def gamma_ridge(M, X, *, ridge):
    """Ridge regression of stacked posterior means M (D, K-1) on covariates X (D, P)."""
    X = np.asarray(X, dtype=np.float64); M = np.asarray(M, dtype=np.float64)
    P = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(P), X.T @ M)


def beta_dirichlet_mean(word_topic_stats, *, eta):
    """Row-normalized Dirichlet posterior mean of the (K,V) topic-word matrix."""
    lam = np.asarray(word_topic_stats, dtype=np.float64) + eta
    return lam / lam.sum(axis=1, keepdims=True)


def _elog_sigmoid(m, v, sign):
    """Delta-method E[log sigma(sign*psi)] under psi~N(m, v), sign in {+1, -1}
    (+1 -> E[log sigma(psi)], -1 -> E[log(1-sigma(psi))] = E[log sigma(-psi)]).
    Fourth-order Taylor expansion of log-sigma (Kendall & Stuart, "The Advanced Theory of
    Statistics" Vol 1, ch.10 - higher-order delta method; Bickel & Doksum, "Mathematical
    Statistics" - for smooth f and X~N(mu,v): E[f(X)] ~ f(mu) + f''(mu) v/2 + f''''(mu) (3 v^2)/4!,
    using E[(X-mu)^4]=3v^2 for Gaussian X).

    For f(x)=log sigma(x): f''(x) = -s(x), f''''(x) = -s(x)(1-2 sigma(x))^2 + 2 s(x)^2, where
    s(x)=sigma(x)(1-sigma(x)). For g(x)=log(1-sigma(x))=log sigma(-x), g''=f''(-x)=-s(x) and
    g''''=f''''(-x)=f''''(x) (both are even in the (1-2 sigma) term), so both branches share the
    same s and 4th-order coefficient q evaluated at m - only the base log-sigma term flips.

    NOTE: the second-order-only truncation (drop the q term) is NOT sufficient here - verified
    against high-precision Gauss-Hermite quadrature (not just Monte-Carlo noise), its bias grows
    ~v^2/8 * f''''(m) and exceeds a 3e-3 tolerance once v gtrsim 0.4-0.5 (e.g. bias ~-0.0047 at
    m=0.1, v=0.6). The 4th-order term is required to track the true expectation to the precision
    demanded by test_expected_log_theta_matches_quadrature / test_gated_expected_log_theta_matches_quadrature."""
    m = np.asarray(m, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    sig = expit(m); s = sig * (1.0 - sig)
    q = -s * (1.0 - 2.0 * sig) ** 2 + 2.0 * s ** 2   # f''''(m), shared by log-sig & log(1-sig)
    corr = -0.5 * v * s + (v ** 2 / 8.0) * q         # 2nd + 4th order delta-method correction
    base = np.log(sig) if sign > 0 else np.log1p(-sig)
    return base + corr


def expected_log_theta(m, v):
    """Delta-method E[log theta] under q(psi_k)=N(m_k, v_k), composed from the per-stick
    E[log sigma]/E[log(1-sigma)] terms via _elog_sigmoid (see its docstring for the delta-method
    derivation and the 4th-order-term necessity)."""
    m = np.asarray(m, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    ls_plus = _elog_sigmoid(m, v, +1)     # E[log sigma(psi_k)]
    ls_minus = _elog_sigmoid(m, v, -1)    # E[log (1-sigma(psi_k))]
    K = m.shape[0] + 1
    out = np.empty(K, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(ls_minus)])   # cum[k] = sum_{j<k} ls_minus_j
    out[:K - 1] = ls_plus + cum[:K - 1]
    out[K - 1] = cum[K - 1]                               # = sum_j ls_minus_j
    return out


def gated_theta(psi_bg, psi_gate, psi_fg):
    """Nested stick-breaking composition: a per-group gate stick splits background vs
    foreground mass, then a flat stick_to_simplex runs within each block. This is what keeps
    gating consistent under stick-breaking (a single flat sequence isn't closed under
    subsetting the allowed topics - see docs/superpowers/specs/2026-07-11-pg-stm-inference-core-design.md).

    theta = concat(sigma(psi_gate) * stick_to_simplex(psi_bg),
                    (1-sigma(psi_gate)) * stick_to_simplex(psi_fg)), length B+m_g."""
    gate = expit(psi_gate)
    theta_bg = gate * stick_to_simplex(psi_bg)
    theta_fg = (1.0 - gate) * stick_to_simplex(psi_fg)
    return np.concatenate([theta_bg, theta_fg])


def gated_expected_log_theta(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg):
    """Composed E[log theta] for the nested gated stick-breaking: the gate contributes an
    E[log sigma(psi_gate)] (resp. E[log(1-sigma(psi_gate))]) term, added to every background
    (resp. foreground) entry's flat expected_log_theta. The gate term uses the SAME delta-method
    helper (_elog_sigmoid) as the within-block sticks, so the gate's approximation accuracy
    matches the sticks' exactly - see _elog_sigmoid's docstring for the derivation."""
    eg_bg = _elog_sigmoid(m_gate, v_gate, +1)     # E[log sigma(psi_gate)], scalar
    eg_fg = _elog_sigmoid(m_gate, v_gate, -1)     # E[log(1-sigma(psi_gate))], scalar
    elog_bg = eg_bg + expected_log_theta(m_bg, v_bg)
    elog_fg = eg_fg + expected_log_theta(m_fg, v_fg)
    return np.concatenate([elog_bg, elog_fg])


def gated_counts(n_bg, n_fg):
    """Per-group sufficient stats for the nested gate + flat-block PG augmentation. The gate is
    one binomial (N_bg successes out of N_bg+N_fg trials); each block's within-block sticks use
    the flat stick_trials count (Task 1)."""
    n_bg = np.asarray(n_bg, dtype=np.float64); n_fg = np.asarray(n_fg, dtype=np.float64)
    gate_a = n_bg.sum()
    gate_b = n_bg.sum() + n_fg.sum()
    b_bg = stick_trials(n_bg)
    b_fg = stick_trials(n_fg)
    return gate_a, gate_b, b_bg, b_fg


def token_responsibilities(doc_indices, elog_theta, elog_beta, allowed, *, counts):
    """LDA-style responsibilities restricted to the allowed topic set.
    phi_{n,k} ∝ exp(elog_theta_k + elog_beta_{k, w_n}) for k in allowed, else 0."""
    K = elog_theta.shape[0]
    log_unnorm = elog_theta[None, :] + elog_beta[:, doc_indices].T   # (n_tok, K)
    mask = np.full(K, -np.inf); mask[np.asarray(allowed)] = 0.0
    log_unnorm = log_unnorm + mask[None, :]
    log_unnorm -= log_unnorm.max(axis=1, keepdims=True)
    phi = np.exp(log_unnorm); phi /= phi.sum(axis=1, keepdims=True)
    n = (phi * np.asarray(counts, dtype=np.float64)[:, None]).sum(axis=0)
    return phi, n


# --------------------------------------------------------------------------- #
# Task 6: gated full-batch PG-VI driver + block-Sigma IW M-step.
# --------------------------------------------------------------------------- #

_PSI_CLIP = 30.0   # numerical guard (Task-1 watch-item): keep sigmoid off 0/1


def stick_layout(partition):
    """Global stick layout (dimension K-1) for the nested gated stick-breaking model.

    Background sticks = global indices 0..B-2 (B = #background topics -> B-1 sticks).
    Group g (in ``partition.groups`` order) occupies m_g consecutive indices starting
    at ``B-1 + sum_{g'<g} m_{g'}``: the first is the GATE stick, the next m_g-1 are the
    group's foreground sticks. A group-g doc's ACTIVE global stick indices, in order,
    are ``[background_sticks (0..B-2), gate_g, fg_g_sticks]``; its allowed TOPIC indices
    are ``partition.allowed_indices({g})`` = background topics then group-g topics.

    Returns a dict::

        {"B": B, "bg_sticks": (B-1,) int array,
         "groups": {g: {"m_g", "gate", "fg_sticks", "active", "allowed"}}}
    """
    B = int(partition.background_k)
    bg_sticks = np.arange(0, B - 1, dtype=np.int64)      # 0..B-2  (B-1 sticks)
    groups = {}
    offset = B - 1
    for g, m_g in partition.foreground:
        gate = offset
        fg_sticks = np.arange(offset + 1, offset + m_g, dtype=np.int64)  # m_g-1 sticks
        active = np.concatenate([bg_sticks, np.array([gate], dtype=np.int64),
                                 fg_sticks]).astype(np.int64)
        allowed = partition.allowed_indices(frozenset({g})).astype(np.int64)
        groups[g] = {"m_g": int(m_g), "gate": int(gate), "fg_sticks": fg_sticks,
                     "active": active, "allowed": allowed}
        offset += m_g
    return {"B": B, "bg_sticks": bg_sticks, "groups": groups}


class PGSTMVI:
    """Full-batch mean-field coordinate ascent for the gated, nested stick-breaking
    logistic-normal topic model under Polya-Gamma augmentation (design
    2026-07-11-pg-stm-inference-core-design.md). Composes the Task 1-5 primitives.

    Every update is a pure function of (globals, per-doc sufficient stats), so the
    E-step / stat accumulation ports verbatim to a stochastic (minibatch) driver.

    Parameters
    ----------
    K, V : int
        #topics and vocabulary size.
    partition : TopicBlockPartition
        Background/foreground gating layout (defines the stick layout + allowed sets).
    P : int
        Covariate dimension.
    n_iter : int
        Outer coordinate-ascent sweeps.
    Psi0_scale, nu0 : IW prior scale (Psi0 = Psi0_scale * I) and dof. ``nu0=None`` ->
        (K-1)+2 (proper: nu0 > dim+1 for every block, so a finite PD mean even at
        n_docs=0 — the runaway cure).
    gamma_ridge, beta_eta : ridge for the Gamma regression / Dirichlet smoothing for beta.
    sigma_mode : ``"iw"`` (production) block IW posterior mean, or ``"mle"`` un-regularized
        ``scatter/n`` per block (Task-8 isolation only).
    """

    def __init__(self, K, V, partition, *, P, n_iter=200, Psi0_scale=1.0, nu0=None,
                 gamma_ridge=1e-6, beta_eta=0.1, sigma_mode="iw", seed=0,
                 inner_rounds=8, inner_tol=1e-3):
        if sigma_mode not in ("iw", "mle"):
            raise ValueError(f"sigma_mode must be 'iw' or 'mle', got {sigma_mode!r}")
        self.K = int(K); self.V = int(V); self.partition = partition
        self.P = int(P); self.n_iter = int(n_iter)
        self.Psi0_scale = float(Psi0_scale)
        self.nu0 = float(nu0) if nu0 is not None else float((K - 1) + 2)
        self.gamma_ridge = float(gamma_ridge); self.beta_eta = float(beta_eta)
        self.sigma_mode = sigma_mode; self.seed = int(seed)
        self.inner_rounds = int(inner_rounds); self.inner_tol = float(inner_tol)
        self.layout = stick_layout(partition)
        self._clip_tripped = 0

    # -- per-doc E-step -------------------------------------------------------- #
    def _e_step_doc(self, doc, glay, log_beta, Gamma, Sigma):
        B = self.layout["B"]; m_g = glay["m_g"]
        active = glay["active"]; allowed = glay["allowed"]
        Sigma_inv_active = safe_inverse(Sigma[np.ix_(active, active)])
        mu_active = (Gamma.T @ doc.x)[active]
        A = active.shape[0]
        m = mu_active.copy()
        V = np.eye(A)
        phi = None
        for _ in range(self.inner_rounds):
            m_prev = m
            vdiag = np.diag(V)
            mc = np.clip(m, -_PSI_CLIP, _PSI_CLIP)
            if not np.array_equal(mc, m):
                self._clip_tripped += 1
            # slice active positions into [background | gate | foreground]
            m_bg, v_bg = mc[:B - 1], vdiag[:B - 1]
            m_gate, v_gate = mc[B - 1], vdiag[B - 1]
            m_fg, v_fg = mc[B:], vdiag[B:]
            gelog = gated_expected_log_theta(m_bg, v_bg, m_gate, v_gate, m_fg, v_fg)
            elog_theta = np.full(self.K, -np.inf)
            elog_theta[allowed] = gelog
            phi, n_full = token_responsibilities(
                doc.indices, elog_theta, log_beta, allowed, counts=doc.counts)
            n_allowed = n_full[allowed]
            n_bg, n_fg = n_allowed[:B], n_allowed[B:]
            gate_a, gate_b, b_bg, b_fg = gated_counts(n_bg, n_fg)
            a_active = np.concatenate([n_bg[:B - 1], np.array([gate_a]), n_fg[:m_g - 1]])
            b_active = np.concatenate([b_bg, np.array([gate_b]), b_fg])
            c = np.sqrt(m ** 2 + vdiag)
            omega = omega_expectation(b_active, c)
            m, V = psi_posterior(a_active, b_active, mu_active, Sigma_inv_active, omega)
            if np.max(np.abs(m - m_prev)) < self.inner_tol:
                break
        return m, V, phi, active, allowed, mu_active

    def fit(self, docs):
        rng = np.random.default_rng(self.seed)
        K, V, P = self.K, self.V, self.P
        Ksm1 = K - 1
        # --- init: beta from smoothed random counts, Gamma=0, Sigma=I ---------- #
        beta = rng.random((K, V)) + self.beta_eta
        beta /= beta.sum(axis=1, keepdims=True)
        Gamma = np.zeros((P, Ksm1))
        Sigma = np.eye(Ksm1)

        # group -> doc indices (each doc has exactly one group here)
        group_docs = {g: [] for g in self.partition.groups}
        for d, doc in enumerate(docs):
            (g,) = tuple(doc.groups)
            group_docs[g].append(d)
        group_counts = {g: len(v) for g, v in group_docs.items()}
        D = len(docs)
        X = np.stack([np.asarray(doc.x, dtype=np.float64) for doc in docs])  # (D, P)

        bg_sticks = self.layout["bg_sticks"]
        sigma_max_trace = []

        for _ in range(self.n_iter):
            log_beta = np.log(beta)
            word_topic_stats = np.zeros((K, V))
            M = np.zeros((D, Ksm1))               # stacked active means (inactive at 0)
            S = np.zeros((Ksm1, Ksm1))            # global block scatter
            psi_mean = np.zeros((D, Ksm1))
            psi_var = np.zeros((D, Ksm1))

            for d, doc in enumerate(docs):
                (g,) = tuple(doc.groups)
                glay = self.layout["groups"][g]
                m, Vd, phi, active, allowed, mu_active = self._e_step_doc(
                    doc, glay, log_beta, Gamma, Sigma)
                # word-topic stats: sum_tokens phi * counts at [allowed, doc.indices]
                word_topic_stats[:, doc.indices] += (phi * doc.counts[:, None]).T
                # Gamma / Sigma / output accumulation
                M[d, active] = m
                psi_mean[d, active] = m
                psi_var[d, active] = np.diag(Vd)
                e_active = m - mu_active
                S[np.ix_(active, active)] += np.outer(e_active, e_active) + Vd

            # --- M-step ------------------------------------------------------- #
            beta = beta_dirichlet_mean(word_topic_stats, eta=self.beta_eta)
            Gamma = gamma_ridge(M, X, ridge=self.gamma_ridge)
            Sigma = self._assemble_sigma(S, bg_sticks, group_counts, D)
            sigma_max_trace.append(float(np.max(np.abs(Sigma))))

        if self._clip_tripped:
            import logging
            logging.getLogger(__name__).info(
                "PGSTMVI psi clip (+-%g) tripped %d times during fit",
                _PSI_CLIP, self._clip_tripped)

        return {"beta": beta, "Gamma": Gamma, "Sigma": Sigma,
                "psi_mean": psi_mean, "psi_var": psi_var,
                "sigma_max_trace": sigma_max_trace}

    # -- block-structured Sigma assembly -------------------------------------- #
    def _block_estimate(self, scatter, n_docs, dim):
        if self.sigma_mode == "iw":
            return sigma_iw_posterior_mean(
                scatter, n_docs, Psi0=self.Psi0_scale * np.eye(dim),
                nu0=self.nu0, dim=dim)
        return scatter / max(float(n_docs), 1.0)      # "mle": un-regularized point est.

    def _assemble_sigma(self, S, bg_sticks, group_counts, D):
        Ksm1 = self.K - 1
        Sigma = np.zeros((Ksm1, Ksm1))
        nb = len(bg_sticks)
        # background block: ALL docs are active on background.
        if nb > 0:
            Sigma[np.ix_(bg_sticks, bg_sticks)] = self._block_estimate(
                S[np.ix_(bg_sticks, bg_sticks)], D, nb)
        # each group block [gate, fg] + its background<->gblock cross, from that
        # group's docs only. group<->group' never co-active -> left at 0.
        for g in self.partition.groups:
            glay = self.layout["groups"][g]
            gblock = np.concatenate([np.array([glay["gate"]], dtype=np.int64),
                                     glay["fg_sticks"]]).astype(np.int64)
            joint = np.concatenate([bg_sticks, gblock]).astype(np.int64)
            n_g = group_counts[g]
            # sigma_iw / scatter-over-n is an ELEMENTWISE map with a scalar denom, so
            # the joint estimate's gblock & cross entries depend only on the group's
            # scatter (bg<->gblock and gblock<->gblock in S receive group-g docs only).
            Sig_joint = self._block_estimate(S[np.ix_(joint, joint)], n_g, len(joint))
            gb_local = np.arange(nb, len(joint))
            bg_local = np.arange(0, nb)
            Sigma[np.ix_(gblock, gblock)] = Sig_joint[np.ix_(gb_local, gb_local)]
            if nb > 0:
                cross = Sig_joint[np.ix_(bg_local, gb_local)]
                Sigma[np.ix_(bg_sticks, gblock)] = cross
                Sigma[np.ix_(gblock, bg_sticks)] = cross.T
        Sigma = 0.5 * (Sigma + Sigma.T)
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma = Sigma + 1e-8 * np.eye(Ksm1)
        return Sigma
