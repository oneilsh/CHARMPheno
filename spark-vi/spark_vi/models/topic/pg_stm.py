"""Pólya-Gamma variational core for the gated stick-breaking logistic-normal
topic model (design 2026-07-11-pg-stm-inference-core-design.md). Single machine,
full-batch VI + exact Gibbs cross-check. References: Polson/Scott/Windle 2013 (PG);
Linderman/Johnson/Adams 2015 (stick-breaking multinomial + PG); Blei/Lafferty 2007
(logistic-normal topic model)."""
from __future__ import annotations

import numpy as np
from polyagamma import random_polyagamma
from scipy.special import expit  # logistic sigmoid


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
