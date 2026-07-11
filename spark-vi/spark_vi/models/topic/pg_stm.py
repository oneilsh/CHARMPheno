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
