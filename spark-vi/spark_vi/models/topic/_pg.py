"""Pure-numpy Pólya-Gamma sampler, so the PG-STM engine ships self-contained in
``spark_vi.zip`` with NO native ``polyagamma`` dependency on the Spark executors (only the
driver had it; a module-level ``from polyagamma import ...`` broke the VI path on every
worker even though the VI path never samples PG — it uses the analytic ``omega_expectation``).

Exact PG(1, z) via the Devroye/alternating-series method of Polson, Scott & Windle 2013
("Bayesian inference for logistic models using Pólya-Gamma latent variables", JASA
108:1339-1349; algorithm as in Windle's BayesLogit). PG(h, z) for integer h is a sum of h
iid PG(1, z); for large h a Gaussian CLT approximation (mean h·E[PG(1,z)], variance
h·Var[PG(1,z)]) replaces the O(h) exact sum. Validated against the ``polyagamma`` package
(dev-only dependency) in tests/test_pg_sampler.py.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr        # stable log standard-normal CDF

__TRUNC = 0.64                             # PSW truncation point
_PI = np.pi
_PI2 = _PI * _PI


def _a(n, x):
    """n-th coefficient of the alternating series for the J* density at x."""
    k = n + 0.5
    if x > __TRUNC:
        return _PI * k * np.exp(-k * k * _PI2 * x / 2.0)
    return _PI * k * (2.0 / (_PI * x)) ** 1.5 * np.exp(-2.0 * k * k / x)


def _mass_texpon(z):
    """P(proposal from the truncated-exponential right piece) in the PSW mixture."""
    fz = _PI2 / 8.0 + z * z / 2.0
    b = np.sqrt(1.0 / __TRUNC) * (__TRUNC * z - 1.0)
    a = -np.sqrt(1.0 / __TRUNC) * (__TRUNC * z + 1.0)
    x0 = np.log(fz) + fz * __TRUNC
    xb = x0 - z + log_ndtr(b)
    xa = x0 + z + log_ndtr(a)
    qdivp = 4.0 / _PI * (np.exp(xb) + np.exp(xa))
    return 1.0 / (1.0 + qdivp)


def _rtigauss(z, rng):
    """Sample the left proposal: inverse-Gaussian(1/z, 1) truncated to (0, TRUNC]."""
    z = abs(z)
    mu = 1.0 / z if z > 1e-12 else 1e12
    x = 1.0 + __TRUNC
    if mu > __TRUNC:
        while True:                        # do-while: always draw once (handles z->0)
            e1 = rng.exponential(); e2 = rng.exponential()
            while e1 * e1 > 2.0 * e2 / __TRUNC:
                e1 = rng.exponential(); e2 = rng.exponential()
            x = __TRUNC / (1.0 + __TRUNC * e1) ** 2
            if rng.random() <= np.exp(-0.5 * z * z * x):
                break
    else:
        while x > __TRUNC:
            y = rng.normal() ** 2
            x = mu + 0.5 * mu * mu * y - 0.5 * mu * np.sqrt(4.0 * mu * y + (mu * y) ** 2)
            if rng.random() > mu / (mu + x):
                x = mu * mu / x
    return x


def _pg1(z, rng):
    """One exact draw from PG(1, z)."""
    z = abs(z) * 0.5
    fz = _PI2 / 8.0 + z * z / 2.0
    while True:
        if rng.random() < _mass_texpon(z):
            x = __TRUNC + rng.exponential() / fz
        else:
            x = _rtigauss(z, rng)
        s = _a(0, x)
        y = rng.random() * s
        n = 0
        while True:
            n += 1
            if n % 2 == 1:
                s -= _a(n, x)
                if y <= s:
                    return 0.25 * x
            else:
                s += _a(n, x)
                if y > s:
                    break                  # reject this x, propose again


def _pg1_mean(z):
    z = abs(z)
    if z < 1e-8:
        return 0.25
    return np.tanh(z / 2.0) / (2.0 * z)


def _pg1_var(z):
    z = abs(z)
    if z < 1e-4:
        return 1.0 / 24.0
    # Var(PG(1,z)) = (sinh z - z) / (4 z^3 cosh^2(z/2)); -> 1/24 as z->0.
    return (np.sinh(z) - z) / (4.0 * z ** 3 * np.cosh(z / 2.0) ** 2)


def random_polyagamma(h, z, *, random_state, exact_max=25):
    """Drop-in replacement for ``polyagamma.random_polyagamma(h=..., z=...,
    random_state=...)`` for the shapes this engine uses (1-D h, z of equal length, integer
    h >= 0). Each entry is PG(h_i, z_i): exact sum of h_i iid PG(1, z_i) when h_i <=
    ``exact_max``, else a Gaussian CLT approximation. h_i == 0 -> 0 (degenerate)."""
    rng = random_state
    h_arr = np.atleast_1d(np.asarray(h, dtype=np.float64))
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
    h_b, z_b = np.broadcast_arrays(h_arr, z_arr)
    out = np.zeros(h_b.shape, dtype=np.float64)
    of = out.ravel(); hf = h_b.ravel(); zf = z_b.ravel()
    for i in range(hf.size):
        n = int(round(hf[i]))
        if n <= 0:
            continue
        zi = zf[i]
        if n <= exact_max:
            acc = 0.0
            for _ in range(n):
                acc += _pg1(zi, rng)
            of[i] = acc
        else:
            of[i] = max(1e-12, rng.normal(n * _pg1_mean(zi),
                                          np.sqrt(n * _pg1_var(zi))))
    return out
