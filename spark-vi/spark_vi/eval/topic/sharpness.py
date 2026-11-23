"""How sharp a topic is: its effective vocabulary, as a fraction of the vocabulary.

A topic's word distribution β_k (the row of λ, normalized) is SHARP when it puts its
mass on a few terms and FLAT when it spreads it evenly. The measure here is the
effective support exp(H(β_k)), with H the Shannon entropy: the number of equally
weighted terms that would have the same entropy (the Hill number of order 1; Hill
1973, Jost 2006). A one-hot row has effective support 1; a uniform row over V terms has
V. Dividing by V gives a scale-free `support_fraction` in (0, 1]: about 0 = sharp,
1 = flat.

Why it diagnoses an unsupported topic. Under a symmetric Dirichlet prior η on the
topic-word distributions, a topic that receives no words keeps λ_k = η everywhere,
which is exactly uniform: its support fraction is 1. Under stochastic VI a topic whose
documents stop sending it words decays geometrically back to η (every step keeps
(1 - ρ_t) of the old λ_k and adds ρ_t η), so a fraction near 1 marks a topic the data
did not sustain, whatever it was initialized or seeded with; a fraction well below 1
marks a topic that the data keeps renewing.

Several vocabularies (multi-domain λ, `{m: (K, V_m)}`): each domain is a separate
vocabulary, normalized on its own row, so a topic has one β per domain and one support
fraction per domain (`support_fraction_by_domain`, each over its own V_m). A topic's
single value (`support_fraction`) is its SHARPEST row, the minimum over domains: a topic
is flat only when it is flat in EVERY domain (walkthrough 2026-11-23, replacing the
earlier dominant-domain reading, which let one domain decide and flipped between
domains of similar mass). A domain the topic does not use sits at its uniform prior
and reads flat, so it never makes a sharp topic look flat. The converse hazard, a
nearly empty row that looks sharp because a few stray tokens dominate its tiny prior,
is why `support_fraction_by_domain` also reports each row's DATA mass (λ minus the
prior), so a reader can see what backs a sharp row.

Pure numpy, domain-neutral: nothing here knows what a term or a topic means.
"""
from __future__ import annotations

import numpy as np


def lambda_domains(lam) -> list:
    """λ as a list of (K, V_m) float arrays: a per-domain dict in domain order (keys
    compared as ints, so `{"10": .., "2": ..}` read back from JSON orders as 2, 10), or
    a single (K, V) array as one domain."""
    if isinstance(lam, dict):
        return [np.asarray(lam[k], dtype=np.float64)
                for k in sorted(lam, key=lambda k: int(k))]
    return [np.asarray(lam, dtype=np.float64)]


def support_fraction_by_domain(lam, eta=None):
    """`(frac, mass, data)`, each `(K, M)` over the M domains (M = 1 for a single λ).

    `frac[k, m]`: exp(H(β_km)) / V_m, the support fraction of topic k's row in domain m;
    about 0 = sharp, 1 = flat. An all-zero row reads 1 (it holds nothing; λ is never
    zero in a fit, since the prior keeps it above η). `mass[k, m]`: the row's λ sum.
    `data[k, m]`: the row's DATA mass, `mass - eta_m * V_m` floored at 0 (the expected
    tokens the row was assigned), or None when `eta` is None; `eta` is a scalar or one
    value per domain. Vectorized over topics."""
    doms = lambda_domains(lam)
    K = doms[0].shape[0]
    M = len(doms)
    frac = np.ones((K, M))
    mass = np.stack([d.sum(axis=1) for d in doms], axis=1)
    for m, d in enumerate(doms):
        tot = mass[:, m]
        nz = tot > 0
        p = d[nz] / tot[nz, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
        frac[nz, m] = np.exp(h) / d.shape[1]
    data = None
    if eta is not None:
        e = np.atleast_1d(np.asarray(eta, dtype=np.float64))
        if e.size == 1:
            e = np.full(M, float(e[0]))
        if e.size != M:
            raise ValueError(f"eta has {e.size} values for {M} domain(s)")
        prior = np.array([e[m] * doms[m].shape[1] for m in range(M)])
        data = np.maximum(mass - prior[None, :], 0.0)
    return frac, mass, data


def support_fraction(lam) -> tuple[np.ndarray, np.ndarray]:
    """`(frac, mass)`, each of length K: `frac[k]` is topic k's SHARPEST row, the
    minimum over domains of `support_fraction_by_domain` (flat only when flat in every
    domain); `mass[k]` the topic's total λ mass over domains (its pseudo-count of
    words plus the prior)."""
    frac, mass, _ = support_fraction_by_domain(lam)
    return frac.min(axis=1), mass.sum(axis=1)
