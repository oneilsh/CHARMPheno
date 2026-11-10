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
vocabulary, normalized on its own row, so a topic has one β per domain. The measure
reads the topic's DOMINANT domain (the one holding most of its λ mass), over that
domain's vocabulary size: a topic that writes nearly all its words into one domain is
judged on that domain, not diluted by a light, prior-level row in another (the prior
row of a domain the topic does not use is uniform and would read as flat).

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


def support_fraction(lam) -> tuple[np.ndarray, np.ndarray]:
    """`(frac, mass)`, each of length K.

    `frac[k]`: the effective support exp(H(β_k)) of topic k's row in its dominant
    domain (the domain holding most of its λ mass), over that domain's vocabulary
    size; about 0 = sharp, 1 = flat (the row is at the symmetric prior). `mass[k]`: the
    topic's total λ mass, summed over domains (its pseudo-count of words plus the
    prior). An all-zero row reads as 1 / V (zero entropy), and ties between domains go
    to the first. Vectorized over topics."""
    doms = lambda_domains(lam)
    sums = np.stack([d.sum(axis=1) for d in doms], axis=1)          # (K, M)
    mass = sums.sum(axis=1)
    dom = np.argmax(sums, axis=1)
    frac = np.empty(sums.shape[0])
    for m, d in enumerate(doms):
        rows = np.flatnonzero(dom == m)
        if rows.size == 0:
            continue
        sub = d[rows]
        p = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
        frac[rows] = np.exp(h) / d.shape[1]
    return frac, mass
