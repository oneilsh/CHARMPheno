"""The collapsed-Gibbs placement estimator: the oracle the DAG-gated engine is tested against.

`fit_gated` / `profile` (Griffiths & Steyvers 2004, with the DAG gate) are the
reference the variational engine (`DagGatedOnlineLDA`) is validated against: same
gate, same model, a different inference scheme. Test-only since ADR 0050 removed the
`dag_placement` model class; moved here from `spark_vi/models/topic/dag_placement.py`
with ADR 0051 (its `DagLayout` went to `spark_vi.dag`).
"""
from types import SimpleNamespace

import numpy as np

from spark_vi.models.topic.spectral_init import find_anchors, recover_beta, word_cooccurrence


def _as_counts(doc):
    idx, cnt = np.unique(np.asarray(doc), return_counts=True)
    return SimpleNamespace(indices=idx, counts=cnt.astype(np.float64))


def fit_gated(train_docs, train_labels, lay, V, *, beta_prior=0.02,
              n_iter=150, burn=80, rng=None):
    """Gated topic training by collapsed sampling: each training item is masked to
    allowed(label) = background ∪ blocks along its label's closure, tying topics to nodes
    structurally. Anchor-word spectral init (Arora et al. 2013) seeds beta. Returns posterior-mean
    beta_hat (K, V).

    The per-token conditional uses ONLY the collapsed word-topic factor
    (n_kw + beta_prior) / (n_k + V*beta_prior) of Griffiths & Steyvers (2004), restricted to the
    gated allowed-set. The usual document-topic Dirichlet factor (n_dk + alpha) is intentionally
    omitted: the gate already fixes each document's admissible topics to its label closure, so the
    per-document mixing term is not what we are estimating here — we want the node-tied topic-word
    distributions. This is a supervised topic-word estimator, not the full unsupervised LDA sampler;
    hence there is no `alpha` argument. The returned beta_hat averages the post-burn token counts
    (n_kw + beta_prior) and normalizes once — the averaged-counts point estimate, not the mean of
    per-sweep normalized rows.

    ORACLE / VALIDATOR — NOT the production fit path. This collapsed-Gibbs estimator is the reference
    the production SVI engine (DagGatedOnlineLDA, models/topic/dag_gated_lda.py) is validated
    against; no driver calls it. Used in tests only (its runtime — a Python per-token loop — is not tuned for production scale)."""
    rng = np.random.default_rng() if rng is None else rng
    K = lay.K
    counted = [_as_counts(d) for d in train_docs]
    Q = word_cooccurrence(counted, V)
    beta0 = recover_beta(Q, find_anchors(Q, K))
    if beta0.shape[0] < K:
        pad = np.full((K - beta0.shape[0], V), 1.0 / V)
        beta0 = np.vstack([beta0, pad])
    beta0 = beta0 + 1e-6
    beta0 /= beta0.sum(1, keepdims=True)
    n_kw = np.zeros((K, V))
    n_k = np.zeros(K)
    # Each label may be a scalar node id or a frontier set (comorbid patient). A comorbid patient
    # trains every block along the union of its frontier's closures — strictly better use of data.
    allowed = [lay.allowed_set(y if hasattr(y, "__iter__") else (y,)) for y in train_labels]
    words = [np.asarray(d, dtype=np.int64) for d in train_docs]
    Z = []
    for d in range(len(train_docs)):
        al = allowed[d]
        w = words[d]
        r = beta0[al][:, w].T
        r = r / r.sum(1, keepdims=True)
        zi = al[(rng.random(len(w))[:, None] < np.cumsum(r, 1)).argmax(1)]
        Z.append(zi)
        np.add.at(n_kw, (zi, w), 1.0)
        for k in zi:
            n_k[k] += 1.0
    Vb = V * beta_prior
    acc = np.zeros((K, V))
    nacc = 0
    for it in range(n_iter):
        for d in range(len(train_docs)):
            al = allowed[d]
            w = words[d]
            zi = Z[d]
            for i in range(len(w)):
                wi = w[i]
                k = zi[i]
                n_kw[k, wi] -= 1.0
                n_k[k] -= 1.0
                p = (n_kw[al, wi] + beta_prior) / (n_k[al] + Vb)
                p /= p.sum()
                knew = al[np.searchsorted(np.cumsum(p), rng.random())]
                zi[i] = knew
                n_kw[knew, wi] += 1.0
                n_k[knew] += 1.0
        if it >= burn:
            acc += n_kw + beta_prior
            nacc += 1
    beta_hat = acc / max(nacc, 1)
    beta_hat /= beta_hat.sum(1, keepdims=True)
    return beta_hat


def profile(doc, beta_hat, lay, *, alpha=0.1, n_iter=60, burn=30, rng=None):
    """Unmasked fold-in (topics fixed) -> per-node affinity = posterior mean mass on each node's
    block. The full profile IS the output; do not collapse to a single node.

    ORACLE / VALIDATOR — the Gibbs-oracle fold-in counterpart of `fit_gated`, NOT the production
    read-out. Test/oracle only."""
    rng = np.random.default_rng() if rng is None else rng
    K = lay.K
    w = np.asarray(doc, dtype=np.int64)
    ndk = np.zeros(K)
    zi = rng.integers(K, size=len(w))
    for k in zi:
        ndk[k] += 1.0
    acc = np.zeros(K)
    nacc = 0
    for it in range(n_iter):
        for i in range(len(w)):
            wi = w[i]
            k = zi[i]
            ndk[k] -= 1.0
            p = (ndk + alpha) * beta_hat[:, wi]
            s = p.sum()
            p = p / s if s > 0 else np.full(K, 1.0 / K)
            knew = int(np.searchsorted(np.cumsum(p), rng.random()))
            zi[i] = knew
            ndk[knew] += 1.0
        if it >= burn:
            acc += ndk / max(len(w), 1)
            nacc += 1
    th = acc / max(nacc, 1)
    return {u: float(th[lay.block[u]].sum()) for u in lay.nodes}

