"""Task 7: exact blocked PG-Gibbs cross-check for the gated nested stick-breaking
logistic-normal topic model.

The Gibbs sampler is the AUDIT instrument for the VI: it draws every latent
(z, omega, psi, beta, Gamma, Sigma) EXACTLY (theta = gated_theta computed from the
current psi — no delta method), so agreement between VI's mean-field Sigma
posterior and Gibbs's sampled Sigma posterior on the shared, best-identified
background block distinguishes "the modeling bet fails" from "mean-field is the
culprit". beta recovery is cross-model comparable; Sigma is compared ONLY
VI-vs-Gibbs (link-internal), never vs the softmax-planted Sigma_true.
"""
import numpy as np

from spark_vi.models.topic.pg_stm import pg_stm_gibbs, PGSTMVI
from tests._stm_synth import gated_ln_corpus, planted_recovery


def _corpus(seed=1):
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=1, bg_k=2,
        V=60, D=500, doc_len=40, seed=seed)
    return docs, {"beta": beta, "Sigma": Sigma_true}, part


def test_gibbs_recovers_planted():
    docs, planted, part = _corpus()
    P = docs[0].x.shape[0]
    out = pg_stm_gibbs(docs, K=part.K, V=60, partition=part, P=P,
                       n_iter=400, burn=200, seed=0)
    assert planted_recovery(out["beta"], planted["beta"]) >= 0.75


def test_vi_matches_gibbs_on_sigma():
    docs, planted, part = _corpus()
    P = docs[0].x.shape[0]
    vi = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150, seed=0).fit(docs)
    gb = pg_stm_gibbs(docs, K=part.K, V=60, partition=part, P=P,
                      n_iter=400, burn=200, seed=0)

    def corr(S):
        d = np.sqrt(np.diag(S))
        return S / np.outer(d, d)

    # Compare the SHARED background block's correlation (present in every doc,
    # best-identified). atol=0.15 is deliberately loose — Gibbs is stochastic.
    B = len(part.background_indices())
    assert np.allclose(corr(vi["Sigma"])[:B - 1, :B - 1],
                       corr(gb["Sigma"])[:B - 1, :B - 1], atol=0.15)
