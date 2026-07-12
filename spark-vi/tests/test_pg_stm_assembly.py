"""F1 — block-Sigma assembly must return a positive-definite Sigma.

``PGSTMVI._assemble_sigma`` stitches independently-estimated blocks (a shared
background block from all docs, one [gate, foreground] block per group from that
group's docs, and background<->group cross-terms) into the full (K-1)x(K-1) Sigma.
The group<->group' cross-blocks are never co-active, so they are UNOBSERVED. The
original assembly zero-FILLED them and applied a single non-escalating +1e-8 jitter,
which can return a non-PD Sigma (at bg_k=2 the iw assembly had eigmin ~ -0.0114 and
Cholesky failed) — the "zero the free covariance" anti-pattern that
``_linalg.pd_complete`` (max-determinant PD completion, Dempster 1972) exists to
replace. The fitted Sigma feeds its own prior next iteration and is Cholesky-factored
downstream, so a non-PD assembly is a real defect independent of the estimator.

These tests pin the contract: the assembled Sigma is PD (min eigenvalue > 0) for BOTH
sigma_mode="iw" and the bg_k=2 corner that previously failed."""
import numpy as np
import pytest

from spark_vi.models.topic.pg_stm import PGSTMVI
from tests._stm_synth import gated_ln_corpus


def _fit_sigma(bg_k, seed=0):
    docs, part, _Sigma_true, _beta = gated_ln_corpus(
        group_weights={"A": 0.7, "B": 0.3}, fg_per_group=1, bg_k=bg_k,
        V=40, D=200, doc_len=30, seed=seed)
    P = docs[0].x.shape[0]
    return PGSTMVI(K=part.K, V=40, partition=part, P=P, n_iter=60,
                   sigma_mode="iw", seed=0).fit(docs)["Sigma"]


@pytest.mark.parametrize("bg_k", [2, 4])
def test_assembled_sigma_is_pd(bg_k):
    """The iw-assembled Sigma is positive-definite for bg_k=2 (previously eigmin<0,
    Cholesky failed) AND bg_k=4."""
    Sigma = _fit_sigma(bg_k)
    eigmin = float(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T)).min())
    assert eigmin > 0.0, f"bg_k={bg_k}: assembled Sigma not PD (eigmin={eigmin:.4g})"
    # Cholesky must succeed (the downstream consumer's actual requirement).
    np.linalg.cholesky(Sigma)


def test_assembled_sigma_pd_across_seeds():
    """PD-ness is not a lucky-seed artifact — holds across several corpora."""
    for seed in (0, 1, 2):
        Sigma = _fit_sigma(2, seed=seed)
        eigmin = float(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T)).min())
        assert eigmin > 0.0, f"seed={seed}: eigmin={eigmin:.4g}"
