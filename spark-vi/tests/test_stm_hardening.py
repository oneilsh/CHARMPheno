"""STM hardening tests: ground-truth planted corpus recovery + xfail baseline."""
import numpy as np
import pytest
from _stm_synth import (synthetic_ehr_corpus, synthetic_gated_corpus,
                        planted_recovery, foreground_recovers_group,
                        fit_stm, final_sigma_range)


def test_recovery_perfect_on_ground_truth():
    _, planted = synthetic_ehr_corpus(K_rare=8, V=300, D=200, doc_len=30,
                                      bg_frac=0.7, seed=0)
    assert planted_recovery(planted, planted, thresh=0.5) == 8


def test_gated_corpus_shapes_and_groups():
    docs, planted, part = synthetic_gated_corpus(
        groups=("a", "b"), fg_per_group=2, bg_k=3, V=200, D=300,
        doc_len=30, bg_frac=0.6, seed=0)
    assert planted.shape[0] == part.K == 3 + 2 * 2
    assert any("a" in d.groups for d in docs) and any("b" in d.groups for d in docs)
    # ground-truth beta recovers each group's foreground by construction
    assert foreground_recovers_group(planted, part, "a", planted, thresh=0.5)


def test_unit_diagonal_bounds_sigma_regardless_of_init():
    """Unit-diagonal M-step structurally removes the sigma_init Sigma-blowup the
    old xfail documented (insight 0033): the diagonal is pinned to 1 every
    M-step regardless of sigma_init, so max Sigma entry <= 1 and min entry is a
    valid correlation (>= -1). This is an estimator invariant (holds after any
    M-step, not a convergence property), so a short fit suffices — the long
    250-iter run the old blowup test needed is no longer meaningful. Init-
    independence of topic RECOVERY is covered separately by
    test_spectral_init_makes_fit_init_independent (this file)."""
    docs, planted = synthetic_ehr_corpus(K_rare=8, V=300, D=1500, doc_len=30,
                                         bg_frac=0.7, seed=0)
    smax, smin = [], []
    for si in (1.0, 5.0, 20.0):
        gp = fit_stm(docs, K=40, V=300, sigma_init=si, batch=100, n_iter=30)
        lo, hi = final_sigma_range(gp)
        smin.append(lo); smax.append(hi)
    assert max(smax) <= 1.0 + 1e-9, smax    # diagonal pinned; no sigma_init blowup
    assert min(smin) >= -1.0 - 1e-9, smin   # valid correlation entries


from spark_vi.models.topic.spectral_init import spectral_init_beta
from spark_vi.models.topic.partition import TopicBlockPartition


@pytest.mark.slow
def test_spectral_init_makes_fit_init_independent():
    docs, planted = synthetic_ehr_corpus(K_rare=8, V=300, D=1500, doc_len=30,
                                         bg_frac=0.7, seed=0)
    part = TopicBlockPartition(group_var="", background_k=40, foreground=())
    beta0 = spectral_init_beta(docs, part, 300)
    recos = []
    for si in (1.0, 5.0, 20.0):
        gp = fit_stm(docs, K=40, V=300, sigma_init=si, n_iter=60,
                     init_data={"spectral_beta": beta0})
        beta = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        recos.append(planted_recovery(beta, planted))
    assert min(recos) >= 6   # recovery no longer depends on sigma_init


# test_sigma_prior_reduces_blowup removed: the inverse-Wishart Σ prior
# (sigma_prior_scale / sigma_prior_count) was removed from OnlineSTM. The Σ M-step
# is now block-wise unit-diagonal (ADR 0034): pinning Σ_ii=1 removes the variance
# degree of freedom entirely, so there is no blowup for an IW anchor to reduce.
# Spec: docs/superpowers/specs/2026-07-01-stm-blockwise-unit-diagonal-correlation-design.md
