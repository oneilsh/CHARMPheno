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


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="random-init STM is sigma_init-unstable; "
                   "fixed by spectral init (insight 0029)")
def test_baseline_random_init_is_unstable():
    docs, planted = synthetic_ehr_corpus(K_rare=8, V=300, D=1500, doc_len=30,
                                         bg_frac=0.7, seed=0)
    recos, smax = [], []
    for si in (1.0, 5.0, 20.0):
        gp = fit_stm(docs, K=40, V=300, sigma_init=si, batch=100, n_iter=250)
        beta = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        recos.append(planted_recovery(beta, planted)); smax.append(final_sigma_range(gp)[1])
    assert min(recos) >= 6 and max(smax) < 1e3


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


# test_sigma_prior_reduces_blowup removed (Task 2): the inverse-Wishart Σ prior
# (sigma_prior_scale / sigma_prior_count) was removed from OnlineSTM in favor of
# the pd_complete covariance-selection M-step. Σ conditioning is now controlled
# by the completion + min_pair_support, not an IW shrink lever.
# Spec: docs/superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md
