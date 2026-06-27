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
