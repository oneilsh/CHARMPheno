import numpy as np
from spark_vi.models.topic.spectral_init import (
    word_cooccurrence, find_anchors, recover_beta, spectral_init_beta)
from _stm_synth import (synthetic_ehr_corpus, synthetic_gated_corpus,
                        planted_recovery, foreground_recovers_group)

def test_cooccurrence_normalized_square():
    docs, _ = synthetic_ehr_corpus(K_rare=4, V=40, D=100, doc_len=20, bg_frac=0.5, seed=1)
    Q = word_cooccurrence(docs, 40)
    assert Q.shape == (40, 40) and np.isclose(Q.sum(), 1.0, atol=1e-6)

def test_spectral_init_recovers_nongated_planted():
    from spark_vi.models.topic.partition import TopicBlockPartition
    docs, planted = synthetic_ehr_corpus(K_rare=6, V=120, D=800, doc_len=30,
                                         bg_frac=0.5, seed=2)
    part = TopicBlockPartition(group_var="", background_k=12, foreground=())
    beta0 = spectral_init_beta(docs, part, 120)
    assert beta0.shape == (12, 120)
    assert planted_recovery(beta0, planted, thresh=0.4) >= 4

def test_block_aware_init_recovers_rare_group_foreground():
    """The decisive gated property: a rare group's foreground anchor lands its
    planted phenotype at INIT, because it is found on the within-group Q
    (undiluted by the majority) and deflated against the background span."""
    docs, planted, part = synthetic_gated_corpus(
        groups=("maj", "rare"), fg_per_group=2, bg_k=3, V=240, D=1200,
        doc_len=30, bg_frac=0.6, seed=3)
    # make 'rare' a minority arm
    docs = [d for i, d in enumerate(docs) if ("rare" not in d.groups) or (i % 4 == 0)]
    beta0 = spectral_init_beta(docs, part, 240)
    assert foreground_recovers_group(beta0, part, "rare", planted, thresh=0.4)
