"""Scalable raw-Gaussian spectral recovery vs the dense oracle (ADR 0032).

Task 8 (Phase B) RISK-GATE characterization. The scalable init
(``scalable_spectral_init_beta``) projects the V×V co-occurrence to V×d via a
RAW Gaussian sketch (Johnson–Lindenstrauss), d = ``default_projection_dim`` =
min(V, max(K, 1000)) — a FIXED ~1000 per the Arora et al. 2013 (ICML) and Mimno
anchor-words reference implementation convention. This is APPROXIMATE — the JL
projection distorts the anchor geometry by O(1/sqrt(d)). CHARMPheno's purpose
is RARE-subgroup phenotype discovery, so the load-bearing empirical question is:
does the raw-Gaussian projection recover planted topics — ESPECIALLY a thinned
rare arm — as well as the dense oracle, at a V where d < V?

The dense ``spectral_init.spectral_init_beta`` is the oracle (no sketch).
Recovery bar (both tests): mass-in-planted-block. For the NON-GATED corpus a
planted block is 250/3000 of vocab, so uniform assignment scores ~0.083;
THRESH=0.3 is ~3.6x enrichment over uniform — clearly "recovered", not noise.

Characterization curve at V=3000 (deterministic, seed=31337), recovery bar 0.3
(expressed as explicit d rather than eps for clarity):

    d     d/V    | NG scal/dense | rare scal/dense | rare_mass(scal)
    3203  1.07   |   5 / 6       | True  / True    |   ~0.45
     801  0.27   |   5 / 6       | True  / True    |   ~0.42
     356  0.12   |   5 / 6       | True  / True    |   ~0.40
     201  0.067  |   5 / 6       | True  / True    |    0.371
    1000  0.33   |   5 / 6       | True  / True    |   ~0.42   <- DEFAULT
     129  0.043  |   -           | True  / True    |    0.360
      89  0.030  |   6 / 6       | False / True    |    0.243   <- rare lost

VERDICT: scalable recovers the rare arm at d=1000 (d/V=0.33, a real 3x reduction
from V=3000) with rare-arm mass well above the 0.3 bar — comfortably above the
cliff at d=89. The default is fixed at ~1000 (Mimno/Arora), V-independent.
The cliff test (d=89 loses the rare arm) guards the lower bound; d=1000 is
safely above it.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.spectral_init import spectral_init_beta
from spark_vi.models.topic.spectral_init_scalable import (
    default_projection_dim,
    scalable_spectral_init_beta,
)
from tests._stm_synth import (
    foreground_recovers_group,
    planted_recovery,
    synthetic_ehr_corpus,
    synthetic_gated_corpus,
)

SEED = 31337
# Mass-in-block bar: a planted block is 250/3000 of vocab (uniform ~0.083), so
# 0.3 is ~3.6x enrichment — recovered, not noise. Used for BOTH scal and dense.
THRESH = 0.3


def _nongated_corpus(V):
    """K=6 planted topics, 12 background slots (2x anchor over-provisioning).

    doc_len=60 / bg_frac=0.4 keep per-doc phenotype co-occurrence dense enough at
    V=3000 that the dense oracle cleanly recovers all 6 (the comparison is only
    meaningful where the oracle itself succeeds).
    """
    docs, planted = synthetic_ehr_corpus(
        K_rare=6, V=V, D=4000, doc_len=60, bg_frac=0.4, seed=2
    )
    part = TopicBlockPartition(group_var="", background_k=12, foreground=())
    return docs, planted, part


def _gated_corpus(V):
    """Majority + rare arm, rare thinned to ~1 in 4 of its docs (~19% of corpus).

    Mirrors the dense test_block_aware_init_recovers_rare_group_foreground setup
    at the larger V; doc_len=80 keeps the rare arm's within-group phenotype
    co-occurrence recoverable by the dense oracle.
    """
    docs, planted, part = synthetic_gated_corpus(
        groups=("maj", "rare"), fg_per_group=2, bg_k=3, V=V, D=4000,
        doc_len=80, bg_frac=0.5, seed=3,
    )
    docs = [
        d for i, d in enumerate(docs)
        if ("rare" not in d.groups) or (i % 4 == 0)
    ]
    return docs, planted, part


# ---------------------------------------------------------------------------
# Fast structural sanity (unmarked): at a small V the scalable orchestrator
# produces a valid, non-uniform, spectrally-seeded beta and recovers a clearly
# planted non-gated signal. Cheap guard that the projection path is wired up.
# ---------------------------------------------------------------------------


def test_scalable_recovers_nongated_small_v_structural(spark):
    V = 240
    docs, planted = synthetic_ehr_corpus(
        K_rare=4, V=V, D=600, doc_len=40, bg_frac=0.4, seed=5
    )
    part = TopicBlockPartition(group_var="", background_k=8, foreground=())
    rdd = spark.sparkContext.parallelize(docs, numSlices=4)
    # Use the DEFAULT d (min(V, max(K, 1000))); at V=240 that is d=240 (= V,
    # so no reduction at this small V — structural test, not a reduction test).
    beta = scalable_spectral_init_beta(
        rdd, part, V=V, seed=SEED, min_doc_freq=3
    )

    assert beta.shape == (part.K, V)
    assert (beta >= 0).all()
    for k in range(part.K):
        row = beta[k]
        if row.sum() == 0:
            continue  # short-fill left a zero row; never NaN
        np.testing.assert_allclose(row.sum(), 1.0, atol=1e-9, rtol=0)
        assert np.abs(row - 1.0 / V).max() > 1e-6  # spectral seeding took effect
    # At least most planted topics surface (small-V structural smoke check).
    assert planted_recovery(beta, planted, thresh=THRESH) >= planted.shape[0] - 1


# ---------------------------------------------------------------------------
# Pinned equivalence at the RECOMMENDED eps on V=3000 (d ≪ V). Slow: builds two
# ~4000-doc corpora and runs the dense oracle + scalable Spark passes.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scalable_matches_dense_nongated_at_default_d(spark):
    """Non-gated: scalable planted recovery within 1 of the dense oracle.

    Pins the default-d non-gated equivalence at V=3000: default d=1000 (fixed
    ~1000 per Mimno/Arora; d/V=0.33, a real 3x reduction). The raw-Gaussian NNLS
    recovery spreads phenotype mass a bit more than dense, so we assert the curve's
    tolerance: scalable recovers at least (dense - 1) planted topics under the same
    enrichment bar.
    """
    V = 3000
    docs, planted, part = _nongated_corpus(V)
    d = default_projection_dim(part.K, V)   # min(V, max(K, 1000)) = 1000 at V=3000
    assert d < V  # d=1000 is a real reduction at V=3000

    dense = spectral_init_beta(docs, part, V)
    dense_rec = planted_recovery(dense, planted, thresh=THRESH)
    assert dense_rec == planted.shape[0]  # oracle saturates; comparison is valid

    rdd = spark.sparkContext.parallelize(docs, numSlices=4)
    scal = scalable_spectral_init_beta(
        rdd, part, V=V, seed=SEED, min_doc_freq=3   # d omitted → default d=1000
    )
    scal_rec = planted_recovery(scal, planted, thresh=THRESH)

    assert scal_rec >= dense_rec - 1, (
        f"scalable {scal_rec} vs dense {dense_rec} at default d={d} "
        f"(d/V={d / V:.3f})"
    )


@pytest.mark.slow
def test_scalable_recovers_rare_arm_at_default_d(spark):
    """RISK-GATE: the thinned rare arm is recovered at the default d=1000.

    The load-bearing rare-phenotype check. The rare arm (~19% of docs) is
    recovered by BOTH the dense oracle and the scalable raw-Gaussian sketch at
    the default d=1000 (fixed ~1000 per Mimno/Arora; d/V=0.33, a real 3x
    reduction from V=3000). At d=89 the scalable rare-arm mass falls below the
    bar — d=1000 is comfortably above that cliff.

    GATED CORPUS: the gated foreground block is rest=1500 // n_fg=4 = 375 words
    wide; uniform baseline is 1/8 = 0.125 (bg_k=3 + 4 fg slots, bg owns 3/8 of
    vocab). The 0.3 bar is ~2.4x enrichment over that uniform baseline.
    """
    V = 3000
    docs, planted, part = _gated_corpus(V)
    d = default_projection_dim(part.K, V)   # min(V, max(K, 1000)) = 1000 at V=3000
    assert d < V

    dense = spectral_init_beta(docs, part, V)
    assert foreground_recovers_group(dense, part, "rare", planted, thresh=THRESH)

    rdd = spark.sparkContext.parallelize(docs, numSlices=4)
    scal = scalable_spectral_init_beta(
        rdd, part, V=V, seed=SEED, min_doc_freq=3   # d omitted → default d=1000
    )
    assert foreground_recovers_group(scal, part, "rare", planted, thresh=THRESH), (
        f"rare arm NOT recovered by scalable at default d={d} "
        f"(d/V={d / V:.3f})"
    )


@pytest.mark.slow
def test_rare_arm_lost_at_small_d(spark):
    """Documents the lower-bound cliff: at d=89 the scalable rare arm is LOST.

    The dense oracle still recovers it (no sketch), so this pins the lower bound
    of the projection-dimension range — the raw-Gaussian projection at d=89 (a
    33x reduction from V=3000) is too lossy for the rare phenotype. This is the
    go/no-go boundary; if a future change drops the default d below it, this test
    flips and forces a re-evaluation. d=1000 (the default) is safely above this
    cliff (3x reduction vs 33x, well within the characterization curve).
    """
    V = 3000
    small_d = 89   # the measured cliff: d=89 is too lossy for the rare arm
    docs, planted, part = _gated_corpus(V)
    assert small_d < V

    dense = spectral_init_beta(docs, part, V)
    assert foreground_recovers_group(dense, part, "rare", planted, thresh=THRESH)

    rdd = spark.sparkContext.parallelize(docs, numSlices=4)
    scal = scalable_spectral_init_beta(
        rdd, part, V=V, d=small_d, seed=SEED, min_doc_freq=3
    )
    # The rare arm is lost here — the projection is too lossy at d=89. If this
    # ever flips to recovered, the cliff has moved and the default should be
    # revisited.
    assert not foreground_recovers_group(
        scal, part, "rare", planted, thresh=THRESH
    ), f"rare arm UNEXPECTEDLY recovered at d={small_d}"
