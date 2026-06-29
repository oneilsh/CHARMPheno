"""Scalable raw-Gaussian spectral recovery vs the dense oracle (ADR 0032).

Task 8 (Phase B) RISK-GATE characterization. The scalable init
(``scalable_spectral_init_beta``) projects the V×V co-occurrence to V×d via a
RAW Gaussian sketch (Johnson–Lindenstrauss), d = ``default_projection_dim`` =
max(K, ceil(eps^-2 · ln V)). This is APPROXIMATE — the JL projection distorts
the anchor geometry by O(1/√d). CHARMPheno's purpose is RARE-subgroup phenotype
discovery, so the load-bearing empirical question is: does the raw-Gaussian
projection recover planted topics — ESPECIALLY a thinned rare arm — as well as
the dense oracle, at a V where d ≪ V?

The dense ``spectral_init.spectral_init_beta`` is the eps-INDEPENDENT oracle.
Recovery bar (both tests): mass-in-planted-block. A planted block here is
250/3000 of vocab, so uniform assignment scores ~0.083; THRESH=0.3 is ~3.6x
enrichment over uniform — clearly "recovered", not noise.

Characterization curve at V=3000 (deterministic, seed=31337), recovery bar 0.3:

    eps   d(ng) d/V    | NG scal/dense | rare scal/dense | rare_mass(scal)
    0.05  3203  1.07   |   5 / 6       | True  / True    |   ~0.45
    0.10   801  0.27   |   5 / 6       | True  / True    |   ~0.42
    0.15   356  0.12   |   5 / 6       | True  / True    |   ~0.40
    0.20   201  0.067  |   5 / 6       | True  / True    |    0.371
    0.25   129  0.043  |   -           | True  / True    |    0.360
    0.30    89  0.030  |   6 / 6       | False / True    |    0.243   <- rare lost

VERDICT: scalable recovers the rare arm at a memory-reasonable d. eps=0.20 gives
d=201 (a 15x reduction from V=3000) with the rare-arm mass at 0.371 — a
comfortable margin over the 0.3 bar — while non-gated recovery (5/6) stays within
1 of the dense oracle (6/6). The rare arm is only LOST at eps=0.30 (d=89, 33x
reduction), so eps=0.20 is the recommended default: the LARGEST eps (smallest d,
best memory) at which BOTH conditions hold. RECOMMENDATION: change the production
default eps 0.1 -> 0.20 (Task 7's change, informed here; production defaults are
NOT touched in this characterization task).
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

# Recommended default from the V=3000 sweep below: the largest eps (smallest d)
# at which non-gated scalable recovery is within 1 of dense AND the thinned rare
# arm is still recovered. d = ceil(0.2^-2 · ln 3000) = 201, a ~15x reduction.
RECOMMENDED_EPS = 0.2
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
    beta = scalable_spectral_init_beta(
        rdd, part, V=V, eps=RECOMMENDED_EPS, seed=SEED, min_doc_freq=3
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
def test_scalable_matches_dense_nongated_at_recommended_eps(spark):
    """Non-gated: scalable planted recovery within 1 of the dense oracle.

    Pins the recommended-eps non-gated equivalence at a V where d=201 ≪ V=3000
    (a ~15x reduction). The raw-Gaussian NNLS recovery spreads phenotype mass a
    bit more than dense, so we assert the curve's tolerance: scalable recovers at
    least (dense - 1) planted topics under the same enrichment bar.
    """
    V = 3000
    docs, planted, part = _nongated_corpus(V)
    d = default_projection_dim(part.K, V, RECOMMENDED_EPS)
    assert d < V  # d must be a real reduction at the pinned eps

    dense = spectral_init_beta(docs, part, V)
    dense_rec = planted_recovery(dense, planted, thresh=THRESH)
    assert dense_rec == planted.shape[0]  # oracle saturates; comparison is valid

    rdd = spark.sparkContext.parallelize(docs, numSlices=4)
    scal = scalable_spectral_init_beta(
        rdd, part, V=V, eps=RECOMMENDED_EPS, seed=SEED, min_doc_freq=3
    )
    scal_rec = planted_recovery(scal, planted, thresh=THRESH)

    assert scal_rec >= dense_rec - 1, (
        f"scalable {scal_rec} vs dense {dense_rec} at eps={RECOMMENDED_EPS} "
        f"(d={d}, d/V={d / V:.3f})"
    )


@pytest.mark.slow
def test_scalable_recovers_rare_arm_at_recommended_eps(spark):
    """RISK-GATE: the thinned rare arm is recovered at the recommended eps.

    The load-bearing rare-phenotype check. The rare arm (~19% of docs) is
    recovered by BOTH the dense oracle and the scalable raw-Gaussian sketch at
    eps=0.2 (d=201, ~15x reduction). At the more aggressive eps=0.3 (d=89) the
    scalable rare-arm mass falls below the bar — so 0.2 is the boundary, pinned
    here as the recommended default.
    """
    V = 3000
    docs, planted, part = _gated_corpus(V)
    d = default_projection_dim(part.K, V, RECOMMENDED_EPS)
    assert d < V

    dense = spectral_init_beta(docs, part, V)
    assert foreground_recovers_group(dense, part, "rare", planted, thresh=THRESH)

    rdd = spark.sparkContext.parallelize(docs, numSlices=4)
    scal = scalable_spectral_init_beta(
        rdd, part, V=V, eps=RECOMMENDED_EPS, seed=SEED, min_doc_freq=3
    )
    assert foreground_recovers_group(scal, part, "rare", planted, thresh=THRESH), (
        f"rare arm NOT recovered by scalable at eps={RECOMMENDED_EPS} "
        f"(d={d}, d/V={d / V:.3f})"
    )


@pytest.mark.slow
def test_rare_arm_lost_at_aggressive_eps(spark):
    """Documents the cliff: at eps=0.3 (d=89) the scalable rare arm is LOST.

    The dense oracle still recovers it (eps-independent), so this pins WHY the
    recommended default is 0.2 and not larger — the raw-Gaussian projection at
    d=89 (a 33x reduction) is too lossy for the rare phenotype. This is the
    go/no-go boundary; if a future change pushes the default past it, this test
    flips and forces a re-evaluation.
    """
    V = 3000
    aggressive_eps = 0.3
    docs, planted, part = _gated_corpus(V)
    d = default_projection_dim(part.K, V, aggressive_eps)
    assert d < V

    dense = spectral_init_beta(docs, part, V)
    assert foreground_recovers_group(dense, part, "rare", planted, thresh=THRESH)

    rdd = spark.sparkContext.parallelize(docs, numSlices=4)
    scal = scalable_spectral_init_beta(
        rdd, part, V=V, eps=aggressive_eps, seed=SEED, min_doc_freq=3
    )
    # The rare arm is lost here — the projection is too lossy at d=89. If this
    # ever flips to recovered, the cliff has moved and the recommended default
    # should be revisited.
    assert not foreground_recovers_group(
        scal, part, "rare", planted, thresh=THRESH
    ), f"rare arm UNEXPECTEDLY recovered at eps={aggressive_eps} (d={d})"
