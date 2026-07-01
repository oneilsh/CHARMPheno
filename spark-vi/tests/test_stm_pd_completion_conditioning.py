"""End-to-end conditioning proof for the gated full-Sigma PD completion.

Two layers (see docs/insights/0032 Resolution; design spec
docs/superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md):

  Layer 1 — deterministic block-arrow A/B: the naive zero-COVARIANCE pin on the
  free cross-pair is near-singular; pd_complete (zero free PRECISION) is
  well-conditioned, preserves observed entries, and zeroes free-pair precision.
  This is the on-disk stand-in for exp 0025's headline cond drop (no cluster).

  Layer 2 — the same structure end-to-end through the M-step on a TERM-SHARING
  gated corpus (foreground topics share vocabulary, calibrated to the real HF
  lda_pasc beta's mean pairwise Jaccard ~0.35), so the proof is not an artifact
  of the disjoint-vocab separation baked into the older synthetic generators.

Domain-agnostic: integer token ids only.
"""
from __future__ import annotations
import numpy as np
import pytest

from spark_vi.models.topic._linalg import pd_complete
from _stm_synth import (
    synthetic_gated_corpus_overlap,
    topic_support_jaccard,
)


# --- term-sharing generator guard ------------------------------------------

def test_overlap_generator_actually_shares_terms():
    """The realism premise: the overlapping generator must produce foreground
    topics that genuinely share vocabulary (unlike synthetic_gated_corpus, whose
    per-topic vocab is disjoint). Calibrated against the real HF beta's measured
    mean pairwise topic Jaccard ~0.35; we require materially-nonzero overlap so a
    future swap to a separated generator fails loudly here."""
    _docs, planted, _part = synthetic_gated_corpus_overlap(
        groups=("A", "B"), fg_per_group=2, bg_k=3, V=120, D=40,
        doc_len=60, bg_frac=0.5, shared_frac=0.5, seed=0)
    jac = topic_support_jaccard(planted)
    assert jac >= 0.15, f"topics not sharing terms (mean Jaccard {jac:.3f})"
    # and not the degenerate all-identical case
    assert jac <= 0.95, f"topics nearly identical (mean Jaccard {jac:.3f})"


# --- Layer 1: deterministic block-arrow conditioning A/B -------------------

def _cond(M):
    w = np.linalg.eigvalsh(0.5 * (M + M.T))
    return float(w.max() / w.min()) if w.min() > 0 else float("inf")


@pytest.mark.parametrize("r,naive_regime", [
    (0.7071, "near_singular_pd"),   # det = 1 - 2r^2 -> ~0+  : PD but cond huge
    (0.9, "indefinite"),            # det < 0               : not even PSD
])
def test_blockarrow_naive_pin_illconditioned_pd_complete_fixes(r, naive_regime):
    """Block arrow: one background topic correlates r with BOTH foreground A and
    B; the A<->B pair is free (no joint support). The naive zero-COVARIANCE pin
    drives a vanishing eigenvalue along (0, 1, -1) -> near-singular (or indefinite)
    assembly. pd_complete zeroes the free PRECISION instead and is well-conditioned
    by construction. This is the on-disk analog of exp 0025's cond drop.

    Eigenvalues of [[1,r,r],[r,1,0],[r,0,1]] are {1, 1+r*sqrt(2), 1-r*sqrt(2)};
    the min collapses as r -> 1/sqrt(2)."""
    bg, A, B = 0, 1, 2
    M = np.array([[1.0, r, r], [r, 1.0, 0.0], [r, 0.0, 1.0]])
    observed = np.ones((3, 3), dtype=bool)
    observed[A, B] = observed[B, A] = False        # the only free pair

    naive = M.copy()                                # free entry pinned to 0
    if naive_regime == "near_singular_pd":
        assert np.min(np.linalg.eigvalsh(naive)) > 0          # PD...
        assert _cond(naive) >= 1e3, _cond(naive)              # ...but near-singular
    else:
        assert np.min(np.linalg.eigvalsh(naive)) < 0          # genuinely indefinite

    Sig = pd_complete(M, observed)

    # well-conditioned, SPD, by construction
    assert np.min(np.linalg.eigvalsh(Sig)) > 0
    assert _cond(Sig) <= 50.0, _cond(Sig)
    # observed entries preserved exactly
    for i in range(3):
        for j in range(3):
            if observed[i, j]:
                assert abs(Sig[i, j] - M[i, j]) < 1e-9, (i, j, Sig[i, j], M[i, j])
    # zero precision on the free pair == conditional independence (max-det)
    P = np.linalg.inv(Sig)
    assert abs(P[A, B]) < 1e-7, P[A, B]


def test_gated_shape_naive_indefinite_pd_complete_well_conditioned():
    """Faithful gated assembly (the exp 0021 shape): 4 background topics, a cancer
    foreground block {4,5} and a dementia block {6,7}. Background couples to BOTH
    blocks (cancer and dementia patients share comorbidity background), the two
    foreground blocks are internally correlated but DISTINCT (not collinear), and
    EVERY cancer<->dementia cross-pair is free (no comorbid docs). The naive
    zero-covariance pin on those cross-pairs is genuinely INDEFINITE; pd_complete
    fills them with the zero-precision (conditional-independence) values and is
    well-conditioned, exactly as exp 0025 predicts on the cluster.

    Note: this holds while the observed bg<->fg coupling admits a PD completion.
    Pushed past the feasibility boundary (coupling ~0.49 here) even the max-det
    completion degrades and routes to the min-Frobenius fallback — a real limit,
    not asserted as success."""
    bg = [0, 1, 2, 3]
    cancer, dementia = [4, 5], [6, 7]
    K = 8
    M = np.eye(K)
    for a in bg:                                   # mild background mutual corr
        for b in bg:
            if a < b:
                M[a, b] = M[b, a] = 0.15
    for a in bg:                                   # bg -> both foreground blocks
        for c in cancer + dementia:
            M[a, c] = M[c, a] = 0.40
    for blk in (cancer, dementia):                 # internal block correlation
        i, j = blk
        M[i, j] = M[j, i] = 0.30
    observed = np.ones((K, K), dtype=bool)
    for c in cancer:                               # cancer<->dementia all free
        for d in dementia:
            observed[c, d] = observed[d, c] = False

    naive = M.copy()                               # cross-pairs pinned to 0
    assert np.min(np.linalg.eigvalsh(naive)) < 0, "naive zero-pin must be indefinite"

    Sig = pd_complete(M, observed)
    assert np.allclose(Sig, Sig.T)
    assert np.min(np.linalg.eigvalsh(Sig)) > 0          # SPD where naive was not
    assert _cond(Sig) <= 50.0, _cond(Sig)               # well-conditioned
    # observed entries preserved exactly; free cross-pairs zero-precision
    for i in range(K):
        for j in range(K):
            if observed[i, j]:
                assert abs(Sig[i, j] - M[i, j]) < 1e-8, (i, j)
    P = np.linalg.inv(Sig)
    for c in cancer:
        for d in dementia:
            assert abs(P[c, d]) < 1e-6, (c, d, P[c, d])


# --- Layer 2: end-to-end recovery invariance to full-Sigma condition --------

def test_recovery_invariant_to_full_sigma_condition_number():
    """Unit-diagonal M-step: Sigma is a correlation matrix each seed; recovery
    is stable and the diagonal is pinned to 1 (no variance runaway to track)."""
    from _stm_synth import (synthetic_gated_corpus_overlap, fit_stm,
                            planted_recovery)

    recs, gps = [], []
    for seed in range(4):
        docs, planted, part = synthetic_gated_corpus_overlap(
            groups=("A", "B"), fg_per_group=2, bg_k=4, V=80, D=150,
            doc_len=70, bg_frac=0.4, shared_frac=0.5, seed=seed)
        # n_iter=100: recovery-stability is a property of the CONVERGED fit. The
        # unit-diagonal M-step clamps supported off-diagonals to [-1,1] (a valid
        # correlation), which on this shared-term overlap corpus removes the
        # spurious |r|>1 entries a shorter fit had leaned on and slightly slows
        # convergence for one seed; by 100 iters all seeds settle within 1 topic.
        gp = fit_stm(docs, K=part.K, V=80, sigma_init=1.0, n_iter=100,
                     seed=42, partition=part, reference_topic=False)
        beta = gp["lambda"] / gp["lambda"].sum(axis=1, keepdims=True)
        recs.append(planted_recovery(beta, planted))
        gps.append(gp)

    # recovery is stable across all seeds (at convergence).
    assert min(recs) >= max(recs) - 1, recs                  # within 1 topic
    for gp in gps:
        np.testing.assert_allclose(np.diag(gp["Sigma"]), 1.0, atol=1e-12)
