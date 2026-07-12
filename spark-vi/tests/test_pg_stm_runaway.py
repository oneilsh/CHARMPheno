"""Task 8 — the milestone-1 gate: does a PROPER inverse-Wishart posterior cure the
Sigma instability that an un-regularized point estimate (scatter/n, fed back each
iteration) produces on a scarce-group gated corpus?

Estimator isolation: the two fits below share the SAME gated nested stick-breaking
model and the SAME per-doc E-step; they differ ONLY in the Sigma M-step
(``sigma_mode``): ``"mle"`` = un-regularized ``scatter/n`` point estimate, ``"iw"`` =
block inverse-Wishart posterior mean (proper prior nu0>dim+1 => finite PD mean even
at n_docs->0). Any divergence is therefore attributable to the ESTIMATOR, not the
link — the whole point of the checkpoint.

MEASURED RESULT, HONEST REFRAME (exp 0049; F1/F3 after the assembly fix). Two things
the toy scale does NOT show, recorded so the next stage is not built on an overclaim:

  1. The PG + stick-breaking MLE does NOT reproduce the softmax point-EM's 10^10
     magnitude blow-up (insight 0033) — it converges to a modestly-inflated fixed
     point (max|Sigma| ~ 4, no 1e3 divergence).
  2. After ``_assemble_sigma`` was fixed to complete the unobserved group<->group'
     cross-blocks with the max-determinant PD completion (``_linalg.pd_complete``)
     instead of ZERO-FILLING them, the un-regularized MLE Sigma is NO LONGER
     indefinite. The earlier "MLE eigmin ~ -0.30, Cholesky fails" contrast was a
     block-STITCHING artifact of the zero-fill (the whole-branch review proved the
     -0.30 neg-eigenvector loaded on the zeroed cross-block), NOT an estimator
     variance runaway. With the fix, BOTH estimators are PD at toy scale.

What SURVIVES as a genuine (but MILD, non-decisive) estimator signal at toy scale:
the raw ``scatter/n`` MLE blocks sit ON the PD boundary — the completion's
min-Frobenius fallback has to floor them, so MLE eigmin lands at ~+0.000 — while the
IW posterior blocks are strictly PD-completable and land comfortably interior
(eigmin ~ +0.012..+0.018), with a uniformly smaller max|Sigma| (IW more bounded). So
IW is better-conditioned, but the toy K=6 / D=1000 corpus does NOT decisively
separate the two the way the confounded zero-fill contrast appeared to.

THE DECISIVE runaway-cure test therefore moves to SCALE / real data (the insight-0033
10^10 regime is larger-K / no-reference-topic / real corpus) — sub-project #2's
distributed PG-SVI on exp-0027. This file now asserts the honestly-supportable toy
claims: IW is PD + bounded + recovers beta (machinery validated), both estimators are
PD after the fix (no toy runaway), and IW is the better-conditioned of the two.

The corpus is bg_k=4 (a genuine 3-stick background CORRELATION block estimated from
all D docs). Historically bg_k=2 was chosen to dodge the gateA<->gateB zero-fill; that
dodge is now unnecessary (pd_complete handles the cross-block), but bg_k=4 is kept so
the IW posterior has an identified multi-stick background covariance to regularize."""
import numpy as np
import pytest

from spark_vi.models.topic.pg_stm import PGSTMVI
from tests._stm_synth import gated_ln_corpus, fit_stm, final_sigma_range, planted_recovery


def _scarce_corpus(seed=0):
    # Group B's foreground used by ~3% of docs (ess ~ 20-30): the weakly-identified-
    # variance regime that drove the point-EM runaway (insight 0033). bg_k=4 gives a
    # real background correlation block (see module docstring for why not bg_k=2).
    return gated_ln_corpus(group_weights={"A": 0.97, "B": 0.03}, fg_per_group=1, bg_k=4,
                           V=60, D=1000, doc_len=40, seed=seed)


@pytest.fixture(scope="module")
def scarce_fits():
    """Fit MLE and IW ONCE on the same scarce corpus (same E-step, only sigma_mode
    differs) and share across the estimator-isolation tests."""
    docs, part, Sigma_true, beta = _scarce_corpus()
    P = docs[0].x.shape[0]
    mle = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150,
                  sigma_mode="mle", seed=0).fit(docs)
    iw = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=150,
                 sigma_mode="iw", seed=0).fit(docs)
    return {"part": part, "beta": beta, "mle": mle, "iw": iw}


def test_iw_posterior_is_pd_bounded_and_finite(scarce_fits):
    """MACHINERY GATE (the honestly-supportable toy claim). The block inverse-Wishart
    posterior mean over the scarce-group E-step is finite, bounded (max|Sigma| < 100),
    and strictly positive-definite (Cholesky succeeds). This is what a proper prior
    buys — a usable covariance on a corpus where an un-regularized scatter estimate is
    only marginally conditioned."""
    iw = scarce_fits["iw"]
    iw_eigmin = float(np.linalg.eigvalsh(iw["Sigma"]).min())
    assert np.all(np.isfinite(iw["Sigma"]))
    assert np.max(np.abs(iw["Sigma"])) < 1e2, "IW Sigma must stay bounded"
    assert iw_eigmin > 0.0, f"expected IW Sigma PD, got eigmin={iw_eigmin}"
    np.linalg.cholesky(iw["Sigma"])          # canonical PD check (raises if not PD)


def test_both_estimators_pd_after_assembly_fix_no_toy_runaway(scarce_fits):
    """HONEST REFRAME (F3). After ``_assemble_sigma`` completes the unobserved
    group<->group' cross-blocks with the max-det PD completion instead of zero-filling
    them, BOTH sigma_modes yield a PD, bounded Sigma at toy scale — the earlier
    "MLE indefinite" contrast was the zero-fill stitching artifact, not an estimator
    runaway (see module docstring + exp 0049). Neither reaches the softmax 10^10 regime.

    The MILD surviving estimator signal, asserted here as a non-decisive regression
    guard (NOT the runaway cure — that moves to scale): the IW posterior is the
    better-conditioned of the two — interior-PD (its blocks are strictly PD-completable,
    so eigmin stays comfortably > 0), where the raw scatter/n MLE sits ON the PD boundary
    (its blocks are floored by the completion, eigmin ~ +0.000). Measured across bg_k=4
    seeds 0/1/2 and bg_k=2 seeds 0/1/2: MLE eigmin is at the floor in every seed while IW
    is interior (+0.007..+0.018). (max|Sigma| is NOT a reliable discriminator — both are
    O(1) and their ordering flips seed-to-seed — so it is not asserted as a direction.)"""
    mle, iw = scarce_fits["mle"], scarce_fits["iw"]
    mle_eigmin = float(np.linalg.eigvalsh(mle["Sigma"]).min())
    iw_eigmin = float(np.linalg.eigvalsh(iw["Sigma"]).min())

    # Both PD after the assembly fix (no toy runaway under either estimator).
    assert mle_eigmin >= 0.0, f"MLE Sigma should be PD after fix, got {mle_eigmin}"
    assert iw_eigmin > 0.0, f"IW Sigma should be PD, got {iw_eigmin}"
    np.linalg.cholesky(mle["Sigma"])
    np.linalg.cholesky(iw["Sigma"])

    # Neither estimator reaches the softmax point-EM 10^10 runaway (both ~O(1)).
    assert np.max(np.abs(mle["Sigma"])) < 1e2
    assert np.max(np.abs(iw["Sigma"])) < 1e2

    # Mild, non-decisive: IW is interior-PD where the raw scatter/n MLE is boundary-PD.
    assert iw_eigmin >= mle_eigmin, (
        f"IW should be at least as interior-PD as MLE: iw={iw_eigmin}, mle={mle_eigmin}")


def test_scarce_topic_gets_wide_not_divergent_posterior_under_iw(scarce_fits):
    """Under the IW posterior the scarce group's per-doc stick variance is ELEVATED
    (the weakly-identified sticks get a wide posterior) but BOUNDED — not divergent."""
    pv = scarce_fits["iw"]["psi_var"]
    assert np.all(np.isfinite(pv))
    assert pv.max() < 1e2 and pv.max() > pv.mean()   # elevated but bounded


def test_iw_still_recovers_beta_on_the_scarce_corpus(scarce_fits):
    """The IW regularization is not paid for by wrecking recovery: beta still recovers
    the planted topics (cross-model comparable half of the gate) on the same corpus."""
    iw, part, beta = scarce_fits["iw"], scarce_fits["part"], scarce_fits["beta"]
    # scarce group B is weakly represented (~3% of docs); require the majority of the
    # K planted topics to be recovered (measured 5/6 at seed 0).
    assert planted_recovery(iw["beta"], beta) >= part.K - 2


def test_CONTEXT_current_softmax_point_em_here():
    """CONTEXT ONLY (link AND estimator both differ from PG-VI): the softmax point-EM
    diagonal-Σ path (OnlineSTM, estimate_sigma_diagonal). MEASURED: at THIS
    bg_k=4/K=6/D=1000 synthetic it does NOT blow up (final Σ range ~[-0.003, 1.614],
    hi ≪ 1e3) — the documented 10^10 softmax runaway (insight 0033) is a larger-K /
    real-data / no-reference-topic property, not this small synthetic. So the brief's
    `hi > 1e3` does not hold here; this is recorded, NOT gated (the PG-VI MLE-vs-IW
    isolation above is the milestone gate). We assert only finiteness so the number
    stays visible and reproducible without a false gate."""
    docs, part, Sigma_true, beta = _scarce_corpus()
    gp = fit_stm(docs, K=part.K, V=60, sigma_init=1.0, n_iter=200, seed=0,
                 partition=part, estimate_sigma_diagonal=True)
    lo, hi = final_sigma_range(gp)
    assert np.isfinite(lo) and np.isfinite(hi)
