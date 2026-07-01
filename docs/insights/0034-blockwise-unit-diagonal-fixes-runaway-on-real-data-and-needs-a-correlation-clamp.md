# 0034 — Block-wise unit-diagonal Σ fixes the runaway on real data (exp 27); per-cell standardization needs a [-1,1] correlation clamp

**Date:** 2026-07-01
**Topic:** stm | svi | gating | conditioning | reporting
**Status:** Confirmed (real-cohort cluster run exp 0027 + local measurement of the clamp)

Insight [0033](0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)
diagnosed the gated full-Σ variance runaway as an identifiability failure with two
ingredients — a weakly-identified topic and a free prior variance — and predicted that
pinning Σ_ii = 1 (block-wise unit-diagonal correlation Σ, ADR
[0034](../decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md)) would remove
the second ingredient and fix it by construction. Exp
[0027](../experiments/0027-stm-comorbid-blockwise-unit-diagonal.md) confirms the
prediction on the real `cancer_or_dementia` cohort, and building the estimator surfaced a
second, non-obvious finding about standardizing per-pair scatter.

## Finding 1 — the runaway is gone on real data, and the sub-phenotypes it was drowning survive

Exp 0027 (same cohort/config as the runaway diagnosis exp 0026, block-wise engine,
max_iter 100) converged at iter 52 with `Σ_var[min=1 max=1]` every iteration — the
diagonal is pinned, so there is no variance to run away. Contrast exp 0026, where one
topic's Σ_kk reached 2.71e5 by iter 28 with the ELBO diverging to −1.56e7. Exp 0027's
ELBO recovered to −1.63e6 (near the ≈−1.59e6 target), and the per-iteration
`M-step pd_complete: …s` cost is gone (completion retired from the fit path).

The dementia sub-phenotypes that prevalence-only STM had collapsed into a single
near-uniform "dementia + baseline" blend (peak β ≈ 0.003) came through crisp: an
**amnestic/Alzheimer's** topic (Amnesia 0.073, Alzheimer's disease 0.028, minimal
cognitive impairment 0.031), a **vascular** topic (atherosclerosis + dementia + atrial
fibrillation), a dementia-core topic, and a background **epilepsy/seizure/amnesia** topic
— peak β 0.03–0.12. Block-aware NPMI means were healthy (background +0.190, cancer
+0.190, dementia +0.180; 0 unrated topics). The full Σ was indefinite
(`Σ_eig[min=−0.718 max=8.52]`) exactly as designed — it is never inverted whole; only the
fully-observed E-step marginals Σ[background ∪ one group] are inverted, and those are PD.
The lesson of insight 0033 holds end-to-end: unit-diagonal is structural insurance
against the runaway that costs nothing in topic quality.

## Finding 2 — per-cell standardization is NOT bounded to [-1,1], and it fires in practice

The block-wise M-step standardizes the observed per-pair scatter to a correlation with
per-cell support counts:

    R_ij = (S_ij / N_ij) / sqrt( (S_ii / N_ii) · (S_jj / N_jj) )

This is NOT Cauchy-Schwarz-bounded to [-1,1], because the covariance and the two
variances are averaged over DIFFERENT document sets. Under single-label gating a
background↔foreground pair is the canonical case: N_ii spans all documents (background
topic i is always active), while N_ij spans only the one group's documents (foreground
topic j is active only there). A background topic with small overall variance but strong
co-variation inside that one group's documents produces |R_ij| > 1 — a nonsensical
"correlation". Concretely, S = [[10, 8], [8, 10]], N = [[100, 10], [10, 100]] gives
R_01 = 0.8 / sqrt(0.1 · 0.1) = 8.0.

This is not merely theoretical. On the shared-term overlap synthetic corpus, adding a
per-entry clamp to [-1,1] measurably CHANGED the fit (the recovery-invariant test's
per-seed recovery shifted, and needed more iterations to re-converge) — proving |r| > 1
was occurring during the un-clamped fit and its invalid values were entering the E-step
prior. The fix is the minimal projection onto the valid correlation range: clip supported
off-diagonals to [-1,1] (cf. Higham 2002 nearest-correlation), applied per entry. With
the clip plus the pinned diagonal, Σ is a valid correlation matrix by construction, and
the exported correlation report cannot contain out-of-range entries. Note that exp 0027
itself ran on the pre-clamp engine, so its exported correlation.json may carry a few
|r| > 1 background↔foreground cells; the runaway fix and topic recovery are independent
of the clamp, but a clamp-clean correlation report needs a re-run on the clamp commit.

## Implications

1. **The gated STM stack is complete for single-label cohorts.** Reference topic +
   spectral init (insight 0029/0030) + gating + block-wise unit-diagonal Σ recovers
   minority sub-phenotypes without a runaway and without any tuning knob. The falsified
   variance levers (IW prior, diag-shrink — insight
   [0032](0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
   Findings 4-6) are not missed.
2. **Pairwise-deletion standardization always needs a range guard.** Any estimator that
   standardizes a covariance assembled from inconsistent document subsets (the essence of
   the lazy per-pair / block-wise M-step) can exceed [-1,1]; the guard belongs at the
   single point where the correlation is formed, not downstream. The dashboard ramp
   (`d3.scaleLinear().domain([-1,0,1]).clamp(true)`) is a robustness backstop, not the
   fix.
3. **The correlation export needed no change.** Because a unit-diagonal Σ IS a correlation
   matrix, `topic_correlation(Σ) = Σ` (a no-op standardization) and
   `topic_correlation_identified` NAs cross-group pairs via n_pairs = 0 — the Plan-1
   correlation-reporting export handles block-wise Σ as-is.

## Relationship to prior insights

Confirms the prediction of insight
[0033](0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)
(unit-diagonal removes the runaway) on real data, and closes the arc opened by insight
[0032](0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
(sub-phenotype recovery + the assembly-conditioning reporting artifact). Governed by ADR
[0034](../decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md).
