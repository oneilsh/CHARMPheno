# 0032 — Gated full-covariance STM recovers Alzheimer's vs vascular dementia in the 19% minority arm; the predicted SPD-assembly ill-conditioning appears on real data and is gracefully handled

**Date:** 2026-06-30
**Topic:** stm | full-covariance | gating | rare-phenotype | conditioning | diagnostics | phenotyping
**Status:** Confirmed (cancer_or_dementia cohort, exp 0020 non-gated vs exp 0021 gated)

The full-covariance Σ arc (ADR 0033) replaced STM's diagonal mean-field
covariance with a genuine (K−1)×(K−1) matrix to model topic correlations. Two
cluster runs validate it: exp 0020 (non-gated cancer) and exp 0021 (gated
multi-group cancer + dementia). This insight records both the scientific payoff
— rare dementia sub-phenotype recovery in a minority arm — and the first
real-data appearance of the SPD-assembly conditioning risk that ADR 0033
predicted and built three layers of mitigation for.

## Setup

- **exp 0020** — non-gated full-Σ, `first_cancer_year`, K=40, ~10.8k docs,
  V=3691, σ_init=1, reference + spectral, 100 iters.
- **exp 0021** — gated multi-group full-Σ, `cancer_or_dementia` cohort,
  K=50 = 30 background + 10 cancer + 10 dementia, `group_var=source_cohort`,
  `min_pair_support=10`, ~13.3k docs, V=4422, otherwise identical. Comorbid
  patients (cancer AND dementia) contribute one document per cohort, giving
  multi-group membership. Block sizes: background 13,295 / cancer 10,819 /
  dementia 2,476 docs (dementia ≈ 19% — the minority arm).

## Finding 1 — non-gated full-Σ is clean and well-conditioned, and adds the correlation feature for free

exp 0020 is the best STM run on the cancer cohort to date AND the first to carry
topic correlations:

| metric | exp 0015 (diag dense) | exp 0017 (diag scalable) | exp 0020 (full-Σ) |
|---|---|---|---|
| ELBO | −1.10e6 | −1.165e6 | **−1.095e6** |
| Σ conditioning | (diagonal) | one topic Σ→8e5 | **cond 13.3** (eig 0.8–10.6) |
| max off-diag corr | n/a (diagonal) | n/a | **0.459** |
| NPMI mean | +0.173 | +0.166 | +0.167 |
| topics resolved | 40 | 40 | 40 |

The full model at σ_init=1 lands at condition number **13.3** — the richer model
is BETTER behaved than the diagonal approximation (which blew one topic's variance
to 8e5 in exp 0017), not worse, while delivering what the diagonal model
structurally cannot: a topic-correlation matrix (max off-diagonal 0.46 — topics
genuinely correlated but nowhere near collinear). The inert reference row is
`[1, 0, 0, …]` exactly as designed. No regression on topics (NPMI +0.167, all 40
resolved). Full Σ gives a stable covariance AND the correlation feature AND held
topic quality, simultaneously.

## Finding 2 — gated full-Σ recovers dementia sub-phenotypes in the minority arm (the thesis payoff)

exp 0021's three blocks all resolve distinct phenotypes (block-aware NPMI:
background +0.19, cancer +0.18, dementia +0.17; 0 unrated). The decisive result
is the **dementia block** — only 2,476 docs, dwarfed 4:1 by cancer — which in the
old combined STM washed into a uniform "dementia + baseline comorbidity" blend
(insight [0028](0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md),
peak word prob ~0.003). Under the gated full-Σ model it splits into genuine
sub-phenotypes:

- **topic 41 — Alzheimer's / amnestic dementia:** Amnesia, Dementia,
  Dementia-associated-with-another-disease, Minimal cognitive impairment,
  Alzheimer's disease.
- **topic 49 — vascular dementia:** Atherosclerosis of coronary artery,
  Essential hypertension, Atrial fibrillation, Dementia, Heart failure, mitral
  stenosis.
- topic 44 — a dementia symptom cluster (Dementia + Anxiety/Pain/Neurological
  deficit/Seizure); topics 42/46/47 — psychiatric comorbidity (bipolar +
  schizophrenia; PTSD + alcohol; anxiety + depression).

Recovering the **Alzheimer's-vs-vascular-dementia distinction in a 19% minority
arm** is the core thesis of the gated approach (surface rare-subgroup phenotypes
a majority-dominated shared model washes out). Gating provides the block
structure; the full-Σ stack (reference + spectral at σ=1) keeps the minority
foreground from collapsing.

## Finding 3 — the predicted SPD-assembly ill-conditioning appears on real data, and is gracefully handled

exp 0021's Σ diagnostics: `Σ_eig[min=1e-06 max=32.8 cond=3.28e7]`,
`max_offdiag=0.80`. The minimum eigenvalue is **exactly SIGMA_FLOOR (1e-6)** —
`nearest_spd` floored a near-zero/negative eigenvalue. This is the SPD-assembly
inconsistency ADR 0033 (decision 5) called "guaranteed to arise, not
hypothetical," now confirmed on data:

- **Mechanism:** comorbid (cancer AND dementia) patients are rare in this cohort,
  so the cancer↔dementia cross-foreground block has little/no support and is
  pinned to the prior (0 off-diagonal) under the `min_pair_support=10` floor. But
  background topics appear in EVERY document's allowed set, so they correlate
  strongly with BOTH foreground blocks. Strong background↔cancer and
  background↔dementia with a pinned cancer↔dementia is internally inconsistent
  (transitively implies a cancer↔dementia correlation that was forced to zero),
  producing a near-singular assembled Σ.

The crucial point: **the three-layer mitigation behaved exactly as designed.**
`nearest_spd` kept Σ valid (positive-definite, no crash), the topics came out
fine (the E-step was robust to the near-singular direction), AND the new
full-matrix diagnostics CORRECTLY FLAGGED the problem (condition number 3.3e7,
far above the 1e4 threshold the exp 0021 decision tree watches for). The failure
mode was anticipated, contained, and observable — the design worked. Contrast
exp 0020 (non-gated, every topic fully supported, no cross-block inconsistency):
condition number 13.3, no floored eigenvalue. The ill-conditioning is specific to
the gated thin-comorbid regime, exactly where the model predicts it.

The consequence is bounded: the **correlation readout** is not trustworthy at
cond 3.3e7 (the near-singular direction pollutes the off-diagonal structure, hence
the inflated max_offdiag 0.80), and a near-singular Σ would also impair the
ADR-0028-B logistic-normal sampler — but the topics and the block structure are
sound.

## Implication — the inverse-Wishart prior is the prescribed, targeted fix

ADR 0033 built the inverse-Wishart prior for exactly this regime. Its MAP M-step
is PER-ENTRY — Σ_ij = (S_ij + ν·scale·δ_ij)/(N_ij + ν) — so a moderate
pseudo-count ν regularizes each entry weighted against that entry's real support.
With background entries at N_ij ≈ thousands and thin cross-foreground cells at
N_ij ≈ 0, ν = 100 is negligible on the well-supported blocks (100 ≪ 13,295) but
dominant on the thin cross-cells (100 ≫ a handful of comorbid patients) — it
regularizes precisely where the inconsistency lives without flattening the real
correlations the topics depend on. exp
[0022](../experiments/0022-stm-comorbid-fullcov-gated-iwprior.md) tests this
(`sigma_prior_scale=2.0`, `sigma_prior_count=100`); the expected result is the
condition number dropping orders of magnitude with the dementia sub-phenotypes
preserved. There is a conditioning-vs-correlation tradeoff (too-large ν shrinks Σ
toward diagonal, back toward a washed minority arm), so the right ν is a sweet
spot to find.

## Implications

1. **Full-Σ is validated as STM's covariance model** — well-conditioned on the
   non-gated cohort (exp 0020), no topic-quality regression, and it delivers the
   correlation feature. The diagonal results (exp 0015/0017, insight 0030) are
   superseded as the baseline.
2. **Gated full-Σ recovers rare sub-phenotypes** (Alzheimer's vs vascular
   dementia in a 19% arm) — the gated approach's reason for existing, now shown
   under the genuine correlated model.
3. **The SPD-assembly risk is real but anticipated and handled** — for gated
   cohorts with thin cross-group comorbidity, the IW prior (`sigma_prior_count`)
   is the conditioning lever; a mild `sigma_diag_shrink` is a second lever. Plan
   to turn the IW prior ON by default for gated full-Σ runs once exp 0022 fixes
   the conditioning.
4. **The new diagnostics earn their keep** — `Σ_eig[cond]` and
   `max_abs_offdiag_corr` made an otherwise-silent near-singular assembly
   immediately visible; without them the ill-conditioning would have surfaced
   only as a confusing correlation readout.

## Relationship to prior insights

Builds on insight [0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md)
(diagonal-Σ spectral result, now the prior baseline) and
[0031](0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md).
Delivers the rare-phenotype recovery that insight
[0028](0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)
found the combined logistic-normal STM could not — here the gated block structure
plus the full-Σ stabilizer stack recovers the dementia sub-phenotypes the shared
model washed out. Governed by ADR 0033 (full-covariance Σ); the SPD-assembly
three-layer mitigation is decision 5 of that ADR, confirmed on data here.
