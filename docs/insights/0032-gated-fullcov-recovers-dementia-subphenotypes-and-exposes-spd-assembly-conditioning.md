# 0032 — Gated full-covariance STM recovers Alzheimer's vs vascular dementia in the 19% minority arm; the predicted SPD-assembly ill-conditioning appears on real data and is gracefully handled

**Date:** 2026-06-30
**Topic:** stm | full-covariance | gating | rare-phenotype | conditioning | diagnostics | phenotyping
**Status:** Confirmed (cancer_or_dementia cohort, exp 0020 non-gated vs exp 0021 gated; exp 0022 — IW alone can't fix the min-eigenvalue near-singularity (Finding 4); exp 0023 — diag-shrink fixes the min end but triggers a max-eigenvalue variance runaway, so conditioning needs BOTH levers (Finding 5); exp 0024 tests the combination)

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

## Finding 4 — the inverse-Wishart prior is the WRONG conditioning lever; the singularity lives in the well-supported entries (exp 0022)

The original prescription was the inverse-Wishart prior: ADR 0033 built it for the
thin-cross-cell regime, and its MAP M-step is PER-ENTRY —
Σ_ij = (S_ij + Ψ_ij)/(N_ij + ν), Ψ = ν·scale·I — so a pseudo-count ν was expected
to regularize each entry weighted against that entry's real support. exp
[0022](../experiments/0022-stm-comorbid-fullcov-gated-iwprior.md)
(`sigma_prior_scale=2.0`, `sigma_prior_count=100`) tested it and **falsified it as
the conditioning fix.** Head-to-head with exp 0021:

| metric | exp 0021 (no prior) | exp 0022 (IW ν=100) |
|---|---|---|
| Σ_eig cond | 3.28e7 | **3.05e7** (unchanged) |
| Σ_eig min | 1e-06 (floored) | **1e-06 (still floored)** |
| max off-diag corr | 0.801 | **0.798** (unchanged) |
| ELBO | −1.5897e6 | −1.5907e6 (a hair worse) |
| dementia sub-phenotypes | Alz(41)+vasc(49) ✓ | preserved ✓ |

The prior did exactly what it was coded to do — the first diagonal entry moved
1.0 → 1.979, pulled toward `scale=2` — it simply doesn't touch the singular
direction. The reason is in the parameterization: **Ψ is diagonal**, so Ψ_ij = 0
for off-diagonals and Σ_ij = S_ij/(N_ij + ν). The prior acts on correlations ONLY
through the denominator, a fractional shrink of N_ij/(N_ij + ν). But the
near-singular direction is built from WELL-SUPPORTED entries — background↔cancer
(N ≈ 10,800) and background↔dementia (N ≈ 2,500), strongly coupled while
cancer↔dementia is structurally zeroed. Against N ≈ 10,800 a pseudo-count ν = 100
is a <1% shrink. To bite there ν would have to be thousands, flattening every real
correlation. The earlier reasoning ("ν = 100 ≪ 13,295 → negligible on the
well-supported blocks") was correct about the magnitude but drew the wrong
conclusion: being negligible on the well-supported blocks is exactly the FAILURE,
because the singularity is IN those blocks, not in the thin cross-cells.

The deeper point: **a diagonal-Ψ inverse-Wishart is a VARIANCE regularizer, not a
CORRELATION regularizer.** It pulls variances toward `scale`; it can only shrink
correlations weakly, via the shared denominator. The correct conditioning lever is
the N-INDEPENDENT fractional shrink (Roberts et al. `sigma.prior`, exposed as
`sigma_diag_shrink`):

Σ ← (1−w)·Σ + w·diag(diag(Σ))

This pulls EVERY off-diagonal toward zero by factor (1−w) regardless of support, so
it reaches the well-supported couplings the IW prior cannot. For the near-singular
direction v (with vᵀΣv ≈ 1e-6 but vᵀdiag(Σ)v ≈ O(1)), the blend gives
vᵀ[(1−w)Σ + w·diag(Σ)]v ≈ w·O(1) — even a modest w lifts that direction's variance
off the floor by orders of magnitude, while the diagonal variances carrying the
foreground topic signal are untouched. exp
[0023](../experiments/0023-stm-comorbid-fullcov-gated-diagshrink.md)
(`sigma_diag_shrink=0.2`, IW off) ran and **confirmed this half of the picture
exactly** — and exposed a second, opposite-end problem. See Finding 5.

## Finding 5 — diag-shrink fixes the min-eigenvalue end but triggers a max-eigenvalue variance runaway: conditioning has TWO heads, needs BOTH levers (exp 0023)

exp 0023 turned `sigma_diag_shrink=0.2` on (IW off). The near-singular end was
fixed exactly as Finding 4 predicted — and the failure mode FLIPPED to the other
end of the spectrum:

| metric | 0021 (neither) | 0022 (IW ν=100) | 0023 (diag-shrink 0.2) |
|---|---|---|---|
| Σ_eig **min** | 1e-6 (floored) | 1e-6 (floored) | **1.0 (off the floor)** ✓ |
| Σ_eig **max** | 32.8 | 30.5 | **2.23e7 (blew up)** ✗ |
| Σ_var max | 8.86 | 8.72 | **2.19e7** |
| max off-diag corr | 0.801 | 0.798 | **0.177** |
| Σ_eig cond | 3.28e7 | 3.05e7 | 2.23e7 (now a MAX problem) |
| ELBO | −1.5897e6 | −1.5907e6 | **−2.252e6** (much worse) |
| dementia sub-phenotypes | Alz(41)+vasc(49) ✓ | ✓ | **✓ still** |

Two things happened. (1) **The min-eigenvalue near-singularity is gone**: min eig
1e-6 → 1.0 (off the floor) and max off-diagonal correlation 0.80 → 0.177. That
0.80 was almost entirely the near-singular inflation, exactly as the root-cause
analysis claimed — the genuine correlations in this gated cohort are small (≤ 0.18).
Finding 4 is **confirmed**: diag-shrink owns the min-eigenvalue / correlation end,
and the IW prior could never reach it. (2) **A single topic's η-variance ran away to
2.2e7** (Σ_var max = 2.19e7 ≈ Σ_eig max), so the condition number stayed ~2e7 — but
now driven entirely by the MAX eigenvalue, the opposite end. ELBO degraded badly
(the Gaussian η-KL logdet/trace terms blow up with a 2.2e7 variance direction),
while — per insight 0030's decoupling — topic quality and the dementia
sub-phenotypes held (topic 41 Alzheimer's, topic 49 vascular, block NPMI
0.187/0.190/0.154, all 50 resolved).

**Why diag-shrink triggers a variance runaway it cannot itself cause.** Diag-shrink
does NOT touch the diagonal — `(1−w)·diag + w·diag = diag`
([stm.py:739](../../spark-vi/spark_vi/models/topic/stm.py#L739)); it only scales
off-diagonals by (1−w). The blowup is therefore a DYNAMICAL feedback over
iterations, not a direct effect: shrinking the off-diagonals changes the gated
E-step's marginal precisions inv(Σ_AA), which changes η̂ and ν_d, which changes the
residuals accumulated back into the diagonal next M-step. The off-diagonal
correlation structure was **load-bearing for variance stability** — it regularized a
weakly-identified topic's variance by tying it (through correlation) to
well-estimated topics. Decorrelating freed that topic to run away, which is exactly
insight [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)'s
runaway-Σ mechanism, re-triggered. (exps 0021/0022, off-diagonals intact, had max
Σ_var ≈ 9 — no runaway.)

**The cure is the IW variance-anchor, applied every M-step.** This is precisely the
diagonal-Σ runaway that insight
[0031](0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md)
tamed with `sigma_prior_count=2000`: the per-entry MAP
Σ_ii = (S_ii + ν·scale)/(N_ii + ν) is a per-iteration contraction toward `scale`
that prevents the variance from ever building up. So the two regularizers are not
"one right, one wrong" — **each owns one end of the spectrum**: diag-shrink (Roberts
`sigma.prior`, N-independent) fixes the MIN-eigenvalue / correlation near-singularity;
the IW prior (`sigma_prior_count`, the variance anchor) caps the MAX-eigenvalue /
variance runaway. Conditioning is the RATIO, so a well-conditioned gated Σ needs
BOTH. exp [0024](../experiments/0024-stm-comorbid-fullcov-gated-both-levers.md)
tests the combination (`sigma_diag_shrink=0.1` + `sigma_prior_scale=2`,
`sigma_prior_count=2000`).

## Implications

1. **Full-Σ is validated as STM's covariance model** — well-conditioned on the
   non-gated cohort (exp 0020), no topic-quality regression, and it delivers the
   correlation feature. The diagonal results (exp 0015/0017, insight 0030) are
   superseded as the baseline.
2. **Gated full-Σ recovers rare sub-phenotypes** (Alzheimer's vs vascular
   dementia in a 19% arm) — the gated approach's reason for existing, now shown
   under the genuine correlated model.
3. **Gated-Σ conditioning has TWO heads and needs BOTH levers, one per spectral
   end.** The min-eigenvalue near-singularity (off-diagonal SPD-assembly
   inconsistency) is fixed by `sigma_diag_shrink` (Finding 4: IW can't reach it,
   Finding 5: diag-shrink does — min eig 1e-6 → 1.0). The max-eigenvalue variance
   runaway (a weakly-identified topic freed by decorrelation — insight 0029's
   mechanism) is capped by the IW prior `sigma_prior_count` (insight 0031's
   established cure, count=2000). Neither lever alone conditions the gated Σ;
   diag-shrink alone trades the min-end problem for a max-end one (exp 0023). exp
   0024 tests the combination; plan the gated full-Σ default around BOTH knobs
   once 0024 lands a well-conditioned Σ with sub-phenotypes preserved.
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
