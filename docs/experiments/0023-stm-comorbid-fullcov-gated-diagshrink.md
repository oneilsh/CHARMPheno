---
id: 23
slug: stm-comorbid-fullcov-gated-diagshrink
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 50
background_k: 30
foreground: "cancer:10,dementia:10"
group_var: source_cohort
max_iter: 100
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
sigma_diag_shrink: 0.2
---

# STM comorbid (cancer + dementia): gated full-Σ + diagonal-shrink (the right conditioning lever)

The conditioning follow-up to exp 0022. **Identical to exp 0021 in every way
except the diagonal-shrink lever is engaged** (`sigma_diag_shrink: 0.2`), with the
inverse-Wishart prior left OFF. This is a clean A/B against exp 0021 on the
`sigma_diag_shrink` lever, mirroring how exp 0022 was a clean A/B on the IW prior.

## Why — exp 0022 falsified the IW prior as the conditioning lever here

Exp 0022 turned the inverse-Wishart prior on (`sigma_prior_count=100`,
`sigma_prior_scale=2.0`) and the condition number barely moved: 3.28e7 → 3.05e7,
minimum eigenvalue still pinned at SIGMA_FLOOR=1e-6, max off-diagonal correlation
still 0.80. The prior did exactly what it was designed to do (the diagonal entry
moved 1.0 → 1.979, pulled toward `scale=2`) — it just doesn't touch the singular
direction. The root cause is in the parameterization:

The IW MAP M-step is `Σ_ij = (S_ij + Ψ_ij) / (N_ij + ν)`, and **Ψ is diagonal**
(`Ψ_ij = 0` for off-diagonals). So for an off-diagonal entry:

Σ_ij = S_ij / (N_ij + ν)

The prior acts on correlations ONLY through the denominator — a fractional shrink
of `N_ij / (N_ij + ν)`. The near-singular direction is the SPD-assembly
inconsistency of insight 0032: background↔cancer (N≈10,800) and
background↔dementia (N≈2,500) are strongly coupled while cancer↔dementia is
structurally zeroed (thin comorbid support + `min_pair_support=10`). A zero
cross-block under two strong couplings forces a near-zero-variance eigen-direction.
But the entries that CREATE that inconsistency are WELL-SUPPORTED, and against
N≈10,800 a pseudo-count of ν=100 is a <1% shrink. To bite there ν would have to be
thousands — which would flatten every real correlation. **Wrong lever: the IW
prior is N-weighted, designed for the thin-cell regime; this singularity lives in
the well-supported entries.**

A diagonal-Ψ inverse-Wishart is structurally a VARIANCE regularizer (it pulls
variances toward `scale`), not a CORRELATION regularizer. The correct conditioning
lever is the N-INDEPENDENT fractional shrink (Roberts et al. `sigma.prior`):

Σ ← (1−w)·Σ + w·diag(diag(Σ))

This pulls EVERY off-diagonal toward zero by factor `(1−w)` regardless of support,
so it attacks the well-supported background↔foreground couplings the IW prior
can't reach. For the near-singular direction v (with vᵀΣv ≈ 1e-6 but
vᵀ diag(Σ) v ≈ O(variance) ≈ O(1)), the blend gives
vᵀ[(1−w)Σ + w·D]v ≈ w·O(1) — so even a modest w lifts that direction's variance
off the floor by orders of magnitude, while the diagonal variances that carry the
foreground topic signal are untouched.

## Hypothesis

With `sigma_diag_shrink=0.2` and the IW prior off, the gated full-Σ fit:

(a) brings the condition number DOWN by orders of magnitude — from exp 0021/0022's
    ~3e7 toward O(1e2-1e3) — with the minimum eigenvalue lifted well off
    SIGMA_FLOOR (no longer a floored near-singular direction). This is the direct
    test that the N-independent lever fixes what the N-weighted prior could not;
(b) PRESERVES the dementia sub-phenotype recovery (topic 41 Alzheimer's/amnestic:
    Amnesia + Dementia + MCI + Alzheimer's; topic 49 vascular: Atherosclerosis +
    AFib + Dementia) and per-block topic quality of exp 0021 — a fractional
    off-diagonal shrink leaves the diagonal variances and therefore the foreground
    signal intact;
(c) yields a TRUSTWORTHY (now well-conditioned) correlation matrix R, uniformly
    attenuated by ~(1−w): the background↔cancer and background↔dementia structure
    survives at reduced magnitude, and the relative ordering of correlations — the
    interpretable content — is preserved.

## What to watch

- **Σ_eig[cond] — THE headline.** Did it drop from ~3e7 toward O(1e2-1e3)? Is
  `sigma_eig_min` now well above 1e-6 (off the floor)? This is the direct test of
  `sigma_diag_shrink` as the conditioning lever and the falsification check on the
  exp 0022 root-cause analysis (singularity in well-supported entries, not thin
  cells).
- **max_abs_offdiag_corr** — should drop from 0.80 toward ~0.80·(1−w) ≈ 0.64 if
  the max-correlation pair is a genuine pairwise correlation, or lower if it was
  inflated by the near-singular direction. A value near 0.5-0.65 with the
  eigenvalue off the floor is the target.
- **Per-block topic quality** — block-aware NPMI per block must hold near exp
  0021's levels (background ~+0.19, cancer ~+0.18, dementia ~+0.16); the dementia
  block must STILL show the Alzheimer's/amnestic (topic with Amnesia + Alzheimer's
  + MCI) and vascular (atherosclerosis + AFib + dementia) sub-phenotypes. If they
  collapse, w is too large (over-shrink toward diagonal — back toward a washed
  minority arm).
- **ELBO + Σ_var** — should stay in the same regime as 0021 (Σ_var O(1-10)); a
  large ELBO drop indicates the shrink is too strong.

## Decision

- **cond drops to O(1e2-1e3) + dementia sub-phenotypes preserved + correlations
  uniformly attenuated but interpretable** → `sigma_diag_shrink` is the validated
  conditioning lever for the gated/thin-comorbid regime, NOT the IW prior. Record
  `sigma_diag_shrink ≈ 0.2` as the default for gated full-Σ runs. Update insight
  0032 and ADR 0033's regularizer guidance: IW prior = variance/thin-cell magnitude
  (N-weighted); diag-shrink = correlation/conditioning (N-independent); they are
  complementary, and conditioning is the diag-shrink's job. Log the cross-block
  correlation findings as an insight.
- **cond improves but is still > 1e4** → w=0.2 under-shrinks; raise
  `sigma_diag_shrink` (0.3-0.5). The conditioning-vs-correlation tradeoff has a
  sweet spot to find.
- **cond fixed but dementia sub-phenotypes collapse / block NPMI drops** → w is
  over-shrinking toward diagonal; lower `sigma_diag_shrink` (0.05-0.1).
- **No change in conditioning** → would be surprising and would falsify the
  root-cause analysis; the singular direction is not an off-diagonal-correlation
  effect at all. Re-examine the assembled Σ eigenvector of the floored eigenvalue
  directly (which topics load on it) before any further lever.

## Run

```
make exp ID=23
```

Compare head-to-head with exp 0021 (same config, both regularizers off) and exp
0022 (IW prior on, diag-shrink off). The three-way delta in `Σ_eig[cond]` —
3.28e7 (none) vs 3.05e7 (IW) vs 0023 (diag-shrink) — is the direct measurement of
which lever owns the gated-conditioning problem (ADR 0033). Result feeds insight
0032.
