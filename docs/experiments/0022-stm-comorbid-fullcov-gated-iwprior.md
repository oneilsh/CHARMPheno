---
id: 22
slug: stm-comorbid-fullcov-gated-iwprior
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
sigma_prior_scale: 2.0
sigma_prior_count: 100.0
---

# STM comorbid (cancer + dementia): gated full-Σ + inverse-Wishart prior (conditioning fix)

The conditioning follow-up to exp 0021. **Identical to exp 0021 in every way
except the inverse-Wishart Σ prior is turned ON** (`sigma_prior_scale: 2.0`,
`sigma_prior_count: 100`). Exp 0021 validated the gated multi-group full-Σ path
(the dementia minority arm recovered Alzheimer's/amnestic and vascular
sub-phenotypes — insight 0032) but the assembled Σ was ill-conditioned
(`Σ_eig cond=3.3e7`, min eigenvalue floored at SIGMA_FLOOR=1e-6): the predicted
SPD-assembly inconsistency, because comorbid (cancer AND dementia) patients are
rare so the cancer↔dementia cross-foreground block is prior-pinned while the
background correlates strongly with both blocks. This run engages the lever
ADR 0033 designed for exactly that regime.

## Why

The exp 0021 decision tree's "ill-conditioned Σ after SPD-repair (cond > 1e4)"
branch prescribes increasing the inverse-Wishart prior weight to regularize the
thin cross-foreground cells. The IW MAP M-step is PER-ENTRY:

Σ_ij = (S_ij + Ψ_ij) / (N_ij + ν),   Ψ = ν · sigma_prior_scale · I,
ν = sigma_prior_count

so the prior pseudo-count ν acts independently on each entry weighted against
that entry's real support N_ij. This is the key property: with the background
block entries at N_ij ≈ thousands (13,295 background docs) and the thin
cross-foreground cells at N_ij ≈ 0, a MODERATE ν = 100 is

- negligible on the well-supported background and within-group entries
  (100 ≪ 13,295 → those entries stay at essentially their MLE), and
- dominant on the thin cross-foreground cells (100 ≫ their handful of comorbid
  patients → those cells are pulled to the coherent prior scale).

So the prior regularizes EXACTLY where the inconsistency lives, without flattening
the real correlations the well-supported blocks carry. `sigma_prior_scale: 2.0`
targets the data's η-variance scale (exp 0021 had Σ_var max 8.86, the bulk O(1-3)).

## Hypothesis

With the IW prior on, the gated full-Σ fit:

(a) brings the condition number down by orders of magnitude — from exp 0021's
    3.3e7 toward O(1e2-1e3) — with the minimum eigenvalue lifted off the
    SIGMA_FLOOR (no longer a floored near-singular direction);
(b) PRESERVES the dementia sub-phenotype recovery (Alzheimer's/amnestic, vascular)
    and the per-block topic quality of exp 0021 (block NPMI: background ~+0.19,
    cancer ~+0.18, dementia ~+0.17) — the prior is targeted at thin cells, not
    the well-supported blocks that carry the topics;
(c) yields a TRUSTWORTHY cross-block correlation structure: with Σ well-conditioned,
    the background↔cancer and background↔dementia correlations and the (regularized)
    cancer↔dementia entries can be read off `correlation.npy` and interpreted.

## What to watch

- **Σ_eig[cond] — THE headline.** Did it drop from 3.3e7 toward O(1e2-1e3)? Is
  `sigma_eig_min` now well above 1e-6 (off the floor)? This is the direct test of
  the IW prior as the conditioning lever.
- **max_abs_offdiag_corr** — exp 0021's 0.80 reflected a near-degenerate direction.
  A moderate value (≈ 0.3-0.6) after regularization indicates real correlation
  without near-collinearity; a value still near 0.8+ with a floored eigenvalue
  means ν is too small (raise it).
- **Per-block topic quality** — block-aware NPMI per block should hold near exp
  0021's levels; the dementia block must STILL show Alzheimer's/amnestic (topic
  with Amnesia + Alzheimer's + MCI) and vascular (atherosclerosis + AFib +
  dementia) sub-phenotypes. If they collapse, ν is too large (over-shrink toward
  diagonal — back toward a washed minority arm).
- **Cross-foreground (cancer↔dementia) sub-block of R** — now that Σ is
  conditioned, inspect the 10×10 cancer↔dementia sub-block of `correlation.npy`:
  regularized thin cells sit near the prior; any cell with genuine comorbid
  support shows signal.
- **ELBO + Σ_var** — should stay in the same regime as 0021 (Σ_var O(1-10)); a
  large ELBO drop would indicate the prior is too strong.

## Decision

- **cond drops to O(1e2-1e3) + dementia sub-phenotypes preserved + sensible
  cross-block correlations** → the IW prior is the validated conditioning lever
  for the gated/thin-comorbid regime; record `sigma_prior_count ≈ 100` as the
  default for gated full-Σ runs and treat the gated correlation structure as
  usable. Log the cross-block correlation findings as an insight.
- **cond improves but is still > 1e4** → ν = 100 under-regularizes; raise
  `sigma_prior_count` (e.g. 300-1000) and/or add a mild `sigma_diag_shrink`
  (0.1-0.3) as a second conditioning lever.
- **cond fixed but dementia sub-phenotypes collapse / block NPMI drops** → the
  prior is over-shrinking toward diagonal; lower `sigma_prior_count`. The
  conditioning-vs-correlation tradeoff has a sweet spot to find.
- **No change in conditioning** → the near-singular direction is not coming from
  the thin cross-foreground cells; re-examine which topic pair drives
  max_offdiag_corr=0.80 (could be a genuine within-block near-collinearity), and
  reconsider whether the SPD-assembly model holds for this cohort.

## Run

```
make exp ID=22
```

Compare head-to-head with exp 0021 (same config, IW prior OFF). The delta in
`Σ_eig[cond]` is the direct measurement of the inverse-Wishart prior as the
gated-conditioning lever (ADR 0033). Result feeds insight 0032.
