---
id: 21
slug: stm-comorbid-fullcov-gated-multigroup
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
---

# STM comorbid (cancer + dementia): gated multi-group full-covariance Σ validation

Validates the gated multi-group path of the full-covariance Σ arc (ADR 0033):
cross-group covariance is estimated only from comorbid documents that co-activate
both groups, thin cross-group cells fall back to the prior via the min_pair_support
floor, and the assembled Σ is SPD-repaired. The cohort is cancer + dementia (two
foreground blocks), mirroring the gated-STM setup established in exp 0004.

## Why

Exp 0020 validates the non-gated (single-group) full-Σ path: a single allowed set
covering all K−1 free topics, all entries estimated from the full corpus, no SPD
assembly risk. The gated multi-group path is the harder case and the central
numerical risk identified in ADR 0033: background topics appear in every document's
allowed set and correlate with every foreground block, but cross-foreground entries
(cancer↔dementia) are estimated only from comorbid patients who co-activate both
blocks. Pinning cross-foreground cells that have no support while background-to-both-
foreground cells are freely estimated produces an inconsistent assembled matrix that
need not be positive definite.

ADR 0033 addresses this with three layers: the inverse-Wishart prior fills uninformed
cells with a coherent SPD scale, the nearest-SPD eigenvalue-floor projection imputes
the minimum-perturbation completion, and the sigma_ridge floor provides the final
backstop. The min_pair_support floor (set here to 10) zeros out the scatter from
cross-group cells backed by fewer than 10 co-activating patients before the IW blend,
so those cells are prior-dominated rather than driven by a handful of comorbid patients
— both a robustness guard and a small-cell privacy control.

The cancer + dementia cohort (cohort_def: cancer_or_dementia, doc_unit: patient_cohort)
is the natural vehicle: the same cohort used in exp 0003/0004, which established that
comorbid patients contribute one document per cohort (a patient with both conditions
appears as both a cancer doc and a dementia doc). With K=50 = 30 background + 10
cancer + 10 dementia, the background block provides the shared cross-group channel
through which comorbid signal can flow, while the foreground blocks capture
group-distinctive phenotypes.

## Hypothesis

The gated multi-group full-Σ fit:

(a) Estimates background-to-cancer and background-to-dementia correlation entries
    from the large within-group populations, while cross-foreground (cancer↔dementia)
    entries are populated only where comorbid patient support exists and suppressed
    (prior-fallback) where it does not.
(b) The min_pair_support floor correctly suppresses thin cross-foreground cells:
    the imputed_fraction diagnostic reports the share of entries that fell below
    the floor, and the suppressed cells carry the IW prior value rather than a noisy
    few-patient estimate.
(c) The assembled Σ is SPD (condition number bounded after the SPD-repair step;
    no degenerate eigenvalues).
(d) Topic quality per block holds at an acceptable level: cancer and dementia
    foreground blocks show clinically distinct top-word profiles, background block
    shows broad comorbidity vocabulary.

## What to watch

- **Cross-group (cancer↔dementia) correlation entries** — are they populated only
  where comorbid patients exist? Inspect the (10×10) cancer-foreground ↔ dementia-
  foreground sub-block of R: entries with high N_ij (many comorbid patients) should
  show signal; entries with low N_ij should be near the prior value (close to
  diagonal, not a raw few-patient estimate). This is the end-to-end test of the
  per-pair lazy update and min_pair_support floor.
- **imputed_fraction diagnostic** — the share of Σ entries that fell below
  min_pair_support=10 and were zeroed before the IW blend. Expect a non-trivial
  fraction for the cancer↔dementia cross-foreground block (many pairs have few
  comorbid patients); near-zero would mean every pair is well-supported and the
  floor is not biting.
- **Σ condition number (Σ_eig[cond])** — must stay bounded after the SPD-repair
  step. A condition number climbing above 1e4 indicates the repair is struggling
  and the IW prior weight should be increased.
- **Topic quality per block** — count distinctly resolved topics in each block
  (cancer foreground, dementia foreground, background). The gated block structure
  means dementia's 21% corpus share is not overwhelmed by cancer's majority when
  estimating dementia foreground topics (the same motivation as exp 0004).
  Success = each foreground block has identifiable, block-distinctive phenotypes.
- **R background sub-block** — the 30×30 background-topic correlation sub-block
  should show sensible comorbidity structure (background topics shared across both
  cohorts); this is the highest-support sub-block and provides a quality reference.
- **Sigma[min … max] + eigenvalue range** — per-iter traces to catch Σ blowup
  analogous to the pre-stabilizer runs. Bounded O(1–100) is the target.

## Decision

- **Cross-foreground entries populated only where comorbid support exists + imputed
  fraction nonzero and plausible + SPD Σ (cond bounded) + sensible per-block topics**
  → the gated multi-group full-Σ path is validated end-to-end. ADR 0033 multi-group
  path is confirmed. Record the imputed fraction and condition number as an insight.
- **Cross-foreground entries near identical to prior everywhere (no comorbid signal
  even where many patients exist)** → the per-pair lazy update is not scattering
  into the cross-group block; check the A_d set construction and scatter indexing.
- **Ill-conditioned Σ after SPD-repair (cond > 1e4)** → the SPD repair is
  insufficient for this degree of inconsistency; increase the IW prior weight
  (`sigma_prior_count`) to regularize thin cells more strongly, and/or raise
  min_pair_support. Record the failure regime as an insight.
- **Foreground blocks degenerate (topics collapse or blend across groups)** →
  the marginal sub-block prior Σ_{A_d,A_d} is not isolating the foreground E-step
  correctly; review the sub-block precision extraction against the ADR 0033 design
  spec.

## Run

```
make exp ID=21
```

Compare cross-group R sub-block against exp 0020 (non-gated, single group): the
delta between 0020 and 0021 is the gating effect — multi-group structure, marginal
sub-block priors, and per-pair support. The exp 0004 (gated diagonal-Σ, pending)
would be the gated-diagonal baseline if it is ever run; otherwise exp 0020's
diagonal precedent (exp 0015/0017) serves as the topic-quality reference.
