---
id: 27
slug: stm-comorbid-blockwise-unit-diagonal
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

# STM comorbid (cancer + dementia): block-wise unit-diagonal correlation Σ

Re-runs the exp [0025](0025-stm-comorbid-fullcov-gated-pdcompletion.md) / [0026](0026-stm-comorbid-fullcov-runaway-diagnosis.md)
cohort configuration (gated multi-group, K=50 = 30 background + 10 cancer + 10
dementia) on the **block-wise unit-diagonal correlation Σ** engine
([ADR 0034](../decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md)).
This is the **fix** run: exp 0026 attributed the Σ max-eigenvalue variance runaway
(insight [0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md))
to a weakly-identified minority dementia sub-phenotype whose prior variance was free
to grow; the block-wise design severs that feedback loop by construction (pins Σ_ii=1
every M-step) and retires `pd_complete` from the fit path.

## Why

Validates the design of
[docs/superpowers/specs/2026-07-01-stm-blockwise-unit-diagonal-correlation-design.md](../superpowers/specs/2026-07-01-stm-blockwise-unit-diagonal-correlation-design.md)
on the real cohort: the runaway is fixed structurally rather than tuned away, and the
completion machinery (max-determinant `pd_complete` / Dykstra) is removed from the
per-iteration M-step. `max_iter` is raised to 100 (a full validation fit, vs the
diagnostic's 40) because we now want convergence, not just to watch the blowup develop.

The only change from exp 0026 is the engine: the Σ M-step
([stm.py](../../spark-vi/spark_vi/models/topic/stm.py)) now standardizes the observed
per-pair scatter to correlations, lazy-keeps unsupported pairs, and pins the diagonal
to 1 — no PD completion. Under single-label gating each E-step still only inverts a
fully-observed marginal Σ[background ∪ one group] (`safe_inverse`), so nothing needs the
cross-group block completed; it stays NA. No frontmatter flag selects this — the engine
change is unconditional.

## Success criteria

- **No variance runaway:** max Σ_var ≈ 1 throughout the fit (the diagonal is pinned to
  1 by construction). Watch the per-iter `maxvar[topic=... peak=... ess=...]` line — the
  named highest-variance topic's Σ_kk must stay at 1, never climb (contrast exp 0026:
  Σ_kk → 2.71e5 by iter 28).
- **Sub-phenotypes preserved:** the Alzheimer's/amnestic and vascular dementia
  sub-phenotypes (insight [0032](../insights/0032-gated-fullcov-recovers-dementia-subphenotypes-and-exposes-spd-assembly-conditioning.md)
  Finding 2) survive — the minority arm is still resolved, since unit-diagonal is
  insurance against the runaway, not a topic-quality change (insight 0033 Finding 3).
- **Trustworthy correlation report:** the within-group and background↔foreground
  correlation matrix is honest (the fitted Σ is already a correlation matrix); cross-group
  pairs are NA (`topic_correlation_identified` masks unsupported pairs — n_pairs=0 for
  single-label cross-group).
- **ELBO recovered:** near −1.59e6 (contrast exp 0026's runaway ELBO of −1.56e7).
- **No pd_complete cost:** the per-iter driver log no longer carries the
  `M-step pd_complete: ...s sweeps=...` line (that call was removed from the fit path);
  per-iteration wall-clock is free of the serial covariance-selection completion.

## Result

_(pending — run `make exp ID=27`)_
