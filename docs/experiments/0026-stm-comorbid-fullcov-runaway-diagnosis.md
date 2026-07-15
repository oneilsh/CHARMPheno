---
id: 26
slug: stm-comorbid-fullcov-runaway-diagnosis
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
max_iter: 40
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# STM comorbid (cancer + dementia): diagnose the Σ variance runaway

Re-runs the exp 0025 configuration **unchanged** (gated multi-group full-Σ, K=50 =
30 background + 10 cancer + 10 dementia, pd_complete assembly) to reproduce and
**attribute** the max-eigenvalue variance runaway observed on the cluster (insight
[0033](../insights/0033-gated-fullcov-variance-runaway-is-an-init-identifiability-failure.md)):
one topic's Σ_kk climbed to 2.71e5 by iter 28, ELBO to −1.56e7.

This is a **diagnostic** run, not a fix. The engine change since exp 0025 is that
`OnlineSTM.iteration_summary`
([stm.py](../../spark-vi/spark_vi/models/topic/stm.py)) now appends a `maxvar[...]`
segment to every per-iter driver line, naming the highest-variance topic and its β
coherence. `max_iter` is dropped to 40 because the runaway is fully developed and
attributable well before then (it was clear by iter 28 on the earlier run); no need
to pay for 100 iterations of a diverging fit.

## Why

The local controlled reproduction (insight 0033) established that the runaway needs
two ingredients — a **weakly-identified topic** AND a **prior variance free to grow**
— and that with good topic identification the fit is stable even at a 5% minority
arm. It also showed the runaway topic in the synthetic is *diffuse* (β near the
corpus marginal). What we do not yet know is **which** topic runs away on the real
cohort and **whether it is diffuse** (→ an init/identifiability fix: spectral-init
quality or K sizing) or a **coherent phenotype** (→ the cause is elsewhere). This run
answers that directly from the driver log.

## What to watch

Each per-iter summary line now ends with, e.g.:

    ..., maxvar[topic=K label peak=0.0039 ess=298], blocks[bg=... cancer=... dementia=...]

- **topic / label** — the index and block (`background` / `cancer` / `dementia`) of
  the current highest-variance topic. Watch whether it is stable (one topic runs away)
  or wanders, and which block it sits in.
- **peak** — the maximum single-term probability of that topic's β. A coherent
  phenotype peaks ≈ 0.01–0.14 (insight
  [0028](../insights/0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md));
  a diffuse/unidentified topic peaks ≈ 0.003.
- **ess** — the effective number of terms (exp of β entropy). A coherent topic has a
  small ess (tens); a diffuse one spans most of the vocabulary (thousands).

## Success criteria

Not a pass/fail fit. Success = we can read off, from the log, the runaway topic's
identity, block, and coherence (peak / ess), and the iter at which its variance takes
off. That determines the fix per insight 0033:

- **diffuse runaway topic** (low peak, high ess) → identification failure → pursue
  better spectral init / K sizing (or the unit-diagonal Σ insurance, Option B).
- **coherent runaway topic** (normal peak/ess) → the cause is not identification →
  re-open the mechanism (SVI noise, a specific covariate interaction, etc.).

The `scratchpad/diagnose_runaway.py` tool gives the same attribution post-hoc from a
saved model dir if a full re-run is not wanted.
