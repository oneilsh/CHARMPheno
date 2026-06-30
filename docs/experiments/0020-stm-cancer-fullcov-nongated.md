---
id: 20
slug: stm-cancer-fullcov-nongated
status: done
model_class: stm
cohort: cancer
cohort_def: first_cancer_year
prior_obs_days: 0
person_mod: 4
doc_unit: patient
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 40
max_iter: 100
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
---

# STM cancer demo: first full-covariance Σ run — non-gated

The feature run for the full-covariance Σ arc (ADR 0033). Identical to exp 0017
in cohort, covariates, K=40, seed 42, sigma_init=1, reference + dense spectral on,
except that Σ is now the full (K−1)×(K−1) covariance matrix (CTM/STM treatment,
Blei & Lafferty 2007) rather than the diagonal vector. The scientific deliverable is
the topic correlation matrix R: which cancer phenotypes co-occur.

## Why

The diagonal-Σ stack was validated in exp 0015 (dense spectral, insight 0030) and
exp 0017 (scalable spectral equivalence, insight 0031): 40 resolved topics, bounded
Σ O(1–10), NPMI +0.173 / +0.166. That diagonal result is the prior baseline for
this arc. ADR 0033 replaces the K-vector of per-topic variances with a full
(K−1)×(K−1) positive-definite Σ — the covariance that the CTM/STM model always
intended — so that topic correlations are estimated and exported rather than
structurally suppressed.

In the non-gated (single-group) case the full-Σ M-step reduces to the classical
CTM update: Σ = (1/D) Σ_d [(η̂_d − μ_d)(η̂_d − μ_d)ᵀ + ν_d], the mean
outer-product of residuals plus the mean Laplace covariance. No cross-group logic,
no support floor complications — the cleanest path to verifying that the
implementation recovers a sensible, well-conditioned correlation structure on real
data. Dense spectral init is used (rather than scalable) to hold the β seed identical
to exp 0015, isolating the Σ change as the only difference.

## Hypothesis

The full-Σ fit on the cancer cohort recovers a sensible phenotype correlation
structure while holding topic quality at least as good as the diagonal baseline
(exp 0015/0017):

(a) The correlation matrix R shows clinically interpretable structure — comorbidity
    clusters among solid tumors, shared metabolic/cardiovascular burden, etc. — not
    a near-identity or a degenerate near-rank-1 structure.
(b) Topic quality holds: ~40 resolved topics, NPMI comparable to exp 0017 (+0.166).
    The full-Σ M-step should not degrade the β fixed point.
(c) Σ stays well-conditioned: eigenvalue condition number < 1e4 (rough threshold;
    a condition number near 1e10 would indicate a near-singular Σ driving η inference
    off a cliff), max |off-diagonal correlation| < 0.9 (below perfect co-occurrence).
(d) The reference topic 0 retains real baseline mass (not the ~0.0001 dead floor of
    pre-stabilizer runs).

## What to watch

- **R (topic correlation matrix)** — THE headline output. Inspect the off-diagonal
  structure: are the strongest positive correlations between clinically related cancer
  phenotypes (e.g. breast + hormone-receptor, colon + GI comorbidities)? Are there
  expected near-zero correlations between anatomically distinct cancers? A near-identity
  R (all correlations near zero) would mean the data has insufficient signal to estimate
  cross-topic covariance; a near-rank-1 R (all topics highly correlated) would suggest
  a runaway convergence issue.
- **Σ_eig[cond] (condition number)** — the new per-iter diagnostic (ADR 0033 /
  ADR 0030 generalization). Should remain bounded across iterations; a climbing
  condition number indicates Σ is becoming ill-conditioned and the IW prior or
  diagonal-shrink lever is needed.
- **Σ_corr[max_offdiag] (max |off-diagonal correlation|)** — bounded below 1.0.
  A value near 1 indicates (near-) perfect collinearity between two topics; inspect
  which pair and whether it is real comorbidity signal or a degenerate collapse.
- **Topic resolution vs 0015/0017** — count distinct phenotypes; success = ~40
  resolved topics. A drop back toward ~14 (the random-init collapse) would indicate
  the full-Σ M-step is destabilizing the β fixed point.
- **NPMI mean** — relative to exp 0017 (+0.166). Read per-active-topic (marginal-blend
  dead topics inflate the mean). Expect comparable or slightly improved NPMI; a large
  drop would indicate quality regression under full Σ.
- **Σ[min … max] trace (ADR 0030)** — individual diagonal entries (topic variances)
  should remain O(1–10) as in exp 0015/0017; the new eigenvalue range and condition
  number generalize this check to the full matrix.
- **Convergence + ELBO** — comparable to exp 0017 (ELBO ≈ −1.10e6); the Gaussian
  η-KL now uses the full-matrix trace and logdet terms, so a large ELBO gap from
  0017 warrants investigation (numerical issue vs genuine model change).

## Decision

- **Sensible correlation structure + topic quality holds (~40 topics, NPMI ≈ 0017) +
  well-conditioned Σ (cond < 1e4)** → full-covariance Σ validated as the new default
  on the non-gated cancer cohort. Mark ADR 0033 Accepted + Implemented for the
  non-gated path. Proceed to exp 0021 (gated multi-group validation).
- **Degenerate R (near-identity or near-rank-1) + topic quality otherwise intact** →
  the prior estimate is too weak or the ridge is dominating; investigate the IW prior
  lever (`sigma_prior_scale`, `sigma_prior_count`) and report as an insight before
  re-running.
- **Ill-conditioned Σ (cond > 1e4 or climbing)** → the unconstrained full-Σ MLE
  is not self-regularizing on this corpus at K=40; try IW prior (exp TBD) and/or
  diagonal-shrink. Record the stability regime as an insight.
- **Topic quality degrades vs 0017 (well below ~40 topics or NPMI drop > 0.05)** →
  investigate whether the full-Σ prior Hessian is adversely affecting the E-step
  η updates; check the implementation against the ADR 0033 design spec before
  further runs.

## Run

```
make exp ID=20
```

Compare head-to-head with exp 0015 (dense spectral, diagonal Σ, 300 iters — the
definitive diagonal baseline) and exp 0017 (scalable spectral, 300 iters — the
topic-quality reference). The delta between 0017 and 0020 is the full-Σ effect on
topic quality and convergence; the R matrix is the new deliverable that has no
diagonal-run counterpart.
