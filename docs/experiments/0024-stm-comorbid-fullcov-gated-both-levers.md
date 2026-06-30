---
id: 24
slug: stm-comorbid-fullcov-gated-both-levers
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
sigma_diag_shrink: 0.1
sigma_prior_scale: 2.0
sigma_prior_count: 2000.0
---

# STM comorbid (cancer + dementia): gated full-Σ + BOTH conditioning levers (one per spectral end)

The conditioning resolution for the gated full-Σ arc. Exps 0021-0023 dissected the
ill-conditioning into TWO independent problems at opposite ends of the Σ spectrum;
this run engages the lever that owns each end, together.

## Why — exps 0021/0022/0023 proved conditioning has two heads

| metric | 0021 (neither) | 0022 (IW ν=100) | 0023 (diag-shrink 0.2) |
|---|---|---|---|
| Σ_eig **min** | 1e-6 (floored) | 1e-6 (floored) | **1.0 (off floor)** ✓ |
| Σ_eig **max** | 32.8 | 30.5 | **2.23e7 (runaway)** ✗ |
| max off-diag corr | 0.801 | 0.798 | 0.177 |
| Σ_eig cond | 3.28e7 | 3.05e7 | 2.23e7 |
| ELBO | −1.5897e6 | −1.5907e6 | −2.252e6 |

- **The MIN-eigenvalue end** is the SPD-assembly near-singularity (insight 0032
  Finding 3): background↔cancer and background↔dementia strongly coupled while
  cancer↔dementia is structurally zeroed forces a near-zero-variance direction. It
  is an OFF-DIAGONAL / CORRELATION effect, fixed by the N-INDEPENDENT
  `sigma_diag_shrink` (exp 0023: min eig 1e-6 → 1.0, max off-diag 0.80 → 0.177).
  The N-weighted IW prior cannot reach it (exp 0022: ν=100 left it at 1e-6).
- **The MAX-eigenvalue end** is a single topic's η-variance running away to 2.2e7
  (exp 0023). `sigma_diag_shrink` does not touch the diagonal
  (`(1−w)·diag + w·diag = diag`,
  [stm.py:739](../../spark-vi/spark_vi/models/topic/stm.py#L739)); the runaway is a
  DYNAMICAL feedback — shrinking the off-diagonals removes the correlation coupling
  that had been stabilizing a weakly-identified topic's variance (by tying it to
  well-estimated topics), and it runs away (insight 0029's mechanism). The cure is
  the VARIANCE anchor: the IW prior's per-entry diagonal MAP
  Σ_ii = (S_ii + ν·scale)/(N_ii + ν) is a per-iteration contraction toward `scale`
  — exactly insight 0031's `sigma_prior_count=2000`, which tamed the diagonal-Σ
  runaway.

Conditioning is the RATIO max/min, so a well-conditioned gated Σ needs BOTH levers,
one per end. This run sets `sigma_diag_shrink=0.1` (gentler than 0023's 0.2 — the
0.80 was mostly near-singular inflation, genuine correlations are ≤ 0.18, so a
light shrink suffices to lift the min end while preserving more real correlation
signal) and `sigma_prior_count=2000`, `sigma_prior_scale=2.0` (insight 0031's
established anchor to cap the max end).

## Hypothesis

With both levers engaged, the gated full-Σ fit:

(a) **min eigenvalue off the floor** (diag-shrink lifts the near-singular direction;
    `sigma_eig_min` ≫ 1e-6), AND **max eigenvalue bounded** O(1-10) (the IW anchor
    caps the variance runaway; `sigma_eig_max` no longer 2e7) — so **cond drops to
    O(1e1-1e3)** for the first time on the gated cohort;
(b) ELBO recovers to the 0021 regime (~−1.59e6, not 0023's −2.25e6) — the variance
    runaway that wrecked the η-KL is gone;
(c) the dementia sub-phenotypes (topic with Amnesia + Alzheimer's + MCI; topic with
    atherosclerosis + AFib + dementia) and per-block NPMI hold near 0021/0023 levels
    — both levers are targeted at Σ conditioning, not the β/topic content;
(d) a trustworthy, well-conditioned correlation matrix R: genuine cross-block
    correlations (small, ≤ ~0.2 after the light shrink) readable off `correlation.npy`.

## What to watch

- **Σ_eig[min] AND Σ_eig[max] — both ends.** Success is min ≫ 1e-6 AND max O(1-10),
  giving cond O(1e1-1e3). A floored min → raise `sigma_diag_shrink`; a max still
  ≫ 100 → raise `sigma_prior_count` (the runaway topic is better-supported than
  ν=2000 anchors, or the anchor is being outrun).
- **ELBO** — should climb back to ~−1.59e6. If it stays near −2.25e6 the variance
  runaway is not actually capped (investigate which topic and its N_ii).
- **Per-block topic quality + dementia sub-phenotypes** — block NPMI background
  ~+0.19, cancer ~+0.19, dementia ~+0.15; topics 41 (Alzheimer's/amnestic) and 49
  (vascular) must survive. If they collapse, the combined shrink is over-flattening
  the minority arm — lower `sigma_prior_count` and/or `sigma_diag_shrink`.
- **max_abs_offdiag_corr** — expect a modest value (~0.15-0.4): the light
  diag-shrink (0.1) preserves more real correlation than 0023's 0.2, while the IW
  anchor adds some variance-side regularization.

## Decision

- **cond O(1e1-1e3) (both ends controlled) + ELBO ~−1.59e6 + sub-phenotypes
  preserved** → the two-lever picture is validated: `sigma_diag_shrink` for the
  min/correlation end, `sigma_prior_count` for the max/variance end. Record
  `sigma_diag_shrink≈0.1` + `sigma_prior_count≈2000` (scale 2) as the gated full-Σ
  default. Update insight 0032 (Finding 5 → confirmed) and ADR 0033's regularizer
  guidance; treat the gated correlation matrix as usable. Log the cross-block
  correlation structure as a follow-up insight.
- **min fixed but max still runs away (Σ_eig max ≫ 100)** → ν=2000 is outrun by the
  runaway topic; raise `sigma_prior_count` (4000-8000) and inspect which topic /
  N_ii drives Σ_var max.
- **max capped but min back on the floor** → `sigma_diag_shrink=0.1` is too gentle
  once the IW prior's 16%-ish off-diagonal shrink is also acting; raise it (0.15-0.2).
- **cond fixed but dementia sub-phenotypes collapse / block NPMI drops** → combined
  over-shrink; back off `sigma_prior_count` first (it pulls hardest on the thin
  dementia arm), then `sigma_diag_shrink`.
- **max runaway persists at any ν** → the runaway may not be a thin-support topic
  the per-entry anchor can catch; inspect the Σ-diagonal to identify the topic and
  its support, and reconsider whether diag-shrink should be applied more gently
  (smaller w) to avoid triggering the runaway in the first place.

## Run

```
make exp ID=24
```

Four-way comparison: 0021 (neither) / 0022 (IW only) / 0023 (diag-shrink only) /
0024 (both). The headline is whether 0024 is the first gated run to control BOTH
eigenvalue ends simultaneously (cond O(1e1-1e3)). Result feeds insight 0032.
