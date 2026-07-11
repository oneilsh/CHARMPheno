---
id: 48
slug: stm-population-cancer-tprior-scale
status: planned
model_class: stm
cohort: population_cancer
cohort_def: population_cancer
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 60
background_k: 40
foreground: "cancer:20"
group_var: source_cohort
max_iter: 200
subsampling_rate: 0.1
tau0: 128
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# Experiment 0048 — Population-cancer gated STM: multivariate-t per-document scale sweep

## Goal

A **dashboard-only re-export** of the exp 0028 population-cancer gated STM fit
(same frozen checkpoint exp 0047 reused), to run the real-corpus multivariate-t
per-document generative-scale diagnostic
(`corpus_tprior_scale_sweep_gated_rdd`, design doc
`docs/superpowers/specs/2026-07-10-tprior-per-document-scale-design.md`). This
is a SEPARATE experiment slot so it does NOT overwrite the exp 0028 / exp 0047
population-cancer dashboard bundles.

**No re-fit.** The fit frontmatter above is byte-identical to exp 0028/0047
(same cohort, seed 42, K=60, hardening stack, cache_uri) so the covariate cache
and model are reused as-is; only the id/slug differ.

Insights 0044/0045 established that the residual held-out-scale drift left
after bias-correcting the MAP-plug-in estimator (exp 0047's "both drift"
result) is genuine per-document concentration heterogeneity, not a fixable
estimator artifact — the fix is prior-side: give each document its own scale
s_d, drawn from a corpus-level Inverse-Gamma(ν/2, ν/2), which marginally makes
η_d a multivariate-t_ν(μ_d, c·R). This run calibrates (c, ν) on the frozen
population_cancer fit and tests the two falsifiable predictions the design doc
commits to before the numbers are known.

## Config

Identical fit to exp 0047 (see frontmatter above — cohort, seed, K=60,
hardening stack, cache_uri all unchanged). Adds the new flagged diagnostic:

```
BUILD_T_PRIOR_SCALE=1
BUILD_T_PRIOR_SCALE_DOC_FRAC=0.05        # default; corpus sample frac
BUILD_T_PRIOR_SCALE_NU_GRID=2.5,5,10,20,inf   # default
BUILD_T_PRIOR_SCALE_C_GRID=2,3,4,6,8,12       # default
```

All three env knobs are optional overrides on top of the defaults baked into
`analysis/cloud/build_dashboard_cloud.py`'s `BUILD_T_PRIOR_SCALE` block; the
values above are what ships if left unset. No hardcoded scale: the grids are
inputs and the sweep emits (c*, ν*) — never a magic number.

## Command

Reuse exp 0028's fit under this experiment's slug (dir copy, NOT a re-fit),
same pattern as exp 0047:

```
git pull && git rev-parse --short HEAD          # confirm the instrument commit is present

cp -r  "$RUNS_DIR/0028-stm-population-cancer-gated"  "$RUNS_DIR/0048-stm-population-cancer-tprior-scale"
rm -rf "$RUNS_DIR/0048-stm-population-cancer-tprior-scale/dashboard_bundle"   # write a fresh bundle

export BUILD_T_PRIOR_SCALE=1
# optional overrides (defaults shown):
# export BUILD_T_PRIOR_SCALE_DOC_FRAC=0.05
# export BUILD_T_PRIOR_SCALE_NU_GRID=2.5,5,10,20,inf
# export BUILD_T_PRIOR_SCALE_C_GRID=2,3,4,6,8,12

make build-dashboard-exp ID=48 2>&1 | tee ~/build_0048.log
```

If the runner objects to the copied checkpoint's manifest `id`/`slug`, either
edit the copied `manifest.json` to match this experiment, or use the explicit-
checkpoint fallback (as exp 0047 documents):
`make build-dashboard-bundle CHECKPOINT="$RUNS_DIR/0028-stm-population-cancer-gated"
BUNDLE_ARGS='--model-class stm --out-dir <0048-bundle-dir> --zip-name 0048-stm-population-cancer-tprior-scale-dashboard.zip <the STM/gated/covariate args build-dashboard-exp normally supplies>'`
— the tracked `build-dashboard-exp ID=48` path is preferred because it
constructs the gated/covariate/cache args from this frontmatter automatically.

Nothing about what SHIPS changes in this run: `eta_scale` still ships the
MAP-plug-in smoothed c* at holdout 0.5 (unchanged from exp 0028/0047). The
t-prior sweep is added under a NEW top-level bundle file, `t_prior_scale.json`
— an off-by-default, never-fatal diagnostic.

## Falsifiable predictions (verbatim from the design doc)

Both readouts emit numbers and bake no thresholds; the design doc commits to
these predictions before the run:

1. **Check #1 — f-drift collapse.** The 1-D c-sweep at several holdout
   fractions f ∈ {0.2, 0.3, 0.5} under ν = ∞ (spread of c*(f)) vs under ν = ν*
   (spread of c*(f | ν*)). **Prediction: the spread collapses** (the tprior
   spread should be smaller than the gaussian spread). Emit both spreads; no
   verdict.
2. **Check #2 — ŝ_d reproduces the implied scales.** At (c*, ν*), infer ŝ_d
   for every doc on the full doc (no split), emit the ŝ_d·c* distribution
   (quantiles). **Prediction: its spread is consistent with the bias-corrected
   per-f implied scales 6.95 / 5.60 / 5.41** (insight 0044). Emit the
   distribution; the insight interprets it.

## What to read back

- **Driver log:** `t-prior scale (n=..., ): argmax c*=... nu*=... | drift
  gauss=... tprior=... | sd*c* p50=...`.
- **Bundle:** `t_prior_scale.json` → `grid` / `argmax` (c*, ν*) / `drift`
  (`gaussian_spread` vs `tprior_spread`, check #1) / `sd_readout`
  (`sd_c_quantiles`, check #2).
- `cp` the bundle/zip to a uniquely-named file before downloading (repeated
  runs overwrite the default path).

## Result

TODO: fill after cluster run.

## Related

- Instrument: `corpus_tprior_scale_sweep_gated{,_rdd}` +
  `_stm_doc_inference_tprior` (branch `stm`); design
  `docs/superpowers/specs/2026-07-10-tprior-per-document-scale-design.md`;
  plan `docs/superpowers/plans/2026-07-10-tprior-per-document-scale.md`.
- Predecessor: exp 0047 (`docs/experiments/0047-stm-population-cancer-scale-diagnostic.md`)
  — the MAP-vs-marginalized held-out scale test whose "both drift" result
  motivated this prior-side fix.
- Fit provenance: exp 0028 (this reuses its checkpoint unchanged).
- Insights: 0044 (heterogeneity reframe + bias inversion), 0045 (dedup gate →
  prior-side verdict).
