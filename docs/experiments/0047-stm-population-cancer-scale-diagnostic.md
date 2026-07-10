---
id: 47
slug: stm-population-cancer-scale-diagnostic
status: done
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

# Experiment 0047 — Population-cancer gated STM: MAP-vs-marginalized held-out scale cluster test

## Goal

A **dashboard-only re-export** of the exp 0028 population-cancer gated STM fit,
to run the real-corpus MAP-vs-marginalized held-out generative-scale diagnostic
(the `marginalized_diagnostic` block added in the cluster-test instrument) and
compare it against the synthetic decomposition (exp 0046). This is a SEPARATE
experiment slot so it does NOT overwrite the exp 0028 population-cancer dashboard
bundle.

**No re-fit.** The fit frontmatter above is byte-identical to exp 0028 (same
cohort, seed 42, K=60, hardening stack, cache_uri) so the covariate cache and
model are reused as-is; only the id/slug differ.

## The question

The synthetic decomposition (exp 0046) showed that plain Laplace-Monte-Carlo
marginalization of the held-out predictive fixes the MAP-plug-in scale drift in
low dimensions (K=8) but INVERTS at production dimensionality (K=60, well-specified
synthetic data): there MAP is holdout-flat (drift ~0.12) and the marginalized
estimator drifts hard (~1.0), a confirmed Laplace-approximation bias (a
controller probe showed it is stable across seeds and GROWS with sample count, so
it is bias not Monte-Carlo variance). But the synthetic corpus is well-specified,
while the real cancer corpus is misspecified — and on real data the MAP c* is
KNOWN to drift (4.58 at holdout 0.5 -> 3.65 at holdout 0.95), which the synthetic
K=60 MAP does NOT reproduce. So the real corpus is the decisive test. Three
outcomes, each with a clear reading:

- **MAP drifts, marginalized flat** -> the real drift was the MAP artifact after
  all; marginalization is the fix (resurrect the production flip).
- **MAP flat, marginalized drifts** (matches synthetic) -> marginalization is
  worse at production scale; the real MAP drift is misspecification. Keep MAP,
  write the negative-result insight, drop the flip.
- **Both drift** -> misspecification dominates; the scale is genuinely
  holdout-dependent on real data. The honest answer is a documented holdout
  choice; neither estimator "fixes" it.

Nothing about what SHIPS changes in this run: `eta_scale` still ships the
MAP-plug-in smoothed c* at holdout 0.5. The marginalized comparison is added under
`correlation.json` -> `eta_scale_diagnostic.marginalized_diagnostic`.

## How to run (Dataproc master, from `analysis/cloud`)

The diagnostic lives INSIDE the real held-out-LL sweep block, so this run must let
that sweep run — do NOT set `BUILD_ETA_SCALE_OVERRIDE` (that override short-circuits
the whole sweep block and the diagnostic with it). It therefore pays the ~13-min MAP
sweep plus the marginalized diagnostic (on a sampled corpus for cost control).

```
git pull && git rev-parse --short HEAD          # confirm the instrument commit is present

# Reuse exp 0028's fit under this experiment's slug (dir copy, NOT a re-fit).
# $RUNS_DIR is the runs dir build-dashboard-exp resolves (see run_experiment.py
# DEFAULT_RUNS_DIR / the Makefile RUNS_DIR); use your actual value.
cp -r  "$RUNS_DIR/0028-stm-population-cancer-gated"  "$RUNS_DIR/0047-stm-population-cancer-scale-diagnostic"
rm -rf "$RUNS_DIR/0047-stm-population-cancer-scale-diagnostic/dashboard_bundle"   # write a fresh bundle

# Turn the diagnostic ON. Leave BUILD_ETA_SCALE_OVERRIDE UNSET.
export BUILD_MARGINALIZE_SCALE_DIAGNOSTIC=1
export BUILD_MARGINALIZE_SCALE_SAMPLES=64        # optional; S per doc (default 64)
export BUILD_MARGINALIZE_SCALE_DOC_FRAC=0.02     # optional; corpus sample frac (default 0.02)

make build-dashboard-exp ID=47 2>&1 | tee ~/build_0047.log
```

If the runner objects to the copied checkpoint's manifest `id`/`slug` (it should
not for a build-only run), either edit the copied `manifest.json` `id`/`slug` to
match this experiment, or use the explicit-checkpoint fallback:
`make build-dashboard-bundle CHECKPOINT="$RUNS_DIR/0028-stm-population-cancer-gated"
BUNDLE_ARGS='--model-class stm --out-dir <0047-bundle-dir> --zip-name 0047-stm-population-cancer-scale-diagnostic-dashboard.zip <the STM/gated/covariate args build-dashboard-exp normally supplies>'`
— the tracked `build-dashboard-exp ID=47` path is preferred because it constructs
the gated/covariate/cache args from this frontmatter automatically.

## What to read back

- **Driver log:** `STM marginalized-scale diagnostic (sample n=..., S=64): MAP
  drift=... marg drift=...; MAP c*=... marg c*=...`.
- **Bundle:** `correlation.json` -> `eta_scale_diagnostic.marginalized_diagnostic`
  -> `map_residual_drift` vs `marg_residual_drift`, plus both `*_cstar_by_holdout`
  curves. Compare `map_cstar_by_holdout` here to the full-corpus MAP
  `robustness_argmax_by_holdout` in the same `eta_scale_diagnostic` (sanity: the
  2% sample should not move MAP much).
- `cp` the bundle/zip to a uniquely-named file before downloading (repeated runs
  overwrite the default path).

## Result (2026-07-10)

Shipped `eta_scale` = **4.61** (MAP smoothed c\* @ holdout 0.5, unchanged). The
`marginalized_diagnostic` (n=967-doc sample, S=64) vs the full-corpus MAP robustness:

| holdout f | MAP c\* (full) | MAP c\* (sample) | marginalized c\* (sample) |
|---|---|---|---|
| 0.5 | 4.61 | 5.30 | 2.36 |
| 0.8 | 3.75 | 3.90 | 2.65 |
| 0.95 | 3.65 | 3.80 | 3.76 |
| drift | 0.96 | 1.49 | 1.40 |

**Outcome = "both drift" (the third fork).** On the real (misspecified) corpus the MAP
drifts substantially (unlike the well-specified K=60 synthetic where it was flat), and
marginalization does NOT remove it — the marginalized drift (1.40) ≈ the MAP drift (1.49)
and runs in the OPPOSITE direction, the two crossing near f=0.95 (≈3.8). Marginalization
trades one drift for an equal-and-opposite one, so it is not shipped. The residual drift is
genuine per-document concentration heterogeneity (misspecification), the f-drift is its
observable signature, and the real fix is a per-document scale (multivariate-t prior),
gated on a burstiness/dedup check. Full analysis: insight 0044; domain-agnostic report
`docs/superpowers/specs/2026-07-10-marginalized-heldout-scale-findings-report.md`.

## Related

- Instrument: `corpus_heldout_scale_sweep_gated{,_rdd}(marginalize=...)` +
  `build_marginalized_scale_diagnostic` (branch `stm`, commits 32fdb9f..3aa961f);
  design `docs/superpowers/specs/2026-07-10-marginalized-heldout-scale-calibration-design.md`.
- Synthetic decomposition: exp 0046 (`docs/experiments/0046-marginalized-scale-decomposition/`).
- Fit provenance: exp 0028 (this reuses its checkpoint unchanged).
