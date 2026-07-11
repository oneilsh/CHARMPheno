# Multivariate-t Per-Document Scale (c, ν) — Design

**Date:** 2026-07-10
**Branch:** stm
**Thread:** generative concentration-scale calibration (`eta_scale`), with collaborator "Fable"
**Status:** design approved (this doc), pre-plan
**Precedes:** implementation plan `docs/superpowers/plans/2026-07-10-tprior-per-document-scale.md`

## Motivation

Insights 0044/0045 established that, after bias-correcting the MAP-plug-in
held-out scale estimator, a residual drift in the calibrated generative scale
across holdout fractions f remains, and that this drift is **genuine
per-document concentration heterogeneity** (the dedup gate ruled out the
within-document burstiness confound: spread_ratio 0.98, rank_corr 0.86,
burstiness_corr 0.009). The fix is therefore **prior-side**: give each document
its own scale.

A single global generative scale c (shipped `eta_scale`, the 1-D held-out
argmax) cannot sit right for both diffuse and peaky patients simultaneously —
that mis-fit is exactly what makes c* drift with f. Letting each document carry
its own scale s_d, drawn from a corpus-level distribution, removes the drift
while keeping a single pair of shippable, data-calibrated hyperparameters.

## The model

Per document d, over its allowed topic block (background ∪ its group), with
K′ = |allowed| − 1 free dims (reference topic fixed at 0):

```
η_d | s_d  ~  Normal(μ_d, s_d · c · R)      μ_d = Γᵀ x_d,  R = the fit's correlation
s_d        ~  Inverse-Gamma(ν/2, ν/2)
```

Integrating s_d out gives η_d ~ **multivariate-t_ν(μ_d, c·R)** — the
scale-mixture-of-normals representation of the multivariate t (Kotz & Nadarajah
2004; Gelman et al., BDA3, §17.1). The IG(ν/2, ν/2) parameterization makes the
marginal scale matrix exactly c·R (E[s_d] = ν/(ν−2) for ν>2). As **ν → ∞,
s_d → 1** and the model recovers today's Normal(μ_d, c·R) exactly — the nesting.

### Roles of the two hyperparameters (they do NOT duplicate s_d)

This is a random-effects / hierarchical structure. s_d is a *per-document*
latent (the random effect: "how concentrated is THIS patient"); c and ν are the
*variance components* of the distribution s_d is drawn from:

- **c** — the location of the per-doc scale cloud (since E[s_d] ≈ 1, s_d·c
  centers on c). Plays the same role as today's 1-D c*.
- **ν** — the spread of the cloud, i.e. the corpus's concentration
  heterogeneity. Small ν = heavy tails = docs differ a lot in peakiness;
  ν → ∞ = homogeneous = today's model. ν is the quantitative measure of the
  heterogeneity insights 0044/0045 identified.

(c, ν) is what transfers from train to a held-out document whose s_d has not
been seen; s_d itself never leaves the document.

## Why calibrate held-out, not in the fit

Deliberate. The obstacle to fitting (c, ν) is not the machinery (the s_d update
is closed-form conjugate; c, ν could be updated by MoM/Newton; SVI could carry
an IG factor) — it is the **objective**. Maximizing in-sample ELBO over the
scale gives a systematically **compressed** estimate: insight 0037 (every
fit-anchored scale under-concentrated, 1.0/2.36/3.67 below the 5–7 held-out
band) and Fable's Caveat 2 (the Laplace/mean-field posterior over η is
under-dispersed, so the ELBO re-estimates the generative scale down to match its
own too-tight posterior). Held-out within-document prediction has no
in-sample-overfit incentive, so its argmax is unbiased — that is the entire
reason the held-out-calibration lineage (insights 0037/0038) exists. Fitting ν
in-sample is worse: heavier tails buy in-sample freedom to absorb training
noise, over-estimating heterogeneity in the same direction; and the t
degrees-of-freedom is notoriously ill-identified by ML regardless (flat
likelihood, boundary-seeking). So (c, ν) is calibrated on a **frozen fit** at
inference/export time. Promotion into the fit is deferred and gated on the
per-topic adequacy check (out of scope here); if it ever happens the held-out
sweep remains the referee.

## Per-document inference — explicit EM over (η, s_d)

Chosen over marginal-t MAP because s_d is a first-class inferred latent, which
is exactly the object of falsifiable check #2. Given (c, ν), for one document,
coordinate-ascent to the joint MAP:

- **η-step** (given s_d): the *existing* `_stm_doc_inference` L-BFGS E-step,
  unchanged, with `Sigma_inv_allowed = (1/(s_d·c)) · Rinv_allowed`. A scaled
  precision — no new optimizer. Conditional on s_d the objective is log-concave
  (multinomial data term + Gaussian prior), so the mode is unique and warm-start
  is result-preserving.
- **s_d-step** (given η): closed form. With q = (η−μ)ᵀ R⁻¹ (η−μ) (the
  correlation-Mahalanobis over the free dims), the complete-data posterior is
  s_d ~ Inverse-Gamma((ν+K′)/2, (ν + q/c)/2), and the joint-MAP update is its
  **mode**:  ŝ_d = (ν + q/c) / (ν + K′ + 2).
- Initialize s_d = 1 (the Gaussian solution), alternate to convergence
  (`sd_tol` on |Δs_d|, `sd_max_iter` cap). s_d converges in a handful of sweeps;
  each EM η-step warm-starts from the previous sweep's η̂, so only the first is a
  full solve.

**Scoring stays MAP-plug-in.** After inferring (η̂, ŝ_d) from the *visible*
half, score the held-out tokens with θ = softmax(η̂ mode) via `_predictive_loglik`
on E[β] — identical to the current MAP sweep. Marginalization is dead (insight
0044); the fix is on the prior, and the bias-inversion showed MAP is a clean
invertible instrument. The t-prior simply lets the instrument stop drifting.

## The 2-D sweep + the two falsifiable readouts

New `corpus_tprior_scale_sweep_gated` and `..._rdd`, mirroring
`corpus_heldout_scale_sweep_gated` (`spark_vi/mllib/topic/stm.py`): same
held-out split (`heldout_split`, seed independent of c/ν/f so every knob sees
the identical split — controlled comparison), same inference-vs-scoring role
split (expElogbeta for inference, E[β] via `_predictive_loglik` for scoring),
same short-doc skip guards. Emits, into `t_prior_scale.json`:

1. **(c, ν) grid of mean-per-token held-out LL → argmax (c*, ν*)** — the
   calibrated scale + heterogeneity for this corpus. The ν = ∞ column is the
   current single-c model, so the grid contains its own null: if the corpus is
   homogeneous the argmax lands at ν = ∞ and the t-prior buys nothing.
2. **Check #1 — f-drift collapse:** the 1-D c-sweep at several holdout fractions
   f ∈ {0.2, 0.3, 0.5} under ν = ∞ (spread of c*(f)) vs under ν = ν* (spread of
   c*(f | ν*)). Prediction: the spread collapses. Emit both spreads; no verdict.
3. **Check #2 — ŝ_d reproduces the implied scales:** at (c*, ν*), infer ŝ_d for
   every doc on the full doc (no split), emit the ŝ_d·c* distribution
   (quantiles). Prediction: its spread is consistent with the bias-corrected
   per-f implied scales 6.95 / 5.60 / 5.41 (insight 0044). Emit the
   distribution; the insight interprets it.

Both readouts **emit numbers and bake no thresholds** — same no-verdict contract
as the dedup/heterogeneity gate, and the same no-magic-number principle: the
sweep emits (c*, ν*) per corpus, never a hardcoded constant.

## Cost control (first-class requirements)

The naive count |c|·|ν|·|f|·(EM) is the number to avoid. Four requirements keep
the real cost at ~2–4× the current 1-D sweep on the same 5% sample:

1. **Warm-start** η̂ (and ŝ_d) across adjacent grid points (order the grid, seed
   from the neighbour) and across EM sweeps (seed from the previous sweep).
   Result-preserving (log-concave conditional η-step). Also speeds the existing
   sweep — the current cold re-solve per c is wasteful.
2. **f enters only the 1-D drift readout**, never the 2-D grid. The main
   (c, ν) sweep runs at a single f (0.3).
3. **Reuse the ν = ∞ column** from the existing Gaussian sweep; do not
   recompute. It doubles as the grid null and check #1's baseline.
4. **Coarse ν grid** {2.5, 5, 10, 20, ∞} (4 new columns) and a **c-range
   centered on the known c*** rather than swept wide. The ŝ_d readout is a
   single pass at (c*, ν*).

If the cluster run drags, the ν grid is the sole coarsening knob.

## Components / files

- `_stm_doc_inference_tprior(...)` — new, beside `_stm_doc_inference` in
  `spark_vi/mllib/topic/stm.py`. The per-doc EM wrapper: loops the existing
  η-step + the closed-form s_d update with warm-start; returns (η̂, ŝ_d, nu_d).
  Reuses the gated allowed/reference contract intact.
- `corpus_tprior_scale_sweep_gated` + `..._rdd` — the 2-D sweep + the three
  readouts. `_json_safe` at the distributed-function boundary (last session's
  lesson: numpy arrays in the summary broke the cluster run).
- `BUILD_T_PRIOR_SCALE` flag in `analysis/cloud/build_dashboard_cloud.py` →
  writes `t_prior_scale.json`, **and its filename added to the zip
  optional-files loop** (the exact omission that broke two prior cluster runs —
  on the checklist).
- Experiment doc cloning the exp 0047 population_cancer fit config with the flag
  on: `docs/experiments/0048-stm-population-cancer-tprior-scale.md`.

## Data flow

frozen fit bundle (lambda/Gamma/Sigma, cohort docs) → sample docs (5%) →
per-doc heldout_split → for (c, ν) in grid: per-doc EM inference on visible →
MAP score on held → mean-per-token LL grid → argmax (c*, ν*) → 1-D drift readout
at f-grid under {∞, ν*} → ŝ_d readout at (c*, ν*) → `_json_safe` → driver writes
`t_prior_scale.json` → zipped into the bundle → interpreted in an insight.

## Testing (TDD)

1. **Nesting** — the ν = ∞ path of the 2-D sweep reproduces
   `corpus_heldout_scale_sweep_gated`'s c-sweep within tight tol (the load-bearing
   equivalence).
2. **s_d closed form** — ŝ_d = (ν + q/c)/(ν + K′ + 2) matches a brute-force 1-D
   maximization of the joint objective in s_d at fixed η.
3. **EM convergence** — the joint (η, s_d) objective is monotone non-decreasing
   across EM sweeps; converges under the tol.
4. **Warm-start invariance** — warm-started and cold-started EM reach the same
   (η̂, ŝ_d) within tol (guards the speed optimization against changing results).
5. **numpy/RDD parity** — including docs skipped by `heldout_split`.
6. **JSON-safe** — `json.dumps(summary)` round-trips (no numpy leakage).

## Out of scope

- Promoting (c, ν) into the fit (deferred, gated on the per-topic adequacy
  check — memory step 3).
- Changing the shipped `eta_scale` export (this is a parallel flagged
  diagnostic; promotion to drive the generative scale is a later decision after
  the falsifiable checks report).
- Dirichlet-compound-multinomial / burstiness emission (retired with evidence,
  insight 0045).

## References

- Gelman et al., *Bayesian Data Analysis* 3rd ed., §17.1 (t as scale-mixture of
  normals; robust inference via the auxiliary scale).
- Kotz & Nadarajah 2004, *Multivariate t Distributions and Their Applications*.
- Wallach, Murray, Salakhutdinov, Mimno 2009, "Evaluation methods for topic
  models", ICML (held-out document-completion evaluation).
- Blei & Lafferty 2007 (logistic-normal topic model; the per-doc Laplace E-step).
- insights 0037/0038 (held-out-LL scale calibration lineage), 0044 (the
  heterogeneity reframe + bias inversion), 0045 (the dedup gate → prior-side
  verdict). Project memory: `project_concentration_scale_thread`.
