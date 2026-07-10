# Marginalized (Laplace-sample) held-out scale calibration — design

**Date:** 2026-07-10
**Branch:** stm
**Status:** Design (approved for planning)
**No LaTeX** — Unicode Greek throughout (η, θ, Σ, Γ, β, R, c).

---

## 1. Goal

Replace the MAP-plug-in held-out predictive likelihood used to calibrate the
exported generative scale `eta_scale` (c) with the **marginalized (Laplace-sample)
posterior-predictive**, so the recovered c stops drifting with the held-out
fraction and equals the data's true generative concentration under a
well-specified model. Validate the fix as a **decomposition** — confirm the drift
is a MAP artifact on synthetic data, then read whatever residual drift survives on
the real corpus as a *measurement of model misspecification*, no longer an
estimator artifact.

---

## 2. Background — two objects were conflated

The dashboard's generative simulator draws patients η ~ Normal(Γᵀx, c·R), θ =
softmax(η), and needs c at the data's true η-scale. Because the gated fit pins Σ to
a unit-diagonal correlation R for stability (ADR 0034 — every free-variance and
variance-prior path was falsified in the gated setting; insight 0033), the scale is
calibrated separately at export by a held-out predictive-LL sweep over c
(`corpus_heldout_scale_sweep_gated`, validated in principle by insight 0038).

That sweep currently conflates two different quantities:

1. **The true generative scale** — the actual variance of η across the patient
   population. One fixed number (property of the data). Prefix-independent. This is
   what generation should use.
2. **The held-out-MAP-optimal c** — the regularization strength that makes a *point
   estimate* θ̂ best predict held-out tokens given n visible tokens. Optimal
   regularization strength depends on how much data you condition on. This is what
   the sweep actually finds, and it is **prefix-dependent**.

Observed drift on the real corpus: c\* = 4.58 at holdout 0.5, c\* = 3.65 at holdout
0.95. Mechanism (textbook bias–variance): more visible tokens → likelihood
dominates → less shrinkage needed → larger c rewarded; fewer visible tokens → the
MAP θ̂ would overfit the handful it saw → the sweep compensates with a smaller c.
The sweep's c is doing double duty as prior scale *and* as a shrinkage dial for a
point estimate, and the second job contaminates the first. The true population
variance of η cannot depend on how many tokens you chose to hide, so any estimator
whose answer does is measuring something else mixed in.

### The fix, and why it works

Score the **Bayesian held-out predictive** — marginalize θ over its posterior
P(η | visible, c) instead of plugging in the MAP mode. This turns the sweep's
objective into (a per-token form of) the marginal likelihood of the held-out tokens
given the visible ones. Marginal likelihood has the consistency property the MAP
plug-in lacks: under a well-specified model it is maximized in expectation at the
true hyperparameter **regardless of how much data you condition on**, because the
posterior's width already carries the "I only saw n tokens" uncertainty that MAP was
forcing c to fake. The prefix-adaptation ("more visible tokens → tighter posterior")
moves out of the hyperparameter and into the posterior, where it belongs.

This also retires the "return a sweep of c\* / per-holdout lookup table" idea: that
is a poor-man's discretization of Bayesian updating (a smaller c at small prefixes
to mimic the extra shrinkage that proper posterior uncertainty provides for free),
and it makes model parameters depend on the query. Ship one scale.

---

## 3. The estimator (Laplace Monte-Carlo, log-of-average)

Exact marginalization over a logistic-normal posterior is intractable — the whole
pipeline is Laplace for that reason. The implementable estimator, per document, per
candidate c:

1. Infer the visible-token MAP η̂ **and its Laplace covariance ν_d** — the existing
   Fisher-scoring E-step already returns both (`_stm_doc_inference` returns
   `(eta_hat, nu_d, iters)`; the sweep currently discards ν_d as `_`). ν_d is H⁻¹ at
   the mode over the free (allowed, non-reference) topics; the reference row/col is 0
   and disallowed topics carry no variance. No H rebuild is needed.
2. Draw S samples η_s ~ Normal(η̂, ν_d) over the free topics via a Cholesky factor of
   the free-topic sub-block of ν_d, seeded deterministically (`seed + doc_index`,
   independent of c) so a c-sweep and the Spark/numpy paths see identical draws.
   Assemble each θ(η_s) with the reference at η=0 and disallowed topics at 0 (the
   same assembly as `_gated_mode_theta`).
4. Score each **held-out token** w by the **log of the average** over samples:

   ```
   score_w = n_w · log[ (1/S) · Σ_s ( Σ_k θ_k(η_s) · β_kw ) ]
   ```

   Sum over held tokens; divide the corpus total by the corpus held-token count for
   a mean-per-token LL comparable across c.

**The ordering is the entire fix.** `log( mean_s p )` (average inside the log), NOT
`mean_s( log p )` (average of the log). The second reproduces a plug-in-flavored
pathology by Jensen's inequality. A dedicated test pins this: the estimator must be
log-of-average, and a deliberately-swapped average-of-log variant must reintroduce
the holdout drift on the synthetic plant.

This is document-completion evaluation done correctly (Wallach, Murray,
Salakhutdinov, Mimno 2009, "Evaluation methods for topic models", ICML — which
warns specifically that the point-estimate plug-in is biased). Inference is
unchanged; only the scoring functional changes. Cost is one extra factor of S on the
sweep, embarrassingly parallel exactly as the current sweep is.

### Caveat — a second-order residual survives (Laplace under-dispersion)

The Laplace Gaussian understates the true posterior width (the compression lesson).
So the marginalized sweep removes the **first-order** MAP artifact but retains a
**second-order** under-dispersion, in the same drift direction. Prediction: the
drift mostly collapses, with a small residual. The synthetic plant measures its
size. If it is not negligible, self-normalized importance sampling de-biases it
cheaply at these dimensions:

```
proposal q = the Laplace Gaussian N(η̂, H⁻¹)
weight  w_s = P(η_s | visible, c) / q(η_s)   (unnormalized joint over proposal)
p(w_held) ≈ Σ_s w_s ( Σ_k θ_k(η_s) β_kw ) / Σ_s w_s
```

This is a **conditional** task, gated on the synthetic residual measurement — built
only if the measured residual is material. Decision surfaced to the human at that
point.

---

## 4. Architecture / file map

Frozen-β, export-time, single-pooled-scalar architecture unchanged (ADR 0034/0036).
Only the *scoring functional* inside the sweep changes, plus a synthetic diagnostic
and a real-data run.

- **`spark-vi/spark_vi/eval/topic/concentration_recovery.py`** — home of the
  frozen-β synthetic harness and `_predictive_loglik` / `heldout_split`. Add:
  `_marginalized_predictive_loglik` (the S-sample log-of-average scorer), a Laplace
  H-builder + Cholesky-sampler helper, and `stm_marginalized_heldout_ll` /
  `sweep_heldout_marginalized` mirroring the existing plug-in functions. The
  existing plug-in path stays intact (it is the thing the decomposition compares
  against).
- **`spark-vi/spark_vi/mllib/topic/stm.py`** — `corpus_heldout_scale_sweep_gated`
  (:868) and `..._rdd` (:972): add a `marginalize: bool` (and `n_samples: int`)
  parameter that routes scoring through the marginalized scorer while reusing the
  identical per-doc split, allowed-set caching, and inference (now capturing ν_d
  instead of discarding it). The reducer `smooth_scale_log_quadratic` (~:1090, local
  log-quadratic interpolation of the flat LL shelf to a smoothed c\*) is unchanged.
- **`docs/experiments/0046-marginalized-scale-decomposition/`** — synthetic
  decomposition results (MAP-vs-marginalized × holdout fraction) + the real-corpus
  cancer before/after run.
- **Drivers** (`analysis/local/build_dashboard.py`,
  `analysis/cloud/build_dashboard_cloud.py`): flip the export sweep to
  `marginalize=True`; ship the single argmax c\* as `eta_scale`; record method +
  per-holdout c\* (now expected flat) in `eta_scale_diagnostic`.
- **Sampler** (`dashboard/src/lib/conditioning/recordPosterior.ts`): NO structural
  change — it already samples η from the Laplace posterior. It consumes the single
  recalibrated `eta_scale`. IS de-bias applied here only if §3's conditional task
  ran.

---

## 5. Validation — a decomposition, not a before/after

**Task order (Fable's run order):**

1. **Synthetic plant, 3 holdout fractions (well-specified).** Reuse the
   `plant_corpus` harness. Plant at a KNOWN generative scale; sweep c under BOTH the
   MAP plug-in and the marginalized estimator at holdout ∈ {0.5, 0.7, 0.95} (and a
   realistic regime: K=60, V≈5000, doc_len≈44). **Acceptance:** the MAP sweep's c\*
   drifts across holdout fractions; the marginalized sweep's c\* is (a) flat across
   holdout fractions and (b) centered on the planted scale within tolerance. This one
   figure isolates and confirms the MAP artifact and the fix. Also **measure the
   residual drift** of the marginalized estimator — its size decides the conditional
   IS task.
2. **Log-of-average ordering test.** A unit test asserting the estimator is
   log-of-average and that an average-of-log variant reintroduces drift.
3. **Conditional — importance-sampling de-bias.** Only if the §5.1 residual is
   material: add self-normalized IS (§3), re-run §5.1, show the residual shrinks.
   Decision surfaced to the human before building.
4. **Real corpus (population_cancer), separate model for before/after.** Re-run the
   export sweep (MAP and marginalized) on the cancer fit. Report both c\* curves vs
   holdout. **Reading:** the marginalized c\* should be flatter than MAP; whatever
   residual drift remains is now a **measurement of misspecification** (most likely
   concentration heterogeneity — a corpus with a genuine spread of true
   concentrations weights differently at different holdout fractions), *not* an
   estimator artifact. Record the shipped c\* = the marginalized argmax at the
   canonical holdout (or a documented aggregate across holdouts if the residual is
   negligible, since it is then holdout-independent by construction).
5. **Insight write-up** documenting the decomposition result and the shipped number.

All synthetic work runs in-process (numpy, no Spark); the real-corpus run is a
cluster export (`make build-dashboard-exp`).

---

## 6. Out of scope / rejected

- **Fit-time free variance or a Σ-shrinkage prior ("exp 0018") to recover the scale
  directly.** Rejected — falsified path (IW anchor made the gated runaway worse,
  exps 0022/0024 → 6.08e8; free diagonal blew up, exp 0032; insight 0033: a
  document-scarce minority topic makes fit-time free variance fundamentally
  unidentifiable and outruns any usable prior strength). Pin-and-calibrate is the
  conclusion, not a stopgap.
- **Per-topic export scale.** Backed off (user). Single pooled scalar.
- **Per-holdout c\* lookup table shipped to the dashboard.** Retired by construction
  — one prefix-independent scale; the posterior does the prefix-adaptation.
- **Restructuring the sampler.** It already posterior-samples; it only consumes the
  new c\* (+ optional IS).

---

## 7. Global constraints (bind every task)

- No LaTeX in any prose/UI/docstring; Unicode Greek only.
- Cite literature for any method/default/constant (Wallach et al. 2009 for
  document-completion; Hill 1973 / Jost 2006 for concentration metrics; Blei &
  Lafferty 2007 for the logistic-normal + Laplace posterior).
- TDD: failing test first, watched to fail, minimal code to pass.
- Numpy/Spark parity: per-doc split and per-doc sample seeds are `seed + index`,
  independent of c, so the sweep is a controlled comparison and the RDD path is
  byte-for-byte the numpy oracle.
- Inference vs scoring split preserved: inference uses `expElogbeta` (exp-digamma of
  λ); scoring uses `beta_prob` = E[β] (λ-normalized). Do not conflate.
- The estimator is **log-of-average**, per held token. Guard with a test.
- Hash IDs in any row-level log output; aggregates/probabilities may be raw.
- Existing plug-in sweep functions stay intact (the decomposition baseline).

---

## 8. References

- Roberts, Stewart & Airoldi (2016) — Structural Topic Model.
- Blei & Lafferty (2007), *Annals of Applied Statistics* 1(1) — correlated topic
  model / logistic-normal prior + Laplace posterior.
- Wallach, Murray, Salakhutdinov & Mimno (2009), ICML — "Evaluation methods for
  topic models" (document completion; the point-estimate plug-in is biased).
- Hill (1973), *Ecology* 54(2); Jost (2006), *Oikos* 113(2) — concentration /
  diversity number (top_mass, effective #topics).
- Project: insights 0033 (gated runaway = document scarcity), 0034 (unit-diagonal),
  0036 (export decoupling), 0037 (fit-anchored scales under-concentrate), 0038
  (held-out-LL recovers true concentration — validates the sweep in principle);
  ADRs 0034 (unit-diagonal correlation), 0036 (record-completion + eta_scale
  addendum); the open-problem brief
  `docs/superpowers/specs/2026-07-03-stm-generative-variance-scale-open-problem.md`.
