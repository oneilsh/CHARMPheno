---
id: 15
slug: stm-cancer-reference-spectral-sigma1
status: pending
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
max_iter: 300
sigma_init: 1.0
reference_topic: true
spectral_init: true
---

# STM cancer demo: reference + spectral init at the default sigma_init=1 (the decisive test)

The experiment the whole hardening arc was built for: **all three published-STM
stabilizers at once** (insight 0029) — K-1 reference (ADR 0031) + anchor-word
spectral init (ADR 0031/0032) — at the out-of-the-box sigma_init=1.0.
**Identical to exp 0012 (reference, sigma_init=1) except `spectral_init: true`.**
Same cohort, covariates, K=40, seed 42, max_iter 300. Dense spectral path
(V=3691 collects to the driver in ~18s; the large-V scalable rewrite is a
separate arc).

## Why

Three cancer-cohort runs framed the problem:

- **exp 0008** (full-K, sigma=1): collapse — 2 catch-alls + ~36 marginal clones,
  NPMI +0.055.
- **exp 0012** (reference, sigma=1): reference *escapes the collapse* but only
  ~14 of 40 topics resolve (under-resolution) and Sigma is still improper (~5e10).
- **exp 0013** (reference, sigma=5): ~28 rich, clean phenotypes — but Sigma still
  blows up (~7e9 converged) AND the **reference topic 0 goes dead** (E[β]≈0.0001):
  once the free topics' Sigma blows up, their η saturate and squeeze the pinned
  η₀≡0 baseline to ~0 mass, so the reference mechanism is defeated by the very
  blowup it was meant to be protected against.

The locked diagnosis (insight 0029): the **Sigma blowup and the sigma=1
under-resolution are the same root cause — random β init.** With random β the
only way topics differentiate is by pushing η to the simplex corners → softmax
saturation → the residual-variance M-step inflates Sigma without bound. Crisp
topics and bounded Sigma are in tension under random init; the Sigma-prior only
trades blowup for collapse. **Spectral init breaks the tension by putting the
differentiation in β from iteration 0**, so η/Sigma can stay moderate while
topics stay distinct. Reference is the identifiability prerequisite; spectral is
the piece that actually bounds Sigma. Synthetic ablation 2 confirmed:
spectral+reference at sigma=1 recovered with Sigma≈3.7, not 1e10.

This run tests that claim on real cancer data at the default sigma_init.

## Hypothesis

`spectral_init=True` + `reference_topic=True` at sigma_init=1 yields BOTH:
(a) the ~28 rich, clean phenotypes of exp 0013 (sigma=5), AND
(b) a **bounded, proper Sigma** (O(1–100), not 1e10),
with the reference topic 0 retaining real baseline mass (not dead) — all at the
out-of-the-box sigma_init.

## What to watch

- **Sigma[min … max] trace** (ADR 0030) — THE headline. Does Sigma stay O(1–100)
  rather than running to ~1e10 (0012/0013)? A bounded Sigma at sigma=1 is the
  proof spectral closes the blowup.
- **Topic resolution** — count distinct phenotypes vs 0012's ~14 and 0013's ~28.
  Success = ~28-class richness at sigma=1 (spectral kills the sigma_init knob).
- **Reference topic 0** — does it hold meaningful baseline mass (E[β] well above
  the ~0.0001 dead-floor of 0013), i.e. does a moderate Sigma revive the pinned
  baseline?
- **NPMI mean** — relative to 0012 (+0.084) and 0013 (+0.191). Read per-active-
  topic; the marginal-blend dead topics inflate the mean (insight 0026/0029).
- **Convergence** — does it reach the ELBO criterion (like 0010/0012) rather than
  running the full 300 without settling (0013, Sigma still creeping)?
- **Peak word prob** — usable phenotypes reach 0.01–0.14 (insight 0028); collapse
  clones sit at ~0.003.

Caveat to keep in mind: spectral candidate anchors are currently gated by
`min_marginal_frac=1.0` (≈ above-average co-occurrence mass, ~13% of words at
this V). If resolution is good but a few rare-but-pure phenotypes are missing,
the candidate filter (not spectral itself) is the first suspect — the ADR-0032
absolute document-frequency floor is the tracked follow-on, deliberately left out
of this run so it is not an introduced confound.

## Decision

- **Rich topics (~28) + bounded Sigma (O(1–100)) at sigma=1** → spectral closes
  BOTH remaining pathologies. Green-light to (i) flip `reference_topic` +
  `spectral_init` defaults on (validated stack), (ii) invest in the scalable
  spectral arc (ADR 0032) for large V, (iii) treat STM's Sigma as a real,
  sample-able covariance again (un-park the faithful sampler, ADR 0028-B).
- **Rich topics but Sigma still blows up** → spectral helps resolution but not the
  blowup on real data (diverges from synthetic); investigate the residual-variance
  M-step / add the Sigma-prior as a top-up cell (0016) before any default flip.
- **Still under-resolved (~14)** → spectral β seed is not surviving EM at this
  corpus scale; check the candidate filter (min_marginal_frac) first, then the
  minibatch/learning-rate schedule.

Optional follow-on **exp 0016**: same config + a moderate Sigma-prior
(`sigma_prior_scale ~20`, `sigma_prior_count` large enough to bind) as a top-up
if Sigma is bounded-but-large.

## Run

```
make exp ID=15
```

Compare head-to-head with exp 0012 (reference-only, sigma=1) and exp 0013
(reference, sigma=5). Result feeds the insight 0029 follow-up and the
`reference_topic` / `spectral_init` default decision.
