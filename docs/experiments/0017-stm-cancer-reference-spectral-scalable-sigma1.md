---
id: 17
slug: stm-cancer-reference-spectral-scalable-sigma1
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
spectral_method: scalable
---

# STM cancer demo: SCALABLE (random-projection) spectral init — the equivalence re-run

The cluster equivalence test for the scalable spectral-init path (ADR 0032).
**Identical to exp 0015 in every way except `spectral_method: scalable`** — same
cohort, covariates, K=40, seed 42, max_iter 300, sigma_init=1.0, reference +
spectral on. Exp 0015 used the DENSE anchor-word init (materializes the V×V
co-occurrence matrix on the driver); this run uses the random-projection sketch
that never forms V×V, so the only intended difference between the two fits is the
Johnson-Lindenstrauss approximation error of the projected co-occurrence.

## Why

Exp 0015 (insight 0030) was the decisive science win: reference + DENSE spectral
init at the default sigma_init=1 brought Sigma from ~1e10 down to 7.56, resolved
all 40 topics with no dead floor, and revived the reference topic. But the dense
path collects a V×V co-occurrence matrix to the driver — fine at the cancer scale
(V=3691, ~109MB) but ~80GB at the V=100k vocabularies spark-vi targets. ADR 0032
decided to scale via random projection of the co-occurrence rows (Arora et al.
2013; Mimno random-projections-for-anchors): project each word's co-occurrence
row to a fixed d≈1000 dimensions (min(V, max(K, 1000)), the Mimno/Arora
convention — three independent reference implementations hardcode 1000), run the
greedy anchor selection and NNLS recovery in projected space, and accumulate the
whole sketch in ONE distributed pass (`scalable_spectral_init_beta`).

The synthetic equivalence suite already established the approximation holds where
it matters: at a representative V=3000 the planted rare arm is recovered down to
d=201 (eps=0.2), and at the adopted d≈1000 the scalable β matches the dense β on
planted-recovery. This run is the **real-data** confirmation: does the projected
init reproduce exp 0015's bounded Sigma and full topic resolution on the actual
cancer corpus, where the dense path proved the science?

At V=3691 the projection dim defaults to min(3691, max(40, 1000)) = 1000, a ~3.7x
row compression — modest here (the point of this cohort is correctness parity,
not memory savings), but the same code path is what runs in bounded memory at
V=100k. The genuine large-V memory validation needs a higher-V cohort and is
tracked as a separate follow-on (out of scope for this equivalence run).

## Hypothesis

`spectral_method: scalable` at sigma_init=1, reference on, reproduces exp 0015's
dense result within JL approximation tolerance:
(a) a **bounded, proper Sigma** O(1–10) (vs 0015's 7.56), NOT the ~1e10 of the
    random-init runs,
(b) all (or nearly all) 40 topics resolved with no marginal-clone dead floor,
(c) the reference topic 0 retaining real baseline mass (not the ~0.0001
    dead-floor of 0013),
(d) clean phenotype separation comparable to 0015 (breast / prostate / thyroid /
    melanoma / kidney / bladder / colon / pancreatic / ovarian / lymphoma).

## What to watch

- **Sigma[min … max] trace** (ADR 0030) — THE headline, same as 0015. Does the
  projected init still bound Sigma to O(1–10) at sigma=1? A bounded Sigma here is
  the proof that the JL-approximated β seed carries the same EM-stabilizing
  signal as the exact dense seed.
- **Topic resolution vs 0015** — count distinct phenotypes; success = ~40
  resolved, matching 0015 (min Sigma-lambda well above the dead-floor). A handful
  fewer is acceptable JL error; a collapse back to ~14 (0012) would mean the
  projected seed is not surviving EM.
- **Reference topic 0** — E[β] well above the ~0.0001 dead-floor (0015 revived it
  to 0.0076).
- **NPMI mean** — relative to 0015 (+0.173). Read per-active-topic (the
  marginal-blend dead topics inflate the mean, insight 0026/0029); expect it in
  the same band as 0015, not 0013's NPMI-biased +0.191.
- **Phenotype quality** — spot-check the same clean cancers 0015 separated; the
  projected NNLS recovery should land the same top-word sets.
- **Convergence + ELBO** — comparable to 0015 (ELBO ≈ −1.10e6); a much worse ELBO
  would indicate the approximate seed degraded the fixed point.

Note: the scalable path uses the ADR-0032 absolute document-frequency floor
(`spectral_min_doc_freq`, default 5) for anchor candidates, NOT the dense path's
`min_marginal_frac=1.0`. This is by design (rare-phenotype-friendly), but it is a
second difference from 0015 beyond the projection itself — if topic resolution
differs, the candidate floor is a co-suspect alongside the JL error. The default
floor of 5 is permissive at this corpus size.

## Decision

- **Bounded Sigma O(1–10) + ~40 resolved topics matching 0015** → the scalable
  random-projection init transfers on real data; mark ADR 0032 Accepted +
  Implemented and treat the scalable path as the validated large-V option (dense
  remains the exact default for small V). Green-light a higher-V cohort for the
  genuine memory-scaling validation.
- **Sigma bounded but topics noticeably degraded (well below ~40, or muddier
  separation)** → the JL approximation at d≈1000 loses real phenotype signal on
  this corpus; raise d (or revisit the candidate floor) before relying on the
  scalable path, and record the d/quality tradeoff as an insight.
- **Sigma blows up (~1e10) like the random-init runs** → the projected seed is
  NOT carrying the dense seed's stabilizing structure through EM; treat as a
  correctness regression in the scalable path (re-check the projected
  co-occurrence / anchor / recovery against the dense oracle) before any further
  investment.

## Run

```
make exp ID=17
```

Compare head-to-head with exp 0015 (dense spectral, otherwise identical). The
delta between 0015 and 0017 IS the random-projection approximation error on real
data — the result that finalizes ADR 0032's status and the scalable-arc go/no-go.
