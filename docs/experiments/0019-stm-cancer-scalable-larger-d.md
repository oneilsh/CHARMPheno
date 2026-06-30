---
id: 19
slug: stm-cancer-scalable-larger-d
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
spectral_d: 2000
---

# STM cancer demo: scalable spectral at d=2000 — JL approximation pressure test

The projection-dimension isolation experiment for the scalable spectral path.
**Identical to exp 0017 in every way except `spectral_d: 2000`** — same cohort,
covariates, K=40, seed 42, max_iter 300, σ_init=1.0, reference + spectral on,
no Σ-prior. Exp 0017 used d=1000 (the default, min(V, max(K, 1000)) = 1000 at
V=3691, a ~3.7x compression); this run halves the compression to d=2000 (2000/3691
= 0.54 vs 0017's 0.27), moving the projected seed closer to the exact dense fixed
point of exp 0015.

## Why

Exp 0017 (insight [0031](../insights/0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md))
produced a split result: all 40 cancer phenotypes were recovered (topic quality
on par with dense exp 0015), but one dominant topic — essential hypertension /
pure hypercholesterolemia (topic 2) — ran Σ to ~8.3e5 while the other 39 held
at O(1–20). Dense 0015 at the IDENTICAL configuration did NOT blow up (Σ_max
7.56). The only material difference between 0015 and 0017 is the Johnson-
Lindenstrauss approximation: d=V (dense, exact) vs d=1000 (scalable, ~3.7x
compression).

This experiment isolates whether the Σ blowup is a **JL approximation artifact
controllable by d**, or something else (e.g. the absolute document-frequency floor
in the scalable path vs 0015's `min_marginal_frac=1.0`). The hypothesis is that as
d rises toward V, the scalable β seed approaches the dense fixed point and the
runaway topic's blowup weakens or disappears.

## The JL argument

The Johnson-Lindenstrauss lemma guarantees that a random projection to d dimensions
preserves all pairwise distances in the word-row set within multiplicative factor
(1 ± ε) with high probability, where ε ~ sqrt(log(V) / d). For V=3691:

- d=1000: ε ~ sqrt(8.2 / 1000) ~ 0.09 (9% distortion)
- d=2000: ε ~ sqrt(8.2 / 2000) ~ 0.064 (6.4% distortion)
- d=V=3691: ε ~ 0 (exact, dense)

The greedy anchor-selection step is a farthest-point search on these rows; a 9%
distance distortion can shift the anchor assignment for a topic that is geometrically
close to another in the projected space. For the MOST DOMINANT topic (topic 2), which
is also the most-peaked cluster, a small geometric shift in the projected β seed may
be enough to leave its β insufficiently distinct from the next-most-dominant cluster,
driving η-saturation and the Σ blowup. Halving the compression (d=2000) should
sharpen the geometry and move topic 2's anchor assignment closer to the dense result.

## Hypothesis

At d=2000:
(a) **Σ_max drops from ~1.08e6 toward O(10–100)** — specifically, topic 2's Σ
    (the sole runaway in 0017) falls substantially, demonstrating the blowup is
    JL-approximation-induced and controllable by d.
(b) **The same 40 cancer phenotypes are recovered** with NPMI on par with exps
    0015 (+0.173) and 0017 (+0.166) — topic quality was already fine at d=1000
    and should not regress at d=2000.
(c) If Σ_max reaches O(1–10), the scalable path at d=2000 is equivalent to dense
    end-to-end, and the gap in d between stability (dense) and blowup (d=1000)
    is bounded to the interval [1000, 2000), quantifying the safe production d.

## What to watch

- **Σ[min … max] trace and per-topic Σ vector** — the headline. Does Σ_2 (the
  runaway at ~8.3e5 in 0017) drop toward O(10)? The ideal result is all 40 topics
  at O(1–20) matching dense 0015's per-topic profile.
- **Σ_max vs 0017 (1.08e6) and 0015 (7.56)** — the critical comparison. Even a
  partial drop (e.g. from 1.08e6 to 1e3) confirms the d sensitivity and points
  toward a safe d between 2000 and V.
- **Topic quality (NPMI, phenotype spot-check)** — should be unchanged from 0017
  (already fine at d=1000).
- **Convergence iter vs 0017 (iter 88)** — a more accurate seed should produce
  similar or faster convergence.
- **Per-topic Σ for all 40 topics** — confirm no NEW runaways emerge (i.e., that
  increasing d does not accidentally shift a different topic into a marginal
  geometry and introduce a new blowup while fixing topic 2).

## Decision tree

- **(a) Σ_max drops toward O(10) with topics preserved** (the confidence-restoring
  result): the blowup is JL-approximation-induced and controllable by d. Document
  the d/Σ-stability tradeoff; pick a safe production d (likely 2000 at this V, or
  express as a V-fraction). The scalable path at d=2000 is the large-V equivalent
  of dense at d=V; update ADR 0032's scalable default accordingly.
- **(b) Σ_max still ~1e6 (blowup persists at d=2000)**: the blowup is NOT mainly
  a JL distance-distortion artifact. The candidate-set difference (absolute
  `min_doc_freq=5` in scalable vs `min_marginal_frac=1.0` in dense) is the
  next suspect — a permissive floor may allow a near-pure spurious anchor for
  topic 2 that the tighter `min_marginal_frac` floor excluded. Investigate the
  anchor selection for topic 2 specifically across 0015/0017/0019 configs before
  adjusting d further.

## Run

```
make exp ID=19
```

Compare head-to-head with exp 0017 (d=1000, no prior) and exp 0015 (dense,
d=V, no prior). The delta 0017 → 0019 IS the projection-dimension effect on Σ;
the delta 0015 → 0019 measures whether d=2000 closes the gap to the dense fixed
point. Cross-link: insight [0031](../insights/0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md),
exp 0017, exp [0018](0018-stm-cancer-scalable-sigma-prior.md).
