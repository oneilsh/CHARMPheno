# 0029 — Our online STM's σ_init-selected collapse↔Σ-blowup is a missing-stabilizer artifact, not a property of STM; published STM avoids it by construction (spectral init + reference-topic identifiability + Σ shrinkage)

**Date:** 2026-06-26
**Topic:** stm | priors | svi | initialization | diagnostics | phenotyping | prior-art
**Status:** Confirmed (+ literature-confirmed)

Re-fitting non-gated STM on the cancer cohort for a covariate demo surfaced a
sharp, two-sided sensitivity to `sigma_init` (the initial diagonal of the
logistic-normal prior covariance Σ). Too small → the model collapses to the
corpus marginal; large enough → it produces crisp phenotypes but drives Σ to an
absurd ~10^10. The crisp regime is real and usable, but the knife-edge and the
absurd Σ are **artifacts of our simplified online STM** (random init, full-K η
with no reference topic, `sigma_ridge=1e-6`, stochastic minibatch EM) — *not*
properties of STM as published. The original CTM/STM avoid this by construction.

## The sweep (cancer cohort, K=40, `~ C(sex) + age`, seed 42)

| Run | `sigma_init` | Outcome | NPMI mean | Σ diagonal (final) | Converged |
|---|---|---|---|---|---|
| [0008](../experiments/0008-stm-cancer-demo.md) | 1.0 (default) | **collapse**: 2 catch-alls hold ~77% mass, ~36 topics clone the corpus marginal at E[β]≈0.0057 | +0.055 | pinned at ≈1.0 | iter 26 (degenerate) |
| [0009](../experiments/0009-stm-cancer-sigma5.md) | 5.0 | **escape**: ~28 distinct phenotypes (breast/prostate/thyroid/melanoma/lung/AF/HF/CKD) | +0.204 | 7.6e7 → 3.9e10 | hit max_iter (100) |
| [0010](../experiments/0010-stm-cancer-sigma5-300iter.md) | 5.0, 300 iter | escape, **converged**; even richer (epilepsy, HCV→cirrhosis→HCC, lymphoma, pancreatic, Crohn's) | +0.216 | ≈2.2e10 (stable, not inflating) | iter 136 |

(`sigma_init=10`, [0011](../experiments/0011-stm-cancer-sigma10.md), is queued to
confirm the escape regime is init-independent; predicted to also stall at ~10^10.)

## Two basins, and σ_init only picks which one

θ_d = softmax(η_d), η_d ~ N(Γᵀx_d, Σ). The optimizer has two attractors:

- **Low-Σ collapse (symmetric basin).** With Σ≈1, prior SD(η)≈1, so η stays within
  ±1 of its mean and softmax(η) is forced near-uniform. Every document spreads θ
  across all K topics → every topic accumulates the corpus marginal → no topic
  specializes. This is self-reinforcing: undifferentiated β keeps η near its mean,
  the residual-variance update keeps Σ≈init, the prior stays tight. The fit
  "converges" fast (iter 26) into this degenerate fixed point. **This is the same
  collapse as insight [0028](0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md)**,
  here reproduced on a clean single cohort with no minority-imbalance pressure —
  strengthening 0028's claim that it is the prior family, not the cohort.

- **High-Σ blowup (boundary run).** With a larger init, η escapes the symmetric
  saddle and starts to peak θ. But nothing then bounds Σ: the residual-variance
  M-step `Σ_k = mean_d[(η_dk − Γᵀx)² + ν_dk]` feeds back — bigger Σ → weaker prior
  penalty (η²/Σ) → MAP η for a document's dominant topic grows almost unchecked →
  bigger residuals → bigger Σ. The loop halts only when **softmax saturates** (θ
  hits the simplex corners, so larger η no longer changes the likelihood) and the
  ELBO flatlines. That stall point is ~10^10 — i.e. SD(η)≈10^5, which is not a
  meaningful log-odds variance. **The crisp topics are a byproduct of near-one-hot
  θ (clean β sufficient statistics), not evidence that Σ≈10^10 is a real
  covariance.** 0010 converging with Σ slightly *below* 0009 confirms it is a
  bounded boundary regime, not a true statistical fixed point.

The scale arithmetic explains why `sigma_init=1` is the actual "magic number":
to load a document on a few of K=40 topics you need η-spread of order log(0.5/0.01)
≈ 4, so prior SD √Σ must be a few → Σ of order 10–50. The default 1.0 is an order
of magnitude too small and **structurally guarantees collapse**; 5 is merely the
right order of magnitude, not a tuned operating point.

## Root cause: three stabilizers we dropped

Reading [`spark-vi/spark_vi/models/topic/stm.py`](../../spark-vi/spark_vi/models/topic/stm.py):
full-K η with `softmax` (no reference topic), diagonal Σ via residual variance,
`sigma_ridge=1e-6` (effectively no regularization), random init, online stochastic
EM. The published logistic-normal topic models (CTM: Blei & Lafferty 2007; STM:
Roberts, Stewart, Airoldi / the `stm` R package) avoid both basins via three
guards we lack:

1. **Reference-topic identifiability.** CTM/STM fix the last topic's η to 0 and
   work in **K−1 dimensions**, removing the softmax translation degeneracy
   (softmax(η) = softmax(η + c·1)). Our full-K η leaves a likelihood-flat all-ones
   direction controlled only by the weak prior, feeding the drift.
2. **Σ shrinkage.** `stm`'s `sigma.prior` ∈ [0,1] shrinks the covariance toward a
   diagonal (convex combination of diagonalized covariance and MLE). It defaults to
   0 and is aimed at over-correlation rather than blowup — so this is the *minor*
   guard — but the mechanism for taming Σ exists; our `sigma_ridge=1e-6` is none.
3. **Spectral (anchor-word) initialization — the decisive one.** `stm`'s default
   `init.type = "Spectral"` is a **deterministic** method-of-moments init (Arora et
   al. 2014) that lands β near a coherent, well-separated solution before any EM.
   Because the start is good, η residuals are sane from iteration 0, Σ is estimated
   at a reasonable scale, and **there is no random-init basin to fall into — so the
   σ_init knob does not exist**. (`stm` also runs batch, not stochastic minibatch,
   EM.) Spectral init: build the V×V word co-occurrence matrix Q, row-normalize so
   each word's row is a convex combination of anchor-word rows, find the anchors as
   the convex-hull vertices (greedy farthest-point after an SVD/projection), then
   recover P(topic|word) via convex weights and Bayes-flip to β.

`stm` does not random-initialize, so it never exhibits our collapse-or-explode
behavior. We rediscovered why all three guards are there by removing them.

## Implications

1. **For the demo:** [0010](../experiments/0010-stm-cancer-sigma5-300iter.md) is
   usable — the topics are crisp and the sex→breast/prostate covariate story holds.
   But **do not present the exported Σ as a real covariance**, and do not feed it to
   a faithful logistic-normal sampler (the parked
   [ADR 0028-B](../decisions/0028-dashboard-conditioned-dirichlet-prior.md) sampler
   would draw garbage at Σ≈10^10). The dashboard's Dirichlet-approximation
   conditioning — which uses Γ's mean-shift and ignores Σ's scale — is, somewhat by
   luck, the right call for a degenerate Σ.
2. **For the method (post-demo):** the principled fix is **spectral initialization +
   the K−1 reference-topic parameterization** (and a real `sigma_ridge`), not
   hunting σ_init. This warrants an ADR/plan to harden spark-vi's STM. Bumping
   `sigma_ridge` is a cheap empirical probe (does Σ tame to O(10–100) with topics
   still crisp?) but a band-aid relative to spectral init.
3. **Relationship to 0028:** this is additional evidence in the same direction. The
   Dirichlet (LDA/PLDA) needs none of this — α<1 bakes in document sparsity and a
   benign random init. The logistic-normal only reaches the crisp regime via a
   knife-edge init and an under-regularized Σ. The σ_init fragility is a
   logistic-normal *tax*, reinforcing "use Dirichlet for content discovery."

## Ablation (controlled synthetic sweep)

Harness: `synthetic_ehr_corpus(K_rare=8, V=300, D=1500, doc_len=30, bg_frac=0.7)`.
Four configurations × sigma_init ∈ {1, 5, 20}; each cell shows (planted_recovery/8, Σ_max).
200 minibatch-EM iterations, batch=150, seed=42.

| Config | sigma_init=1 | sigma_init=5 | sigma_init=20 |
|---|---|---|---|
| random-init (baseline) | (0/8, 6.6e+00) | (0/8, 6.9e+00) | (0/8, 7.9e+00) |
| +Σ-prior (scale=2, count=500) | (0/8, 4.2e+00) | (0/8, 4.5e+00) | (0/8, 4.9e+00) |
| +spectral | (0/8, 3.0e+00) | (2/8, 3.5e+00) | (2/8, 4.7e+00) |
| +spectral+Σ-prior | (0/8, 2.7e+00) | (2/8, 2.9e+00) | (2/8, 3.5e+00) |

Observations:
- Random init (with or without Σ-prior) achieves 0/8 recovery at all sigma_init
  values; the bg_frac=0.7 corpus is genuinely noisy (70% background tokens) so
  random init never escapes the collapsed basin at this batch scale.
- Spectral init breaks from 0 to 2/8 recovery at sigma_init=5 and 20,
  confirming that the deterministic β seed provides the necessary lift. The 0/8
  at sigma_init=1 is consistent with the collapse story: with Σ initialized to
  1, prior SD(η)=1 keeps the η distribution near-uniform even with a good β start.
  There is **no non-monotonic 1-bad/5-blowup/20-clean pattern** at this corpus
  scale; recovery is 0 at sigma=1 and 2/8 at sigma=5 and 20 for spectral rows.
- Σ-prior alone does not improve recovery but tames Σ_max from ~7 to ~5 across
  random-init rows, and from ~3-5 to ~2.6-3.5 for spectral rows.
- The best configuration at this scale is **+spectral+Σ-prior**: 2/8 recovery
  (init-stable at sigma_init=5 and 20) with the lowest Σ_max of any config.

Gated minority block: `synthetic_gated_corpus` with groups (maj, rare), rare
thinned to ~10% of the majority count (401 maj / 25 rare docs; bg_frac=0.5,
doc_len=40). Block-aware spectral init correctly recovers the rare group's
planted foreground (foreground_recovers_group("rare") = True, Σ_max=1.3e+01);
random-init does not (False, Σ_max=1.0e+01). This is the load-bearing result:
the within-group Q + background deflation seam lets the spectral init surface the
rare foreground phenotype from a handful of minority documents, before any EM,
precisely the scenario where random init is dominated by the majority signal.

## Caveats

- The controlled ablation (above) uses a synthetic corpus; the cancer-cohort
  sweep (0008–0010) confirms the collapse↔blowup pattern on real data. The two
  together characterize the failure mode and its fix.
- Recovery of only 2/8 at the best non-gated config reflects the difficulty of
  a heavily background-dominated corpus (bg_frac=0.7, 18-word fg vocabulary per
  topic, minibatch scale). Spectral init is a necessary but not sufficient fix
  for this extreme noise level; a larger document count or lower bg_frac would
  increase the count further.
- The three root causes (spectral init, K-1 reference-topic, Σ shrinkage) are
  ablated here for init and Σ shrinkage; the K-1 reference-topic
  parameterization remains unimplemented and its independent contribution is
  from the literature, not a controlled run in this stack.
- `sigma.prior`'s default of 0 in the published stm R package means even the
  reference implementation leans on init + the K-1 parameterization for
  stability — so Σ-prior alone (without spectral init) does not improve recovery.

## Setting context

Experiments 0008–0011: non-gated STM on the `cancer` cohort (`first_cancer_year`,
prior_obs_days=0, condition_era, person_mod=4, ~10.8k patient docs), K=40,
`covariate_formula: ~ C(sex) + age`, seed 42, online VI in Spark. The load-bearing
diagnostics are the per-iter `Σ[min…max]` trace (ADR 0030 persists the 2-D Σ/Γ
traces), per-topic peak β / E[β] / Σλ spread, and NPMI (Röder et al. 2015) read as
a relative signal across the sweep, not an absolute. Literature checked via
targeted search: `stm` reference manual + source (`sigma.prior`, `init.type`),
Roberts/Stewart/Tingley JSS 2019, Blei & Lafferty 2007 (CTM K−1 identifiability),
Arora et al. 2014 (spectral/anchor-word init).
