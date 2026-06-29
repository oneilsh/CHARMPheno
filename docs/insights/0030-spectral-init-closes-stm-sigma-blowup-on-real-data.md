# 0030 — On real cancer data, spectral init brings STM's Σ from ~10^10 to ~7.6 and resolves all K topics at the default σ_init=1; the K−1 reference alone does not — the Σ blowup defeats the reference topic itself

**Date:** 2026-06-29
**Topic:** stm | priors | svi | initialization | diagnostics | phenotyping
**Status:** Confirmed (cancer cohort, exps 0012/0013 vs 0015)

Insight [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)
established (synthetically + by theory) that our online STM's σ_init knife-edge —
collapse at small σ_init, Σ→~10^10 blowup at large — is an artifact of three
dropped stabilizers (K−1 reference identifiability, Σ shrinkage, spectral init),
and predicted spectral init as the decisive fix. This insight records the
**real-data confirmation on the cancer cohort**, and one finding the synthetic
harness could not surface: on real data the **K−1 reference parameterization
alone does not tame Σ — the Σ blowup actively defeats the reference topic.**

## Setup

Cancer cohort (`first_cancer_year`, prior_obs_days 0, person_mod 4,
condition_era, ~10.8k patient docs, V=3691), K=40, `~ C(sex) + age`, seed 42,
max_iter 300. All opt-in stabilizers now wired through the cluster fit path
(`StreamingSTM` → `OnlineSTM`). Load-bearing diagnostics: the per-iter
Σ[min…max] trace (ADR 0030), per-topic E[β] / Σλ / peak β, and NPMI
(Röder et al. 2015) read relatively.

## The progression

| exp | config | Σ max (final) | active topics | reference topic 0 E[β] | NPMI mean | ELBO |
|---|---|---|---|---|---|---|
| 0008 | full-K, σ=1 | ≈1 (collapse) | ~2 catch-alls + ~36 clones | n/a | +0.055 | — |
| 0012 | reference, σ=1 | ~5e10 | ~14 | (alive, under-resolved) | +0.084 | — |
| 0013 | reference, σ=5 | 7.16e9 | ~28 | **0.0001 (dead)** | +0.191 | −1.32e6 |
| 0010 | full-K, σ=5 | 2.2e10 | ~28 | n/a | +0.216 | — |
| **0015** | **reference + spectral, σ=1** | **7.56** | **40 (no dead floor)** | **0.0076 (alive)** | +0.173 | **−1.10e6** |

## Finding 1 — reference alone escapes collapse but NOT the Σ blowup, and the blowup then defeats the reference topic

The synthetic ablation (0029 Ablation 2) showed reference keeping Σ bounded
(Σ_max ≈ σ_init, no runaway). **That did not transfer to the real cancer
corpus.** Reference at σ=1 (0012) escapes the 0008 collapse — but only ~14 of 40
topics resolve and Σ still runs to ~5e10; reference at σ=5 (0013) recovers ~28
rich phenotypes but Σ still blows to 7.16e9. So on real V, reference fixes the
*level/collapse* degeneracy but is **necessary-not-sufficient** for the blowup —
exactly the synthetic "+reference" row (0/8), now confirmed on data.

The non-obvious part is the mechanism by which it fails. In 0013 the **reference
topic 0 went dead** (E[β]=0.0001, peak β=0.000), and the per-topic Σ trace shows
why: topic 0 sits at its init (Σ₀=5, correct — η₀≡0 ⟹ zero residual) while every
*free* topic is at ~10⁹. Once the free topics' Σ blows up, their η saturate to
the simplex corners, and softmax([0, huge, huge, …]) crushes the pinned η₀≡0
baseline to ≈0 mass. A de-facto background topic (0013 topic 15, E[β]=0.352)
absorbs the role instead. **The Σ blowup defeats the very reference mechanism that
was supposed to prevent it** — the identifiability pin only holds if something
else keeps η (hence Σ) moderate. That something is spectral init.

## Finding 2 — reference + spectral at the DEFAULT σ_init=1 closes both pathologies

exp 0015 (add spectral init to 0012's config — same σ=1) is the decisive cell:

- **Σ bounded and proper: max 7.56** (per-topic trace 1–3.4), with Γ moderate
  (|Γ| max 5.22, mean 0.54). Nine orders of magnitude below 0012/0013. Σ is a
  real, estimated, sample-able covariance again — not a softmax-saturation
  byproduct. This is the synthetic prediction (spectral+reference σ=1 → Σ≈3.7)
  transferring to real data, where reference-alone did not.
- **All 40 topics resolve** — minimum Σλ 6.11e3 (not the 93.7 dead-floor of
  0013); **no marginal clones**. Spectral didn't merely match 0013's ~28 at
  σ=5, it used the full K at the default σ=1. The σ_init knob is gone.
- **The reference topic revived** — topic 0: E[β] 0.0001→0.0076, peak β
  0.000→0.048, real content. With Σ moderate the free-topic η no longer saturate,
  so the pinned baseline keeps a real (if small) share. (This also de-prioritizes
  the idea of *forcing* the reference to carry mass — moderate Σ revives it on its
  own.)
- **Higher ELBO** (−1.10e6 vs 0013's −1.32e6): a strictly better fit, not a
  quality trade for the bounded Σ.

Mechanistically (per 0029): random β forces topics to differentiate only by
pushing η to extremes → saturation → the residual-variance M-step
Σ_k = mean_d[(η_dk − Γᵀx)² + ν_dk] inflates without bound. Spectral init puts the
differentiation in **β** from iteration 0 (anchor words → distinct topic-word
rows), so η/Σ stay moderate while topics stay distinct. This is how the published
`stm` keeps Σ proper with its default `sigma.prior=0`. **Spectral is the fix;
reference is the identifiability prerequisite that lets the good β seed survive.**

## Caveat — NPMI mean is slightly LOWER at 0015 but this is not a regression

0015's NPMI mean (+0.173) sits just below 0013's (+0.191), which looks backwards
given 0015 is the better fit. It is a property of the metric, not the model
(cf. insight [0026](0026-stm-prevalence-gives-prevalence-not-content-fidelity.md)):
in 0013 the ~12 *dead* topics were pure corpus-marginal blends, and
marginal-blend top-word sets score *high* NPMI (0013 topic 16: NPMI +0.276 at
peak β=0.000) — inflating its mean. In 0015 every topic carries specialized
content, so mass spreads across 40 genuine phenotypes whose top words are
individually rarer (lower per-topic NPMI) but more specific. Min NPMI +0.072,
0/40 unrated. Read peak β / Σλ and the ELBO, not the NPMI mean, to compare these
regimes — NPMI rewards common-word co-occurrence, which finer topics have less of.

## Implications

1. **`reference_topic` + `spectral_init` are the validated default stack** for
   STM on this data — the σ_init knob no longer exists once both are on.
2. **STM's Σ is usable again.** At O(1–10) it can feed a faithful logistic-normal
   sampler; the dashboard's Dirichlet-approximation conditioning (parked
   [ADR 0028-B](../decisions/0028-dashboard-conditioned-dirichlet-prior.md)) is no
   longer forced by a degenerate Σ.
3. **K=40 may now be under-sized.** With all 40 slots crisply used (no graceful
   unused capacity, cf. insight
   [0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md)),
   a K=60/80 spectral run would show whether more structure remains to recover.
4. **The dense spectral path proved the science**; the large-V scalable rewrite
   (distributed co-occurrence + random projection, ADR 0032) is now worth the
   investment.

## Open / pending

- **High-σ init robustness (exp 0016, queued):** reference+spectral at σ_init=20.
  Prediction: Σ comes *down* to the same ~O(1–10) fixed point as 0015's 7.56,
  demonstrating the M-step is now a proper init-independent estimator on both
  sides of the old knife-edge (the spectral analog of the now-moot exp 0014).
- exp 0015 hit max_iter (300) rather than early-converging; Σ is stable but the
  ELBO was still slowly creeping. Not a concern at this Σ scale, but a longer run
  or convergence-tol check would confirm the fixed point.

## Relationship to prior insights

Confirms and closes the loop on [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)
(the artifact + its predicted fix) with real-data evidence, and adds the
reference-defeated-by-blowup mechanism. Consistent with
[0028](0028-dirichlet-vs-logistic-normal-rare-phenotype-recovery-gated-lda-is-plda.md):
the logistic-normal needs this whole stabilizer stack to reach a regime the
Dirichlet (LDA/PLDA) gets for free via α<1 — STM's win remains
covariate-prevalence, not rare-phenotype content.
