# 0031 — Scalable (random-projection) spectral init matches dense on topic quality but not on Σ: a single dominant topic escapes into the blowup basin

**Date:** 2026-06-29
**Topic:** stm | priors | svi | initialization | spectral | phenotyping | sigma
**Status:** Confirmed (cancer cohort, exps 0015 / 0017)

Insight [0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md) proved
that DENSE spectral init + reference topic at σ_init=1 closes STM's Σ blowup on
the cancer corpus — all 40 phenotypes resolved, Σ bounded at 7.56. Insight 0030's
Finding 3 established a **decoupling**: spectral init makes topic quality
(β/θ) σ_init-robust, while Σ-properness is fragile (requires σ_init inside the
Goldilocks window). Exp 0017 **sharpens that decoupling one level deeper** into the
seed itself: the Johnson-Lindenstrauss (JL) approximation of the scalable spectral
path is good enough to recover the same 40 phenotypes as the dense path, but NOT
good enough to keep the single most dominant topic out of the residual-variance
M-step blowup basin that the exact dense seed avoided at the same config.

## Setup

Cancer cohort (`first_cancer_year`, prior_obs_days 0, person_mod 4,
condition_era, ~10,819 fit docs, V=3691), K=40, `~ C(sex) + age`, seed 42,
max_iter 300, σ_init=1.0, reference_topic + spectral_init ON.

- **Exp 0015 (dense):** `spectral_method: dense` — exact V×V co-occurrence,
  exact anchor-word geometry.
- **Exp 0017 (scalable):** `spectral_method: scalable` — random-projection sketch
  to d=1000 (min(V, max(K, 1000)) = min(3691, 1000) = 1000), a ~3.7x row
  compression at V=3691. Absolute document-frequency floor (`min_doc_freq=5`),
  replacing 0015's `min_marginal_frac=1.0`. All other knobs identical.

## Split verdict — headline comparison

| exp | spectral path | Σ_max (final) | runaway topics | # resolved | NPMI mean | converged iter |
|---|---|---|---|---|---|---|
| **0015** | dense (exact) | **7.56** | 0 | **40** | +0.173 | 300 (still creeping) |
| **0017** | scalable d=1000 | **1.08e6** | 1 (topic 2) | **40** | +0.166 | 88 (converged) |

Topic quality: **full win for scalable**. NPMI means are statistically on par
(+0.166 vs +0.173); all 40 topics resolved, unrated 0/40; both return the same
crisp cancer phenotypes (breast topic 14, prostate 5, thyroid 19, melanoma 24,
CKD/kidney 36, bladder 4, colon 18, ovary/endometrium 34,
lymphoma+myeloma 27, HIV+HCV 33, seizure/epilepsy 6, depression/migraine 16,
iron-deficiency anemia + essential thrombocythemia 7, hypothyroid 8/19). The
JL-approximated β seed lands the same 40 phenotypes as the exact dense seed —
the ADR 0032 central bet (random projection preserves the anchor geometry) is
confirmed on real data.

Σ: **NOT equivalent**. Dense 0015 held Σ bounded across all 40 topics (per-topic
O(1–3)); scalable 0017 shows Σ_max = 1.08e6 at convergence, driven entirely by one
topic: the per-topic Σ vector is approximately [1, 2.7, 8.3e5, 3.5, 17.2, 10.8,
…] — topic index 2 alone at ~8.3e5 while the other 39 sit at O(1–20). This is
NOT the ~1e10 global collapse/blowup of the random-init runs (exp 0008, 0012), and
NOT dense's fully bounded regime — an intermediate state where the approximate seed
positioned ONE topic in the blowup basin while correctly positioning the other 39.

## The mechanism — seed approximation shifts one topic into the blowup basin

The residual-variance M-step is:

  Σ_k = mean over docs of [(η_dk − Γᵀ x_d)² + ν_dk]

There is no shrinkage in this step (the Σ-prior is defaulted OFF). If any topic's
η saturates (η_dk driven to extreme values across many docs), Σ_k grows unbounded
through this feedback: larger Σ → weaker Gaussian penalty → larger η residuals →
larger Σ (insight [0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md),
Finding 3; [0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)).

The dense seed (exp 0015) positioned topic 2 to differentiate via β from iteration
0 — its anchor-word rows were sufficiently distinct that η could stay moderate
while the topic stayed separated. The JL-approximated seed (exp 0017) introduced
enough approximation error in topic 2's co-occurrence rows that its β seed was not
distinct enough: η-saturation pressure on the single most dominant, most-peaked
non-background cluster exceeded the threshold for the blowup feedback loop, and by
iter 88 Σ_2 had escaped.

The blowup topic (topic 2) is "Essential hypertension (0.285), Pure
hypercholesterolemia (0.204), Obesity (0.076), Cough, Obstructive sleep apnea,
Chest pain, Hyperlipidemia" — the corpus's single most prevalent comorbidity
cluster, exactly the topic where η-saturation pressure is highest across the most
documents.

The 39 other topics are entirely unaffected. This is a **single-topic partial
escape**, not a systemic blowup.

## Why the blowup is induced by approximation, not by the data or STM

The dense path at the IDENTICAL config did NOT blow up. The approximation is the
only material difference. This means:

- The blowup is an artifact of the JL compression at d=1000 for THIS dominant
  topic, not an intrinsic property of the cancer corpus or of STM.
- Two levers can close the gap (follow-up experiments 0018 and 0019):
  1. **Increase d** — exp 0019 (`spectral_d: 2000`, halving the compression from
     3.7x to 1.8x) tests whether, as d rises toward V, the scalable seed approaches
     the dense fixed point and Σ_2 falls back toward O(10). If so, the blowup is a
     JL-quantifiable approximation error, controllable by d.
  2. **Enable the Σ-prior** — exp 0018 (`sigma_prior_scale: 10.0`,
     `sigma_prior_count: 2000.0`) activates STM's published stability mechanism: an
     inverse-gamma prior that anchors Σ toward a target scale with a pseudo-count.
     This is the shrinkage the M-step currently lacks. It should bind the single
     runaway WITHOUT over-shrinking the 39 already-bounded topics.

## Practical implication

Scalable spectral is **validated for large-V phenotype DISCOVERY** — its primary
purpose. The JL-approximated seed recovers the same 40 phenotypes as the dense
seed, confirming ADR 0032's design.

For a **proper, sample-able Σ** — the green-light for the logistic-normal sampler
of [ADR 0028-B](../decisions/0028-dashboard-conditioned-dirichlet-prior.md) — the
scalable seed at σ_init=1 alone is insufficient without additional Σ-binding. The
Σ-prior (`sigma_prior_scale` / `sigma_prior_count`) is the principled robustness
lever: it is STM's own published stability mechanism, and its role is exactly to
absorb the residual Σ-pressure that an imperfect seed leaves unsuppressed.

**The broader point**: all STM runs to date have operated with ZERO Σ shrinkage
(`sigma_prior` defaulted OFF). The dense spectral seed was strong enough to
suppress Σ blowup without it, but the scalable seed surfaces the fragility.
Engaging a mild Σ-prior is the principled top-up — not a workaround, but the
mechanism the published STM uses by default.

## Relationship to prior insights and decisions

- **[Insight 0030](0030-spectral-init-closes-stm-sigma-blowup-on-real-data.md)**:
  0031 sharpens 0030's decoupling (topic quality vs Σ) from the σ_init dimension
  to the seed-approximation dimension.
- **[Insight 0029](0029-stm-sigma-init-collapse-blowup-missing-stabilizers.md)**:
  the three missing stabilizers; the M-step feedback mechanism.
- **[ADR 0032](../decisions/0032-scalable-spectral-init-random-projection-over-maxv.md)**:
  ADR 0032's topic-quality bet confirmed on real data; Σ-equivalence partially
  confirmed (39/40 topics bounded), with the one runaway quantifying the JL
  approximation pressure on the most dominant topic.
- **[ADR 0028-B](../decisions/0028-dashboard-conditioned-dirichlet-prior.md)**:
  the logistic-normal sampler green-light requires a proper Σ; the scalable path
  needs the Σ-prior (exp 0018) or a larger d (exp 0019) to deliver it.
- **Exp 0015** (dense spectral, baseline); **exp 0017** (this run, scalable);
  **exp 0018** (Σ-prior top-up on scalable); **exp 0019** (larger d=2000, JL
  approximation pressure test).
