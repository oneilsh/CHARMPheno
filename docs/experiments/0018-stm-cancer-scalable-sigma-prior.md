---
id: 18
slug: stm-cancer-scalable-sigma-prior
status: done
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
sigma_prior_scale: 10.0
sigma_prior_count: 2000.0
---

# STM cancer demo: scalable spectral + Σ-prior top-up (the shrinkage fix for the one runaway topic)

The Σ-prior stability test for the scalable spectral path. **Identical to exp 0017
in every way except the addition of `sigma_prior_scale: 10.0` and
`sigma_prior_count: 2000.0`.** Exp 0017 recovered all 40 cancer phenotypes with
NPMI on par with the dense baseline (exp 0015), but one dominant topic — the
hypertension/cholesterol comorbidity cluster (topic 2) — ran Σ to ~8.3e5 while
the other 39 topics held at O(1–20). The Σ-prior is STM's published stability
mechanism; this run tests whether it is the correct top-up for the single runaway.

## Why

Exp 0017 (insight [0031](../insights/0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md))
surfaced a partial blowup: the scalable (random-projection) spectral seed correctly
positioned 39/40 topics in bounded Σ regimes, but the JL approximation error left
the most dominant topic (essential hypertension / pure hypercholesterolemia) with
insufficient β differentiation, causing its η to saturate and driving Σ_2 → ~8.3e5
through the residual-variance M-step feedback loop (Σ_k = mean over docs of
[(η_dk − Γᵀ x_d)² + ν_dk] — no shrinkage by default).

All STM runs to date have used ZERO Σ shrinkage (`sigma_prior` defaulted OFF). The
dense spectral seed was strong enough to avoid blowup without it; the scalable seed
surfaces the fragility of running with no Σ anchor. The inverse-gamma Σ-prior
anchors each Σ_k toward a target scale with a pseudo-count:

  posterior Σ_k = (sum of residuals + scale × count) / (N_docs + count)

where `sigma_prior_scale` is the target Σ scale and `sigma_prior_count` is the
number of pseudo-observations pulling toward it. This is STM's own published
stability mechanism — not a workaround, the principled lever.

## Parameter rationale

- **`sigma_prior_scale: 10.0`** — targets Σ near the data's natural η-scale. Dense
  0015's Σ_max was 7.56 with per-topic O(1–3); a target of 10.0 is comfortably
  near the dense fixed point and well above the O(1) σ_init=1.0 starting point.
- **`sigma_prior_count: 2000.0`** — approximately 18% of the 10,819 fit documents
  (2000 / 10819 ≈ 0.185). At this pseudo-count the prior is informative enough to
  bind a runaway Σ (~8.3e5, driven by a single dominant topic) while being weak
  enough that the 39 already-bounded topics (Σ O(1–20)) are not significantly
  over-shrunk. The runaway topic's residual is large (its η saturates), so it
  dominates the M-step sum — a moderate count is needed to pull it back without
  crushing the well-behaved topics. If 2000 proves insufficient, the natural
  follow-on is to raise count toward N (stronger shrinkage); but watch for topic 2's
  NPMI dropping (over-shrink signal).

## Hypothesis

The Σ-prior at scale=10, count=2000 binds the single runaway (topic 2 Σ from
~8.3e5 back toward O(10)) without materially degrading the 39 already-bounded
topics, so that:
(a) Σ_max falls toward O(10–100), NOT O(1e5+),
(b) the same 40 cancer phenotypes are recovered with NPMI on par with exp 0017
    (+0.166) and exp 0015 (+0.173),
(c) the runaway topic 2 (hypertension/cholesterol) retains its identity and
    structure — only its Σ is pulled in, not its β.

## What to watch

- **Σ[min … max] trace and per-topic Σ vector** — the headline. Does Σ_2 drop from
  ~8.3e5 toward O(10)? Does any of the 39 already-bounded topics see its Σ pushed
  unnaturally LOW (toward 1 or below), signalling over-shrink?
- **Topic 2 content (hypertension/cholesterol cluster)** — spot-check that peak β
  words are unchanged. If the prior over-shrinks this topic's η, its content may
  blur toward the corpus marginal.
- **NPMI mean vs 0017 (+0.166) and 0015 (+0.173)** — a sharp drop would indicate
  the prior is pulling topics toward over-shrink; a maintained or improved NPMI
  confirms the prior is acting as a Σ anchor, not a quality drag.
- **Convergence iter vs 0017 (iter 88)** — a prior modifies the M-step but should
  not dramatically change the convergence profile; large changes are diagnostic.
- **All 40 topics resolved** — success criterion same as 0017.

## Decision tree

- **(a) Σ bounded + topics preserved** (Σ_max O(10–100), NPMI ~0.166, all 40
  phenotypes): the Σ-prior is the scalable stability answer. A mild prior should
  be considered default-ON for any scalable spectral run. Document scale/count as
  the recommended starting point; green-light the logistic-normal sampler
  (ADR 0028-B) for the scalable path.
- **(b) Σ bounded but topics degraded** (NPMI drops meaningfully, topic 2 blurs):
  count=2000 over-shrinks. The prior is pulling too hard on the η residuals — lower
  the count (try 500–1000) and re-run before treating the prior as the production
  fix.
- **(c) Σ still blows up** (Σ_max remains ~1e5+): the prior at count=2000 cannot
  bind a true η-saturation blowup. The seed approximation is the root cause and the
  d lever (exp 0019, larger projection dimension) is the correct fix — the prior is
  insufficient at reasonable pseudo-counts when the per-doc residuals are extreme.

## Run

```
make exp ID=18
```

Compare head-to-head with exp 0017 (scalable, no prior) and exp 0015 (dense,
no prior). The delta 0017 → 0018 IS the Σ-prior contribution; the delta 0015 →
0018 measures whether prior-stabilized scalable matches the exact dense baseline
end-to-end. Cross-link: insight [0031](../insights/0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md),
exp 0017, exp [0019](0019-stm-cancer-scalable-larger-d.md).

## Result

Ran on the cluster 2026-07-09 with the **latest code** — which changed what the run
measures. The two things exp 0018 was built around no longer exist:

1. **The Σ-prior is gone.** `sigma_prior_scale` / `sigma_prior_count` (the
   inverse-gamma / IW anchor) were removed from the engine and from `build_stm_args`
   — a falsified class: the IW anchor *worsened* the gated runaway (exps 0022/0024,
   2.2e7 → 6.08e8) and free variance is unidentifiable at fit for document-scarce
   topics (insight 0033); see `analysis/local/stm_ablation.py`. This doc's frontmatter
   keys `sigma_prior_scale: 10.0` / `sigma_prior_count: 2000.0` are now HISTORICAL keys
   that `build_stm_args` silently ignores (pinned by a regression test in
   `scripts/tests/test_run_experiment.py`). No Σ-prior reached the fit.
2. **No free variance to bind.** The run is UNIT-DIAGONAL (`Σ_var[min=1 max=1]`) — the
   block-wise unit-diagonal correlation Σ (ADR 0034) is now the default, and exp 0018
   set no `estimate_sigma_diagonal`. Σ_ii is pinned to 1 by construction, so exp 0017's
   topic-2 escape to ~8.3e5 (a property of the FREE-variance parameterization) cannot
   occur here.

So exp 0018 does not (cannot) test its stated Σ-prior hypothesis. What it actually is:
**exp 0017 re-run under the current unit-diagonal default** — a clean scalable spectral
+ unit-diagonal cancer fit.

Outcome: converged 300/300 (ELBO −1.136e6, 598s). All 40 topics resolved, NPMI mean
**+0.174** (median +0.148, max +0.556) — on par with the DENSE baseline exp 0015
(+0.173) and slightly ABOVE the free-variance scalable exp 0017 (+0.166). Σ bounded and
proper by construction (unit-diagonal; eig 0.0296–22.8; the eval's
`runaway = topic 39 Σ_ii=1.000` is the argmax-over-a-constant-diagonal artifact, not a
blowup). Reference topic 0 alive (E[β]=0.0126, chest-pain / gastritis / abdominal-pain).
Crisp cancer sub-phenotypes throughout: breast(14), prostate(5), thyroid(19),
melanoma(24), skin/actinic(31), lung(11), CKD/kidney(36), bladder(4), colon(18),
pancreatic/ovarian(25,34), lymphoma/myeloma(27), head-and-neck(21), HIV/HepC/liver(33),
chemotherapy-toxicity(26).

**Verdict — the Σ-prior question is MOOT and the scalable arc's Σ concern is CLOSED.**
None of the Decision-tree branches (a/b/c) apply — each presumes a free-variance Σ and a
live Σ-prior. The real conclusion is stronger: unit-diagonal Σ (ADR 0034) REMOVES the
scalable path's single-topic runaway by construction, so exp 0017's topic-2 escape cannot
recur and no Σ-prior is needed. This resolves insight 0031's "Σ-equivalence PARTIAL"
caveat — under the production default (unit-diagonal), scalable spectral is not merely
topic-quality-equivalent to dense, it is fully clean (no escape), NPMI matching dense. By
the same logic the other Σ-stability follow-up, exp
[0019](0019-stm-cancer-scalable-larger-d.md) (larger projection dim d to shrink the JL
escape), is also moot — there is no escape to shrink under unit-diagonal.

Housekeeping: the Parameter-rationale / Hypothesis / Decision-tree sections above are
kept as the record of the ORIGINAL (now-obsolete) intent — they describe a Σ-prior +
free-variance regime the current code no longer supports. See insight 0031, exp 0017
(scalable, free-variance), exp 0015 (dense baseline).
