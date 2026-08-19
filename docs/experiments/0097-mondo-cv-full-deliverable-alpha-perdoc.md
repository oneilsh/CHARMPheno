---
id: 97
slug: mondo-cv-full-deliverable-alpha-perdoc
status: pending
model_class: gated_pc
cohort: population_mondo_cardiovascular
cohort_def: population_mondo_cardiovascular
disease: rare_priority
# 0097 — Mondo cardiovascular: full deliverable (alpha=0.5 + per-doc-mean, stable)

The first full-scale run of a STABLE shaping PC fit — the payoff after the whole
neutral-PC saga resolved to: alpha=1/K collapsed the CAVI Jacobian (exp 0095), and the
newton head ran away on an absolute ridge at 1e5 docs (exp 0095 blowup). Fixes:
doc_concentration=0.5 (Jacobian alive) + per-doc-mean newton (|w| corpus-bounded). exp
0096 confirmed stability at 8 iters (|w|~1.45, corr~0.7%/step, ELBO rising). This is the
full 100-iter fit WITH the unsup twin + pc_topics_lr readout — the deliverable.

## What to read (`make -C analysis/cloud report ID=97`)

1. **HEADLINE `gated_pc vs unsup_gated` (pc_topics_lr)** — THE deliverable. A POSITIVE
   delta = PC shaping lifts the readout over unsupervised topics = the method delivers.
   Every prior run was Δ≈0 because shaping was DEAD (alpha-collapse); this is the first
   run where it can actually move.
2. **FIT-HEALTH trajectory** — confirm 0096's stability holds over 100 iters: `|w_CK|max`
   bounded (~1-3, no 1e5), `corr_relΔλ` steady (~0.005-0.01), `||grad_y||` finite, ELBO
   rising. If it destabilizes late, note where.
3. **conditional readout (P(child|parent) by depth)** — the case-finding metric.

## Sequence after this

- **Positive lift**: sweep `weight_y` UP (4, 8) for MORE shaping — now SAFE, the head
  can't run away (per-doc-mean). weight_y=2 gives a gentle ~0.7%/step; there is lots of
  headroom. Find the shaping-vs-stability sweet spot, then re-confirm the lift.
- **Flat lift despite stable shaping**: shaping is too gentle at weight_y=2 — raise it
  before concluding anything.
- Then: the deferred refinements (uniform-beta A/B via gamma_shape; decoupled/floored
  alpha only if we later co-fit) and the whole-Mondo K≈3800 scale-up.

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=97
make -C analysis/cloud report ID=97
```

## Run log

_(pending first run)_
