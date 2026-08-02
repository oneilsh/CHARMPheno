# 0082 — No-roll-up SNOMED hierarchy recovers case-finding to the flat baseline: roll-up flooding was the entire villain

**Date:** 2026-08-02
**Topic:** hierarchy | pooling | rollup | case-finding | multidomain | decision
**Status:** Confirmed on exp 0083 (SNOMED hierarchy, NO roll-up, spectral init,
clean fit). Completes the {roll-up, init} 2x2 (exps 0080/0081/0082/0083).

## The 2x2 is complete and the story is clean

Condition macro median AP (fast parameter-free readout), all same layout
(rare_priority, cond+drug+measurement, SNOMED hierarchy, K structure, seed 42):

| | random init | spectral init |
|---|---|---|
| **roll-up** | 0080: 0.006 (under-fit) | 0082: **0.006** (clean fit) |
| **no roll-up** | 0081: AP lost to OOM (classes alive) | 0083: **0.021** (clean fit) |

Flat baseline ~0.020. So:
- **Roll-up collapses case-finding ~3x** (0.020 -> 0.006), and it is structural
  (spectral 0082 clean fit reproduces it exactly) — insight 0080.
- **Removing roll-up recovers case-finding all the way back to flat** (0.021 ~
  0.020) on a clean well-conditioned fit. **The roll-up flooding was the entire
  villain.** Routing every descendant patient up to a class node made the class
  topics common-disease topics; pooling only the anchor-routed patients (a
  rare-flavored class mean) removes the harm.

## But the hierarchy is neutral, not a win, for case-finding

0083 == flat at the macro (0.021 vs 0.020). So fit-time hierarchical pooling does
not *lift* rare-disease case-finding overall — it is a wash. It is not uniform:
- **Helps where the class is coherent:** Ehlers-Danlos 0.060 -> **0.083**;
  Scleroderma surfaces strongly (0.086 cond / 0.058 meas) under the
  connective-tissue class; the class topic is genuinely rare-autoimmune
  (Sjogren's / systemic sclerosis / ANA / complement / DMARDs — 0081 fit card).
- **Dilutes elsewhere:** SLE and MS drop somewhat. Net wash at the macro.

So the classes pool usefully when the sibling anchors share a real phenotype and
dilute when they do not — averaging to flat.

## Decision

1. **Roll-up attestation is closed for case-finding.** Do not ship it. (Still
   mechanically correct and gives clean *class* phenotypes — insight 0080 — just
   the wrong pooling prior for rare-disease targets.)
2. **The hierarchy WITHOUT roll-up is safe to keep** (parity with flat, no
   regression), which matters because it is the scaffold for the higher-value
   use: **eval-time within-class ranking** (the colleague's "rank within
   connective-tissue / EDS-class" ask). That readout scores against the class
   structure at evaluation time and does not depend on fit-time pooling, so this
   neutral fit-time result does not touch it. **That readout is the next build.**
3. **Fit-time pooling to *beat* flat is not demonstrated.** The open question —
   does more class *capacity* (data-driven K_v, insight 0081) turn the coherent
   classes' parity into a win? — is now exploratory, not critical-path: the
   effrank probe is wired into the gated init (it silently did not fire on 0083
   because it was in the wrong init function; fixed), so any future spectral fit
   emits the per-node effective-rank table. Chase it only if the within-class
   readout motivates richer class topics.

## Fit health note (exp 0083)

Clean spectral fit. A handful of thin descendant nodes had zero/degenerate
sketches and stayed at the 1e-9 floor (nodes 32/33/34/78/94/102/150) — expected
for sparse deep branches, not the class layer. ELBO fell -177M -> -105M over the
first iterations and settled; persist succeeded (138s) on the n2-standard-16
master. See exp 0083 for the fit card (summarize-exp).

**Setting context.** Exps 0080/0081/0082/0083, rare_priority,
cond+drug+measurement, SNOMED hierarchy (restrict-under 4274025,
max_class_fraction 0.6), spectral_proj_dim 400, K=350, seed 42. Fast --fixed-only
readout, WEIGHTING_JOBS=4. See insight 0080 (roll-up flooding mechanism), 0081
(effrank capacity method), 0079 (measurement specialist), 0076 (pooling not worth
it — now refined: roll-up pooling hurts, no-roll-up pooling is neutral).
