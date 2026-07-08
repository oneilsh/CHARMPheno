# 0041 — A drug-anchored active-comparator gated STM separates drug-specific structure from the shared indication

**Date:** 2026-07-08
**Topic:** stm | gating | cohorts | drug-anchor | pharmacoepi
**Status:** Confirmed (real-cohort cluster run exp 0044)

Exp [0044](../experiments/0044-stm-population-glp1-comparator.md) is the first cohort on
the **drug-anchor track** (everything prior anchored on a *diagnosis*; here the anchor is
a *drug era*, while documents stay condition content). It fits a whole-population
background against two incident-new-user drug foreground arms — GLP-1 receptor agonists
and SGLT2 inhibitors — chosen as an **active-comparator pair**: both are prescribed for
the same T2DM / cardiometabolic indication, so a same-indication comparator controls
confounding-by-indication in a way a population-only contrast cannot. It answers whether
the gate can carve *drug-specific* structure out from the shared *why-they-were-prescribed*
structure.

## Finding 1 — the gate separates drug-specific comorbidity on top of a shared indication spine

The full-population fit (`person_mod: 1`, 294,485 persons → 177,359 docs, K=110 = 80 bg +
15 + 15) produced a clinically faithful split that a pharmacologist would name:

- **SGLT2i arm → cardiorenal.** Two heart-failure topics (systolic/diastolic CHF,
  cardiomyopathy, hypertensive HF), a coronary-disease topic (coronary atherosclerosis,
  old MI, aortocoronary bypass graft), and CKD/proteinuria topics — the evidence-based
  HFrEF/HFpEF + cardiovascular-risk indication the class is actually prescribed for. This
  *exceeded* the pre-fit guess ("genitourinary / volume").
- **GLP-1 arm → obesity + its signature GI adverse-effect footprint.** A topic pairing
  morbid obesity with nausea / vomiting / constipation / GERD (the classic GLP-1 GI
  side-effect cluster), plus pure obesity/metabolic topics (severe obesity, prediabetes,
  steatosis, OSA, PCOS, metabolic syndrome).
- **Shared T2DM spine in both blocks.** Type-2-diabetes topics appear in *both* foreground
  arms. That is the confounding-by-indication being controlled, made visible: the
  indication sits in both arms, and the HF/CAD-vs-obesity/GI difference is precisely what
  the gate isolates on top of it.

The key methodological point: because both arms share the indication, the *difference*
between the arms is closer to drug-specific structure than either arm's raw content is. A
GLP-1-vs-population contrast would have surfaced "who gets a GLP-1" (their T2DM/obesity
indication); the active comparator subtracts most of that, leaving drug-linked structure.

## Finding 2 — the drug-anchor track behaves exactly like the disease track under the gate

No new instability from anchoring on a drug instead of a diagnosis. Σ stayed bounded (all
Σ_ii = 1, block-wise unit-diagonal, ADR [0034](../decisions/0034-stm-blockwise-unit-diagonal-correlation-sigma.md);
`blocks[bg=7.7e6 glp1_ra=1.41e5 sglt2i=7.95e4]`; the eval's `runaway = topic 0` is the
argmax-over-constant-diagonal artifact, same benign readout as insight
[0031](0031-scalable-spectral-topic-quality-matches-dense-but-sigma-splits-one-runaway.md)),
background NPMI mean +0.183. This confirms the drug-anchored primitive
(`apply_population_drug_cohort`: `drug_era` → first-era index → new-user observation
bracket → condition documents) is a drop-in sibling of the disease primitive, not a new
regime — the gating math is anchor-agnostic.

## Corollaries that fell out of the build

- **Descendant concept sets are load-bearing for drugs too.** Name-only RxNorm-ingredient
  matching under-counted tirzepatide to 128 docs; descendant expansion of the ATC class
  seeds (GLP-1 1403 concepts / 20,722 persons, SGLT2i 1006 / 13,528) fixed the drug arms
  the same way ancestor descendants define the disease arms. The lone remaining name/pin
  dependency (tirzepatide 779705) is only used to *exclude* those users.
- **A both-user routing rule preserves power without a combination arm.** Routing wide-gap
  (>365d) GLP-1+SGLT2i both-users to their earlier *monotherapy* year recovered 3,848
  otherwise-excluded documents while keeping each index year single-class; only the ~2,546
  within-365d co-initiators were dropped. Thin arms (tirzepatide 128, combo) were removed
  rather than modeled — merging arms post-hoc is cheap, un-mixing a contaminated contrast
  is not.

## Takeaway

The gated STM generalizes from disease anchors to **drug anchors**, and pairing a drug arm
with a same-indication active comparator turns the gate into an indication-controlled
drug-contrast instrument: the shared indication lands in both blocks and the drug-specific
structure separates on top of it. This is the pharmacoepi analogue of the rare-disease
result (insight [0035](0035-rare-disease-gated-foreground-recovers-eds-subphenotypes-on-full-population.md)) —
same architecture, new anchor domain, faithful structure.
