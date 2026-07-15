---
id: 5
slug: gated-stm-dementia-only-foreground
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 40
background_k: 30
foreground: "dementia:10"
group_var: source_cohort
max_iter: 100
---

# Gated STM: drop the majority (cancer) foreground block

A/B follow-up to experiment [0004](0004-gated-stm-cancer-dementia.md), isolating
a single lever from insight
[0027](../insights/0027-gated-stm-imbalanced-arms-majority-foreground-collapses.md):
**remove the majority arm's foreground block.** Everything else is held
identical to 0004 (same cohort, `prior_obs_days`, `person_mod`, doc-unit,
formula, seed, iters, and `background_k: 30`) so the only change is dropping the
10 cancer foreground topics. `K` falls from 50 to 40.

`cancer` is no longer named in `foreground`, so under the gating partition
(ADR [0026](../decisions/0026-gated-stm-hard-masking.md)) cancer documents
become **background-only**: they express the 30 shared background topics and no
foreground block. This is the documented "large common cohort informs the
background while only the rarer group carries foreground topics" use of the
partition. `group_var: source_cohort` is unchanged, and `source_cohort` remains
deliberately absent from `covariate_formula` (it is the gating variable).

## Why

Insight 0027 found that on the imbalanced 81% cancer / 19% dementia cohort the
cancer foreground block collapsed into 10 degenerate baseline-echo slots
(Σλ≈320, NPMI≈0.03, near-identical top-N): the crisp cancer phenotypes
(breast — 0004 topic 22; skin/melanoma — 0004 topic 4) settled in the all-docs
**background** instead, because a background topic may fire on every document
and cancer is the majority. The cancer-only foreground was therefore occupied
but contentless — wasted capacity.

This run tests that diagnosis directly: if the cancer foreground was genuinely
redundant, removing it should leave the rest of the model essentially intact.

## Success criteria / what to watch

1. **Dementia foreground is preserved.** Its reference (the 2,476 dementia
   documents) is identical to 0004, so dementia-block NPMI is directly
   comparable run-to-run. Expect roughly the same profile as 0004 (~1 real
   musculoskeletal/aging cluster + several "Dementia + Postconcussion +
   baseline" near-duplicates, NPMI ~0.13–0.24). A large change here would be
   surprising and worth investigating (the dementia block is separately masked
   and should not depend on the cancer block).
2. **Background still carries the cancer phenotypes.** The full-corpus
   background reference (13,295 docs) is also identical to 0004, so background
   NPMI is comparable. Confirm the breast-cancer and skin/derm clusters still
   appear as background topics. The ~3.2k token-mass the cancer foreground held
   in 0004 returns to the background on cancer documents, so the background
   anchors and crisp-cancer topics may **sharpen slightly** (higher Σλ / peak) —
   note any such shift.
3. **Verdict on the wasted-capacity claim.** If (1) and (2) hold, the cancer
   foreground in 0004 was wasted capacity and gating should reserve foreground
   blocks for minority arms only. If instead removing it materially changes the
   background or dementia blocks, the blocks interact more than 0027 assumed.

Holding off on the companion "more dementia foreground slots" run (would be
0006: `30 bg / 0 cancer / 20 dementia`) until this A/B lands.
