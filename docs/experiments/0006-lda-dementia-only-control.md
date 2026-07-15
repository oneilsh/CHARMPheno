---
id: 6
slug: lda-dementia-only-control
status: pending
model_class: lda
cohort: dementia
cohort_def: first_dementia_year
prior_obs_days: 0
person_mod: 4
doc_unit: patient
doc_min_length: 20
K: 40
max_iter: 100
vocab_size: 10000
min_df: 20
print_topics_every: 10
top_n_tokens: 8
seed: 42
optimize_doc_concentration: true
optimize_topic_concentration: false
---

# Control: plain LDA on the dementia arm alone (isolate gating vs. corpus size)

Diagnostic control for insight
[0027](../insights/0027-gated-stm-imbalanced-arms-majority-foreground-collapses.md)
and experiment [0005](0005-gated-stm-dementia-only-foreground.md). The gated-STM
dementia foreground collapsed to 10 near-uniform clones (peak β ≈ 0.003, all
"Dementia + Postconcussion + baseline"), surfacing no within-dementia
sub-phenotype. Insight [0021](../insights/0021-cohort-corpora-two-anchor-mass-concentration.md)
showed that **non-gated LDA on the dementia cohort alone** *does* recover ~15–18
crisp dementia sub-phenotypes (HIV-dementia peak 0.140, Lewy-body REM 0.090,
epilepsy 0.060, scleroderma 0.108, psoriasis 0.080, …) — but at **person_mod=1**
(full set), a 365d post-dx window, and vocab 10k. So 0021 alone cannot tell us
whether 0005's collapse is caused by **gating** (the foreground competes with a
30-topic, 81%-cancer-led background that absorbs all shared structure) or by
**corpus size / windowing** (at person_mod=4 the rare phenotypes, which sat at
the α-floor in 0021, may drop below resolvability).

This run holds the corpus as close to 0005's dementia arm as possible and
removes gating, changing only what the control requires:

- **No gating, no cancer arm.** Plain LDA on `first_dementia_year` only, so all
  K topics are fit on dementia documents (anchors *and* the rare-phenotype tail
  come from dementia data). This is the standalone analogue of 0005's 10-topic
  dementia foreground.
- **Corpus matched to 0005:** `person_mod=4`, `prior_obs_days=0`, `patient`
  doc-unit (one lifetime doc per dementia patient — the single-cohort analogue
  of 0005's `patient_cohort` granularity), `doc_min_length=20`, condition_era,
  `vocab_size=10000`, `min_df=20`.
- **K=40, asymmetric-α optimization on**, matching 0021 deliberately: 0021
  established that lowering K collapses the rare-phenotype tail into the anchors
  without changing anchor mass, so K must be generous to give sub-phenotypes a
  chance. `max_iter=100` (0021's rare phenotypes sharpened by iter ~30–50).

## What the result decides

1. **If the tail resolves crisp dementia sub-phenotypes** (Lewy-body / epilepsy /
   vascular / HIV-dementia, peak β well above the ~0.003 the gated foreground
   produced) → the mod=4 corpus is rich enough standalone, and **gating is what
   starves the foreground.** That points the fix at the gating mechanism
   (balanced-background SVI so the background isn't cancer-led; or reserve more
   foreground capacity; or a two-pass scheme).
2. **If the tail also collapses into the anchors** (no crisp sub-phenotype above
   the noise floor) → the limitation is **corpus size / windowing at person_mod=4**,
   not gating per se. Follow-up would be a person_mod=1 standalone run
   reproducing 0021 on this windowing to confirm the rare phenotypes return with
   ~4× the documents.

Either way the comparison is read against 0021 (same model, same cohort family,
person_mod=1) and 0005 (same corpus, gated). NPMI is reported but — per
[0010](../insights/0010-npmi-not-comparable-across-doc-units.md) and 0027's
finding that NPMI can reward degenerate uniform topics — the load-bearing
readout here is per-topic **peak β / Σλ spread / top-term diversity** (now
visible in the per-iter fit log), not the NPMI mean.
