# 0021 — Cohort-filtered corpora concentrate 90%+ of mass into two universal-anchor topics regardless of K
**Date:** 2026-05-18
**Topic:** lda | doc-units | diagnostics

When LDA is fit on a clinical-cohort corpus (patients sharing a
common index condition, windowed to a fixed post-index period),
the corpus-level topic mass concentrates dramatically into two
anchor topics: one absorbing the **universal-symptom baseline**
(Pain, Chest pain, Nausea, SoB, Vomiting, Anxiety) and one
absorbing the **chronic-comorbidity baseline** (HTN, HLD, T2DM,
GERD, Chronic pain, Anxiety disorder). At K=40 on the
first-dementia-year cohort, by iter 50:

- topic 18 (symptom anchor): α=0.127, E[β]=0.705, Σλ=8.5×10⁵
- topic 13 (comorbidity anchor): α=0.119, E[β]=0.223, Σλ=2.7×10⁵
- **Combined: 93% of corpus mass.** Remaining 38 topics sit
  between α=0.017 and α=0.021 (essentially the asymmetric-α floor).

α-optimization actively amplifies the concentration over iterations
(at iter 10, anchors were α≈0.031; at iter 47 they were α≈0.069;
at iter 50 they're α≈0.12 — a 4× amplification of the gap to the
non-anchor topics).

## Why this is a corpus property, not a K problem

The cancer cohort run shows the same pattern at lower amplitude
(its post-dx year corpus is much larger and more heterogeneous —
many tumor types × treatment regimens). The full-corpus
patient-year run from [0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md)
decomposes the same baseline into *three* distinct catch-alls
(acute-presentation, generic-chronic, cardiometabolic) at ~14–29%
each, totaling ~60% — still concentrated, but spread.

The cohort case is more extreme because filtering on a single
index condition + a fixed time window enforces a shared coding
load across all docs: every patient gets BP checks, lipid panels,
generic-symptom encounters during the workup. The phenotype
*variance* is compressed; the baseline *commonality* is
amplified.

**Empirically verified that this is K-invariant:** lowering K
would collapse the rare phenotypes (epilepsy, HIV/AIDS-dementia,
Lewy body subtype, diabetic retinopathy, psychotic features,
psoriasis, IBD, scleroderma) into the anchors without changing
the anchors' total mass. Raising K would create more low-α
near-empty slots without redistributing mass away from the
anchors (the α optimizer pulls in the wrong direction for that).

## What does work

Despite the 93% anchor concentration, the **small-α tail topics
resolve to crisp rare phenotypes** by mid-to-late iters. At
iter 50 on dementia K=40, ~15–18 topics with E[β] in the
0.001–0.01 range carry sharp signatures (epilepsy peak 0.060,
HIV peak 0.140, Lewy body REM-behavior-disorder 0.090, cataract
surgery 0.093, psoriasis 0.080, scleroderma 0.108, etc.). The
labeler classifies anchors as `anchor`/`background` and these
crisp rare topics as `phenotype` — the dashboard story works
once that classification is applied.

## Implications

1. **Don't size K based on mass concentration.** The 90% anchor
   reading is alarming but is a property of the cohort filter,
   not a misfit. K=40 was correct for the dementia cohort; K=20
   would have destroyed the rare phenotype tail without changing
   the anchor mass.
2. **Don't size K based on iter-10/iter-20 snapshots.** The rare
   phenotype topics look like noise early — peak word
   probabilities below 0.01 — and sharpen between iter 30 and
   iter 50. The mass distribution stabilizes before per-topic
   vocabulary does.
3. **Anchor concentration scales inversely with cohort
   heterogeneity.** Narrow cohorts (one syndrome, fixed window)
   concentrate harder than broad cohorts (many tumor types, same
   window) which concentrate harder than the full corpus.
   Reading anchor mass percentages across cohorts is informative
   about heterogeneity.
4. **The K-sizing decision for cohorts should target the
   phenotype tail richness, not the anchor mass.** "How many
   rare-but-sharp clusters do I expect to find?" → set K to
   roughly 2 × that number to give the model room.

## Relationship to [0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md)

[0019](0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md)
showed that large K on the full corpus gracefully under-uses
excess slots without micro-cluster artifacts. The cohort case
here is consistent: at K=40 on dementia we get ~2 anchors + ~15
phenotype + ~23 gracefully-unused (the unused topics show peak
word probs <0.005 and look like noise-floor echoes of
HTN/Dementia/Pain). The same parametric-LDA-at-generous-K
recipe holds, just with a more extreme mass distribution.

**Setting context:** Online VI LDA, K=40, asymmetric-α optimization
on, condition_era doc-unit, `first_dementia_year` cohort (descendants
of SNOMED 4182210, 365d post-dx window, observation_period bracketing
±365d), person_mod=1 (full participant set), vocab_size=10000,
min_df=10, batch_fraction=0.2, τ_0=64, κ=0.7, 50 iters. Pattern
also visible in the `first_cancer_year` K=40 run at lower amplitude.
