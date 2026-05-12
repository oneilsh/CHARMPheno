# 0015 — Crisp topics can regress in late iters when K is undersized
**Date:** 2026-05-12
**Topic:** lda | diagnostics
**Status:** Observed

On the K=25 patient-year LDA run, two phenotype topics were crisp at
iter 12 and contaminated by iter 20:

- **Pregnancy** (iter 12): Finding related to pregnancy, Complication
  occurring during pregnancy, High risk pregnancy, Third trimester,
  Second trimester, Disorder of pregnancy. Clean.
  → (iter 20): Visual disturbance, Influenza-like illness, High risk
  pregnancy, Third trimester, IgE-mediated allergic asthma, Disorder
  of pregnancy. Contaminated — pregnancy concepts still present but
  blended with unrelated visual/respiratory content.

- **Sickle cell** (iter 12): Hemoglobin SS × 2, Retinal detachment,
  Type 2 DM, Pain. Reasonably clean.
  → (iter 20): Systolic heart failure, Hemoglobin SS, Chronic fatigue,
  Glaucoma, HTN, Tear film insufficiency, Transplanted cornea present,
  Changes in skin texture. Diluted — Hemoglobin SS still present but
  the topic has drifted into a broader "complicated chronic patient"
  shape.

Interpretation: at K=25 the model identified more phenotype-shaped
clusters than it had topic slots for. As later iters refined the
high-mass topics, low-mass phenotype topics absorbed remaining
unmatched events, getting diluted in the process. The model isn't
"losing" the phenotype — it's compressing 26+ phenotype classes into
25 slots.

**Implications.** Watch for late-iter topic contamination as a signal
that K is too small for the corpus's effective phenotype count. If
a topic that was crisp mid-training drifts toward mixed content by
the end, the response is to raise K, not to extend iter count.

For the patient-year condition-era corpus at K=25 the threshold seems
to be K ≈ 25–30 phenotype classes plus 3 background flavors plus 2
acute clusters — roughly 30–35. Worth trying K=50 next to give the
rare-phenotype topics room.

**Setting context.** K=25 LDA, online VI, patient-year condition-era
docs, min_doc_length=30, 20 iters. Iter-12 vs final (iter 20) topic
inspection. Same hyperparameters as the run analyzed in
[0014](0014-patient-year-npmi-bimodal-vs-lifetime-unimodal.md).
