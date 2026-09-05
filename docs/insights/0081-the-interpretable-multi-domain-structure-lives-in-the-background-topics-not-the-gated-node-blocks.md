# 0081 — At whole-Mondo scale the interpretable, multi-domain-*discriminative* structure lives in the 8 background topics, not the gated node blocks; measurement earns its keep there, and `n_bg` is not a performance lever

**Date:** 2026-09-05
**Topic:** lda
**Status:** Observed

**Relates to / grounded in:** 0005 (LDA decomposes background into flavors —
chronic / acute / metabolic — on patient-year docs; this is that finding at
whole-Mondo scale with the fused multi-domain vocab), 0019 (generous-K flat LDA →
catch-alls + phenotypes), 0062 (n_bg is a NULL lever for detection; binding
constraint is information = multi-domain features, not capacity), 0079/0080 (the
node blocks: catch-all shallow, starved deep, non-redundant where fed).

## Observation

Reading the 8 shared background topics of the 0111 episode fit
(`inspect_topics.py`, all domains named), each `n_bg` topic is a coherent,
**multi-domain, discriminative** care archetype — conditions, labs, AND drugs
cohere:

- **BG0** — stable cardiometabolic outpatient: HLD / HTN / coronary atherosclerosis
  + normal CBC + aspirin / atorvastatin / simvastatin / lisinopril / metoprolol.
- **BG1** — acute / ED: pain / N&V / chest pain / SOB + vitals `[measured]` + IV
  fluids / ondansetron / fentanyl / oxycodone + flu antigens.
- **BG2** — primary-care lab visit: HTN / anxiety / HIV + a full BMP+CBC
  `[measured]` + amoxicillin / azithromycin / fluticasone.
- **BG3** — endocrine-oncology / women's health: acquired hypothyroidism / breast
  cancer / osteoporosis / vit-D deficiency + **levothyroxine** + cholecalciferol.
- **BG4** — complex renal-hepatic-transplant: anemia / ESRD / CKD / cirrhosis / HIV
  + **abnormal labs — calcium `[low]`, hemoglobin `[low]`, phosphate, low GFR** +
  heparin / vancomycin / hydromorphone.

Two things this settles:

1. **The multi-domain fusion's payoff shows up in the BACKGROUND, not the node
   blocks.** The measurement value-state tokenization *discriminates* here — BG4's
   `[low]` calcium/Hgb/phosphate is the renal-anemia signature, BG1's
   vitals-`[measured]` is the acute signature — where in the (starved) node topics
   measurement is `[normal]`-panel everywhere (0079/0080). Drugs likewise separate
   archetypes (BG3 levothyroxine, BG0 statins, BG4 vancomycin). So "measurement adds
   nothing" (a first-draft 0080 claim) is wrong: it adds nothing *in the starved
   node blocks*, but carries real signal *in the background*.

2. **The fitted model is ~8 rich background archetypes + a gated node overlay** that
   is catch-all at shallow depth, well-differentiated among fed siblings (0080), and
   starved below depth 5 (0079). The interpretable content is the background flavors
   + the shallow node increments; deep node-specific phenotypes are out of reach at
   the monolith.

## Interpretation & implications

- This is 0005 ("background decomposes into flavors") at whole-Mondo scale, and the
  multi-domain vocab makes the flavors *discriminative* rather than just
  condition-shaped. A genuine, if not novel, win — the shared/pooled representation
  works.
- **Do NOT reach for `n_bg` as a performance lever.** It is tempting to raise `n_bg`
  for more archetypes, but 0062 tested exactly that (n_bg 40→80) and it is NULL on
  detection — FP 13158→12992 (noise), FN 276→276 (identical), AUC flat — because the
  binding constraint is information, not capacity, and the information fix 0062 named
  (multi-domain features) is already in 0111. Raising `n_bg` is at most an
  *interpretability* lever (more background flavors to read), never a task one, and
  should be reached for only with that history in view.
- The honest deliverable framing: the phenotype product is the background archetypes
  + shallow node increments, on the calibrated per-node readout. Deep node-specific
  discovery needs the cascade (0071), not a monolith knob.

**Setting context:** exp 0111 episode arm — gated-PC `weight_y=0` (unsupervised
topics + separate readout), whole-Mondo native DAG (C=2714, K=2721, **n_bg=8**,
tpn=1), 3 domains (condition/measurement/drug), episode index (gap 90d, cap 3, 365d
label), lookback 1825d. Read from the fit's saved globals + bundle meta via
`inspect_topics.py`; matched-random control (0112) not yet fit.
