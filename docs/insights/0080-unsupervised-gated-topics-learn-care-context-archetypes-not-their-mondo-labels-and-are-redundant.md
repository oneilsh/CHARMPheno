# 0080 — Unsupervised (weight_y=0) gated topics learn coherent care-context archetypes, not their Mondo disease labels, and are redundant

**Date:** 2026-09-05
**Topic:** lda
**Status:** Observed

## Observation

Reading the 0111 episode arm's well-fed shallow node topics (depths 1–4, the
only fed ones — insight 0079) via the `inspect_topics.py` tree tour, all
domains shown, the fed topics are **coherent, interpretable multi-domain
clusters** — and the fused vocabulary works: conditions, labs, and drugs cohere
within each topic.

- `d2` *disease by body system or component* → a **cardiometabolic** cluster:
  T2DM / hypertension / hyperlipidemia / CKD-3 / coronary atherosclerosis +
  LDL/cholesterol/triglyceride labs + metformin / atorvastatin / insulin
  glargine / lisinopril / amlodipine.
- `d2` *disease by etiologic mechanism* → an **acute-care / ED** cluster:
  pain / nausea / vomiting / bleeding + vitals + fentanyl / midazolam / propofol
  / IV fluids / naloxone.
- `d3` *inflammatory disease* → **outpatient respiratory-infection / allergy**:
  URI / asthma / sinusitis / allergic rhinitis + CBC + amoxicillin / albuterol /
  azithromycin / fluticasone.
- `d3` *nervous system disorder* → **psychiatric / substance / HIV**:
  depression / bipolar / PTSD / tobacco dependence / HIV + lorazepam / trazodone
  / nicotine / quetiapine.

Three failure modes accompany the coherence:

1. **The topic content does not match the node's Mondo label.** A node named
   *inflammatory disease* learned a respiratory-infection topic; *nervous system
   disorder* learned a psychiatric one; *infectious disease* and *central
   nervous system disorder* learned the **same** generic acute-care topic. The
   block captures the care-context / comorbidity signature of the patient
   population gated to the node, not the node's disease.
2. **Redundancy.** ≥3 shallow nodes re-learned the same acute-care archetype
   (IV fluids + opioids + antiemetics + vitals). The effective number of
   distinct topics is a handful (~5–8), not the 182 fed nodes.
3. **Measurement is `[normal]`-panel-dominated** across nearly every topic and
   all 8 background topics (evidence 4–7e6, measurement-dominant). The
   value-state tokenization adds little discrimination beyond "labs were drawn."

## Interpretation

This arm is **`weight_y = 0` — the topics are UNSUPERVISED** (the label does not
pull the topic during CAVI; the readout is a separate downstream fit). So the
topics have no incentive to align with the node's disease; they model raw
co-occurrence in the gated documents, which is dominated by care context. And
Mondo's top-level ontological partition (*by body system / by etiologic
mechanism / inflammatory / nervous-system*) does not align with EHR
co-occurrence structure, so the topics learn the EHR-natural clusters (care
archetypes) and drape them over whichever nodes the closure gating routes
documents through. Combined with 0079's depth-5 starvation, the fitted model is
effectively a ~5–8-archetype care-context model wearing a 2714-node
hierarchical mask.

## Implications

- This is exactly what prediction-constrained fitting (`weight_y > 0`) exists to
  correct — pull each topic toward discriminating its label. This run is the
  honest **unsupervised baseline**; a `weight_y > 0` arm is the natural next
  probe of whether supervision recovers disease-specific shallow topics.
- Supervision cannot rescue 0079's starvation, though: it can sharpen a
  low-token topic but not invent signal where a deep node contributes no
  distinctive tokens. The two findings are separable — 0080 is about the fed
  shallow topics' *alignment*, 0079 about the deep topics' *existence*.
- Per-node topic-word interpretation is misleading under `weight_y = 0`: the
  words describe the node's patient population, not its disease. `inspect_topics`
  reports this honestly (the label↔content mismatch is visible in the tour).
- If interpretable disease phenotypes are the goal, candidate directions
  (each its own experiment/ADR): `weight_y > 0` supervision; an
  empirically-derived scaffold (co-occurrence clustering) rather than the Mondo
  ontology; or accepting care-context archetypes as the honest unsupervised
  output.

**Setting context:** exp 0111 episode arm — gated-PC **weight_y = 0**
(unsupervised topics + separate L-BFGS readout, `skip_unsup_gated: true`),
whole-Mondo native DAG (C=2714, K=2721, n_bg=8, tpn=1), 3 domains
(condition/measurement/drug), episode index (gap 90d, cap 3, 365d label),
lookback 1825d, doc_concentration 0.5. Read from the fit's saved globals +
bundle meta via the `inspect_topics.py --tour` tree tour; matched-random control
(0112) not yet fit.
