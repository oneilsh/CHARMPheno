# 0070 — Whole-Mondo places 97.9% of coded AoU patients via ancestor roll-up (despite 8.2% direct SNOMED mapping); the unplaced 1.18% are HPO-domain symptoms, not mapping gaps

**Date:** 2026-08-17
**Topic:** case-finding, ontology, mondo, snomed, hpo, cohort-definition

**Status:** Confirmed on exp 0087 (whole-Mondo → OMOP, AoU CDR R2024Q3R8, 626,396 persons)

The Mondo-backbone redesign (detection = top edge of one all-conditional tree) needs the
disease ontology to actually reach the coded population. Exp 0087 mapped the WHOLE Mondo
disease ontology → OMOP standard Condition and rolled every patient's condition codes up
to it. It does, decisively.

**Direct SNOMED mapping is sparse but that's a red herring.** Of 112,051 standard SNOMED
Condition concepts, only **9,164 (8.2%) carry a direct Mondo xref**. But SNOMED's 112k are
mostly ultra-granular variants ("Pain of left knee region", "Laceration of thumb") that
**roll up** the `concept_ancestor` graph to a mapped disease ancestor. The SNOMED-climb
placement (a patient lands on Mondo if ANY ancestor of any of their codes is mapped) is
what matters, and it achieves near-complete coverage from the sparse direct map.

**Whole-Mondo places 97.88% of coded patients.** Ladder over 626,396 persons: 55.85% have
≥1 condition code (ceiling); **54.66% place on some Mondo node** (= 97.88% of the coded);
only **1.18% are coded-but-unplaced**; 44.15% have no code at all. (The 760-rare-seed
placed 11.71%, exp 0086 — so whole-Mondo is the lever, not a bigger rare list.) Mondo is a
near-complete disease backbone for AoU's coded population.

**The 1.18% unplaced are CORRECTLY unplaced — they are HPO-domain phenotypes, not
diseases.** The top unplaced condition concepts are almost entirely the SNOMED "Finding"
branch — signs/symptoms/observations: "Finding relating to sexual activity" (n=3319),
Chronic pain, Low back pain, Fatigue, Chest pain, Cough, Dyspnea, Palpitations, Tachycardia
— plus non-clinical codes ("Clinical finding" root, "Worried well", "Postoperative state",
"Spiritual beliefs conflicting with healthcare plan"). Mondo is a DISEASE ontology and
correctly excludes phenotypes; the Human Phenotype Ontology (HPO) is their home. These are
patients coded with symptoms but no disease.

**Two consequences for the redesign:**

1. **Feature vs label — no signal is lost.** The finding codes are Condition-domain, so
   they remain **input tokens** in the corpus (they feed θ); they simply don't get a
   disease *node* in the label DAG. Symptoms inform the representation; they don't define
   an outcome. Both correct — nothing to fix.

2. **The residual is TWO classes, and the smaller one is the target.** Split the
   "no disease dx" residual into **44.15% no-code** (truly healthy/undocumented) vs
   **1.18% symptom-only** (symptomatic-UNDIAGNOSED). The symptom-only group is not noise —
   it is arguably the case-finding TARGET: symptoms present, disease not yet coded, exactly
   who a diagnostic-suggestion model should score. A cheap, meaningful cohort-definition
   distinction for the top-level class, not new modeling.

A handful of genuine diseases fell through (Inguinal hernia n=43, Prediabetes n=24) — minor
Mondo xref gaps, immaterial at this coverage.

**Implication.** Whole-Mondo is validated as the disease backbone. HPO is the correct (and
already-used-as-features) home for the symptom fall-through; the only refinement worth
making is separating healthy-no-code from symptomatic-undiagnosed in the cohort definition.
No HPO-as-node build is needed for the disease-diagnosis backbone.
