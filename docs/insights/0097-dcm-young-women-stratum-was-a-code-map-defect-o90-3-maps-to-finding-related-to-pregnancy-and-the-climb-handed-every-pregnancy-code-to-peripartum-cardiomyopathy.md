# 0097 — The DCM "young-women stratum" was a code-map defect: ICD-10-CM O90.3 maps to "Finding related to pregnancy", and the climb handed every pregnancy code to peripartum cardiomyopathy

**Date:** 2026-10-09
**Topic:** data-quality, code-map, mondo-native, multi-map, peripartum-cardiomyopathy, same-vocabulary-child, exp 0135, exp 0136
**Status:** Mechanism confirmed against the vocabulary (OMOP `Maps to`, Mondo 2026-06-02 `same_as`) and the 0135 code map; the fix (native-mondo-v2) is built and unit-tested, not yet re-run.

## Observation

1. **Peripartum cardiomyopathy (MONDO:0018920) was attested by 106 standard codes, all
   pregnancy findings** — "Finding related to pregnancy", every "Gestation period, N
   weeks", the trimesters, primigravida, high-risk and unplanned pregnancy (0135
   `code_map.tsv`). Its block's three topics were pregnancy and its evidence was four
   times DCM's.
2. **The chain.** Mondo's `same_as` for the disorder includes `ICD10CM:O90.3`. OMOP maps
   O90.3 to TWO standard concepts: *Dilated peripartum cardiomyopathy* (4037495) and
   *Finding related to pregnancy* (444094). The exact rung therefore made 444094 a
   standard concept OF the disorder, and the climb rung (nearest mapped standard
   ancestor, for codes with no exact term) sent the whole pregnancy subtree to it.
   Every pregnant woman in the data carried a peripartum-cardiomyopathy label.
3. **That is the "same-vocabulary child".** On every nested native-Mondo branch run
   (0127–0134) peripartum CM sat under DCM, so DCM's closure cohort contained every
   pregnant woman and DCM's block grew the young-women / pregnancy stratum that
   insights 0094–0095 read as forward deflation. Insight 0096 credited the flat ribbon
   with "rescuing" DCM; what the flat fit plus R1a actually did was move the pregnancy
   documents off DCM onto the mislabelled peripartum node. Flattening did not fix the
   defect; it relocated it.

4. **This was already known on `main`.** Main's insight 0076 (2026-08-24, a different
   numbering line from this branch's 0076) found exactly this case in the whole-Mondo
   usage export: peripartum cardiomyopathy at 24,629 patients, 444094 shared with
   preeclampsia and severe pre-eclampsia, "ICD `Maps to` decomposition inflates
   standard-space exact counts". It recommended source-space counting for the usage
   report and kept standard space "for the ontology-gated modeling" — which is the path
   that never got a guard. The dashboard's `source_climb` default credits a
   source-exact `condition_source_concept_id` first, but its standard-exact and climb
   rungs still reach 444094, so the usage dashboard shows the same inflation.

## Why it matters

- **Scope of the contamination, honestly.** The native-Mondo code map (exp 0110 on) is
  affected wherever an xref'd source code multi-maps to a broad context concept that
  the climb can reach. The DCM / peripartum narrative of 0124–0134 and insight 0096's
  mechanism claim are wrong as stated. What is NOT explained by this defect: starvation
  and deflation measured over hundreds of nodes (insight 0091 and its predecessors),
  the α results, the spectral-init result, and every readout-instrument finding
  (ridge, convergence, top-m, the closure mask, the stacked head). The SNOMED-path runs
  before 0110 used a different attestation (SNOMED descendants of anchors) and do not
  go through this rung.
- **Other nodes are suspect until audited.** Attesting-code counts on 0135 include
  osteochondrosis 211, thrombophilia 162, colorectal cancer 132, lymphoma 122, uveitis
  115, preeclampsia 111, tularemia 93 and chorioamnionitis 62. Many will be legitimate
  (a family of specific codes), but tularemia at 93 codes is not plausible, and the
  pregnancy disorders share peripartum's shape.
- **The fix (`mondo_native_dag.drop_ancestor_multimap_targets`, native-mondo-v2):**
  per source code, drop any `Maps to` target that is an is-a ancestor of another target
  of the same source code. It removes 444094 from O90.3 and keeps the disorder; it is
  general (every ICD-10-CM "disorder + context finding" multi-map), and the powering
  receipt now prints how many source codes multi-map and how many targets were dropped.
  A second test from main's 0076 (FAN-IN): a multi-map target that is the exact
  concept of two Mondo terms unrelated by is-a (444094 <- peripartum CM, preeclampsia)
  is dropped from that source code too, so the guard does not depend on SNOMED placing
  the context finding above the disorder.
  It does not catch a broad concept that is a source code's ONLY target; the
  attesting-code count per node (0136's audit list) is the check for that.
- **Process lesson.** The census's own-code mark (†) is what exposed this: a topic whose
  "own codes" are generic is a code-map question before it is a model question. Every
  label-space change should be followed by the per-node attesting-code audit before
  any topic is read.

## Addendum (2026-10-09, 0137 launch-1 audit): single broad targets, and the anchor test

The v2 guard fired narrowly (8 links, 2 concepts: Disorder of pregnancy, Finding related
to pregnancy) and the pregnancy captures were gone, but the audit's most-attested list
exposed the same defect WITHOUT a multi-map: tularemia's 93 codes were "Disorder of
gastrointestinal tract" and its subtree, thrombophilia's 162 were every deep-vein-
thrombosis code, osteochondrosis's 211 were spine fractures and scoliosis — an ICD xref
whose ONLY `Maps to` target is a broad concept, filled by the climb. v3 adds main's
anchor test (`anchor_corroborated_rows`): for a term with a SNOMED `same_as`, a target
from another vocabulary is kept only if it is that concept, its descendant, or agreed
by >= 2 of the term's source codes. 0137 launch 1 was stopped for it.

**v3 was withdrawn on its own receipts (0137 launch 2).** The anchor test dropped 151
targets over 147 terms and with them dilated cardiomyopathy, Down syndrome, Lyme disease,
giardiasis, coccidioidomycosis, atrial septal defect, IPAH and MALT lymphoma from the
label set (their main ICD code maps to a sibling or synonym of their SNOMED xref, not a
descendant), while tularemia (93) and thrombophilia (165) were untouched — those captures
arrive through Mondo DESCENDANTS of the ribbon member that have no SNOMED xref, whose
broad ICD target rolls up. **v4 replaces it with a hierarchy-consistency test**
(`overbroad_exact_rows`): a non-SNOMED target is dropped when SNOMED places it above the
exact concepts of >= 3 Mondo terms that Mondo does not place under the term. It asks the
question the captures get wrong (is this concept broader than the disease, by the disease
hierarchy's own account?) and spares DCM (its concept subsumes Mondo's own DCM
children). SNOMED `same_as` rows are never dropped.

## Reproduce

```bash
# the vocabulary side (no patient data)
bq query --use_legacy_sql=false "SELECT c1.vocabulary_id, c1.concept_code, c2.concept_id, c2.concept_name FROM \`${WORKSPACE_CDR}.concept_relationship\` r JOIN \`${WORKSPACE_CDR}.concept\` c1 ON c1.concept_id=r.concept_id_1 JOIN \`${WORKSPACE_CDR}.concept\` c2 ON c2.concept_id=r.concept_id_2 WHERE r.relationship_id='Maps to' AND c1.vocabulary_id='ICD10CM' AND c1.concept_code='O90.3'"
```
