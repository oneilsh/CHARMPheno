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
  It does not catch a broad concept that is a source code's ONLY target; the
  attesting-code count per node (0136's audit list) is the check for that.
- **Process lesson.** The census's own-code mark (†) is what exposed this: a topic whose
  "own codes" are generic is a code-map question before it is a model question. Every
  label-space change should be followed by the per-node attesting-code audit before
  any topic is read.

## Reproduce

```bash
# the vocabulary side (no patient data)
bq query --use_legacy_sql=false "SELECT c1.vocabulary_id, c1.concept_code, c2.concept_id, c2.concept_name FROM \`${WORKSPACE_CDR}.concept_relationship\` r JOIN \`${WORKSPACE_CDR}.concept\` c1 ON c1.concept_id=r.concept_id_1 JOIN \`${WORKSPACE_CDR}.concept\` c2 ON c2.concept_id=r.concept_id_2 WHERE r.relationship_id='Maps to' AND c1.vocabulary_id='ICD10CM' AND c1.concept_code='O90.3'"
```
