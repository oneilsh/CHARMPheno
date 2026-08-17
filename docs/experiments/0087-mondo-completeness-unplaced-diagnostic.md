---
id: 87
slug: mondo-completeness-unplaced-diagnostic
status: pending
model_class: mondo_completeness
# NOT a fit — whole-Mondo mapping-completeness + unplaced-condition diagnostic
# (analysis/cloud/mondo_completeness_cloud.py). Follows exp 0086 (760-seed ladder,
# which showed ~79% of coded patients unplaced by the RARE seed). This maps ALL of
# Mondo (restrict=None, scale-fixed with broadcast joins) and asks: how much of the
# SNOMED disease space does Mondo cover, how many patients land on ANY Mondo node,
# and — for those who still don't — WHAT conditions are they coded with?
cohort: population_rare_priority   # vestigial (validation only; driver ignores it)
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
top_unplaced: 100                  # how many top unplaced condition concepts to list
---

# 0087 — Whole-Mondo completeness + "what falls through" diagnostic

Maps the **whole** Mondo disease ontology → OMOP standard Condition concepts (the
faithful `mondo2omop` port, `restrict_mondo_ids=None`), then reports three things.
All patient counts are **AoU small-cell suppressed**: any count in (0, 20) prints as
`<20`.

## What it reports

1. **`[mapping]` completeness** — Mondo disease terms with an OMOP-resolvable xref →
   matched OMOP source concepts → distinct standard Condition anchors; and the
   **SNOMED-condition-space coverage** (of all standard SNOMED Condition concepts, how
   many are directly Mondo-mapped).
2. **`[coverage]` ladder** (whole-Mondo) — total / has-code (ceiling) / **placed on ANY
   Mondo node** (via the SNOMED-climb roll-up) / **coded-but-unplaced** / no-code floor.
3. **`[unplaced]` diagnostic** — for the coded-but-unplaced patients, the top
   `top_unplaced` condition concepts they carry (patient count, concept_id, domain,
   standard flag, name). This is *what falls through*: SNOMED disease concepts with no
   Mondo mapping, plus non-disease codes miscoded as conditions.

Artifacts in the run dir: `mondo_omop_mapping.tsv` (the full mapping) and
`unplaced_top_conditions.tsv` (suppressed).

## What to read

- **`placed on ANY Mondo node` vs 0086's `has >=1 condition code` (55.85%)** — how much
  of the codeable population Mondo actually reaches. Close to the ceiling ⇒ Mondo is a
  near-complete backbone; a big gap ⇒ real mapping holes.
- **The `[unplaced]` table is the actionable output.** If the top rows are *real diseases*
  (e.g. common conditions with no Mondo xref), that's a mapping gap to fix / a reason to
  supplement Mondo. If they're *findings/symptoms/administrative* codes, that's the
  genuine non-disease residual (correctly unplaced). This tells us whether the 44%
  no-Mondo gap from 0086 is fixable coverage or irreducible non-disease.

## Scale note

Whole-Mondo pulls ~10^4 source concepts and ~10^4 target anchors — the driver uses
broadcast JOINS (not IN-lists) for the `concept_relationship` and `concept_ancestor`
filters. The `concept` SNOMED/ICD/MeSH slice is collected to the driver as pandas (same
pattern anchor_selection_cloud already runs).

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  make -C analysis/cloud exp ID=87
```

Copy/paste the `[mapping]`, `[coverage]`, and `[unplaced]` blocks back.

## Run log

### Run 1 (whole-Mondo, v2026-06-02) — near-complete disease backbone: 97.9% of coded patients placed; the 1.18% unplaced are HPO-domain symptoms/findings, NOT mapping gaps

CDR R2024Q3R8, 626,396 persons. Whole-Mondo mapping resolved cleanly.

**Mapping:** 12,239 Mondo disease terms with an OMOP xref → 10,807 matched source
concepts → **9,164 distinct OMOP standard Condition anchors**. Of 112,051 standard SNOMED
Condition concepts, only **9,164 (8.2%) are DIRECTLY Mondo-mapped** — but that 8.2% is a
red herring: SNOMED's 112k are mostly ultra-granular variants ("Pain of left knee region")
that **roll UP** to a mapped disease ancestor. The number that matters is patient placement.

**Coverage ladder:**
```
persons (total)               626396  100.00%
has >=1 condition code         349815   55.85%   ceiling
placed on ANY Mondo node       342394   54.66%   = 97.88% of CODED patients
coded but UNPLACED               7421    1.18%
residual: no condition code    276581   44.15%   healthy/undocumented floor
```
Whole-Mondo places **97.88% of coded patients** (vs the 760-rare-seed's 11.71%, exp 0086).
Mondo (via the SNOMED-climb roll-up) is a near-complete disease backbone for AoU's coded
population — the detection-as-top-of-tree redesign is on solid ontological ground.

**The 1.18% unplaced are correctly unplaced — they're PHENOTYPES, not diseases.** The top
unplaced conditions are almost all SNOMED "Finding" branch, i.e. HPO-domain signs/symptoms:
"Finding relating to sexual activity" (3319), Chronic pain (745), Low back pain, Fatigue,
Chest pain, Cough, Dyspnea, Palpitations, Dizziness, Tachycardia, ... plus non-clinical
codes ("Clinical finding" root itself, "Worried well", "Spiritual beliefs conflicting with
healthcare plan", "Postoperative state"). Mondo is a DISEASE ontology and correctly
excludes these. They are patients coded with SYMPTOMS but no disease.

- **No signal is lost:** these finding codes are Condition-domain and remain **input
  tokens** in the corpus (they feed θ); they simply don't get a disease *node* in the
  label DAG. Feature vs label — correct on both.
- **The residual is two distinct classes, not one:** 44.15% no-code (truly
  healthy/undocumented) vs 1.18% **symptom-only (symptomatic-UNDIAGNOSED)**. The latter is
  not noise — it is arguably the case-finding TARGET (symptoms present, disease not yet
  coded). A cheap, meaningful cohort-definition split for the redesign's top-level "no
  disease dx" class.
- **A handful of genuine diseases fell through** (Inguinal hernia n=43, Prediabetes n=24) —
  minor Mondo xref gaps, not worth chasing at this coverage.

**Read:** whole-Mondo is validated as the disease backbone (97.9% coded placement). HPO
phenotypes are the correct home for the fall-through, and they're already used as features;
the only refinement worth making is splitting healthy-no-code from symptomatic-undiagnosed.
Recorded as insight 0070.
