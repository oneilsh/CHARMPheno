---
id: 86
slug: mondo-snomed-coverage-probe
status: pending
model_class: anchor_select
# NOT a model fit — the expanded-SNOMED anchor-selection pipeline
# (analysis/cloud/anchor_selection_cloud.py; design spec
# 2026-07-31-expanded-snomed-anchor-selection-design.md) run as an exp. Maps the
# Monarch dismech #1079 seed (760 MONDO ids) -> OMOP standard-condition anchors via
# the faithful mondo2omop port, counts distinct persons per anchor, and (added here)
# prints a WHOLE-POPULATION COVERAGE LADDER: total / placeable-ceiling / placed-under-
# seed / healthy-residual / ceiling-seed gap. Sizes the Mondo-backbone redesign
# (detection = top-of-tree conditional) before we build it.
cohort: population_rare_priority   # vestigial (validation only; driver ignores it)
min_positives: 100                 # anchor power floor (spec default)
mondo_version: "2026-06-02"        # Mondo release the driver auto-downloads
mondo_cache_dir: "data/mondo"
# seed_tsv defaults to analysis/cloud/anchor_selection_data/priority_seed.tsv
---

# 0086 — Mondo/SNOMED anchor-selection + placement-coverage ladder

Reuses the existing on-cluster anchor-selection pipeline (cherry-picked from
`claude/hybrid-domain-reliability-review-ckn2bq`) rather than a hand-rolled probe.
The driver auto-downloads the Mondo release, so there is **no manual SSSOM staging** —
the one external dependency (MONDO↔OMOP) is handled by the faithful `mondo2omop` port.

Answers, before we build the redesign: **how many patients get placed on a real disease
node, and how big is the truly-healthy residual?** (See the use-case discussion:
detection = the top edge of one all-conditional tree.)

## What it produces

1. `candidates_with_counts.tsv` (+ `.mapping.tsv`, `.ancestry.tsv`) in the run dir:
   one row per OMOP anchor — MONDO ids/labels/#1079 categories, `positive_count`
   (distinct persons with ≥1 in-subtree condition), `clears_floor`, `is_maximal`.
2. A **coverage ladder** in the run log (`[coverage]` lines, teed to `summary.md`):

```
persons (total)                100.00%
has >=1 condition code         placeability CEILING (needs no Mondo; complement = no-code residual)
placed under the N seed anchors current priority-rare foreground
residual: no condition code    irreducible healthy floor
gap: ceiling - seed            placeable but NOT under the 760-disease seed -> whole-Mondo opportunity
```

## What to read

- **`gap: ceiling − seed`** is the headline: patients who have a codeable disease but
  are NOT captured by the 760 rare-disease seed — the population the whole-Mondo backbone
  would place (and today are dumped in the synthetic background). Large gap ⇒ the
  detection-as-top-of-tree redesign has real payoff.
- **healthy residual** (no condition code) sizes the irreducible "no-disease" class —
  the NOS/exclude decision.
- **Validate the mapping first:** the per-anchor counts must reproduce the known rare6
  numbers (SLE ~6500, MG ~1100, amyloidosis ~1500) — a regression check on the port
  before trusting new-anchor counts (spec §Testing).

## Scope note

This run is the **760-disease priority seed** (`restrict_mondo_ids=seed`). It validates
the mapping + gives the ceiling/residual, but "placed under seed" is only the rare-disease
foreground — the ceiling−seed gap is what **whole-Mondo** (`restrict_mondo_ids=None`) would
fill. Whole-Mondo is the follow-up once this validates (it needs the
`concept_relationship` filter switched from an IN-list to a join for scale — flagged, not
yet done).

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  make -C analysis/cloud exp ID=86
```

Copy/paste the `[coverage]` block + the top of `candidates_with_counts.tsv` back.

## Run log

### Run 1 (760-seed, min_positives=100) — the redesign premise is strongly validated: ~79% of coded patients are NOT placed by the rare-disease seed (the whole-Mondo opportunity)

CDR R2024Q3R8, 626,396 persons. Pipeline ran clean; mapping resolved the 760 MONDO seed
ids → **268 distinct OMOP standard-condition anchors**.

```
persons (total)                     626396   100.00%
has >=1 condition code              349815    55.85%   placeability CEILING
placed under the 268 seed anchors    73335    11.71%   current priority-rare foreground
residual: no condition code         276581    44.15%   irreducible healthy/undocumented floor
gap: ceiling - seed                 276480    44.14%   placeable but NOT under seed
```

- **The headline: 276,480 patients (44% of all; ~79% of the 349,815 *coded* patients)
  have a codeable disease but are NOT placed by the 760-disease rare seed.** These are
  exactly the common-disease patients today dumped in synthetic background. A whole-Mondo
  tree would give them real homes → the detection-as-top-of-tree redesign has large
  headroom, and it's the population that could clean up the FP wall (insight 0064) by
  routing confusable common-disease patients into their own branches.
- **44% have no condition code at all** — the irreducible "healthy/undocumented" floor.
  This is AoU-specific: a large fraction of participants are survey/PPI/genomic without
  linked (or with sparse) EHR. This is the residual the NOS/exclude decision applies to;
  it is bigger than a pure-EHR cohort's would be.
- **Rare-disease power problem, quantified:** of 268 anchors only **32 clear ≥100
  positives**, 30 survive nesting. Most priority rare diseases are simply too rare in AoU
  to fit their own node — the min-patient filter does the pruning the spec expected.
  (Note: `placed under seed` = ≥1 in-subtree code, no first-dx/lookback filter, so it
  overcounts vs a real fit's positive set.)
- **Neurodevelopmental washed out** (0 powered; Cardiac 22 / Neuroimmune 6 /
  Neurodegenerative 4) — the spec's predicted risk ("adult coding coverage") confirmed in
  adult AoU.

**Read:** whole-Mondo is worth building — the gap is the whole ballgame. Next: the
whole-Mondo run (`restrict_mondo_ids=None`) to see how much of that 276k the full disease
ontology actually places (and at what node granularity), which needs the
`concept_relationship` IN-list → join scale fix.

**Housekeeping:** the harness auto-ran NPMI eval and errored (`--model-class anchor_select`
not a choice) — non-fatal (results were saved first); fixed by adding `anchor_select` to
the self-contained skip-eval set. The tail `Could not find CoarseGrainedScheduler` is a
benign Spark shutdown race.

**Pending:** paste the top rows of `candidates_with_counts.tsv` to spot-check the mapping
against known rare6 counts (SLE ~6500, MG ~1100, amyloidosis ~1500).
