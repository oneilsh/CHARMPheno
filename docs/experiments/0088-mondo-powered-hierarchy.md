---
id: 88
slug: mondo-powered-hierarchy
status: pending
model_class: mondo_hierarchy
# NOT a fit — builds the whole-Mondo POWERED label DAG (analysis/cloud/
# mondo_hierarchy_cloud.py) and reports its size / implied K. Follows 0087
# (whole-Mondo places 97.9% of coded patients). Map whole Mondo -> OMOP anchors,
# power-count each (distinct persons/subtree), keep those clearing the min-patient
# floor, reduce the Mondo is-a DAG over the powered anchors to its compact branch-
# point hierarchy, and report powered-nodes + class-nodes => K.
cohort: population_rare_priority   # vestigial (validation only; driver ignores it)
min_positives: 100                 # the min-patient floor (a node no one populates has no topic)
min_class_size: 2                  # a class node must group >=2 powered anchors
max_class_fraction: 1.0
tpn: 1
mondo_version: "2026-06-02"
mondo_cache_dir: "data/mondo"
---

# 0088 — The whole-Mondo powered hierarchy (the DAG a full-Mondo fit would use)

Turns the validated whole-Mondo backbone (0087) into the **actual label DAG** and sizes
it. This is the "how big is the model" answer before committing to a whole-Mondo fit.

## What it does

1. **Map** whole Mondo → OMOP standard Condition anchors (9,164 in 0087; faithful
   `mondo2omop`, `restrict_mondo_ids=None`, broadcast-join scale fix).
2. **Power-count** each anchor: distinct persons with ≥1 in-subtree condition (the same
   proxy anchor_selection uses; a real fit's first-dx/lookback filter only shrinks these).
3. **Filter** to anchors clearing `min_positives` — the min-patient floor.
4. **Reduce** the Mondo is-a DAG over the powered anchors to the compact **branch-point**
   hierarchy (`reduce_to_anchor_hierarchy`): keeps only class nodes that are the common
   ancestor of ≥`min_class_size` powered anchors (O(#anchors), not the raw closure).
5. **Report** `K = n_bg + (powered_anchors + class_nodes) × tpn`.

All patient counts small-cell suppressed (<20 → `<20`). Artifact:
`mondo_powered_hierarchy.tsv` (a ready DagLayout `parent_of`).

## What to read

- **`powered (>= floor)`** — how many Mondo anchors actually have the data to be a node.
  (0086 saw only 32/268 of the *rare* seed clear 100; whole-Mondo includes common
  diseases, so expect far more — this is the real node count.)
- **`compact class nodes kept`** — the intermediate Mondo classes (e.g. "cardiovascular
  disorder") that give the tree its depth for conditional sharpening.
- **`K at n_bg=…`** — the implied topic count. This is the go/no-go on compute: if K is a
  few thousand it's a big-but-buildable fit; if it's 10k+ we cap depth / raise the floor.
- The top class nodes by size show whether the branch points are clinically sensible
  umbrellas (the sharpening structure).

## Run

```bash
cd ~/repos/CHARMPheno && git pull origin claude/spectral-anchor-topic-k-200nqp && \
  make -C analysis/cloud exp ID=88
```

Sweep the floor via `min_positives` in the frontmatter if K comes back too large.
Copy/paste the `WHOLE-MONDO POWERED HIERARCHY` block + the top `[class]` lines back.

## Run log

### Run 1 (floor=100) — K≈3,800; clean clinical class tree (Mondo vindicated); too big for ONE co-fit-Newton fit → stage by body-system branch

9,164 mapped anchors → **2,513 powered** (≥100 positives) + **1,306 compact class nodes**
= **3,819 layout nodes, K≈3,821–3,827**. (3,000 raw Mondo ancestors avoided by the
branch-point reduction.)

- **The class tree is clinically excellent — Mondo delivered exactly the umbrellas we
  wanted.** The branch points read as real clinical categories: cardiovascular disorder
  (273), nervous system disorder (454), respiratory (170), endocrine (181), immune (180),
  psychiatric (126), hematologic (125), musculoskeletal (225), cancer/neoplasm (469/446/
  444/205), infectious (231), connective tissue (95), heart disorder (100)… This is the
  conditional-sharpening structure, and it's *clean* — vindicating "rely on Mondo, less on
  SNOMED" (SNOMED's ragged graph gave us the SLE→SLE artifacts in 0085; Mondo gives
  "cardiovascular disorder → …").
- **A few classes are abstract Mondo meta-groupings**, not clinical umbrellas: "disease by
  body system or component" (2216), "disease by etiologic mechanism" (995), "disease by
  developmental or physiological process" (746). These sit near the root and would be the
  top of the tree; `max_class_fraction < 1.0` drops such over-general classes.
- **K≈3,800 is ~38× our biggest fit (K=101, exp 0081).** As ONE fit it is NOT viable —
  specifically the **co-fit ridge-Newton head** (ADR 0043's unified head) is
  **O(K³·C) compute and O(C·K²) memory**: the per-node Fisher stack alone is C·K·K ≈
  3819³ floats ≈ **~850 GB**. The topic model itself and the two-stage readout LR would
  scale (the readout is just C logistic fits on K features), but the *unified Newton head*
  does not. So whole-Mondo forces either two-stage-only or staging.
- **The tree decomposes NATURALLY by body system.** The ~20–30 top-level classes
  (cardiovascular, nervous, respiratory, …) each carry ~100–500 nodes — a buildable sub-fit
  at the K~few-hundred scale we already run. This is exactly the **detection-at-top ×
  per-branch-conditional cascade** we've been designing toward — now motivated by *compute*,
  not just concept: fit a top-level "which body system" detector + per-system conditional
  models, compose at inference.

**Read:** whole-Mondo is a clean, buildable backbone — but as a **staged cascade** (top-level
system detector × per-branch conditional fits), not a single K≈3,800 model. Recorded as
insight 0071. Levers if a single fit is ever wanted: raise `min_positives` (drops rare
nodes) + lower `max_class_fraction` (drops the abstract "disease by body system" top).
