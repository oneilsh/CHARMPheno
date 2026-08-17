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

_(pending)_
