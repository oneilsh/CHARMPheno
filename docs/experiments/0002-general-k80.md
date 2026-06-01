---
id: 2
slug: general-k80
status: pending
model_class: lda
cohort: general
created: 2026-06-01
K: 80
---

# Experiment 0002 — general-k80

## Intent
First full-scale LDA fit on the **general** (whole-population) cohort at K=80.
Follows the pilot (0001), which validated the experiment-tracking pipeline at a
tiny K; this is the first real model size on the unfiltered population.

Only `K` is overridden in frontmatter — everything else inherits the standard
config chain (`_base.yaml` → `general.yaml`), i.e. the same learning rates and
shapes we've been using:

- `cohort_def: none` — no cohort filter (whole population)
- `doc_unit: patient_year`, `doc_min_length: 20` — one document per patient-year (1-year windows)
- `vocab_size: 10000`, `min_df: 20`, `min_patient_count: 20`
- online variational LDA learning rates: `subsampling_rate: 0.2`, `tau0: 64`, `kappa: 0.7`
- `max_iter: 20` — iterations to run *this call* (not cumulative; resumable by
  re-running `make exp ID=2`), `optimize_doc_concentration: true`
- `seed: 42`

## Fit history
- (pending first run)
