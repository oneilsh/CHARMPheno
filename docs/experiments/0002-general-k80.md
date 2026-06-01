---
id: 2
slug: general-k80
status: pending
model_class: lda
cohort: general
created: 2026-06-01
K: 80
max_iter: 100
print_topics_every: 10
---

# Experiment 0002 — general-k80

## Intent
First full-scale LDA fit on the **general** (whole-population) cohort at K=80.
Follows the pilot (0001), which validated the experiment-tracking pipeline at a
tiny K; this is the first real model size on the unfiltered population.

Frontmatter overrides `K`, `max_iter`, and `print_topics_every`; everything
else inherits the standard config chain (`_base.yaml` → `general.yaml`), i.e.
the same learning rates and shapes we've been using:

- `cohort_def: none` — no cohort filter (whole population)
- `doc_unit: patient_year`, `doc_min_length: 20` — one document per patient-year (1-year windows)
- `vocab_size: 10000`, `min_df: 20`, `min_patient_count: 20`
- online variational LDA learning rates: `subsampling_rate: 0.2`, `tau0: 64`, `kappa: 0.7`
- `max_iter: 100` — per-fit ceiling (not cumulative); early-stops at
  `convergence_tol: 1e-4` relative ELBO improvement. Resumable: re-run
  `make exp ID=2` to continue from the latest checkpoint.
- `print_topics_every: 10` — print the full topic dump every 10th iter (K=80 ×
  every iter floods the summary).
- `optimize_doc_concentration: true`, `seed: 42`

## Fit history
- 2026-06-01 — **Session 1** (`max_iter: 20`): ran the full 20-iter ceiling
  without converging — ELBO still descending steeply (−13.20M → −10.77M →
  −10.40M over iters 1–3). Bumped `max_iter` 20→100 and `print_topics_every`
  1→10; resume with `make exp ID=2` from the iter-20 checkpoint.
