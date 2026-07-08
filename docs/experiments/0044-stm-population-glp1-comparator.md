---
id: 44
slug: stm-population-glp1-comparator
status: pending
model_class: stm
cohort: population_glp1
cohort_def: population_glp1
prior_obs_days: 365
person_mod: 1
doc_unit: patient_cohort
doc_min_length: 10
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
known_sex_only: true
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 140
background_k: 80
foreground: "glp1_ra:15,sglt2i:15,tirzepatide:15,glp1_sglt2_combo:15"
group_var: source_cohort
max_iter: 300
subsampling_rate: 0.1
tau0: 256
kappa: 0.7
sigma_init: 1.0
reference_topic: true
spectral_init: true
spectral_method: dense
min_pair_support: 10
---

# Experiment 0044 — Population + GLP-1 & active-comparator gated STM

## Goal

What does the year after starting a GLP-1 receptor agonist look like — against
the general population AND against an active comparator (SGLT2 inhibitors)
started by a similar patient population — in one gated fit? The gated Σ yields
GLP-1↔comparator and GLP-1↔background topic (anti-)correlations. Contrasting
against a same-indication comparator controls for confounding-by-indication, so
what separates the drug foreground blocks is closer to drug-specific structure.

## Cohort

New drug-anchored `population_glp1` cohort (`apply_population_drug_cohort`),
one document per person, `source_cohort ∈ {glp1_ra, sglt2i, tirzepatide,
glp1_sglt2_combo, general}`:

- **glp1_ra / sglt2i** — incident new-users of that class only (never the other
  tracked classes); 365d prior coverage, 365d observed follow-up.
- **tirzepatide** — new-users of tirzepatide (dual GIP/GLP-1; precedence over
  the other arms).
- **glp1_sglt2_combo** — new-users of both a GLP-1 RA and an SGLT2i co-initiated
  within `combo_max_gap_days` (code default 90; set from the build-time gap
  histogram). Non-combo both-users are excluded from the cohort.
- **general** — no tracked drug exposure; random observed year with the SAME
  1yr-prior + 1yr-follow-up bracket.

Documents are the person's conditions in the post-index year; drugs are the
anchor only.

## Configuration

Full population (`person_mod: 1`) — the thin tirzepatide/combo arms need it.
K=140 = 80 background + 15 × 4 foreground. Incident new-user (`prior_obs_days:
365`), 1-year windows. Otherwise the exp 0043 hardened + slowed stack (subsample
0.1, tau0 256, kappa 0.7, max_iter 300, reference + dense spectral, sigma_init 1,
min_pair_support 10, block-wise unit-diagonal Σ / ADR 0034), `~ C(sex) + age`,
`known_sex_only`.

## Success criteria

- Build diagnostics report non-empty per-arm document counts (watch tirzepatide +
  combo); the co-initiation gap histogram is logged so `combo_max_gap_days` can
  be re-cut.
- Drug foreground arms recover recognizable, distinctive structure (e.g. GI /
  weight / appetite signal for GLP-1; genitourinary / volume for SGLT2i) and
  interpretable GLP-1↔background anti-correlations.
- Σ variance bounded (no runaway); honest correlation report.

## Related

First cohort on the drug-anchor track (parallel to the disease track). Follows
exp 0043 (population_eds) on the same hardened + slowed stack.
