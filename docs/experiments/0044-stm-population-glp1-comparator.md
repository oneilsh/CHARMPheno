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
K: 110
background_k: 80
foreground: "glp1_ra:15,sglt2i:15"
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
one document per person, `source_cohort ∈ {glp1_ra, sglt2i, general}` — a clean
GLP-1-vs-SGLT2i active-comparator contrast:

- **glp1_ra / sglt2i** — incident new-users whose index year is monotherapy for
  that class (never the other, or the other started > 365d away); 365d prior
  coverage, 365d observed follow-up.
- **general** — no tracked drug exposure; random observed year with the SAME
  1yr-prior + 1yr-follow-up bracket.

A both-GLP1+SGLT2i user goes to the **earlier drug's single arm when the two
starts are > 365d apart** (the second drug is outside the index year, so that
year is genuine monotherapy) and is **excluded** otherwise (no combination arm).
**Tirzepatide (dual GIP/GLP-1) users are excluded entirely** — resolved (via its
own descendant set) only to keep them out of the GLP-1 arm and the background,
not fitted as an arm: at 128 docs it was too thin to model, and folding it into
glp1_ra would mix a dual-agonist into the pure-GLP-1-RA contrast.

Drug classes are resolved as `concept_ancestor` **descendants** of their seed
ingredients (RxNorm Ingredient names + any pinned ids; **tirzepatide pinned to
779705** — name-only resolution under-counted it to 128). The build logs each
class's resolved concept-set size + person count. Documents are the person's
conditions in the post-index year; drugs are the anchor only.

**Cache note:** the partition + concept-set logic is NOT part of the corpus cache
key, so a rebuild after these changes must be FORCED (clear the cache entry / use
the force flag) or the stale v1 corpus is reused.

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
