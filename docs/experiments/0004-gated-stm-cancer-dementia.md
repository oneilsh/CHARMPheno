---
id: 4
slug: gated-stm-cancer-dementia
status: pending
model_class: stm
cohort: cancer_or_dementia
cohort_def: cancer_or_dementia
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
covariate_formula: "~ C(sex) + age"
categorical_cols: [sex]
continuous_cols: [age]
random_seed: 42
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
K: 50
background_k: 30
foreground: "cancer:10,dementia:10"
group_var: source_cohort
max_iter: 100
---

# Gated STM validation: cancer/dementia cohort with background + foreground blocks

Validates that the gated-STM partitioning surfaces a dementia-distinctive
phenotype that prevalence-only STM (experiment 0003) could not. In prevalence-
only STM, the topic-word matrix beta is shared across all documents and
estimated by pooled SVI weighted by token mass. Because cancer represents ~79%
of documents, dementia's tokens reinforce the shared anchor topics rather than
carving distinct vocabulary clusters; the `source_cohort` covariate can only
re-weight how much of those shared topics a dementia document expresses, not
create dementia-specific content (insight [0026](../insights/0026-stm-prevalence-gives-prevalence-not-content-fidelity.md)).

The gating mechanism (ADR [0026](../decisions/0026-gated-stm-hard-masking.md))
addresses this through a hard block partition:

- **30 background topics** are shared by all documents (cancer and dementia),
  estimated via the full pooled token mass. These capture the universal anchor
  phenotypes (comorbidity patterns, chronic disease clusters) that insight 0026
  showed are corpus-invariant across model classes.
- **10 cancer foreground topics** are expressed only by documents whose
  `source_cohort` is `cancer`; dementia documents contribute zero weight to
  these slots and vice versa.
- **10 dementia foreground topics** are expressed only by dementia documents.

This partition means dementia's tokens, even at 21% of corpus mass, are not
diluted by cancer's majority when estimating the dementia foreground block.
Success criterion 4 of the gated-STM design: at least one dementia foreground
topic achieves a crisp dementia-distinctive phenotype (NPMI > 0.10, top-N
terms not dominated by universal anchor vocabulary).

`source_cohort` is the gating variable (`group_var: source_cohort`) and is
therefore deliberately absent from `covariate_formula`. Within a foreground
block only one group's documents express those topics, so including
`source_cohort` in the prevalence regression would be rank-deficient (ADR 0026).
The formula retains `C(sex)` and `age` as prevalence covariates unchanged from
the baseline, so the prevalence-covariate path is still exercised.

`K: 50` = `background_k: 30` + 10 (cancer) + 10 (dementia). All other cohort
and sampling settings are identical to experiment 0003 (`prior_obs_days: 0`,
`person_mod: 4`, `max_iter: 100`, same `cache_uri` and `random_seed`) to keep
the corpora as comparable as possible.
