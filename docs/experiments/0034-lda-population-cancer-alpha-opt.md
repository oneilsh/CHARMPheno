---
id: 34
slug: lda-population-cancer-alpha-opt
status: pending
model_class: lda
cohort: population_cancer
cohort_def: population_cancer
prior_obs_days: 0
person_mod: 4
doc_unit: patient_cohort
doc_min_length: 10
K: 60
max_iter: 200
subsampling_rate: 0.1
tau0: 128
kappa: 0.7
print_topics_every: 10
top_n_tokens: 8
seed: 42
optimize_doc_concentration: true
optimize_topic_concentration: false
cache_uri: hdfs:///user/dataproc/charm/covariates_cache
---

# Experiment 0034 — Plain LDA (α-optimized) on the population_cancer corpus (non-gated)

## Goal

Fit plain **LDA with document-concentration (α) optimization** on the SAME
population_cancer corpus that the gated STM fits (exp 0028 / 0033), non-gated
(covariates and gating ignored — LDA reads only the bag-of-words). This is the
real-corpus prior-family comparison arm: Dirichlet-α (LDA) vs logistic-normal-Σ
(STM) per-document concentration on identical documents.

## Why

The prior-family-vs-scale diagnostic (synthetic, real fitted β) established that
STM's over-diffuse patients are a SCALE deficit (STM is faithful at Σ ≈ 5–7),
not a prior-family deficit, and that "LDA-peaky" is substantially a Dirichlet
artifact — small α over-sharpens. This experiment tests that on the real corpus:
does α-optimized LDA on the population_cancer documents land MORE concentrated
(higher top_mass / fewer effective topics) than STM, and if so by how much? The
answer calibrates how much of the STM–LDA concentration gap is prior family vs
recoverable scale.

Important caveat (from the diagnostic, to interpret the LDA reading): on short
documents, mean-field variational α-optimization is known to read HOT
(over-concentrated) — mean-field VB underestimates posterior spread and biases
α̂ downward relative to collapsed/exact inference. So the LDA concentration here
is an UPPER bound on the true per-user concentration, not ground truth. The
planted-corpus α̂-vs-truth bias curve (a diagnostic-script follow-up, not this
run) is what would correct it. Treat exp 0034 as "how peaked does α-optimized
LDA claim patients are", to be read against exp 0033 (STM in-band scale) and exp
0028 (STM unit baseline), not as the true target.

## Configuration

Same corpus as exp 0028 / 0033 — the corpus cache key is determined by
`source_table`, `person_mod` (4), `vocab_size`/`min_df` (inherited from
`_base.yaml`, matching the STM runs), the `patient_cohort` doc-spec with
`doc_min_length: 10`, `cohort: population_cancer`, and `prior_obs_days: 0`. With
those matched, the LDA fit reads the IDENTICAL documents the STM fit used (a
cache hit against the parquet exp 0028 wrote, or a byte-identical rebuild on a
miss). `doc_unit: patient_cohort` is required for this match (it produces the
same `source_cohort:person_id` documents the gated STM fit on); LDA harmlessly
ignores the `source_cohort` prefix and fits all documents non-gated.

| Field | Value | Note |
|---|---|---|
| `model_class` | lda | plain online-VI LDA (Dirichlet prior) |
| `cohort` / `doc_unit` | population_cancer / patient_cohort | same documents as exp 0028/0033 (LDA driver widened to admit these — non-gated) |
| `K` | 60 | matches the STM K (40 background + 20 foreground → 60 flat LDA topics) |
| `optimize_doc_concentration` | true | α-optimization (the Dirichlet analog of STM's global scale) |
| `optimize_topic_concentration` | false | keep η (topic concentration) fixed, isolate the document-concentration comparison |
| schedule | max_iter 200, subsampling 0.1, τ0 128, κ 0.7 | matches exp 0028/0033 for comparable convergence |

## Success criteria

- The fit completes and writes a `concentration_readout` (per-document top_mass +
  eff_topics percentiles, from the variational θ) into checkpoint metadata — the
  same statistic the STM runs emit, enabling a direct comparison.
- Compare the three readouts: exp 0028 (STM unit-diagonal — the over-diffuse
  baseline), exp 0033 (STM + in-band pooled scale), exp 0034 (this, α-opt LDA).
  Expected ordering by concentration: 0028 (most diffuse) < 0033 ≤ 0034, with
  0034 the most peaked (Dirichlet over-sharpening on short docs). The size of the
  0033→0034 gap is the residual "prior-family" contribution AFTER scale is
  corrected — the quantity this arc exists to measure.
- Topic quality (NPMI, top terms) is recorded for context but is not the target
  here; the target is the per-document concentration distribution.

## Related

Comparison arm for exp 0028 (STM unit baseline) and exp 0033 (STM in-band pooled
scale, A1). Diagnostic:
`docs/superpowers/specs/2026-07-04-prior-family-vs-scale-concentration-diagnostic.md`
(prior-family-vs-scale finding + the α̂ mean-field-VB bias caveat). Method
lineage: Hoffman et al. 2010 (online LDA), Wallach et al. 2009 (asymmetric
Dirichlet priors). The concentration metric (top_mass, effective number of
topics = inverse-Simpson / Hill q=2) follows Hill 1973 / Jost 2006. The LDA
driver was extended to fit the population_cancer corpus non-gated (shared
`load_or_build_corpus`, additive argparse widening) for this comparison.
