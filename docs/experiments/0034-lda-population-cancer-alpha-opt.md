---
id: 34
slug: lda-population-cancer-alpha-opt
status: done
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

Second caveat — gating-support asymmetry in eff_topics. The STM readout is the
GATED mode θ (support = background ∪ the doc's own group, so a document's
`eff_topics` is structurally bounded by its allowed-set size: ~40 for a
background-only doc, ~60 for a cancer doc), while this LDA readout is NON-gated θ
over all K=60. So the `top_mass` comparison across 0028/0033/0034 is clean (a max
is a max regardless of support), but the raw `eff_topics` gap between gated STM
(0033) and non-gated LDA (0034) conflates concentration with gating support and
should NOT be read as pure concentration. The clean concentration comparison is
0028↔0033 (both gated STM, identical allowed sets); against 0034, lead with
`top_mass` and treat `eff_topics` qualitatively. State this explicitly when the
result is recorded.

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

## Result — real-corpus reference: patients are peaky (top_mass p50 = 0.513)

LDA fit to iter 200; α optimized DOWN to a small value (mean ≈ 0.018), i.e. the
corpus drove the Dirichlet toward peaky documents — the hot reading Fable
predicted for mean-field VB on short docs. Concentration readout:
**top_mass p50 = 0.513, eff_topics p50 = 2.8** (n = 48,328), versus STM-A1's
0.269 / 8.5 (exp 0033). Leading with top_mass (gating-independent): **LDA
patients are ~2× more peaked than STM-A1's.** As a hot upper bound, 0.513
over-states true concentration — but even discounted, it sits well above the
fit-anchored STM scales (A1 2.36, export 3.67), corroborating the plant-recover
diagnostic that the faithful generative scale is ~5–7. See insight
[0037](../insights/0037-in-band-pooled-scale-converges-below-export-scale-faithful-scale-not-fit-recoverable.md).

(First run silently omitted the readout — the helper imported from the
non-shippable `analysis/` package, absent on the Dataproc path; fixed in 3670b49
by relocating it into `spark_vi.eval.topic.concentration`, then re-run.)

UPDATE (insight 0038): the "α-opt reads HOT" attribution above is REFUTED by a
synthetic plant-and-recover experiment. With β frozen at truth, LDA's own
α-optimization recovers AT or BELOW the true concentration (reads cool), not
above — at both a clean and the real (K=60, 44-token) regime. So this arm's ~2×
peakiness vs STM-A1 is NOT an α-inference artifact; it is STM's fit scale being
too low (insight 0037) plus LDA co-fitting a sharper, more document-specific β
(the synthetic froze β, so it cannot reproduce that). Whether 0.513 over- or
under-states the true concentration is open; the validated held-out-LL calibration
(insight 0038) on the real corpus is what would pin it.

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
