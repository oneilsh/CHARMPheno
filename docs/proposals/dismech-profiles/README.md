# Phenotype Profiles for DisMech — schema proposal

A minimal, growable LinkML schema for attaching **CHARMPheno phenotype profiles**
to **DisMech** disease pages. This is a proof-of-concept proposal intended to be
contributed to [`monarch-initiative/dismech`](https://github.com/monarch-initiative/dismech);
it lives here for now so it can iterate alongside the models that produce the data.

## Files

| File | What it is |
|---|---|
| `profiles.yaml` | The LinkML schema (3 core classes + a `Source` provenance block). |
| `example_ehlers_danlos.yaml` | A worked `ProfileSet` instance for Ehlers-Danlos (MONDO:0020066). |

## What a profile is

A **profile** is a small, human-readable summary of an EHR-derived signal: one or
more categorical **distributions over medical codes** (conditions, drugs, labs, …)
that tend to co-occur within a disease's patient cohort. It is the interpretable
output of CHARMPheno's Bayesian phenotype models, reduced to just what a DisMech
page needs.

```
ProfileSet ──has──> Profile ──has──> CodeDistribution ──has──> WeightedCode
   │                   │                  (per domain)          (OMOP concept_id
 source              disease                                     + name + weight
(provenance)        (MONDO)                                      + value_qualifier?)
```

## Design principles

1. **Simple, and can grow.** Three required classes. A profile is a `label`, a
   `disease` (MONDO), and a list of weighted-code distributions. Nothing else is
   required.
2. **Per-MONDO term; hierarchy is Mondo's.** A profile points at one MONDO term.
   Any nesting/overlap of cohorts is expressed by Mondo's own subclass tree — which
   DisMech already renders — so this schema ships **no** DAG / node / cohort
   vocabulary of its own.
3. **Presentation-agnostic.** The schema carries data (labeled, weighted codes),
   never a visualization. DisMech decides how to display it.
4. **Model-specific detail is quarantined in `source`.** The underlying model is
   iterating fast (ontologized/gated multi-domain topic models today; possibly
   something else tomorrow). None of that leaks into the core: method, dataset,
   modeling description, references, and an open `metadata` map all live in
   `source` and can be reshaped freely without touching the profile contract.
5. **Codes are OMOP `concept_id`s.** Exposing source-vocabulary CURIEs (SNOMED,
   RxNorm, LOINC) directly proved brittle, so every code is an OMOP `concept_id` +
   `concept_name`. The OMOP vocabulary release that resolves them is recorded once
   in `source.omop_vocabulary_version`.
6. **Multi-domain, kept factored.** A profile holds one distribution *per domain*
   and encodes **no** cross-domain combination (no weights, no merged vector). How
   domains combine is an open modeling question and a rendering choice — kept
   deliberately outside the contract.
7. **No patient-level data.** Instances carry only aggregate distributions and
   privacy-safe counts (e.g. `cohort_size`).

## Explicitly out of scope

- **Case-finding / patient-placement readouts** (per-disease domain weights,
  reliability, precision/recall). That is model-evaluation, not disease-page
  content, and it is the fastest-moving part of the pipeline.
- **The model checkpoint itself** (`VIResult`, variational parameters). This schema
  describes the *exported, human-facing* profiles, not the fitted model.

## Mapping to CHARMPheno outputs

Profiles correspond to the per-topic, per-domain code distributions the models
already produce internally (row-stochastic β, trimmed to top codes). A future
`adapt_*` exporter would flatten a fitted model into a `ProfileSet`; this schema is
the target contract for that exporter.

## Validate

```bash
pip install linkml
linkml-validate -s profiles.yaml -C ProfileSet example_ehlers_danlos.yaml
```

## Open questions for the DisMech devs

- **Naming.** `Profile` / `ProfileSet` are placeholders. `PhenotypeProfile` risks
  colliding with DisMech's HP-term "phenotypes"; alternatives: `CooccurrenceProfile`,
  `EHRProfile`, or folding it in as a new `AssociationSignalSource: CHARMPHENO`.
- **Packaging.** One `ProfileSet` per model run (DisMech filters by MONDO) vs. one
  per disease page. This proposal uses the former.
