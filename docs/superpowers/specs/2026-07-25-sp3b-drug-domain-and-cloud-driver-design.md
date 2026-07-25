# SP3b — Drug domain + multi-domain cloud driver — Design

**Date:** 2026-07-25
**Status:** Approved (user, 2026-07-25). Ready for a plan. **Depends on SP3a.**
**Arc:** `docs/superpowers/specs/2026-07-24-multidomain-gated-lda-arc-design.md` (SP3 stub, split into SP3a/SP3b)
**Prerequisite:** SP3a — the shim's `featuresCols` contract and dict-λ persistence are what this produces input for and writes output through.

## Goal

Give the multi-domain gated model a **real second domain and a way to run it**: extend OMOP loading to drug events, assemble a two-domain case-finding corpus over one shared window, and stand up a cloud driver entry point with ω configurable — so SP4 has something to sweep and a real-cohort specificity read to make.

## Layer note

This work lives in `charmpheno/` and `analysis/cloud/`, where **clinical vocabulary is permitted and expected** — the opposite of the `spark_vi/**` constraint. Concept ids, drug eras and disease anchors are the domain language here. The engine stays integer-id and domain-neutral; this layer is where semantics attach.

## Decision: the drug domain is `drug_era`, windowed like conditions

**User decision (2026-07-25):** use `drug_era`, taking whatever falls in the window the cohort and doc spec define. The user also corrected an assumption worth recording: **`drug_era` is not ingredient-only in practice**, whatever the OMOP specification implies, and the design must not depend on an ingredient-class rollup.

Consequences:

- **The drug vocabulary is built empirically** from the `concept_id`s actually observed in `drug_era` within the window — no ingredient-class filter, no assumed rollup. Whatever concept classes the CDR populates are what the vocabulary contains.
- `drug_era` mirrors `condition_era`: both are span-shaped, both carry "active exposure/condition" semantics, both fit the era-replication the doc specs already do. That symmetry is why it needs the least new plumbing.
- Token volume is modest relative to `drug_exposure`, so ω should sit near 1 initially — but ω is exactly the dial SP4 sweeps, so nothing here should assume its value.

## Decision: one window, both domains; the gate stays condition-only

Drug events go through the **same** index-date and lookback split as conditions — `cohorts.case_finding_index_table` then `cohorts.lookback_feature_label_events` — so drug features come from the same pre-index `lookback_days` window as condition features. Anything else makes the two domains incomparable within a document.

**Labels and the gate remain condition-only.** The gate acts on θ's support (which topics a document may use, from its label's DAG closure); the domains act on β's normalizer. They are orthogonal — `gate ⟂ domain` in the arc design. A drug domain therefore needs a **vocabulary, not a DAG**, which is why no drug DAG builder is required and the arc's "depends on the condition/drug DAG builders" note over-stated the dependency. The condition DAG builder already exists (`charmpheno/charmpheno/omop/condition_dag.py`).

## Components

### 1. OMOP loading (`charmpheno/charmpheno/omop/bigquery.py`)

The extension point is already documented there: `_SUPPORTED_CONCEPT_TYPES = ("condition",)` with a `NotImplementedError` for anything else, and a comment naming drug_exposure as the planned follow-on.

- `_SUPPORTED_CONCEPT_TYPES` gains `"drug"`; `_SUPPORTED_SOURCE_TABLES` gains `"drug_era"`.
- Drug rows normalize to the same event shape conditions use (person, concept_id, start, end) so the existing window and doc-spec machinery applies unchanged.
- Schema validation extends to the drug frame.

### 2. Two-domain assembly (`charmpheno/charmpheno/omop/case_finding_assembly.py`)

- `assemble_case_finding_corpus` gains drug-side parameters and emits **two feature columns** matching SP3a's `featuresCols` contract, each a sparse vector over its own vocabulary.
- **Per-domain vocabulary controls** (`vocab_size`, `min_df`, `min_patient_count` per domain): the two domains have very different natural sizes, and one shared threshold would either starve the drug vocabulary or bloat the condition one.
- The leakage strip (`strip_mode`, node-marker codes) applies per domain — a drug that *is* the node marker leaks exactly as a condition code would.
- Frontier/label construction is untouched: condition-only, from the label frame.

### 3. Cloud driver (`analysis/cloud/`)

- A new entry point alongside `dag_placement_cloud.py`, following the existing driver conventions (`_driver_common.py`, the corpus/case-finding cache modules, the `Makefile` target pattern).
- Config surface: the two source tables, per-domain vocabulary controls, `omega`, per-domain `eta`, and the existing window/cohort knobs.
- Writes a `VIResult` through SP3a's dict-λ writer, so the artifact is loadable.

## Validation / acceptance

1. **Assembly unit tests, no CDR required** — well-formed two-domain documents: per-domain vector sizes consistent, ids within their domain's range, the two columns aligned per document, leakage stripped per domain. This is where the real coverage lives; the BigQuery body is cluster-covered only, matching the existing convention that `assemble_from_events` is unit-tested while the BQ path's arg surface is not.
2. **A drug-domain loading test** against the schema-validated frame shape.
3. **Driver smoke on a tiny `person_mod` sample** — cluster, user-run. This is the first end-to-end multi-domain fit on real data, so it is a smoke check, not a quality gate.
4. **Round-trip:** the driver's artifact loads back through SP3a's loader.

## Out of scope

- **The ω sweep and the specificity green light — SP4.** Per insight 0069, the `theta_contribution_by_domain` stat cannot serve as the ω-tuning read; the arc design's validation item 4 is already amended to require a quantity sensitive to where the mass landed (per-domain marginal contribution to fitted θ, or a leave-one-domain-out refit).
- `drug_exposure` at clinical-drug level, and any dose/route/frequency detail.
- A drug DAG. Not needed — see the gate/domain orthogonality above.
- Mid-fit checkpoint/resume (SP3a defers it). **If the planned fit sizes make Dataproc preemption likely, this becomes a real prerequisite** rather than a deferral, because a preempted multi-domain fit currently loses all progress.

## Risks

- **This layer's input contract is the driver's contract**, so a change to the per-domain vocabulary decisions after the driver exists means driver rework. The user accepted this coupling when choosing to stand the driver up in SP3 rather than SP4.
- **Drug-era concept heterogeneity.** Because the vocabulary is empirical, a CDR whose `drug_era` mixes concept classes produces a mixed-granularity vocabulary. That is the honest reflection of the data, but it makes drug-side topic interpretation harder, and it should be *measured* (report the observed concept-class distribution) rather than assumed away.
- **Utilization confounding.** Token counts in either domain track encounter frequency as well as clinical state; ω tempers the θ contribution but does not remove the confound. Documented in the arc design's β/π/ω rationale.
