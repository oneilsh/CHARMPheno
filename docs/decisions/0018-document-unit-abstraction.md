# ADR 0018 — Document-unit abstraction (DocSpec)

**Status:** Accepted
**Date:** 2026-05-12
**Supersedes:** none
**Superseded by:** none

## Context

Until now, "document" in CharmPheno's topic-model pipeline has been hardcoded
as "patient" — `to_bow_dataframe` groups OMOP rows by `person_id` and emits
one BOW vector per patient. This worked for the bootstrap pipeline and the
initial LDA/HDP comparisons, but two pressures argue for making the doc
unit explicit and pluggable.

**Pressure 1: phenotype topics from lifetime bags are noisy.** A patient's
full event history mixes acute and chronic, early-life and late-life,
recovered and active phenotypes. Topics fit against that bag tend to find
"background comorbidity catch-alls" rather than crisp phenotype clusters
(see the HDP K-collapse exploration around iter 6 of the 2026-05-11/12
runs). Year-binned documents — one BOW per (patient, year) — preserve the
temporally-coherent co-occurrence structure that defines individual
phenotypes while letting chronic-condition background absorb into its own
topic rather than dominating every patient's topic mixture.

**Pressure 2: future use cases want different doc units.** The README
mentions "profiles that evolve over time" and "autoregressive generation of
patient profiles." The natural document for those is (patient, time-window),
not (patient, lifetime). Visit-level docs, anchor-event-centered windows,
and event-type-extended bags (conditions + drugs + procedures) are all
plausible follow-ons.

Continuing to hardcode the doc unit means each new shape ripples through
SQL, the BOW aggregator, the cloud and local drivers, the eval driver, and
the corpus-manifest schema — and worse, allows the shape to silently
diverge between fit and eval.

## Decisions

### Introduce a `DocSpec` strategy

A small, frozen-dataclass-style interface lives in
`charmpheno/omop/doc_spec.py`. Each spec:

- Declares the OMOP columns it requires (so the loader can fail fast if
  they're absent).
- Implements `derive_docs(events_df) -> events_with_doc_id_df`, returning
  the same event-level DataFrame with an added `doc_id` column. Each event
  may contribute to multiple docs (era replication across years) or
  exactly one (patient-as-doc, visit-as-doc).
- Implements `manifest()` returning a JSON-serializable summary used in
  `VIResult.metadata['corpus_manifest']`, so eval drivers can reproduce
  the same docs from a checkpoint without re-passing CLI args.
- Implements `from_manifest(d)` (classmethod) for round-tripping.

### Initial set of specs

Ship two concrete implementations:

- **`PatientDocSpec`** — `doc_id = person_id`. Default; reproduces current
  behavior exactly. Required columns: `(person_id, concept_id)`.
- **`PatientYearDocSpec`** — `doc_id = "{person_id}:{year}"`. Required
  columns: `(person_id, concept_id, condition_era_start_date,
  condition_era_end_date)`. With era replication on (default), explodes
  each era across every calendar year it spans, matching the BigQuery
  `UNNEST(GENERATE_ARRAY(...))` pattern explored in
  `analysis/cloud/doc_size_evals.ipynb`. End dates clipped to current year
  to bound the explosion against pathological future-dating. Default
  `min_doc_length=30` per the cutoff curve in that notebook (drops
  too-sparse year-bins before fit).

Deferred for follow-on:

- **`PatientVisitDocSpec`** — trivial given current schema, but most visit
  bags are too small to be informative for K~50-150. Will land when there
  is a concrete use case.
- **Multi-event-type specs** — conditions + drugs + procedures combined.
  Requires loader extension to additional OMOP fact tables; cleaner as
  its own ADR when the event-type-extension shape is decided.
- **Anchor-event-windowed specs** — docs centered on a diagnosis or
  hospitalization, fixed before/after window. Useful for causal/timing
  analyses; out of scope for phenotype-discovery.
- **Hierarchical specs** — patient-as-doc with per-year sub-docs. Possible
  in principle (HDP-derived nested DP) but a substantial modeling change,
  not in scope.

### Loader change: support condition_era

`load_omop_bigquery` gains a `source_table` parameter (default
`"condition_occurrence"` for backward compat). When set to
`"condition_era"`, the loader reads `condition_era` instead, projecting
`condition_era_start_date` and `condition_era_end_date` into the output
DataFrame. Because `condition_era` is derived from `condition_occurrence`
via OMOP's 30-day sliding window, it does not carry `visit_occurrence_id`;
the canonical OMOP schema is relaxed to make `visit_occurrence_id`
optional. `validate()` checks the always-required columns
(`person_id, concept_id, concept_name`) strictly and tolerates absence of
`visit_occurrence_id`.

### Plumbing

- `to_bow_dataframe(events_df, *, doc_spec=PatientDocSpec(), ...)` — the
  doc_spec parameter replaces the prior `doc_col` parameter. The default
  behavior is byte-identical to the prior default.
- LDA/HDP fit drivers (cloud + local) gain a `--doc-unit` CLI flag
  (`patient` | `patient_year`) and a `--doc-min-length N` flag.
- Eval drivers (cloud + local) reconstruct the DocSpec from
  `corpus_manifest['doc_spec']` rather than from CLI args, so eval
  semantics always match fit. The split-contract pattern already
  established in ADR 0017 extends to the doc-unit contract: mismatched
  doc-unit between fit and eval aborts the eval driver.

### corpus_manifest schema extension

`VIResult.metadata['corpus_manifest']` gains a `doc_spec` field:

```json
{
  "source": "bigquery",
  "source_table": "condition_era",
  "cdr": "<project>.<dataset>",
  "person_mod": 10,
  "vocab_size": 10000,
  "min_df": 10,
  "doc_spec": {"name": "patient_year",
               "min_doc_length": 30,
               "replicate_eras": true,
               "date_start_col": "condition_era_start_date",
               "date_end_col": "condition_era_end_date"}
}
```

The `doc_spec` sub-dict is `DocSpec.manifest()` output; eval drivers
re-instantiate the same DocSpec via `DocSpec.from_manifest(d)`.

## Trade-offs

- **Per-doc semantic shift is not free.** NPMI absolute values are not
  comparable across doc units. The corpus_manifest stamping prevents
  silent cross-comparison at the eval driver, but human interpretation
  of NPMI numbers needs to internalize that "this run's +0.14 mean is
  on patient_year docs, not lifetime docs."
- **More code paths.** The abstraction adds ~200 lines of spec/glue code,
  plus the loader extension, plus tests. Justified by the experimental
  appetite for trying multiple doc units; the cost of *not* abstracting
  was unbounded per-doc-unit branching in drivers.
- **"Phenotype profile" semantic shift is deferred.** Patient-year docs
  produce one θ vector per (patient, year). A patient-level profile, if
  downstream consumers want one, must be derived from those — mean,
  most-recent-year, weighted-by-activity, trajectory, etc. This ADR
  chooses *not* to specify that aggregation; it'll be decided when
  the first downstream consumer concretely needs it.

## Consequences

- ✅ Doc-unit experiments become a CLI flag change, not a code change.
- ✅ Fit→eval consistency is automatic via corpus_manifest.
- ✅ Future doc shapes (multi-event-type, anchor-windowed) add a new
  spec class without touching drivers.
- ⚠️ Old fit checkpoints (pre-2026-05-12) lack `corpus_manifest.doc_spec`
  and will be assumed `PatientDocSpec` by eval drivers (matching their
  pre-ADR behavior).
- ⚠️ Cross-run NPMI comparison requires comparing same-doc-unit runs;
  the eval-output banner must surface the doc_unit so this is obvious.
