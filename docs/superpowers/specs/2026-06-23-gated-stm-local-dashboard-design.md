# Gated STM: Local Sim→Fit→Dashboard Validation + Dashboard Gating — Design

**Date:** 2026-06-23
**Status:** Approved design, pending implementation plan(s)
**Branch context:** extends the `stm` branch, on top of the gated-STM model feature
(spec `2026-06-23-gated-stm-background-foreground-design.md`, ADR 0026)

## Goal

Build a fully local pipeline that **simulates** a gated background/foreground STM
corpus, **fits** it in-process (no BigQuery, no Dataproc), and renders the result
in the **dashboard** with gating-aware behavior — so the dashboard gating feature
(Plan 2 of the gated-STM work) can be built and visually validated now, without
waiting for a billed cluster run. The dashboard half is the full Section-4 feature
from the model design: masked prevalence on server and client, per-topic block
labels, and a group selector that makes foreground bubbles appear/vanish by group.

## Why local

The cluster experiment (`docs/experiments/0004-gated-stm-cancer-dementia.md`) is the
real validation, but it is billed and not yet run. The repo already has a local
sim→fit→dashboard chain for LDA/HDP (`scripts/simulate_lda_omop.py`,
`analysis/local/fit_lda_local.py`, `analysis/local/build_dashboard.py`) that runs
in-process under `poetry run`. Extending it for gated STM gives a fast, reusable,
offline validation path and lets the dashboard feature be built against a real
gated checkpoint.

Confirmed during design: under the root `poetry run` venv, local Spark workers run
and the gated `StreamingSTM` fit executes through the MLlib shim (the Python-3.12
`distutils` worker gap that fails some pytest suites is specific to a *different*
poetry venv, not the one the local fitters use). So `fit_stm_local.py` matches the
established local-fit pattern (local Spark + the shim) rather than inventing a new
shape.

## Decomposition (two plans)

- **Plan 2a — local gated-STM sim + fit harness** (produces the checkpoint):
  engine robustness fix, the gated simulator, and `fit_stm_local.py`.
- **Plan 2b — dashboard gating** (consumes the checkpoint): shared-export masked
  prevalence + block labels + block-level k-anon, the local builder's STM-file
  emission, the front-end group selector + masked prevalence, and the one-command
  view script.

```
simulate_gated_omop.py ─► omop.parquet + person.parquet ─► fit_stm_local.py ─► gated checkpoint
   (Plan 2a)                  (source_cohort, sex, age)      (local Spark + shim)       │
                                                                                         ▼
                              build_dashboard.py (local) ──► bundle (+ covariate + gating.json)
   (Plan 2b)                  shared export: masked prevalence + block labels + k-anon    │
                                                                                         ▼
                              front-end: group selector + masked covariatePrevalence ──► npm run dev
```

## Design decisions (resolved during brainstorming)

1. **Sim route:** extend the existing local sim framework (a sibling simulator +
   a local STM fitter), not a one-off fixture — a reusable pipeline mirroring the
   LDA one.
2. **Dashboard scope:** full Plan 2 (masked prevalence on server + client, block
   labels, group selector), validated locally.
3. **Foreground-less group:** a document whose group has no foreground block gets
   **background-only** (engine fix), not a crash. This also lets the simulator
   model the real rare-disease structure (large common background cohort + rare
   foreground).
4. **Block-level k-anon:** a foreground group below the small-cell threshold `k`
   is **fully suppressed** from the export (dropped from `groups`, its foreground
   topics excluded, no β / no prevalence, not selectable).
5. **Fit pattern:** `fit_stm_local.py` uses local Spark + `StreamingSTM` (the
   shim), matching `fit_lda_local.py`.

## Engine fix — foreground-less group is background-only

[`TopicBlockPartition.allowed_indices`](spark-vi/spark_vi/models/topic/partition.py)
currently raises `KeyError` when a document's group is not in the foreground map.
That is wrong for the rare-disease structure (a large common cohort whose group has
no foreground block must see background only) and surfaces as a crash deep in a
Spark worker.

**Change:** `allowed_indices(groups)` skips groups absent from the foreground map
(they contribute no foreground block); the result is `background ∪ ⋃ foreground[g]
for g in groups that have a block`. A document with an empty group set already gets
background-only; this makes a non-empty-but-foreground-less group behave the same.
The Task-1 test `test_unknown_group_in_allowed_indices_raises` is replaced by
`test_foregroundless_group_yields_background_only`. Construction-time validation of
the partition itself is unchanged (background/foreground sizes still validated).

## Gated simulator — `scripts/simulate_gated_omop.py`

A sibling to `simulate_lda_omop.py` (kept separate so the clean LDA generator is
not bloated with gating concepts).

**Generative model.** Vocabulary is partitioned into background concepts (shared)
and per-group distinctive concepts. β is generated programmatically: K_bg
background topics over background concepts; for each group, K_fg foreground topics
over that group's distinctive concepts, with a tunable background-bleed so
foreground docs still emit common codes. Per patient:
- draw a group with configurable proportions (supports both the cancer/dementia
  both-foreground shape and a large common-background + rare-foreground shape);
- draw covariates: `sex ~ Bernoulli`, `age ~ Normal(μ_group, σ)` (group-correlated
  for realism; group is not a covariate so this is fine);
- draw θ over allowed topics only (background ∪ group's foreground), zero on other
  groups' foreground (the gating);
- emit codes via θ, β.

**Outputs:**
- `data/simulated/gated_omop_N<n>_seed<s>.parquet` — events:
  `person_id, visit_occurrence_id, concept_id, concept_name, source_cohort,
  true_topic_id, true_block`.
- `data/simulated/gated_person_N<n>_seed<s>.parquet` — `person_id, source_cohort,
  sex, age` (or `year_of_birth`).
- An oracle sidecar JSON (planted partition + concept→block map) for validation.
  `true_topic_id`/`true_block` are oracle-only; the fitter must not read them.

**Args** mirror the LDA sim plus `--background-k`, `--foreground
group:K[,group:K...]`, `--group-props`, vocab-partition sizes, visit/code means.

## Local STM fitter — `analysis/local/fit_stm_local.py`

Follows `fit_lda_local.py`'s structure (local Spark `local[2]` + the MLlib shim;
"local proxy for the Dataproc submit").

**Flow.** Load the events parquet (`load_omop_parquet`) and build the BOW with the
`patient_cohort` doc-spec (`doc_id = "source_cohort:person_id"`). Load the person
parquet and build the covariate DataFrame via the formulaic path
(`build_patient_covariate_df`, `~ C(sex) + age`, keyed per person — `source_cohort`
is the gating label, NOT a covariate). Derive `source_cohort` from `doc_id` (the
same decoupled materialization the cloud driver does) for the `doc_group_col`. Join,
construct `StreamingSTM(K, topic_blocks=partition, doc_group_col="source_cohort",
covariate_formula="~ C(sex) + age")`, and fit.

**Saves** a `STMModel` checkpoint with full metadata so the dashboard build reads it
uniformly: `corpus_manifest` (`cdr="local"`, `source_table`,
`cohort="cancer_or_dementia"` or the sim's cohort name, `doc_spec` patient_cohort,
`vocab`, `name_by_id`, `min_patient_count` for the k-anon threshold,
**`topic_block_spec`**), `covariate_manifest`, `model_class="stm"`,
`concept_names`/`concept_domains` (from the sim's concept table). It also persists
the covariate design matrix + per-doc group labels locally
(`covariates.parquet`), so the dashboard's masked prevalence is computable offline
without a Spark `cov_df` cache.

**Args:** `--omop`, `--person`, `--background-k`, `--foreground`, `--group-var`
(default `source_cohort`), `--covariate-formula`, `--K`, `--max-iter`, SVI params,
`--out-dir`.

## Dashboard data layer (shared export + local builder)

**A. Gating-aware masked prevalence (pure-numpy, shared).** New helper beside
[`corpus_mean_topic_proportions`](spark-vi/spark_vi/models/topic/stm.py):
`corpus_mean_topic_proportions_gated(Gamma, X, groups_per_doc, partition) -> (K,)`
— for each doc, softmax of Γᵀx over its allowed topics only (background ∪ its
group's foreground), zero elsewhere, averaged over docs. A foreground topic's
corpus-mean prevalence therefore reflects only its group's share. Both builders use
it for the STM `corpus_prevalence`.

**B. Block labels + gating spec — a new optional `gating.json`.** Following the
existing optional-STM-file pattern (`covariate_effects.json`,
`covariate_schema.json`):
```json
{ "group_var": "source_cohort",
  "groups": ["cancer", "dementia"],
  "topic_blocks": ["background", ..., "cancer", ..., "dementia", ...] }
```
`topic_blocks` is length-K aligned to the original topic ids (front-end maps each
phenotype by `original_topic_id`). `model_adapter`/`DashboardExport` carry a
per-topic `topic_blocks: list[str]`.

**C. Block-level k-anon (full suppression).** When building `gating.json` + the
masked prevalence, compute per-group patient counts; a group with count `< k`
(`k = corpus.min_patient_count`) is fully suppressed — dropped from `groups`, its
foreground topics excluded from the bundle (no β top-concepts, no per-group
prevalence), not selectable. One consistent threshold with the existing
covariate-level k-anon. This is the honest information floor for the rare-disease
case: the model may fit a sub-`k` group, but the export must not reveal it.

**D. Local `build_dashboard.py` emits the STM files** (today it emits none): for
`model_class=="stm"`, write `covariate_effects.json` (Γ rows via `adapt_stm`),
`covariate_schema.json` (controls/levels/percentiles from the locally-saved
covariate matrix — no Spark `cov_df`, no cache), `gating.json` (post k-anon), and
the masked `corpus_prevalence`. The shared `charmpheno.export` functions get the
gating extensions, so `build_dashboard_cloud` inherits them too —
`_stm_corpus_prevalence` switches to the gated helper (deriving each cov row's group
from `source_cohort`), and the cloud builder writes `gating.json`. Cloud parity is
included, but only the local path is runnable/validated here (cloud needs BigQuery).

Contract stays clean: Γ remains (P, K); gating is one new optional file + the
masked-prevalence swap, never a reshape of existing arrays.

## Front-end (group selector + masked prevalence)

- **Loader/types:** `loadBundle` optionally fetches `gating.json`; `DashboardBundle`
  gains `gating?: GatingSpec { group_var, groups[], topic_blocks[] }`. No
  `gating.json` → behaves exactly as today (backward compatible).
- **Group selector:** a new Atlas control listing `gating.groups` plus a
  "background only" option — separate from the covariate controls, because
  `group_var` is not a covariate. New `store.ts` writable `selectedGroup`.
- **Masked prevalence:** a gating-aware variant of `covariatePrevalence` — compute
  η = Γᵀx, set η[disallowed] = −∞ for foreground topics whose block label is
  neither `background` nor `selectedGroup`, then softmax (masked *before* the
  softmax, normalized over the allowed set — matches the model). "Background only"
  zeros all foreground. Covariate mode off → the bundle's gating-aware corpus-mean
  `corpus_prevalence`.
- **Atlas:** out-of-group foreground bubbles → prevalence 0 → vanish; background +
  selected-group foreground show; switching group swaps which foreground appears;
  bubbles still resize with sex/age. ("Bubbles resize; foreground appears/vanishes
  with the group.")

## Testing

- **Engine:** the `allowed_indices` fix (foreground-less group → background-only)
  with the flipped Task-1 test.
- **Python:** simulator unit tests (output schema, group/covariate columns, planted
  block coverage); `fit_stm_local` end-to-end recovery test (planted foreground
  recovered on its group's distinctive concepts, majority ≈0); export-helper tests
  — gated masked prevalence (numpy), `gating.json` content, and block-level k-anon
  (a sub-`k` group is fully suppressed from the bundle).
- **Front-end:** vitest for the masked prevalence (out-of-group foreground zeroed +
  renormalized; background-only zeros all foreground) and the non-gated-bundle path
  unchanged.

## Viewing

A one-command local chain (a small script / Make target mirroring the existing
local convenience) runs `simulate_gated_omop → fit_stm_local → build_dashboard →
place bundle in dashboard/public/data/<cohort>/`, then `npm run dev`. The build runs
end-to-end and a screenshot of the Atlas demonstrates the foreground-by-group
behavior.

## Out of scope

- **Cloud-path runtime validation** — the shared-export gating changes propagate to
  `build_dashboard_cloud`, but running it needs BigQuery; only the local path is
  validated in this work.
- **LLM topic-labeler guidance** — still deferred (insight 0024; disambiguate the
  gating "background block" from the labeler's `background` quality category) until
  after a real gated run.
- **Content covariates / SAGE** — out of scope, as before.

## Success criteria

1. `simulate_gated_omop` produces a gated corpus + person/covariate parquet with a
   planted background/foreground structure.
2. `fit_stm_local` fits it in-process and recovers the planted foreground (a
   foreground topic concentrates on its group's distinctive concepts; the majority
   contributes ≈0).
3. The engine fix makes a foreground-less group background-only (no crash).
4. The local dashboard build emits `covariate_effects.json`, `covariate_schema.json`,
   and `gating.json`, with a sub-`k` group fully suppressed.
5. In the running dashboard, selecting a group makes that group's foreground bubbles
   appear and the other group's vanish, while bubbles resize with sex/age; a
   non-gated bundle is unaffected.
