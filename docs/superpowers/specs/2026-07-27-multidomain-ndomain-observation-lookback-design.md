# SP3c — N-domain generalization + observation domain + lookback windowing — Design

**Date:** 2026-07-27
**Status:** Approved (user, 2026-07-27). Ready for a plan. **Depends on SP3a + SP3b.**
**Arc:** `docs/superpowers/specs/2026-07-24-multidomain-gated-lda-arc-design.md`
**Prereq:** SP3b — the two-domain assembler, the multidomain cloud driver, and the
dict-λ persistence (SP3a) are what this generalizes.

## Goal

Extend the multi-domain gated model from a hardwired **two** domains to **N**,
add **`observation`** as a third data domain, and give the driver **lookback**
windowing so a rare6 3-domain (condition + drug_era + observation) case-finding
fit runs at parity with the single-domain rare6 experiments (0061–0065). The
first tracked run is **exp 0071**.

## Layer note

This work lives in `charmpheno/` and `analysis/cloud/`, where clinical
vocabulary is permitted and expected. The engine (`spark_vi/**`) is already
N-domain (dict-λ `{m: (K, V_m)}`, `featuresCols` list, per-domain ω/η lists,
`_concat_domain_features(list, sizes)`) and stays integer-id and domain-neutral.
The driver is the clinical-semantics layer: it knows condition / drug / observation;
the assembler and engine know only ordered integer domains.

## Decision: generalize-and-migrate to N domains (not bolt on a third)

The two-domain layer becomes an N-domain layer, and **exp 0070 migrates onto it
as N=2**. One code path, no duplication (the user's general/complete preference).
The 2-domain seam test + exp 0070 are the N=2 regression that proves the
generalization is behavior-preserving. The rejected alternative — a separate
three-domain module beside the two-domain one — duplicates the assembler and the
driver and would repeat for a fourth domain.

## Decision: lookback windowing via one shared condition index

**User decision (2026-07-27).** The rare6 3-domain fit uses **lookback**
windowing (pre-index features + forward labels, leakage-free by construction),
matching the single-domain rare6 experiments, not exp 0070's forward window.

The mechanism composes from existing, already-domain-neutral cohort functions:

- `cohorts.case_finding_index_table(cond_df, …)` builds **one** `(person_id,
  index_date, source_cohort)` table. The index is **condition-derived** (first
  qualifying dx for the disease arm; a gated random window for the background
  arm) — which is correct, because the gate/label is condition-only.
- `cohorts.lookback_feature_label_events(events_df, index_df, date_col,
  lookback_days, label_window_days)` splits **any** event frame against that
  index into a pre-index feature frame and a forward label frame. It already
  takes `date_col` as a parameter, so it applies unchanged to each domain.

So the driver builds the index **once** from conditions, then calls
`lookback_feature_label_events` **once per domain** (each with that domain's own
`date_col`) to get N pre-index **feature** frames. The **label** frame is taken
**only from conditions** (the forward window) — the gate is condition-only, so
no drug or observation event ever defines a frontier. This is the parked SP3b/SP4
"window all domains against one `case_finding_index_table`" item; it is small
because the split function is already generic.

The forward path (exp 0070) is retained: `apply_population_disease_cohort` +
a domain-neutral `_window_events_to_cohort` (the SP3b `_window_drug_events_to_cohort`
generalized to take a `date_col`).

## Decision: `observation` is a point event; vocabulary is empirical

`observation` has `observation_concept_id` and `observation_date` (a **point**,
no era span), unlike the span-shaped `condition_era` / `drug_era`. It normalizes
to `(person_id, concept_id, observation_date)`; downstream BOW counts each
observation concept as one occurrence per document, and `observation_date` is its
`date_col` for windowing. No era replication.

The observation vocabulary is built **empirically** from the concept_ids observed
in the window — no class filter, no rollup — mirroring the drug_era decision.
`observation` is the most heterogeneous OMOP domain (social history, survey/
questionnaire answers, clinical findings, administrative concepts), so its
mixed-granularity vocabulary is the honest reflection of the data and is tamed by
the per-domain `min_df` / `min_patient_count` / `vocab_size` controls, not
assumed away. The observed heterogeneity should be visible in the topic dump and
is a documented risk, not a blocker (mirrors the drug_era heterogeneity finding).

## Decision: the leakage strip loops over ALL N domain vocabularies

The strip maps the DAG **node-marker concept-ids** through **each** domain's
`vocab_map` and drops those indices from that domain's feature column. This is
already the SP3b design's symmetric strip, now over N domains.

**Why it matters, precisely (user correction, 2026-07-27):** By OMOP convention
each concept carries a single `domain_id` and should be recorded only in its
matching fact table — so a Condition concept (e.g. a rare6 anchor like cutaneous
sarcoidosis) appearing in `observation` is a **spec violation, not the norm**;
under a conforming ETL the condition markers live only in vocab 0. But real CDRs
are **not always spec-conformant** — some ETLs do load condition-domain (SNOMED
finding) concepts into `observation` — so the symmetric strip is a cheap
**defensive** guard: it is a **no-op on conforming data** (a condition marker
simply isn't in a conforming observation vocabulary) and costs nothing, while
guaranteeing that *if* a defining condition does leak into any domain's vocabulary
it is still stripped. The mechanism is sound because concept_ids are globally
**unique**: a given node concept-id denotes the same concept in every table, so
mapping it through each domain's `vocab_map` catches it wherever it landed, with
no per-domain special-casing.

## Components

### 1. OMOP loading — `observation` (`charmpheno/charmpheno/omop/bigquery.py`)

- `_SUPPORTED_CONCEPT_TYPES` gains `"observation"`; `_SUPPORTED_SOURCE_TABLES`
  gains `"observation"`.
- An `_observation_select_cols()` helper + read branch: select `person_id`,
  `observation_concept_id AS concept_id`, `observation_date`. No end date (point
  event). The existing `cohort=` fast-fail guard (SP3b: cohort filtering requires
  a condition source_table) already rejects `cohort=` with `observation`, which
  is correct — the driver never passes `cohort=` for a feature domain.
- Schema validation extends to the observation frame.

### 2. N-domain assembly (`charmpheno/charmpheno/omop/multi_domain.py`)

Rename `two_domain.py` → `multi_domain.py` and generalize (imports updated; the
test file follows). `DomainVocabSpec` is unchanged.

- `MultiDomainBundle`: `train_df` / `test_df` carry feature columns
  `features_0 … features_{N-1}` (index 0 = conditions) plus `frontier`
  (engine-ids) and `source_cohort`. `vocab_maps: list[dict]` (one
  `{concept_id: idx}` per domain, in domain order). `domain_names: list[str]`
  (e.g. `["condition", "drug", "observation"]`) for display/persistence. The
  bridge/receipt fields (`parent_int`, `int2cid`, `cid2int`, `name_by_id`,
  `ledger`) are unchanged — the frontier is condition-only.
- `multidomain_bow(domain_events: list[DataFrame], vocab_specs:
  list[DomainVocabSpec], *, doc_spec) -> (df, list[vocab_map])`: fits each
  domain's BOW via `topic_prep.to_bow_dataframe` (unchanged), then an **N-way
  full-outer join** on `doc_id`, coalescing `person_id` and filling each absent
  per-domain column with an empty `SparseVector` of that domain's fixed vocab
  size (so every per-domain vector size is constant across the corpus — SP3a's
  shim derives `domainBounds` from row 0 and validates every row).
- `assemble_multidomain_from_events(cond_events, extra_events: list[DataFrame],
  before_dag, *, doc_spec, min_n, vocab_specs: list[DomainVocabSpec],
  holdout_frac=0.2, split_salt=None, n_bg=2, tpn=1, strip_mode="test_only",
  label_events=None) -> MultiDomainBundle`: reuses the single-domain
  `assemble_from_events` helpers verbatim (split, frontier, prune, ledger, strip).
  Domain 0 = `cond_events` (also the DAG/frontier source); domains 1…N−1 =
  `extra_events` in order. `vocab_specs` has one spec per domain (length =
  `1 + len(extra_events)`). The same salted split lands a person's rows in every
  domain on the same side; each domain's vocab is fit on TRAIN, frozen for TEST.
  The strip loops over all N `(features_i, vocab_map_i)` pairs.

### 3. Multi-domain lookback windowing (`analysis/cloud/multidomain_cloud.py`)

Driver orchestration reusing `cohorts.case_finding_index_table` +
`cohorts.lookback_feature_label_events` (no new cohort code):

- `--window-mode {forward,lookback}` (default forward, exp-0070 behavior),
  `--lookback-days`, `--label-window-days`.
- **Lookback:** build the condition index once; for each domain call
  `lookback_feature_label_events(domain_raw, index_df, date_col=<domain date col>,
  lookback_days, label_window_days)` and keep the **feature** frame; keep the
  **condition** forward **label** frame; pass `label_events=cond_label` to the
  assembler. `strip_mode` / `prior_obs_days` are moot in lookback (disjoint by
  construction; the ≥1yr gate is intrinsic to the index table).
- **Forward:** `apply_population_disease_cohort` on conditions, then
  `_window_events_to_cohort(cond_windowed, domain_raw, date_col=…, window_days)`
  per extra domain (the SP3b drug windower generalized).

### 4. Driver generalization (`analysis/cloud/multidomain_cloud.py`)

- `--domains` = comma list of extra domains beyond conditions (subset of
  `{drug_era, observation}`; default `drug_era` = exp 0070). Condition is always
  domain 0. The driver builds an ordered domain list and per-domain
  `(source_table, date_col, DomainVocabSpec, display_name)`.
- Per-domain vocab controls: existing `--cond-*` / `--drug-*`, plus `--obs-*`
  (`--obs-vocab-size` / `--obs-min-df` / `--obs-min-patient-count`).
- Load / window / assemble / fit (`featuresCols=["features_0", …]`) / dead-node
  read / topic dump / vocab persistence all loop over the domain list. The
  dead-node cross-domain OR and the topic dump's per-domain top-terms generalize
  from 2 to N. The manifest persists per-domain vocab keyed by **domain name** —
  `vocab_<domain_name>` and `vocab_names_<domain_name>` (e.g. `vocab_condition`,
  `vocab_drug`, `vocab_observation`) — which **supersedes** SP3b's `vocab_a` /
  `vocab_b` / `vocab_names_a` / `vocab_names_b` keys. This is a safe rename: those
  keys are brand-new (SP3a/SP3b, commit 527c5d4) with no downstream consumers yet.
- `--seed` stays required (insight 0070). The dead-node check stays the
  pre-registered smoke assertion.

### 5. `run_experiment` wiring (`scripts/run_experiment.py`)

`build_multidomain_args` emits `--window-mode`, `--lookback-days`,
`--label-window-days`, `--domains`, and the `obs_*` vocab trio, all from the
effective config (defaults mirror the driver). The existing multidomain wiring
(model_class, driver path, eval-skip) is unchanged.

### 6. Exp 0071 (`docs/experiments/0071-multidomain-rare6-cond-drug-obs.md`)

`model_class: multidomain`, `cohort: population_rare6`, `disease: rare6`,
`window_mode: lookback`, `lookback_days` / `label_window_days` matching 0061,
`domains: drug_era,observation`, per-domain vocab trios (cond / drug / obs),
`seed`, `n_bg` / `tpn`. `make exp ID=71` runs it.

### 7. Migrate exp 0070 to the general path (N=2)

Update `0070`'s driver invocation to `domains: drug_era`, forward. Its existing
seam/round-trip behavior is the N=2 regression; the fit is byte-comparable to the
pre-migration two-domain path (same corpus, same vocab, same result shape).

## Validation / acceptance

Per the arc's realism discipline (insights 0067/0068/0070), acceptance asserts
**structural** correctness, not a single recovery number.

1. **N-domain assembly unit tests, no CDR** (the real coverage): a 3-domain doc
   has 3 feature columns; each per-domain vector size is constant across docs;
   ids fall within each domain's own range; the columns are aligned per doc; the
   leakage strip removes a node-marker concept-id from **every** domain whose
   vocab contains it (test with a marker injected into the observation vocab, per
   the user's guarantee).
2. **Multi-domain lookback unit test, no CDR**: one synthetic index table → per
   domain, the feature frame holds only pre-index events and the label frame
   (condition-only) holds only forward events; a post-index drug/observation
   event never enters any feature column.
3. **observation loading test**: the schema-validated observation frame shape
   (person_id, concept_id, observation_date).
4. **N=2 regression**: exp 0070's seam/round-trip test passes through the
   generalized assembler + driver unchanged.
5. **Cluster smoke (user-run)**: exp 0071 fits; `manifest.dead_nodes` empty; 3
   per-domain vocab sizes reported within plausible bands; the topic dump shows
   all 3 domains; the artifact round-trips through SP3a's loader.

## Out of scope

- **The ω sweep + specificity green light — SP4.** Unchanged.
- **An observation DAG.** Not needed — the gate is condition-only.
- **Observation `value_as_*` (numeric/concept values), dose/route for drugs.**
  Concept presence only.
- **A fourth domain** (measurement, procedure). The N-domain generalization makes
  it a config change, but it is not built or tested here (YAGNI).
- **Mid-fit checkpoint/resume** (SP3a defers it). If rare6 3-domain fit sizes make
  Dataproc preemption likely, revisit — a preempted fit currently loses progress.

## Risks

- **Migrating exp 0070 could regress the shipped two-domain path.** Mitigated by
  keeping the N=2 seam/round-trip test as an explicit acceptance gate (item 4).
- **Observation heterogeneity** produces a mixed-granularity vocabulary; measured
  via the topic dump, not assumed away (mirrors drug_era).
- **Three domains × rare6 forest × lookback** is a larger fit than exp 0070; if
  Dataproc preemption bites, resume is the follow-on (out of scope above).
- **Vocabulary volume imbalance across three domains** feeds the shared θ
  unweighted; the higher-volume domain dominates (the exp-0070 observation). This
  is the SP4 ω lever, not fixed here.
