# Case-Finding Cohort + Frontier-Label Assembly — Design

**Date:** 2026-07-15
**Status:** Design approved (brainstorm), pre-implementation
**Piece:** 2 of 3 of the cluster driver (piece 1 = condition DAG builder, done; this piece; then piece 3 = cloud driver + `run_experiment` wiring)
**Consumes:** OMOP (`condition_era`/`condition_occurrence`, `concept`, `concept_ancestor`, `observation_period`), the piece-1 DAG builder (`charmpheno/omop/condition_dag.py`), the existing cohort/BOW machinery (`cohorts.py`, `topic_prep.to_bow_dataframe`, `bigquery.load_omop_bigquery`), and the engine's frontier helpers (`spark_vi.models.topic.dag_placement`).
**Produces:** the `(train, test, DagLayout inputs, vocab_map, name_by_id, ledger)` bundle the gated-SVI MLlib shim (`GatedLDAEstimator`) fits and the engine's `evaluate` scores.

## Goal

Assemble, from OMOP, the labeled corpus for hierarchical case-finding: one document per patient tagged with its set-valued DAG **frontier** (the clinical truth), the pruned label `DagLayout`, and a held-out split with the leakage strip applied — everything the gated-SVI shim needs to `fit` and the engine needs to `evaluate`, produced as a pure, unit-testable transformation over OMOP frames.

## Decisions (brainstorm)

- **Document unit = patient-year, one year per patient.** Documents are patient-years (`PatientYearDocSpec`, so in-window features + the era-replication semantics are available), but **exactly one representative year per patient** so the corpus is not biased toward patients with more history. This maps onto the existing `apply_population_disease_cohort` pattern, which already selects one windowed year per patient.
- **Cohort = diabetes + background population.** Foreground = patients who attest a node under anchor **201820** (their first-diagnosis-year window); background = non-diabetes patients (a random observed year). Tagged by `source_cohort`. Background trains the shared background topics and supplies true negatives.
- **Frontier = per-doc, set-valued, in-window.** Each foreground doc's truth = `frontier_from_coded` over its in-window attested type nodes (most-specific attested nodes; incomparable ones kept as a set). Background docs → empty frontier (ungated).
- **Prune counts = distinct patients per node.** Learnability is "enough distinct patients populate this node," not patient-years. `prune_by_attestation(min_n)` with `min_n` a surfaced knob tuned from the ledger.
- **Pruned attestation rolls up.** A patient attesting a concept whose node is pruned reattaches to the nearest surviving ancestor (the same `_nearest_surviving_ancestors` walk the prune and ledger use) — the patient still contributes a coarser frontier; consistent with the ledger's coarsening accounting.
- **Split by patient, deterministic.** A salted hash of `person_id` assigns train/test (default `holdout_frac=0.2`); with one doc per patient this is one doc per side, but keyed on patient so it stays correct if the doc unit ever changes. Resume-stable (fixed salt, not `F.rand()`), mirroring `_RANDOM_WINDOW_SALT`.
- **Leakage strip at EVAL only.** Training docs keep all tokens (the gate, not token removal, is what welds topics to nodes). Held-out docs have their DAG-node codes stripped from the BOW, so a held-out patient's placement cannot trivially read its own type code — the case-finding test.
- **Test set = foreground + background.** Foreground held-out docs (codes stripped) measure recovery/ranking; background held-out docs measure specificity (should show low affinity / place near root).

## The three id spaces (the main correctness risk)

Every value in this piece lives in exactly one of three integer spaces; the translations must be exact and are the primary thing the tests pin:

1. **Concept-id space** — raw OMOP `concept_id`. The DAG (`ConditionDag`) is built and pruned here; `counts` for the prune are keyed by concept-id; roll-up of a pruned attestation walks the concept-id DAG.
2. **Engine-id space** — the contiguous `0..N` ids from `ConditionDag.to_engine()` (anchor → 0). The `DagLayout`, `frontier_from_coded`, and `label_from_coded` all operate here. Frontiers are emitted in engine-id space (that is what the shim's `labelCol` and `evaluate` expect).
3. **Vocab-index space** — the `[0, V)` BOW indices from `vocab_map: {concept_id: idx}`. The `features` SparseVector and the leakage strip live here; the DAG-node codes stripped at eval are `{vocab_map[cid] for node concept-ids cid in vocab_map}`.

Ordering that respects these spaces:

```
build_condition_dag (concept-id)               # piece 1
  -> counts{concept-id: n_distinct_patients}
  -> prune_by_attestation (concept-id)         # pruned DAG, still concept-id
  -> pruned.to_engine() -> (parent_int, int2cid, cid2int)
  -> per doc: attested concept-ids
       -> roll dropped ones up to nearest surviving ancestor (concept-id, pre-prune DAG)
       -> map survivors via cid2int -> engine-ids
       -> frontier_from_coded(engine-ids, DagLayout) -> frozenset (engine space)
  -> to_bow_dataframe (vocab-index) + attach frontier column
  -> split; strip {vocab_map[cid] for node concept-ids} from held-out features only
```

## Architecture

**New module `charmpheno/charmpheno/omop/case_finding_assembly.py`** that *composes* existing pieces (does not invade them). Spark DataFrame in/out; the pure per-doc frontier logic is factored so it unit-tests without Spark.

Units (names indicative; finalized in the plan):

- **`"diabetes"` entry added to `cohorts._DISEASE_REGISTRY`** (`inclusion_ancestors=(201820,)`, exclusions as needed) so `apply_population_disease_cohort(disease="diabetes", ...)` yields the fg+bg, one-year-per-patient, tagged cohort. This is the only edit to an existing module; everything else is the new module.
- **`load_condition_dag(spark, *, anchor, cdr, billing) -> ConditionDag`** — reads the anchor's standard-condition descendant `node_ids` (from `concept`) and the min-sep-1 `concept_ancestor` edges among them, calls piece-1 `build_condition_dag`. The BQ-loading wrapper around the pure builder.
- **`doc_attested_nodes(events_df, node_cids) -> DataFrame[doc_id, person_id, source_cohort, attested_cids: array<int>]`** — per doc, the in-window condition concept-ids that are DAG nodes (`∩ node_cids`). Foreground docs get their type attestations; background docs get `[]`.
- **`node_patient_counts(attested_df) -> dict[concept-id, int]`** — distinct `person_id` per attested node (for the prune).
- **`doc_frontiers(attested_df, before_dag, keep_cids, cid2int, lay) -> DataFrame[doc_id, frontier: array<int engine-ids>]`** — roll pruned attestations up to nearest surviving ancestor, map to engine ids, `frontier_from_coded`. Empty → `[]`.
- **`assemble_case_finding_corpus(spark, *, anchor=201820, min_n, holdout_frac=0.2, split_salt, vocab_size, min_df, min_patient_count, window_days, prior_obs_days, person_mod, ...) -> CaseFindingBundle`** — the orchestrator threading the pipeline above; returns the bundle.
- **`CaseFindingBundle`** (a small dataclass): `train_df` / `test_df` (`person_id, doc_id, features, frontier: array<int>, source_cohort`), `parent_int` (`{child:[parents]}` for `DagLayout`), `int2cid` / `cid2int`, `vocab_map`, `name_by_id`, `ledger` (from `pruning_ledger`, incl. coarsening).
- **Leakage strip** (`strip_node_features(features, node_vocab_idxs)`): a SparseVector transform dropping the DAG-node vocab dims; applied to `test_df` foreground+background rows only. (The engine's numpy `strip_dag_node_codes` is the token-array analogue used in the local tests; the Spark corpus needs the SparseVector version.)

**Caching / determinism:** reuse the `_corpus_cache` key discipline (fold `min_n`, `anchor`, `holdout_frac`, `split_salt`, `cohort_defs_version()`, `condition_dag`'s module version, and the `doc_spec.manifest()` into the key) so an edit to any assembly input invalidates the cache. The split salt is a fixed constant (like `_RANDOM_WINDOW_SALT`), not `F.rand()`.

## Interfaces (boundaries)

- **In:** the CDR/billing coordinates + anchor concept-id + the knobs (`min_n`, `holdout_frac`, `split_salt`, `vocab_size`, `min_df`, `min_patient_count`, `window_days`, `prior_obs_days`, `person_mod`). The OMOP tables and `concept_ancestor` are read via the existing `bigquery.py` / cohort loaders.
- **Out:** `CaseFindingBundle`. `train_df` → `GatedLDAEstimator(featuresCol="features", labelCol="frontier", parent=parent_int).fit(train_df)`. `test_df` → the model's `transform` → `nodeAffinity`, scored by `dag_placement.evaluate` against the test frontiers. `int2cid`/`name_by_id` carry interpretation back for `render_profile`.
- The engine stays domain-agnostic (integer ids); this module is the domain bridge where concept-ids and the three id spaces are expected.

## Testing

- **Unit (synthetic OMOP frames, small; Spark local where a DataFrame is needed, pure where possible):**
  - `doc_attested_nodes`: only in-window, only DAG-node concepts survive; background docs → `[]`.
  - `node_patient_counts`: distinct patients, not patient-years (a patient coding a node in two years counts once).
  - `doc_frontiers`: single root→node path → most-specific node; incomparable attestations → set; a pruned attestation rolls up to the nearest surviving ancestor; empty → `[]`. All asserted in engine-id space.
  - split determinism: same `person_id` + salt → same side across two runs; a patient's docs never straddle the split; `holdout_frac` roughly honored.
  - leakage strip: held-out foreground features lose exactly the DAG-node vocab dims and nothing else; train features are untouched.
  - bundle round-trip: `parent_int` loads into `DagLayout`; `train_df`/`test_df` schema matches the shim's `fit`/`transform`; the ledger's `K_nodes` equals the `DagLayout` node count.
- **Real-data smoke (skipped without the vocab/CDR, e.g. CI):** assemble on anchor 201820 with a small `person_mod`; assert the DAG has ~127 pre-prune nodes, the bundle loads into the shim, and the ledger reports a sane coarsening rate.

## Scope / deferred

- **In scope:** the new assembly module + the single `_DISEASE_REGISTRY` `"diabetes"` entry, the frontier/counts/split/strip logic, the bundle, and their unit + smoke tests.
- **Deferred (piece 3):** the cloud driver `dag_placement_cloud.py` (fit via the shim + `evaluate` + save), the `run_experiment.py` `model_class: dag_placement` wiring (the four if-chains + a `build_dag_placement_args`), the subsample-scale decision, and the pre-registered real-DAG `init="random"` vs `"spectral"` A/B. The complication anchor (442793) / combined forests remain a one-line anchor change.

## References

- Griffiths & Steyvers (2004), Hoffman et al. (2010) — the engines this feeds.
- The piece-1 design `docs/superpowers/specs/2026-07-15-condition-dag-builder-design.md` and the engine design `docs/superpowers/specs/2026-07-15-gated-svi-placement-engine-design.md`.
- The anchor-first case-finding design `docs/superpowers/specs/2026-07-15-anchor-first-hierarchical-case-finding-design.md` (in-window labeling, leakage strip at eval, per-node + DAG-distance evaluation).
