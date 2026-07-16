# DAG-Placement Cloud Driver + run_experiment Wiring — Design

**Date:** 2026-07-16
**Status:** Design approved (brainstorm), pre-implementation
**Piece:** 3 of 3 of the cluster driver (piece 1 = condition DAG builder; piece 2 = case-finding assembly; this piece = cloud driver + experiment wiring)
**Consumes:** the piece-2 assembly (`charmpheno.omop.case_finding_assembly.assemble_case_finding_corpus` → `CaseFindingBundle`), the gated-SVI MLlib shim (`spark_vi.mllib.topic.gated_lda.GatedLDAEstimator/Model`), the engine's scoring (`spark_vi.models.topic.dag_placement.evaluate`, `DagLayout`, `render_profile`), the gated block-aligned spectral init (`spark_vi.models.topic.gated_init.spectral_block_aligned_lambda`), the corpus-cache discipline (`analysis/cloud/_corpus_cache.py`), and the experiment runner (`scripts/run_experiment.py`).
**Produces:** a runnable `dag_placement` experiment — `make exp ID=N` fits the gated-SVI case-finding engine on real diabetes OMOP data and reports hierarchical placement metrics (AUC-by-depth, MRR, top2) — plus the pre-registered `init="random"` vs `"spectral"` A/B as two experiment configs.

## Goal

Make the gated-SVI hierarchical case-finding engine runnable as a first-class tracked experiment on the cluster: a thin cloud driver that assembles the labeled diabetes corpus, fits the gated shim, scores held-out placement, and saves an artifact; the `run_experiment.py` wiring so `make exp ID=N` drives it; a write-through cache for the (expensive) assembly; and the shim change that makes `init="random"` vs `"spectral"` a config flip so the pre-registered A/B runs on-cluster.

## Decisions (brainstorm)

- **Init A/B via the dense spectral path (this piece); scalable projected variant deferred.** The gated block-aligned spectral init (`gated_init.spectral_block_aligned_lambda`) already exists and runs in the dense, driver-side form; it just isn't threaded through `GatedLDAEstimator`. Wire it exactly as the STM shim wires its dense spectral path (`mllib/topic/stm.py:1524-1541`): expose an `init` param, and when `init="spectral"` collect the training docs to the driver, build `data_summary={"train_docs":…, "train_labels":…}`, and pass it to `VIRunner.fit(rdd, data_summary=…)`. The **scalable projected block-aligned** init (distributed co-occurrence + random projection, for vocabularies too large for a driver-side V×V) is the documented large-V follow-up — the gated analogue of STM's ADR 0032 arc — out of scope here.
- **Build the assembly cache now.** `assemble_case_finding_corpus` currently re-extracts from BigQuery, refits CountVectorizer, and rebuilds/prunes the DAG on every run. Add a write-through bundle cache in the driver layer (mirroring `_corpus_load.load_or_build_corpus` wrapping the domain `to_bow_dataframe`), so `case_finding_assembly.py` stays cache-free and the driver owns caching.
- **Save format = pg_stm pattern.** The NPMI coherence eval cannot score a placement model (`eval_coherence_cloud.py`'s `--model-class` choices exclude it), so eval is inline. Save `np.savez` of the fit arrays (`lambda`, `alpha`) plus a `manifest.json` (metrics, ledger, corpus_manifest, resolved config) — the same "methods-experiment artifact" pattern as `pg_stm_bigquery_cloud.py:244-274`. The `manifest.json` is also what makes `run_experiment`'s `find_most_recent_fit`/resume detection work.
- **`make exp ID=N` drives it generically; no dedicated Makefile target.** Once the four `run_experiment` chains are wired, the generic `exp` target runs a `dag_placement` experiment with no new Makefile recipe. A dedicated `dag-placement-bq-fit-eval` target (ad-hoc, non-tracked flow) is deferred (YAGNI).
- **Config: `_base.yaml` defaults + a diabetes cohort YAML + two experiment files.** New dag_placement knobs live in `_base.yaml` (read by `build_dag_placement_args`); the A/B is two experiment `.md` files differing only in `init`.

## The critical invariant: `nBg`/`tpn` single-sourced

`assemble_case_finding_corpus` builds a `DagLayout(parent_int, n_bg, tpn)` to compute the frontier engine-ids, and `GatedLDAEstimator` rebuilds its OWN `DagLayout(parent, nBg, tpn)` for gating. If the two disagree, the block ranges and `K` diverge and the gate is wrong. The driver MUST pass the SAME `n_bg`/`tpn` (from one config source) to both the assembly and the estimator, and the scoring `DagLayout` in the eval adapter MUST also use them. This is the primary correctness risk of the driver and is asserted in tests.

## Architecture

### A. `init` wiring in `GatedLDAEstimator` (`spark_vi/mllib/topic/gated_lda.py`)

- Add an `init` Param (`"random"` default; `"spectral"` opt-in), validated against `{"random"} | set(gated_init.INIT_STRATEGIES)` at fit time (unknown → ValueError naming `init`).
- Construct `GatedOnlineLDA(lay, V, init=<init>, …)` (replace the hard-coded `init="random"`).
- In `_fit`, when `init != "random"`: `docs = rdd.collect()` and build `data_summary={"train_docs": [token-id array per doc], "train_labels": [frontier frozenset per doc]}`. Reconstruct each token-id array from the gated BOW doc as `np.repeat(indices, counts.astype(int))` (the block-aligned init's `word_cooccurrence`/`_as_counts` expect token arrays). Pass `data_summary` to `VIRunner(...).fit(rdd, data_summary=data_summary)`; for `init="random"` pass no `data_summary` (unchanged path).
- **Dense-only guard:** the block-aligned init builds a driver-side V×V co-occurrence. Add a `spectralMaxVocab` param (default e.g. 8000, ~0.5 GB at float64) — if `init="spectral"` and `V >= spectralMaxVocab`, raise `NotImplementedError` naming the scalable-projected follow-up and suggesting a smaller `vocab_size` or `init="random"`. (Mirrors STM's `resolve_spectral_method` warning, but as an error since the scalable gated path is unbuilt.)
- Record `init` (and resolved dense/deferred) in the model's result metadata for provenance.

The engine side already works: `GatedOnlineLDA.initialize_global(data_summary)` resolves `INIT_STRATEGIES["spectral"](data_summary, lay, V)` and seeds `gp["lambda"]`. `VIRunner.fit(rdd, data_summary=…)` already forwards `data_summary` to `initialize_global` (the STM path proves this). No engine change.

### B. Bundle cache (`analysis/cloud/_case_finding_cache.py` + `load_or_build_case_finding_bundle`)

A write-through cache for the `CaseFindingBundle`, structured like `_corpus_cache.py`:

- `compute_bundle_cache_key(**assembly_params) -> str` — SHA-256 over the assembly-affecting inputs: `source_table, person_mod, vocab_size, min_df, min_patient_count, doc_min_length, prior_obs_days, window_days, anchor, min_n, holdout_frac, split_salt, n_bg, tpn`, plus `cohorts.cohort_defs_version()` and content hashes of `condition_dag.py` + `case_finding_assembly.py` (so any assembly-logic edit auto-invalidates — same discipline as `cohort_defs_version`).
- `save(spark, bundle, cache_uri, key)` — write `train_df`/`test_df` as parquet under `…/key/train`, `…/key/test`; write the python fields (`parent_int, int2cid, cid2int, vocab_map, name_by_id, ledger`) to `…/key/meta.json`. JSON stringifies int keys; on load, restore them to int.
- `try_load(spark, cache_uri, key) -> CaseFindingBundle | None` — read parquet + `meta.json`, rebuild the bundle with int-restored keys; `None` on miss.
- `load_or_build_case_finding_bundle(spark, *, cache_uri, **assembly_params) -> CaseFindingBundle` — HIT: return cached; MISS (or `cache_uri is None`): `assemble_case_finding_corpus(**assembly_params)` then `save`. Prints HIT/MISS like `_corpus_load`.

`case_finding_assembly.py` stays cache-free (no domain → `analysis/cloud` dependency); the driver layer owns caching, exactly as `load_or_build_corpus` wraps `to_bow_dataframe`.

### C. Cloud driver (`analysis/cloud/dag_placement_cloud.py`)

Mirrors the STM/LDA driver skeleton:

1. **argparse** — corpus knobs (`--source-table, --person-mod, --vocab-size, --min-df, --min-patient-count, --doc-min-length, --prior-obs-days`), assembly knobs (`--anchor` default 201820, `--min-n`, `--holdout-frac`, `--window-days`), gating knobs (`--n-bg`, `--tpn`), SVI knobs (`--max-iter`, `--seed`, `--cavi-max-iter`, `--cavi-tol`), `--init {random,spectral}`, `--spectral-max-vocab`, `--cdr`, `--billing`, `--save-dir`, `--cache-uri`. (No `--K` — K is emergent = `n_bg + (#surviving nodes)*tpn`.)
2. **Env + Spark** — `configure_logging()` + `make_spark_session(...)` from `_driver_common` (STM-style `with` block); `--cdr/--billing` supplied by `run_experiment`'s `_require_workspace_env`.
3. **Assemble (cached)** — `bundle = load_or_build_case_finding_bundle(spark, cache_uri=…, anchor=…, person_mod=…, min_n=…, n_bg=…, tpn=…, …)`. Log `bundle.ledger` (kept/dropped/K_nodes/coarsening) — the receipt for the emergent K.
4. **Fit** — `est = GatedLDAEstimator(featuresCol="features", labelCol="frontier", parent=bundle.parent_int, nBg=n_bg, tpn=tpn, maxIter=…, seed=…, init=init)`; `model = est.fit(bundle.train_df)` inside `with _phase("gated-svi fit"):`.
5. **Transform + score (inline eval)** — `scored = model.transform(bundle.test_df)` (adds `nodeAffinity`); adapt to `evaluate` (see the adapter below); `metrics = dag_placement.evaluate(profiles, test_labels, lay)` with `lay = DagLayout(bundle.parent_int, n_bg, tpn)`. Print `auc_by_depth`, `mrr`, `top2`, `mean_hops`, `frontier_size_mean`, `multi_frontier_rate`.
6. **Spot-check render** — for a few held-out foreground docs, `render_profile(profile_dict, lay, names=<engine-id-keyed names>, true_node=<frontier>)`, where the names are remapped `{i: bundle.name_by_id[c] for i, c in bundle.int2cid.items() if c in bundle.name_by_id}` (the concept-id → engine-id remap the bundle docstring warns about). Hash any patient/doc id before printing (row-level log rule).
7. **Save** — `np.savez(save_dir/"dag_placement_result.npz", lambda=…, alpha=…)` + `(save_dir/"manifest.json").write_text(json.dumps({metrics, ledger, corpus_manifest, config, init}))`.

**The transform→evaluate adapter** (factored pure for testing) — `profiles_from_scored_rows(rows, lay) -> (profiles, test_labels)`: each row's `nodeAffinity` is a `DenseVector` ordered by `lay.nodes`, so `profile = dict(zip(lay.nodes, nodeAffinity.toArray()))`; `test_label = set(row["frontier"])`. Collecting the test set to the driver is fine (held-out scale).

### D. `run_experiment.py` wiring (the four chains)

- **Chain 1 — `validate_frontmatter` (~:271):** add `"dag_placement"` to the allowed `model_class` tuple. No STM-style required-field block — the dag_placement knobs all have `_base.yaml` defaults (a missing knob falls back, not errors).
- **Chain 2 — `build_fit_driver_path` (~:286):** `if model_class == "dag_placement": return f"{base}/dag_placement_cloud.py"`.
- **Chain 3 — `build_fit_args` (~:299):** `if model_class == "dag_placement": return build_dag_placement_args(effective, out_dir, resume_from)`. New `build_dag_placement_args` mirrors `build_pg_stm_args` (`:608-657`): corpus + assembly + gating + SVI + `--init` flags from `effective`, `--save-dir out_dir`, `--cache-uri` from `effective.get("cache_uri")`. Resume is NOT supported (GatedLDAModel is not persistable in v1 — `gated_lda.py:112-116`); `build_dag_placement_args` ignores `resume_from` and the driver always fits fresh.
- **Chain 4 — eval dispatch (~:1102):** fold `dag_placement` into the same branch as `pg_stm` (skip NPMI: "eval is inline in the fit driver").

### E. Config

- **`_base.yaml`** — a `# --- dag_placement defaults ---` block: `anchor: 201820, min_n: <TBD from ledger, e.g. 50>, n_bg: 2, tpn: 1, holdout_frac: 0.2, window_days: 365, init: random`. (These keys are read only by `build_dag_placement_args`; harmless to other model classes, same as the STM block.)
- **`experiments/defaults/population_diabetes.yaml`** — thin cohort YAML (`cohort`/`cohort_def`) so `load_defaults` finds a file to merge. (Corpus identity actually comes from `anchor` + the hard-coded diabetes cohort inside `assemble_case_finding_corpus`; `--cohort` is vestigial here, as with the STM gated cohorts.)
- **Two experiment files** `docs/experiments/NNNN-dag-placement-diabetes-{random,spectral}.md` — identical frontmatter except `init: random` vs `init: spectral`, the pre-registered A/B. `model_class: dag_placement`, `cohort: population_diabetes`, `person_mod: <subsample>`, `doc_unit`: n/a (the driver hard-uses PatientCohortDocSpec via the assembly).

## Interfaces (boundaries)

- **In:** the experiment frontmatter + `_base.yaml`/cohort defaults (the knobs above), the workspace env (`--cdr/--billing` from `_require_workspace_env`), and the OMOP CDR + `concept`/`concept_ancestor`/`observation_period` read by the assembly.
- **Out:** `save_dir/{dag_placement_result.npz, manifest.json}` and the printed placement metrics. The engine stays domain-agnostic; the driver + config are where diabetes/concept-ids/the three id spaces live.

## Testing

- **`init` wiring (`spark_vi/tests/test_gated_lda_shim.py`, extend):** `init="spectral"` on a tiny gated corpus fits without error and its `lambda` differs from the random-init seed (spectral seed took effect); `init="banana"` raises ValueError matching "init"; `init="spectral"` with `V >= spectralMaxVocab` raises NotImplementedError naming the scalable follow-up. Do NOT re-run the slow placement-equivalence gate.
- **Bundle cache (`analysis/cloud/tests/…` or `charmpheno/tests/…` per repo convention):** save→try_load round-trips a bundle equal in DataFrame contents (train/test rows) and in every python field (int keys restored, ledger identical); `compute_bundle_cache_key` changes when any of `{anchor, min_n, holdout_frac, person_mod, vocab_size, n_bg, tpn}` changes and is stable otherwise; miss→build→hit path returns the built bundle on the second call. Use synthetic frames + a `tmp_path` `file://` cache_uri.
- **`profiles_from_scored_rows` (pure):** given synthetic rows with `nodeAffinity` DenseVectors + `frontier` arrays and a `DagLayout`, returns `profiles` dicts keyed by `lay.nodes` in the right order and `test_labels` as sets; feeds cleanly into `evaluate`.
- **`build_dag_placement_args` (`scripts/tests/…`):** given an `effective` config dict, returns the expected flag list (corpus + assembly + gating + `--init` + `--save-dir`, no `--K`, no resume flag).
- **`run_experiment` dispatch (`scripts/tests/…`):** `model_class="dag_placement"` passes `validate_frontmatter`, selects `dag_placement_cloud.py`, routes `build_fit_args` to `build_dag_placement_args`, and takes the NPMI-skip eval branch.
- **`nBg`/`tpn` single-source invariant:** a test that the bundle's `DagLayout(parent_int, n_bg, tpn)` and the estimator's rebuilt `DagLayout(parent, nBg, tpn)` produce the same `K` and `block` map when fed the same `n_bg`/`tpn` (guards the driver's must-match contract).
- **Real-data smoke (skipped without CDR):** `dag_placement_cloud.main` importable + arg surface; the assemble→fit→evaluate wiring is covered by the unit pieces above (the end-to-end BQ run is the cluster smoke).

## Scope / deferred

- **In scope:** the `init` shim wiring (dense spectral), the bundle cache + `load_or_build_case_finding_bundle`, `dag_placement_cloud.py` (assemble→fit→transform→inline evaluate→save), the four `run_experiment` chains + `build_dag_placement_args`, the `_base.yaml`/cohort-YAML/two-experiment config, and their tests.
- **Deferred:** the **scalable projected block-aligned** spectral init (large-V; gated analogue of STM ADR 0032) — dense-only with a clear guard for v1; a dedicated `dag-placement-bq-fit-eval` Makefile target (use `make exp`); resume/warm-start (GatedLDAModel not persistable in v1); any dashboard (placement output is AUC/MRR, not topic-word — the topic dashboard drivers hard-exclude this model_class); the complication-anchor (442793) / combined-forest variants (one-line `anchor` change).

## References

- Griffiths & Steyvers (2004), Hoffman et al. (2010) — the engines. Arora et al. (2013) — the anchor-word spectral init the block-aligned strategy generalizes.
- Piece-1 spec `docs/superpowers/specs/2026-07-15-condition-dag-builder-design.md`; piece-2 spec `docs/superpowers/specs/2026-07-15-case-finding-assembly-design.md`; engine spec `docs/superpowers/specs/2026-07-15-gated-svi-placement-engine-design.md`.
- STM dense/scalable spectral precedent: `mllib/topic/stm.py:1498-1541`, ADR 0032/0037. pg_stm artifact-save precedent: `pg_stm_bigquery_cloud.py:244-274`.
