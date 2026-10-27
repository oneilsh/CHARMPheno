# Bolt-on audit: what to keep, cut and move (2026-10-26)

**Status:** proposal for the maintainer; nothing changed yet. Decisions go in the
"Decide" column (or the questions at the end); the cleanup ADR(s) and commits follow.

**Scope.** Everything added since 2026-08-01 (baseline `199935e7`): about 600 commits,
~60k added lines across `spark-vi/`, `charmpheno/`, `analysis/`, `scripts/`. Inventoried
by area against the central configuration (exp 0163: leaves-first, balanced seed, leaf
seed, random init, multi-domain, n_bg 200, tpn 1, `weight_y: 0`, `diag_only: true`) and
against the evaluations in use since 0155 (own-code census, digest, gated NPMI,
new-diagnosis AUC on the ribbon, framings survey). "Last used" is the highest
experiment whose front matter or run commands use a non-default value.

Builds on [2026-10-23 cleanup scope](2026-10-23-spark-vi-cleanup-and-review-scope.md)
(already decided: cut the supervised head; keep spectral init and HPO profile seeds as
options; walkthrough after the API is settled).

Verdicts: **CORE** (the central configuration or a current evaluation needs it),
**OPTION** (keep, off by default), **CUT** (dead or superseded), **MOVE** (keep, wrong home).

## 1. The shape this points to

1. **spark-vi becomes a gated-LDA engine plus generic readout and evaluation.** Once the
   supervised head goes, `OnlinePCLDA` / `OnlinePCLDAEstimator` add nothing over
   `GatedOnlineLDA` at `weight_y = 0`; the two MLlib estimators (`GatedLDAEstimator`,
   `OnlinePCLDAEstimator`) duplicate the init Params, spectral routing, per-domain λ
   conversion and `domainBounds` checks. Merge them into one gated-LDA estimator. The
   generic readout (batched L-BFGS heads, closure-product stacking, ranking metrics,
   the generic readout-DAG core) and topic evaluation (gated NPMI, block-topic census)
   move into spark-vi from `analysis/`. Exact module layout is the API ADR's job.
2. **charmpheno takes the clinical definitions now in `analysis/cloud`:** the native
   Mondo label DAG and code map, the Mondo→OMOP port, the first-attestation sidecar,
   gate-ever attestation, the prevalence rules, the egress floor (defined in 7 files
   today), name resolution, the ribbon reader.
3. **`analysis/cloud` returns to thin drivers:** fit (ends at save), readout,
   inspect/evaluate. Today the fit driver `gated_pc_cloud.py` is 6,005 lines, of which
   ~1,700 run on the 0163 path and ~3,600 are a readout library that
   `gated_pc_readout.py` imports (~20 private functions) and partly re-implements.
4. **Dead lines of work go** (git keeps them; tag the last pre-cleanup commit, e.g.
   `pre-cleanup-2026-10`, so they are one checkout away).

## 2. spark-vi

| Feature | Lines (≈, + tests) | Last used | Verdict | Decide |
|---|---|---|---|---|
| Gated engine, multi-domain dict λ, export format v2 | gated_lda ~350, domains 131, export +126 | 0163 | CORE | |
| Leaves-first schedule (`gateWarm`, heights, capped gate) | ~150 | 0163 | CORE | |
| Open-time seed balanced / subtree; leaf seed | ~200 | 0163 | CORE; flip `gateSeedMode` default to balanced | |
| `docConcentration` reaching the gated engine | small | 0163 | CORE | |
| Batched multi-head L-BFGS (`batched_lr.py`) | 834 | 0163 (readout) | CORE → MOVE to the readout module (no topic-model dependency) | |
| Runner: `_agg_depth`, broadcast `destroy` | small | always | CORE; consolidate the 6 copies of `_destroy_broadcast` | |
| **Supervised head** (λ correction, trust caps, Flat/DAG heads, newton/sgd, ~18 Params, `PCDocument` types) | pc.py ~1,300; tests ~2k; `manual_*` 1.4k | 0120 | CUT (decided) | |
| Localized head support (`localizeHead`, `headSupport`, `allowed_with_*`) | ~200 | inert in 0163 | CUT. Note: 0163 sets `localize_head: true`, so C support sets are built and ride every E-step closure for nothing | |
| L-BFGS co-fit seam + `CofitHeadStatsProvider` | ~200 + ~150 | 0120 | CUT | |
| Warm start with fresh RM counter (`warmStartFrom`, `VIRunner.fit(warm_start_from=)`) | ~150; test 384 | 0075 | CUT (only the supervised two-phase protocol used it) | |
| Spectral init (dense, scalable, bg anchors) | ~1,000 | 0161 | OPTION (decided) | |
| Spectral variants: `anchorScope=frontier`, `spectralTopoOrder=reverse`, `spectralMarginalFloor` | small | 0128–0131 | CUT | |
| Guided (HPO-preferred) anchors, `spectralAnchorCandidates` | ~250; test 234 | 0131 | CUT (the profile *seed* is the kept HPO comparator) | |
| HPO profile seed (`gateSeedProfile`) | ~30 | 0158 | OPTION (decided); take the primary domain as a parameter (hard-codes domain 0) | |
| Profile-η prior (`etaBoost`, `profileEta*`, 5 Params) | ~250; tests ~590 | 0120 | CUT (line closed, insight 0087; different from the profile seed) | |
| `countTransform` | ~40 | 0115 (once) | CUT | |
| α policy: `equalized` / EB α | ~80 | 0121–0122 | CUT equalized; EB α OPTION (0134: keep off) | |
| Per-domain η, ω modality weight | ~150; ~25 tests | never set | CUT | |
| Multi-domain on `GatedLDAEstimator` (node-affinity shim) | +337 | never (dag_placement last 0069) | folds into the merged estimator | |
| Dense multi-domain spectral | ~100 | 0044 | CUT unless a small-V dense option is wanted | |
| Flat `tpn=0` layout | 6 | 0132 | OPTION | |

Domain-generality leaks to fix in the API pass: "condition domain" in Param help
(`gateSeedOwnCodes`, `gateSeedProfile`, `etaBoost`), `hpoa_stage2_probe` in `profileEta`
help, domain 0 hard-coded in the profile seed. `spark-vi/tests/test_pc_lda.py` imports
the `analysis/pc` prototype as its oracle, a test-time spark-vi → analysis dependency.

## 3. Fit driver and runner (`gated_pc_cloud.py`, `run_experiment.py`, Makefile)

| Feature | Last used | Verdict | Decide |
|---|---|---|---|
| Fit, corpus spec / cache key, native Mondo assembly, gate-ever, manifest, fit-only save | 0163 | CORE | |
| In-driver readout after the fit (~430 lines main + calibration second solve + incident arm) | not on the 0163 path (`diag_only` exits at :5434) | CUT from the fit; the fit ends at save, readout is one entry point | |
| `unsup_gated` twin arm, `dag_head` arm, eval-every logger, oracle ladder, `per_node_head_report` | 0103 / 0080 / 0081 | CUT | |
| Readout knobs in the fit (`readout_mode`, `readout_theta_mass`, `readout_l2`, …) | 0163 | MOVE to the readout driver; the manifest stops carrying readout defaults | |
| Old Mondo DAG path (`dag_source: mondo`), `dag_collapse` | 0109 | CUT the driver path; **leave `mondo_dag.py`/`mondo_collapse.py` files** (hashed into every key) | |
| SNOMED gated path, `window_mode: forward`, `min_n` | ≤0089 | CUT from the gated driver (SNOMED stays for LDA/STM/dag_placement) | |
| Episode / matched-random index arms, `gate_frontier_days`, `episode_index.py`, `diag_episode_probe.py` | 0111/0112, never evaluated | CUT (key folds are no-ops when off; leave them) | |
| Pre-index closure (E1) | 0110 | CUT the flag; keep `feature_window_index_table` (the sidecar uses it) | |
| Fit-time `label_set` | 0146 | CUT (the ribbon moved to `--readout-nodes`) | |
| `gate_no_open_seed` | never | CUT | |
| `fit_save_interval`, `disk_telemetry.py` | 0120 / 0110 | OPTION (cheap safety) | |
| `run_experiment` `build_pc_args` (model_class `pc`), `CHARM_DEV` supervised trims | 0075 | CUT / simplify | |
| Makefile: `readout-ab`, `hpoa-stage2-probe`, `report`, `diag-*`, `pc-antidepressant`; pre-baseline `next-exp`, `lr-readout`, `build-dashboard-exp`, `build-covariates`, lda/hdp smokes | 0042–0128 | CUT (check the LDA/HDP ones against the README first) | |

## 4. Readout and evaluation

| Feature | Last used | Verdict | Decide |
|---|---|---|---|
| Batched-LR distributed readout, lean collect (`--eval-path driver`), stacked closure product, ranking/detection metrics, readout-node subset | 0163 | CORE → MOVE generic parts to spark-vi | |
| Readout DAG | 0163 | CORE; split the generic core (spark-vi) from the Mondo builder (charmpheno) | |
| Prevalence check with `--prev-history`, `compare_incident_runs.py` | 0163 | CORE; generic AUC split / clinical prior rule | |
| First-attestation sidecar | 0163 | CORE → MOVE to charmpheno | |
| Gated NPMI (`topic_npmi.py`) | 0163 | CORE → MOVE to spark-vi eval | |
| Own-code census, digest; `--tour`, `--redundancy`, `--grep` | 0163 / 0155 | CORE (generic classification → MOVE); tour/grep OPTION | |
| Framings survey: differential E2/E3/E5/E6, next question, survey table, `survey_analysis.py` | 0163 | CORE | |
| Differential engines E1 (presence subtype), E4 (gated likelihood); `evaluate_next_question` E1/E3 variant; `--diff-train-ever`, `--diff-multi` | 0141–0149 | CUT | |
| Simulated acquisition mode (`--acquisition`, `--acq-*` except tags) | 0139 | CUT the mode; keep `build_pool`, `ModelEngine` (the survey uses them) | |
| VOI tables CLI (`inspect_topics --voi*`) | 0138 | CUT the mode; keep the library functions | |
| Conditional from saved heads, fit-time conditional readout (~480 lines) | 0138 | CUT or OPTION | |
| Look-alikes (`--lookalikes`) | 0142 | CUT; move its `auc()` into prevalence_check first | |
| Feature-mask ablation, incident heads, E2 incident arm, A/B parity, read sampling | 0103–0144 | CUT | |
| Collect-free eval (`--eval-path distributed`) | 0111 | CUT, or OPTION if the collect ever OOMs (it has not at whole Mondo) | |
| Isotonic calibration (default **on** in the fit driver; a second batched solve) | 0079 | CUT | |
| sklearn driver readout (`--readout-mode driver/auto`) | none since 0103 | CUT from the CLI; keep as the test oracle | |
| Raw-code baseline (`--raw-baseline`) | 0147 | OPTION | |
| `inspect_topics` `--compare-*`, `--strip-audit`, `--collinearity`, `--profile-support`, `--profile-align`; `--readout-auc`, `--measurement-states` | 0116–0140 | CUT the first five; OPTION the last two | |
| `conversion_analysis.py`, `diag_incident_census.py` | 0110 | CUT | |
| `analysis/pc/evaluate.py` metric primitives (shipped to executors) | 0163 | CORE → MOVE; its PC-vs-baselines evaluators CUT | |

## 5. Data, DAG and other files

| Feature | Last used | Hashed? | Verdict | Decide |
|---|---|---|---|---|
| Native Mondo DAG + code map, `mondo_collapse`, Monarch port, multi-domain assembler, measurement tokens, population index | 0163 | Y (port, tokens: N) | CORE | |
| `mondo_usage_core.py` (940; ~70% dead) | 0163 (3 functions) | Y | trim only inside a cache drop | |
| **Antidepressant / MDD (Hughes) line**: `cohorts.py` ~900 lines, `pc_antidepressant_cloud.py` 2,193, tests ~1,050, defaults | 0075 | **Y (`cohorts.py`)** | CUT; the `cohorts.py` part needs the cache drop | |
| `anchor_selection*.py`, `anchor_neighborhoods.py`, `anchor_hierarchy_cloud.py`, `mondo_completeness_cloud.py`, `mondo_hierarchy_cloud.py`, `priority_seed.tsv` | 0086–0088 | N | CUT, after moving `_download_cached`/`_read_bq` (imported by hashed `mondo_native_dag.py:854`, so the move is part of the drop; or leave the two helpers where they are) | |
| HPO stage 1 survey, `hpo_profile_marks.py`, `build_profile_seed` | 0152/0158 | N | OPTION (feeds the comparator) | |
| HPO stage 2 probe, eta-boost / anchor-candidate builders | 0116–0131 | N | CUT | |
| `diag_wearables_rare.py`, `diag_sibling_support.py`, `gated_pc_report.py`, `name_spectral_anchors.py` | — / 0110 / 0110 / 0131 | N | CUT (name_spectral_anchors: OPTION with spectral) | |
| `analysis/pc/` Hughes prototype (`model`, `objective`, `slda_reference`, `generative`, `head`, `variants`, diagnostics) | 0075 | N | CUT, after replacing it as the oracle in `test_pc_lda.py` | |
| `analysis/local/sim_leaves_first.py` | report 2026-10-19 | N | OPTION | |
| `spark-vi/tests/manual_*` (10 files) | 2026-08 | N | CUT | |

## 6. The cache drop

Driver-only cuts keep every key byte-identical: the corpus-spec flags fold into the key
only when set (`_multidomain_params`, `gated_pc_cloud.py:3391-3456`). The hashed edits
are few, and none is needed for correctness:

- remove the antidepressant/MDD block from `cohorts.py`;
- trim `mondo_usage_core.py` to its live functions;
- move `_download_cached`/`_read_bq` out of `anchor_selection_cloud.py` (changes an import
  line in `mondo_native_dag.py`);
- moving any hashed module into charmpheno (its import lines change, so its hash does);
- **add** `measurement_tokens.py`, `bigquery.py`'s loaders and `topic_prep.py` to the key:
  they shape the corpus but are not hashed today, so an edit there would poison the cache
  under an unchanged key.

Recommendation: one deliberate drop, after the package moves are decided, so it happens
once. Each bundle rebuild needs the n2-standard-8 master; the sidecar is keyed separately
and survives (moving `conversion_sidecar.py` drops only the sidecar).

## 7. Found along the way

- `differential_survey` hard-codes `prevalence_check_history.json` /
  `prevalence_check_stacked_raw2000.json` (`gated_pc_readout.py:1103-1104`), so its
  framing-3 column is empty on 0161/0163 (`survey_analysis.py` works around it).
- `make exp` does not forward `GPR_SPARK_CONF` (Makefile:385-388; `run_experiment` reads
  `CHARM_SPARK_CONF`), so the `spark.eventLog.enabled=false` in recent fit commands never
  reached the fit.
- The fit driver's `--weight-y` default is 50 (`gated_pc_cloud.py:4514`) and other
  defaults are stale (snomed, rare6, label, forward, test_only, full, prior-obs 365);
  `_base.yaml` supplies `kappa 0.7`, `optimize_doc_concentration: true`, `n_bg 20`,
  `tpn 5`. Runs are correct only because the front matter sets everything. After the
  cut the defaults should be the central configuration.
- 0163 sets `topic_trust`, `head_*`, `grad_cavi_iters`, `localize_head`,
  `baseline_max_iter`: all inert at `weight_y: 0` (and `topic_trust` is unused at any
  weight).
- AGENTS.md: `condition_dag.py` is in `charmpheno/charmpheno/omop/`, not
  `analysis/cloud/`; `preindex_closure.py` only moves 0110's key, not every key; the
  sample cluster command uses `diag-episode-probe`, a CUT candidate.
- Duplicates: `EGRESS_FLOOR` ×7, `_ece` ×3, `resolve_run_dir` ×2, `_curie` ×2, DAG
  depth/closure helpers ×4, the dict-λ → `expElogbeta` fusion ×3, DagLayout built twice
  per fit, a CAVI θ twin beside `lda._cavi_doc_inference` (goes with the head cut).
- The readout solve rarely converges in 200 iterations, but 600 iterations leave the AUCs
  unchanged (exp 0163); its per-pass cost is the θ scan, an argument for the local
  readout in the two-engine design.

## 8. Proposed order

1. **Driver-side cuts** (no cache effect): dead drivers and Makefile targets, dead flags
   and arms in `gated_pc_cloud.py`/`gated_pc_readout.py`, fit ends at save, defaults set
   to the central configuration, the bugs in §7. Tag `pre-cleanup-2026-10` first.
2. **spark-vi cut and merge** (ADR): supervised head, warm start, dead init variants,
   profile-η, ω; one gated-LDA estimator; tests replaced where they used the PC oracle.
3. **API and two-engine ADR + spec**: core vs MLlib shim, local runner and local readout,
   where the generic readout and evaluation live (§1).
4. **Package moves and the single cache drop** (§6), as one announced commit.
5. **Walkthrough** of what remains.

## Decisions so far

- 2026-10-26: **Q1 yes.** Cut the antidepressant/MDD (Hughes) line, including its
  `cohorts.py` block and the `analysis/pc` prototype. **Resurrection point: commit
  `fff7bf4067`** on `claude/dismech-ribbon` (the last commit before any cut; the
  tag `pre-cleanup-2026-10` exists locally only, as tag pushes are refused here).
- 2026-10-26: **Profile-η prior: cut** (the continuous word-side prior; the profile
  *seed* stays). **Fit-time label set: cut** (the ribbon is a post-hoc readout subset).
  **Cache drops: not a cost concern** (clusters are recreated near-daily and caches are
  on HDFS), so hashed edits happen where they belong, each re-pinning the tripwires,
  with a `--bundle-only` rebuild checking corpus counts after any refactor-only edit.
- 2026-10-26: **Pre-index closure: cut** (the label rules are: pre-index features, a
  post-index label year, train-only gating with ungated test transforms, heads trained
  on all positives, incident-only evaluation via the whole-history sidecar; to be
  written down in one place). **Organization: option A** (the `core / dag / models /
  readout / eval / mllib` layout), `readout` kept as the name, `dag/` strictly generic,
  framings stay in `analysis/`, MLlib-compatible flow with a heads Estimator. Collect-
  free metrics must be exact and must not assume anything fits on the driver: one
  method, a distributed sort of the (node, score) cells with per-node ranks (decided over
  positive-anchored counts), and ROC/PR curve output for a node sample.
- Pending:
  collect-free evaluation (the readout needs a scalable metrics path before it moves to
  spark-vi); Q6 organization (option A, spark-vi, leaning; layout to brainstorm).

## Progress

**Step 1 (driver-side cuts), 2026-10-26/27 — done except the items listed below.**
About 21k lines removed across ~70 files (from commit `fff7bf4067`):

- Dead drivers and probes: antidepressant PC driver, Mondo completeness/hierarchy,
  wearables census, sibling support, conversion analysis, run report, HPO stage-2
  probe, incident census, episode index + probe, look-alikes, with tests, defaults
  and Makefile targets.
- Fit driver: ends at its save (no in-driver readout, arms, calibration, eval logger);
  native Mondo only (SNOMED / anchor-Mondo / collapse / pre-index closure / label set /
  episodes / profile-η / guided anchors removed). Native keys byte-identical until the
  `cohorts.py` drop below; a spec asking for a removed option is refused, not re-keyed.
  Both drivers reject abbreviated flags (a stale `--k 4` had parsed as `--kappa 4`).
- Readout: one path (distributed batched solve, lean eval collect, optional stacked
  product); acquisition, look-alikes, conditional, incident heads/arm, feature masks,
  A/B, sklearn driver mode, collect-free cell explode, calibration and co-fit head arm
  removed. `--readout-mode distributed` is accepted as a no-op; `--readout-l2` defaults
  to 100. `inspect_topics` loses strip-audit, collinearity, profile-support,
  profile-align and the VOI CLI.
- Defaults: `population_mondo_all.yaml` is the central configuration (0163's fit argv
  is byte-identical under it). `make exp` forwards `GPR_SPARK_CONF`. The survey's
  framing-3 column reads the current file names. AGENTS.md hashed-module facts fixed.
- **Cache drop:** the antidepressant block left `cohorts.py` (commit `671345b7`); every
  bundle/corpus/covariate key moved and the tripwire pins were re-pinned. The next
  readout of an existing run on a fresh cluster rebuilds its bundle (the drift gate
  checks it matches the fit).
- Tests: no new failures against the pre-cleanup baseline (the remaining failures are
  the environmental local-Spark ones AGENTS.md describes).

**Still open from step 1:** the differential's E1/E4 engines and `train_ever`/`multi`
internals (framings stay in `analysis/` for now); the anchor-selection trio
(`_download_cached`/`_read_bq` are imported by hashed `mondo_native_dag`); the
`analysis/pc` prototype (it is spark-vi's test oracle, so it goes with step 2).

**Step 2 (spark-vi, ADR 0050), 2026-10-27 — done.** About 24k lines removed across
~130 files, one commit per ADR item. One estimator, `DagGatedLDAEstimator` /
`DagGatedLDAModel` over the engine `DagGatedOnlineLDA` (`models/topic/dag_gated_lda.py`),
with the gate, the leaves-first schedule and seeds (`gateSeedMode` default balanced, new
`gateSeedDomain` for the primary domain), `docConcentration`, `optimizeDocConcentration`,
multi-domain columns and the persistence Params; random init plus the gate seeds. Removed:
`OnlinePCLDA` and its shim, the PC document types, every head/label Param, the localized
support helpers, the co-fit provider, `warmStartFrom`/`warm_start_from`, the DAG spectral
init (block-aligned, multi-domain, scalable, bg anchors, guided anchors, anchor dump,
`name_spectral_anchors.py`, the digest's ⚓), the profile-η prior, `countTransform`,
equalized α, ω and per-domain η, the `dag_placement` model class with
`dag_placement_cloud.py`, `lr_readout.py` and the placement scoring, and the `analysis/pc`
reference except `evaluate.py`'s metric primitives and the `batched_lr` re-export. The fit
driver builds the new estimator and no longer saves `w_CK`/`b_CK`; `gated_pc_readout`
rebuilds a `DagGatedLDAModel` from any run's npz (old runs' head arrays are ignored);
`run_experiment` refuses a spec that sets a removed option to a non-inert value. Kept:
`resume_from`, `batched_lr.py`, the flat `spectral_init*.py` (STM default), `DagLayout`
and `frontier_from_coded` at their hashed-module-facing path, and `fit_gated`/`profile`
as the engine's test oracle. One `destroy_broadcast` in `spark_vi.core`; one dict-λ
reader (`expElogbeta_from_lambda`). No hashed module edited; the tripwire suite passes
unchanged (one test of the removed head probe was dropped from that file, no pin moved).
Tests: spark-vi 663 passed (884 before; the difference is removed-feature tests, the
`test_pc_lda.py` collection error is gone); charmpheno 250 passed, 1 skipped; drivers 26
failed / 8 errors, exactly the pre-existing environmental local-Spark set (no new failure
by test id). `make zip` and `make build` succeed for both packages.

**ADR 0051 step 1 (LocalRunner), 2026-10-27 — done.** The iteration loop moved out of
`VIRunner.fit` into `spark_vi.core.loop.run_svi_loop` (schedule, scaling, ELBO,
diagnostics, checkpoint, resume, early stop), which both engines call; `VIRunner` keeps
only the Spark E-step, now destroying each iteration's broadcast as soon as its
treeReduce returns (same destroy count). `LocalRunner` (`spark_vi.core`) samples rows as
`RDD.sample` does (Poisson with replacement, Bernoulli without) and runs `local_update`
in optional chunks merged by `combine_stats`. Eight fast tests
(`test_local_runner.py`, ~1 s) and one slow agreement test against local Spark on the
gated LDA with the leaves-first schedule: full batch equal to 1e-8, minibatch topic
distance Spark-vs-local 0.096 against local-vs-local 0.071.

**ADR 0051 step 2 (readout extraction), 2026-10-27 — done.** `spark_vi.readout`:
`kernels` (the numpy kernels, verbatim from `distributed_readout.py`), `solver` (the
batched L-BFGS, moved from `models/topic/batched_lr.py`), `heads` (`fit_heads`,
checkpoints and the heads npz, from `gated_pc_cloud`), `spark` (`SparkReadoutData`,
the passes and the lean collect), `local` (`LocalReadoutData`, new) and `stacked`
(the closure product); the Spark-action retry went to `spark_vi.core.retry`. The
driver keeps the clinical seams: `distributed_readout.py` became `doc_key.py` (the
doc key only, no longer on `--py-files`), and `gated_pc_cloud.readout_arm(engine=)`
(was `distributed_score_arm`) runs either backend; `gated_pc_readout
--readout-engine {spark,local}` replaces the no-op `--readout-mode`. The pre-index
eligibility run of the lean collect (cut with the pre-index closure) is gone. The
metrics stay in `analysis/` until step 4 (spark-vi has no scikit-learn). Tests:
the kernel and heads tests moved into spark-vi; the readout integration test runs
both engines against the sklearn oracle (it errored in the old baseline and passes
now); local and Spark backends fit the same heads to 1e-6.

**Test tiers, 2026-10-27.** Every test that requests the `spark` fixture is now
marked `slow` by a conftest hook, plus explicit marks on the pure-numpy tests over
~0.3 s: spark-vi's unit suite went from ~9.5 min to 4.7 s (544 tests), charmpheno's
from ~98 s to 1.1 s (148). AGENTS.md "Testing expectations" states the tiers and
when to run each.

**ADR 0051 step 3 (spark_vi.dag, charmpheno.mondo; the cache drop), 2026-10-27 —
done.** `spark_vi.dag`: `layout` (`DagLayout`, `frontier_from_coded`, from
`models/topic/dag_placement.py`) and `ops` (closure, descendants, depth, the transitive
reduction, nearest-kept walks, the closure matrix; verbatim from the Mondo builders,
the readout and the fit driver). The collapsed-Gibbs oracle went to
`spark-vi/tests/_gibbs_oracle.py`, so `dag_placement.py` is gone. `charmpheno.mondo`:
`native_dag` (with the two live functions of `mondo_usage_core`), `collapse`,
`omop_mapping`, `ontology` (the release fetch and CDR reader out of
`anchor_selection_cloud.py`). Removed: `mondo_dag.py` (the anchor build), the
anchor-selection trio and `anchor_hierarchy*`, the `anchor_select` model class,
`priority_seed.tsv`, the rest of `mondo_usage_core` (the HPO parsers went to
`hpoa_profile_survey.py`); the ribbon TSV moved to `analysis/cloud/label_sets/`.
**The cache drop** (one commit): the bundle key folds the moved modules,
`spark_vi.dag` and the new `corpus_code_version()` (bigquery / measurement_tokens /
topic_prep, unhashed before), drops the anchor-Mondo, `dag_collapse`, `label_set` and
pre-index folds (refused by the driver since step 1) and goes to payload v5; the
corpus and covariate keys fold `corpus_code` too. The conversion sidecar's key also
moved (its legacy anchor branch was removed), so it rebuilds once. Tripwires
re-pinned.

**Found while verifying step 3: the full-batch step-size fix was never on this
branch.** `3f71de55` ("full-batch VI must use rho=1", 2026-07-29) lives only on sibling
branches; its slow gate test was cherry-picked here (`43689f35`) without it and has
failed ever since, unnoticed because nothing ran the slow tier. Ported into the shared
loop (`spark_vi.core.loop`), so VIRunner and LocalRunner both take rho = 1 on full
batches. Production fits are mini-batch (subsampling 0.1) and unaffected; any
full-batch fit made on this branch under-converged. Four runner tests that leaned on
the damped step to keep a conjugate model from converging now use
`CountingModel(early_stop=False)`.

## Questions for the maintainer

1. Antidepressant/MDD (Hughes) line, including its `cohorts.py` block and the
   `analysis/pc` prototype: cut, relying on the git tag?
2. Old Mondo DAG path, SNOMED on the gated path, episodes, pre-index closure, fit-time
   label set: cut all five driver paths?
3. Readout modes in §4 marked CUT (acquisition, look-alikes, conditional, calibration,
   feature masks, E1/E4, sklearn driver mode, collect-free eval): cut all, or keep any?
4. Profile-η prior and guided anchors: cut (the profile seed stays as the HPO comparator)?
5. The cache drop: one batched drop at step 4, or none (leave hashed modules frozen and
   only add the missing hashes)?
6. Home for generic readout and evaluation: inside spark-vi (e.g. `spark_vi.readout`,
   `spark_vi.eval`) or a separate package? (Can wait for the API ADR.)
