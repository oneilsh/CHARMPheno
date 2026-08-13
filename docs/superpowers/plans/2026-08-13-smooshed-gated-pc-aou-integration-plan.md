# Smooshed-first Gated-PC → AoU integration — plan + handoff

**Date:** 2026-08-13
**Branch (base for this work):** `claude/spectral-anchor-topic-k-200nqp`
**Status:** Plan approved to start ("smooshed-first"). No integration code written yet.
This doc is a compaction-surviving handoff: state, decisions, the plan, and how to resume.

---

## 1. What this session built (state to carry forward)

The PC / Gated-PC arc is complete and green on this branch. Core pieces:

- **`OnlinePCLDA`** (`spark-vi/spark_vi/models/topic/pc.py`) — prediction-constrained LDA.
  Wraps a topic-engine **delegate**; head is a pluggable `SupervisedHead`
  (`FlatLogisticHead` / `DagClosureHead`).
- **Gated-PC composition** — `OnlinePCLDA(topic_engine=GatedOnlineLDA(lay, ...))`. The head is
  topic-engine-agnostic (reads only `global_params["lambda"]/alpha/K`), and `GatedOnlineLDA`
  *is-a* `OnlineLDA`, so the topic-side DAG gate × label-side head compose by injection (ADR
  0042). New doc type **`GatedPCDocument`** (`types.py`) = features + frontier + y + label_mask.
- **`head_l2` is an ABSOLUTE ridge = Hughes `lambda_w`, default 1e-3** (ADR 0041). NOT
  per-doc×n_docs (that was an ~840× over-regularization bug). `head_l2=0` blows up on the
  separable topics PC creates. Good basin ~1e-4…1e-2.
- **MLlib shim** (`spark-vi/spark_vi/mllib/topic/pc.py`) `OnlinePCLDAEstimator` exposes:
  `gateParent` (JSON DAG map → gated engine), `gateNBg`, `gateTpn`, `frontierCol`,
  `closureParents` (DAG head), `headL2`, `weightY`, `headOptimizer="newton"`.
- Tests green: `test_pc_lda_shim.py` (incl. 2 gated-PC), `test_pc_dag_head.py`, `test_pc_lda.py`,
  `test_gated_lda.py`, `test_mllib_pc_persistence.py`, `analysis/pc` (121).

**Load-bearing empirical findings (do not relearn):**
- **Inject the hierarchy ONCE.** Gate + FLAT head wins; gate + DAG-closure head COLLAPSES to
  chance on realistic β (0.745 vs 0.495; `manual_gated_pc_realistic.py`). Use gate+flat OR
  ungated+DAG-head, never both. (ADR 0042.)
- **Measure representation quality with `pc_topics_lr`** (post-hoc LR on the shaped θ), NOT the
  co-fit head AUC (insight 0066). It's convergence-robust and comparable to the two-stage
  baseline.
- PC only helps in the **hidden-low-mass** regime (rare phenotype the unsupervised fit misses).
  On the high-mass AoU antidepressant task PC was marginal (insight 0066). Rare-disease
  case-finding is the named forward test where PC *should* help.
- Joint vs alternating optimization is NOT the gap (a reference alternating fit matches joint);
  it was the head_l2 miscalibration. Keep Newton, no Adam, no joint rewrite (ADR 0041 / insight
  0068).

ADRs: 0039 (Newton head), 0040 (superseded-in-part), 0041 (absolute head_l2), 0042 (Gated-PC).
Insights: 0065, 0066, 0067, 0068. Runbook: `docs/reports/2026-08-13-gated-pc-real-run-readiness.md`.

---

## 2. Branch map (the divergence to resolve)

- **THIS branch `spectral-anchor-topic-k-200nqp`** — latest PC head fixes + Gated-PC + shim.
  ALSO already carries the OMOP case-finding infra: `charmpheno/charmpheno/omop/`
  (`case_finding_assembly.py`, `condition_dag.py`, `cohorts.py` incl. **`rare6`**),
  `analysis/cloud/dag_placement_cloud.py`, `pc_antidepressant_cloud.py`, `scripts/run_experiment.py`.
  → A smooshed rare6 Gated-PC run needs NO cross-branch merge to start.
- **`claude/hybrid-domain-reliability-review-ckn2bq`** — has, IN ADDITION:
  - **Anchor selection (Mondo→SNOMED)**: `analysis/cloud/anchor_selection.py`,
    `mondo_to_omop_mapping.py` (port of Monarch mondo2omop), `anchor_hierarchy.py`/`_cloud.py`,
    `anchor_neighborhoods.py`, `anchor_selection_data/priority_seed.tsv`. Grows rare6 → ~20–30
    rare-disease anchors in ontology neighborhoods. Spec:
    `docs/superpowers/specs/2026-07-31-expanded-snomed-anchor-selection-design.md`.
    KEY CHOICE: OMOP `concept_class_id='Clinical Finding'` does NOT separate disorders from
    cross-cutting findings; the separator is **descent from SNOMED "Disease" (concept_id
    4274025)** — `--restrict-under 4274025` drops cross-cutting findings.
  - **Multi-domain (MixEHR) explicit betas**: `charmpheno/omop/multi_domain.py`,
    `spark_vi/models/topic/domains.py` (domain_bounds), `analysis/cloud/multidomain_cloud.py`,
    `GatedLDAEstimator(featuresCols=[...])` → per-domain dict-λ; condition-only gate ⟂ domain.
    This is on the UNSUPERVISED gated model only.
- **Does NOT combine** with our supervised PC head. Multi-domain Gated-PC is a follow-on
  (§5): the head reads `global_params["lambda"]` as a single (K,V); a dict-λ would break it.

Merge direction is a later decision (§5). For smooshed-first with rare6, this branch suffices.

---

## 3. The gap to close (smooshed-first)

Existing `assemble_case_finding_corpus(spark, disease="rare6", ...)` → `CaseFindingBundle`
gives `train_df/test_df = [person_id, doc_id, features, frontier(engine-ids), source_cohort]`
+ `parent_int, int2cid, cid2int, vocab_map, name_by_id, ledger`. That feeds the UNSUPERVISED
`GatedLDAEstimator`. To feed **Gated-PC** we additionally need `label` + `labelMask`.

That is the ONLY missing seam. Everything else (gate, head, shim, DAG) exists.

---

## 4. The plan (concrete steps, in order)

### Step A — frontier→(label, labelMask) adapter  [small, pure, unit-testable]
A function that, given a doc's `frontier` (engine node ids) and the DAG (`parent_int` /
`DagLayout`), returns the `(C,)` label + mask.
- `C = len(int2cid)` (engine nodes incl. root 0).
- `label[c] = 1` iff `c ∈ closure(frontier)` (each frontier node + all its ancestors via
  `parent_int`; root 0 ⇒ always 1). Background doc (empty frontier) ⇒ all-zero label.
- **labelMask policy (DEFAULT, revisit):** `ones(C)` for every doc — the coded conditions give
  a full membership vector; a background doc is a known negative for all disease nodes. Caveat:
  coded-absence ≠ true-absence (data-quality, not a mask); an alternative is to observe only the
  closure path + siblings. Make it a flag (`label_mask_mode`).
- **Where it lives:** add an optional path in `charmpheno/omop/case_finding_assembly.py`
  (behind a flag, e.g. `emit_labels=True`) that appends `label`/`labelMask` array columns to
  train_df/test_df, so ONE bundle serves both the gated placement and Gated-PC. Keep the pure
  frontier→(label,mask) helper standalone + unit-tested (no Spark), the column-append thin.
- Test: closure correctness on a diamond DAG; background ⇒ zero label; C matches int2cid.

### Step B — `gated_pc_cloud.py` driver  [mirror dag_placement_cloud.py]
New `analysis/cloud/gated_pc_cloud.py`:
- Assemble bundle (`disease="rare6"`, cached via `_case_finding_cache`), Step-A labels on.
- Build `parent_int` → pass as `gateParent` (JSON), `gateNBg`/`gateTpn` from the layout.
- Fit `OnlinePCLDAEstimator(gateParent=parent_int, labelCol="label", labelMaskCol="labelMask",
  frontierCol="frontier", weightY=<~tokens/doc; tune>, headOptimizer="newton", headL2=1e-3,
  weightYWarmupIters=10, subsamplingRate=<0.01–0.1>, maxIter=100+)`. **Flat head** (inject-once).
- Score three arms for the headline comparison (insight 0066's forward test):
  1. **Incumbent:** unsupervised gated placement (existing `dag_placement_cloud` node_affinity).
  2. **New:** gate + flat PC head — `pc_topics_lr` (post-hoc LR on shaped θ) + head P(node) + node_affinity.
  3. (optional) ungated + DAG-closure head — the label-side-only alternative.
- Eval via `dag_placement.evaluate` (per-node AUC + length-conditioned FDR; insight 0064:
  ranking AUC ≠ FDR-controlled discovery — report both). Save npz + manifest (dag_placement_cloud pattern).
- Unit-test only parse_args + pure helpers (cluster-covered main, per the repo idiom).

### Step C — run + read
`make -C analysis/cloud …` a bq-smoke, then a rare6 fit. Headline question: **does gate + flat
PC head beat unsupervised gated placement (node_affinity) on rare-disease `pc_topics_lr`?**
Expect a gain in the hidden-low-mass regime (that's the whole thesis); if not, it's a data
finding (à la 0066), not a bug.

### Step D — (later) expanded anchors
Swap `rare6` → the hybrid branch's ~20–30-anchor set (`anchor_selection` + `anchor_hierarchy`,
`--restrict-under 4274025`). Requires bringing that infra onto this branch (§5).

---

## 5. Open decisions (need user input before/at each)
1. **labelMask policy** — full-observation `ones(C)` default vs closure-path-only. (Step A)
2. **weight_y scale** — Hughes ≈ tokens/doc, "possibly much larger"; tune on validation.
3. **Merge direction** — to run expanded anchors + eventually multi-domain, either merge the
   hybrid anchor/OMOP/multidomain infra ONTO this branch, or port Gated-PC + head_l2 fix onto
   hybrid. Recommend assessing recency/test-state first. NOT needed for rare6 smooshed start.
4. **Multi-domain Gated-PC (follow-on)** — make `OnlinePCLDA`'s head + topic-correction
   `domain_bounds`/dict-λ aware (mirror SP3a), then inject the multi-domain gated engine. Do
   AFTER a smooshed baseline exists. Motivated by insight 0064 (binding constraint = information
   = meds/labs domains).

---

## 6. How to resume (env + commands)
- Python venv: `/home/user/CHARMPheno/.venv-pc/bin/python`.
- Pyspark tests / manual runs need:
  `JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
   PYSPARK_PYTHON=<venv> PYSPARK_DRIVER_PYTHON=<venv>
   PYTHONPATH=/home/user/CHARMPheno/spark-vi:/home/user/CHARMPheno`.
- Composition sanity (fast, no Spark): `scratchpad/dbg_gated_pc.py` (in-memory wiring).
- Gated-PC end-to-end (Spark): `spark-vi/tests/manual_gated_pc_case_finding.py`,
  `manual_gated_pc_realistic.py`, shim smoke `scratchpad/smoke_gated_pc_shim.py`.
- OMOP assembly entry: `charmpheno/charmpheno/omop/case_finding_assembly.py:
  assemble_case_finding_corpus(spark, disease="rare6", cdr, billing, ...)`.
- Driver template to copy: `analysis/cloud/dag_placement_cloud.py`.
- First code to write: Step A adapter (pure helper + column append), then Step B driver.
