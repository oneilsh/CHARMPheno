# Audit: temporal/incident eval program vs. the codebase — 2026-09-01

**Status:** pre-spec audit, adversarial, read-only. All citations at HEAD `e6209c7`.
**Subject:** the incident/episode evaluation program discussed 2026-09-01 (pre-index
closure primitive; dual prevalent/incident metrics; incident-local cells with
P-stratification; future-conversion analysis; episode-anchored index sampling;
doc repairs). The spec MUST incorporate this audit's findings; the design as
discussed is NOT buildable verbatim.

**Verdicts:** E1 CLEAN-WITH-NOTES · E2 CONFLICT · E3 CLEAN-WITH-NOTES (mostly
already built) · E4 CLEAN-WITH-NOTES · E5 CONFLICT (highest risk) · E6 CLEAN.

---

## Element 1 — Pre-index closure primitive

**Verdict: CLEAN-WITH-NOTES** (mechanically easy; the cost is entirely in cache-key blast radius)

### Seams

The closure machinery is already fully parameterized over "which events frame", so running it a second time is a function call, not new logic:

- `charmpheno/charmpheno/omop/multi_domain.py:242-247` — `_attest(ev)` is a closure over `attested_provider`; it is called once on `train_lab` and once on `test_lab`. Calling it a third/fourth time on `train_doms[0]` / `test_doms[0]` (the condition FEATURE frame) is legal: the feature frame carries `person_id, concept_id, source_cohort` and a date column, which is exactly what `make_mondo_attested_provider`'s `provider()` requires (`analysis/cloud/mondo_dag.py:298-316`).
- `attach_frontiers` (`case_finding_assembly.py:261-274`) and `frontier_to_label` (`:106-160`) are pure functions of `(attested_cids, before_dag, keep, cid2int, lay)` — all available at `multi_domain.py:250-253`.
- `attach_labels` (`:163-191`) is already a generic UDF with `frontier_col=` / `label_col=` / `label_mask_col=` parameters. A second call with `frontier_col="preindex_frontier", label_col="preindexClosure"` is a one-liner.

So the primitive is: one extra `_attest`, one extra `attach_frontiers`, one extra `attach_labels`, one join on `doc_id`. Feasible as described.

### Conflicts / hidden costs

**1. The cache-key tripwire is the real cost, and it is severe.** `compute_bundle_cache_key` folds *source hashes of whole modules*:

- `_case_finding_cache.py:112-113` — `dag_src` = `condition_dag`, `assembly_src` = `case_finding_assembly` (folded on **every** key, SNOMED and Mondo alike)
- `:133` — `multi_domain_src` (folded whenever `multidomain=True`)
- `:152` — `mondo_src`

And `cohort_defs_version()` (`cohorts.py:36-52`) is a source hash of the *entire* `cohorts.py`, folded at `:111`.

Four hashes are **pinned as tripwire tests**: `tests/scripts/test_case_finding_cache_mondo.py:63-65` (`d658ce0a9a7425dd`, `3cf6c7aac6140393`, `275c8e6a76283e86`) and `:87` (`_MONDO_KEY_NO_COLLAPSE = "ca958995cc1cfb17"`), with an explicit comment at `:84-92`: *"this hash is a deliberate TRIPWIRE — any edit to `analysis/cloud/mondo_dag.py` moves it and orphans every cached Mondo bundle in every bucket… re-pin this only alongside a note saying why the caches were dropped."*

Consequence: **editing `case_finding_assembly.py` (where `frontier_to_label`/`attach_labels` live) moves `assembly_src` and orphans every cached bundle in the repo, including exp 0104's record bundle** (~20 min of BigQuery to rebuild per `test_case_finding_cache_mondo.py:83`) and breaks all four pinned tests. Editing `multi_domain.py` orphans every multidomain/Mondo bundle. The bundle *format* has room for an extra column (parquet writes take the frames as they are — `_case_finding_cache.py:216-219`); the *key* does not tolerate the edit.

**2. The established mitigation exists and is documented.** `dag_collapse` is the template: the reduction lives in its own module (`mondo_collapse.py`) and is applied **in the driver, between the DAG build and the assembler** (`gated_pc_cloud.py:1773-1783`), with an explicit comment (`:1774-1779`) saying it was placed there *precisely* to avoid moving `mondo_src`. Note `gated_pc_cloud.py` is **not** source-hashed into any key, so driver-level code is free. The flag folds only when on (`_case_finding_cache.py:160-166`).

The pre-index closure can follow that template *only if implemented as a driver-level post-pass*, because it needs post-prune internals. Those internals are all recoverable from the returned bundle: `bundle.parent_int` → `DagLayout(parent_int, n_bg, tpn)`, `bundle.cid2int`, `keep = set(bundle.int2cid.values())`. That is a viable design; it is not what "run the machinery a second time inside the assembler" implies.

**3. Meta/manifest does not round-trip the new column.** `_meta_dict` (`:171-191`) serializes only `parent_int/int2cid/cid2int/name_by_id/ledger/vocab_map(s)`. The extra column rides in the parquet, so `try_load` returns it silently — but nothing *records* that a given cached bundle has it, so a mixed-vintage cache dir will hand a readout a bundle without the column and fail at `select`. There is no witness field.

**4. Storage.** At C≈3,820, `label` + `labelMask` are already 2×3,820 float64 ≈ 61 KB/row. A third dense array is +50% on the bundle parquet. Stored as an index list instead of dense `array<double>`, negligible — but `attach_labels` emits dense (`case_finding_assembly.py:177-180`), so reusing it verbatim buys the 50%.

**5. Semantic caveat under 0110** — see Sequencing.

---

## Element 2 — Dual metrics (prevalent + incident), eval-time masks

**Verdict: CONFLICT** (the seam is genuinely clean; the *metric semantics* break in three named ways, one of which silently inflates the incident arm)

### Seams — clean

- **Driver path (the one actually used).** `readout_from_proba(proba, y_te, m_te, C, ...)` (`gated_pc_cloud.py:187-208`) takes the mask as an argument and passes it to `_bundle_masked` (`analysis/pc/evaluate.py:222-244`) and `pr_readout` (`gated_pc_cloud.py:116-145`). Calling it a second time with `m_incident` is literally a second call. No solver touched.
- **Spark path.** `score_cells_df(..., mask_col=...)` (`distributed_readout.py:1284-1323`) already takes `mask_col`; `_score_cells_kernel` (`:328-348`) only ever reads `np.flatnonzero(mask)`. But this path is **never called by any driver** — only tests — and its docstring calls it *"The ESCAPE HATCH for when D_te x C outgrows the driver, not the default path"* (`distributed_readout.py:31-32`). Wiring it is new work.
- **Collect.** Getting `m_incident` to the driver means extending the lean block: `_lean_eval_kernel` (`distributed_readout.py:565-643`) packs `(ids, P, y_idx, y_ptr, m_idx, m_ptr)`; a fourth CSR run plus the matching `_densify_lean_blocks` branch (`gated_pc_cloud.py:756-788`) is a mechanical +1 byte/cell. Neither file is source-hashed into a cache key — free.

`results_readout.json` rows come from `run_readout` → `_dump_partial_results` (`gated_pc_readout.py:525-528`), written after *each* arm — an extra `"gated_pc_incident"` key is a natural addition.

### Conflicts

**C2.1 — Degenerate heads become *scored* at 0.5 under incident masking, inflating the incident macro.** The sharpest finding in the audit.

At `gated_pc_cloud.py:856-858`, every node whose *train* cell was degenerate gets a **constant column**: `proba[:, deg] = const[deg]`. Degeneracy is a train-side property `(n_obs<=0)|(n_pos<=0)|(n_pos>=n_obs)` (`diag_sibling_support.py:78`) — 620 nodes post-splice, of which **619 are the subsumed category-anchors with `n_pos == n_obs` exactly**.

Under the *prevalent* mask those nodes are all-positive in test too, so `_score_label` returns `skipped: "degenerate test column (all-positive)"` and `_macro` drops them (`evaluate.py:87-95`, `:116-125`). Under the *incident* mask, prior carriers are removed from **both** classes — exactly the set that made the column all-positive. Some columns acquire negatives, become non-degenerate, and are **scored**: `roc_auc_score(y, const)` = **exactly 0.5**, `skipped: None`, included in the macro. `detection_readout` has a guard (`informative = np.ptp(...) > 0`, `:171`) added as an explicit *"EVAL bug fix, always on"* (`:155-165`) — but deliberately scoped to detection only (`:162-165`). The incident arm re-opens that bug on the ranking axis, with up to 619 forced-0.5 nodes.

**C2.2 — Prevalent and incident macros average different node sets; the headline comparison is not a comparison.** `_macro` averages only non-skipped labels (`evaluate.py:118-119`). At `min_label_count: 20` the prevalent arm scores 2,106 of 3,677; the incident arm scores strictly fewer, a *different* subset. The repo has a named remedy: the 0110 plan §5 mandates macro on BOTH the shared node set and the full space (`plans/2026-08-31…:144-146`). The program must carry that discipline over.

**C2.3 — Standing decision: eval masks must NOT depend on the run.** `conditional_readout`'s docstring (`gated_pc_cloud.py:296-301`): *"MASK-INDEPENDENCE: pass the FULL-closure observation mask (all-ones)… otherwise the closure mask silently makes the conditional eval an easier sibling-only contrast and cross-run numbers are not comparable (exp 0079, Trap 3)."* Incident eligibility is a per-run, corpus-derived mask — it obeys the letter, violates the spirit; the spec must define incident eligibility so it is stable across runs being compared.

**C2.4 — Heads are fit on the prevalent problem.** `_fit_readout_heads` standardizes per node on that node's own observed train rows (`gated_pc_cloud.py:977-979`). The incident metrics therefore report a *prevalent-fit model on an incident cohort* — a legitimate quantity, but not "the incident AUC"; the spec must name it as such. (Train-time incident masking is explicitly deferred; this is the deferral's cost.)

---

## Element 3 — Incident-local cells with P-stratification

**Verdict: CLEAN-WITH-NOTES** (substantially already built — twice)

### Prior art: this exists

**`conditional_readout` (`gated_pc_cloud.py:276-370`) already implements the cell definition verbatim, minus the incident filter:**
- `:310` `cohort = np.where((y[:,p]==1) & (mask[:,p]==1))` — the parent-P cohort
- `:315-320` positives = `y[rows,c]==1`, negatives = the rest of P's cohort = *siblings under P, or P-but-nothing-more-specific*

That is exactly the "local negatives" definition. Pure numpy over `(D,C)` arrays; already reports `cond_auc`, `cond_ap`, `marg_ap`, per-edge and pooled ECE, top-1 vs majority, per-node reliability; already in `results_readout.json` (`:2480`).

**`label_mask_mode="closure"` (`case_finding_assembly.py:148-156`) is the same construction at corpus level** and is the mainline setting (0104 front matter).

### What is actually new

Only the incident filter and the stratum: a third array threaded into `conditional_readout` (eligibility) and a stratum key (pre-index closure of P). Trivial signature change.

### Notes

- Stratifying halves each cell; at `min_count >= 20` on both classes (`:318`) most edges will drop out of one or both strata. Same C2.2 discipline needed.
- The `_ones` mask convention (`:2477-2479`) must be broken deliberately, with the comparability rule it protects re-stated.
- **`min_count` is not only statistical**: counts < 20 are *not disclosable* on All of Us (`evaluate.py:76-78`). Any stratified cell table is an egress surface; keep publishing rules separate from model-internal dials (0110 plan `:59-62`).

---

## Element 4 — Future-conversion analysis

**Verdict: CLEAN-WITH-NOTES**

### Confirmed facts

- **Full history is loaded, no date filter.** `load_omop_bigquery` (`bigquery.py:216-372`) applies only `person_sample_mod` and `concept_id != 0`. Confirmed.
- **Post-label-window events are discarded at windowing, not retained.** `lookback_feature_label_events` (`cohorts.py:1648-1668`) emits only the two windows; the cached bundle holds BOW vectors, not events (`_case_finding_cache.py:21-23`), so **nothing downstream can recover a future date**.

### Cheapest place

The `cond` frame at `multi_domain.py:365-367` — loaded once, full history, before windowing. One aggregation:

```
cond ⋈ climb_sdf on concept_id = descendant_concept_id
  → groupBy(person_id, ancestor_concept_id).agg(min(condition_era_start_date))
```

gives *first attestation date of each frontier node, ever*, per person. "First attestation of `closure(c)`" folds driver-side via `lay.closure`.

### Notes

- **Same source-hash problem as element 1**: `multi_domain.py` is inside the hashed assembler. A driver-level equivalent needs its own load of `cond` (a second full-table scan at `person_mod: 1`) or a **sidecar cache — ADR 0025** (covariate sidecar parquet) is exact prior art, and it explicitly contemplates multi-doc persons. Grain warning: the sidecar there is per person; element 4's artifact is per (person, node); under element 5, eligibility joins are per (person, index) — three grains.
- **Right-censoring is unhandled in the design statement.** "Conversion in later years" needs the denominator gated on `observation_period_end_date` at each horizon (`_window_observed_cohort`, `cohorts.py:693-729`), or the conversion rate is a censoring artifact.
- **The bound will be loose.** 0109 records root prevalence 0.9609 — most nodes' "did not gain c this year" is a weak negative; conversion rates will be high across the board. A finding, not a failure — but do not assume tightness.

---

## Element 5 — Episode-anchored index sampling

**Verdict: CONFLICT** — highest-risk element by a wide margin. Three independent walls: an id-type wall in the readout, a driver-memory wall, and a comparability wall against the 0110 protocol.

### 5a. Machinery reuse — partial

- `_random_event_windows` (`cohorts.py:1083-1145`) **cannot** be reused: it *is* the one-per-person sampler (`row_number() … rn == 1`, `:1136-1143`), eligibility defined on event dates; "just before episode start" is not an event date.
- `_window_observed_cohort` (`cohorts.py:693-729`) **can**: it takes an arbitrary `(person_id, index_date)` frame and preserves N rows per person. It drops other columns; rejoin the episode id as `_mdd_antidepressant_index` does (`cohorts.py:1905`).
- The gap-and-islands sessionization idiom exists at `cohorts.py:2237-2252` (`_stable_drug_intervals`) — a 60–90d episode clusterer is that pattern. Genuine prior art.

### 5b. The observation gate kills a large, non-random share of episodes

`_LOOKBACK_PRIOR_OBS_DAYS = 365` is **hardcoded and deliberately un-overridable** (`case_finding_assembly.py:43-53`). A person's *first* diagnostic episode is by construction at or near record start, and the gate is `index >= op_start + 365` — so the earliest, most unambiguously incident episodes are systematically dropped. **The "100% incident capture" claim is false as stated**; capture is conditional on ≥1 year prior observation, anti-correlated with incidence. The lookback spec's survivorship caveat (`specs/2026-07-18…:60-64`) is *amplified*, not escaped. No measurement of the kill rate exists; the spec must budget an empirical probe. Symmetrically, `index + 365 <= op_end` kills the last episode of every record.

### 5c. `index_date` is dropped before the doc spec sees it

`lookback_feature_label_events` drops `index_date` on both outputs (`cohorts.py:1663,1667`); `DocSpec.derive_docs` only sees the events frame. An `EpisodeDocSpec` needs `doc_id = cohort:person:index`, which requires editing `cohorts.py` — which moves `cohort_defs_version()` and **invalidates every cache key in the repo** (bundle, corpus, covariates) and all four pinned hashes.

### 5d. Join fan-out and driver broadcast growth

`cohorts.py:1659` joins events to a broadcast `index_df` per domain. M=5 indexes at `person_mod: 1` ≈ 2M rows / ~50 MB collected through the driver, three times (three domains). Survivable; linear in the multiplier.

### 5e. Documents from one person overlap ~80%

With `lookback_days: 1825`, consecutive episode docs share most events. Two unaddressed consequences:

- **Insight 0009 is a direct hit** (`docs/insights/0009-year-binning-intensifies-chronic-bg-for-hdp.md`): doc-multiplication for chronic patients drives catch-all topic growth. Worse here — windows overlap rather than partition. Whether `n_bg: 8` absorbs it is an open empirical question.
- **Effective sample size is far below N**: every CI and every `min_count >= 20` threshold silently assumes independent rows.

### 5f. Scale — the readout driver collect does not fit

From 0104's recorded run:

| quantity | 0104 record | ×3 episodes | ×5 episodes |
|---|---|---|---|
| D_te | ≈80k | 240k | 400k |
| lean eval bundle @6 B/cell, C=3,820 | 1.9 GB | 5.7 GB | 9.5 GB |
| `calibrate_per_node` float64 copy | 2.4 GB | 7.3 GB | 12.2 GB |
| observed train cells | 56.2M | 169M | 281M |
| readout pass @ topm=256 | 17.6 s | ~53 s | ~88 s |
| 60-iter solve (×2 with calibration) | ~1.2 h | ~3.5 h | ~6 h |

At ×5 the lean bundle plus the calibration float64 copy exceed a 16 GB driver. The distributed eval (`score_cells_df`/`per_node_metric_rows`) exists but **has never been called by a driver** (`distributed_readout.py:31-32`) — element 5 makes wiring it mandatory. No repo data exists on episodes-per-person; the multiplier is a guess until measured.

### 5g. It breaks the 0110 comparison protocol

Insight 0010: NPMI (and by extension any doc-unit-sensitive number) is not comparable across doc units. The 0110 acceptance protocol makes 0104's 0.6978/0.4845 the control on a shared node set (`plans/2026-08-31…:144-150`). An episode corpus changes the doc unit, the vocabulary (`min_df` is a document count), every node's prevalence, and the base rates — 0104/0109 stop being controls. Element 5 must NOT ride with 0110.

### 5h. Prior art in favor

`TOPIC_STATE_MODELING.md:148-155` names `EpisodeDocSpec` and `WindowedDocSpec` as sanctioned future specs; ADR 0018 anticipates anchor-event-centered windows. Element 5 is inside the designed extension point — but the doc's claim that a new DocSpec is "one class + manifest round-trip; the BOW build, fit drivers, and eval drivers don't change" is **now false**: the ADR-0046 readout stack hard-codes an int64 doc key and a dense driver collect (see seam list).

### 5i. Recorded stance against index fan-out

`cohorts.py:723-728` names over-weighting multi-period patients as a harm to avoid. Episode anchoring does it by design; the spec must state why the trade is now worth it.

---

## Element 6 — Doc repairs

**Verdict: CLEAN** — both claims verified; **three** sites carry the stale claim, not one:

1. `specs/2026-07-18-lookback-window-design.md:22-25` ("no disease code exists before it… strip becomes a no-op")
2. `docs/experiments/0061-dag-placement-rare6-lookback-1yr.md:55-59`
3. **In code**: `case_finding_assembly.py:394-396` — fixing this comment moves `assembly_src` and orphans every cache; ride it along with another assembler change, never alone.

Strip narrowness confirmed exactly: `drop_idxs` from `before_dag.nodes()` (`multi_domain.py:223,293`) = anchor cids + synthetic negatives; attestation climbs from descendants (`mondo_dag.py:298-310`), which are not stripped.

---

## Seams that break under multi-doc persons

1. **`_lean_eval_kernel` hard-codes int64 ids** (`distributed_readout.py:632-633`; driver twin `gated_pc_cloud.py:774`; `id_col="person_id"` at `:794,:1198`). String doc_ids raise. Needs a synthetic int64 doc key or dtype change both sides. **Hard break.**
2. **`readout_ab_report` aligns collects by a dict keyed on person_id** (`gated_pc_cloud.py:1338-1341`) — duplicates silently overwrite; the A/B gate would compare mismatched rows. **Silent wrongness in the correctness gate itself.**
3. **`lookback_feature_label_events` drops `index_date`** (`cohorts.py:1663,1667`). Fixing moves `cohort_defs_version()` → every cache key. **Hard break, maximum blast radius.**
4. **doc_spec is hard-coded in two places and absent from the cache key** (`multi_domain.py:408`, `gated_pc_cloud.py:1769`; `_SPEC_ASSEMBLY_KEYS` `gated_pc_cloud.py:1657-1662` has no doc-spec entry). Changing the driver-side doc spec alone would poison the cache under a byte-identical key. **Must be closed before any doc-unit work.**
5. **Driver-path calibration split is row-level** (`gated_pc_cloud.py:2467-2470`) — under multi-doc a person straddles cal/fit (exp 0079 run-2 failure). Distributed twin is person-keyed and safe (`:2434-2436`).
6. **`readout_ab_report` `sample_frac` is row-level** (`:1295-1298`).
7. **`detection_readout` is per-document** (`gated_pc_cloud.py:169-176`) — silently becomes episode-weighted under multi-doc.
8. **`min_df` is a document count** (`topic_prep.py:222-224`); `min_patient_count` (insight 0025) is the guard and is set — but the vocab still shifts.
9. **`min_doc_length: 10`** drops episode docs non-uniformly toward the incident end (shortest lookbacks).
10. **`node_patient_counts` and Mondo power counts are person-level and index-independent — safe** (`case_finding_assembly.py:250-257`, `mondo_dag.py:258-262`). C and the label DAG stay fixed across doc units: this is what makes an episode-vs-random comparison on a shared node set possible at all.
11. **`split_train_test` is explicitly multi-doc-safe** (`case_finding_assembly.py:277-289`). No change.
12. **`F.first("person_id")` groupings safe** given person-encoding doc_id (invariant documented `case_finding_assembly.py:219-221`).
13. **doc_id prefix parsing** (`split(":").getItem(0)`) survives an APPENDED index component only (`eval_coherence_cloud.py:273` et al.).
14. **`multidomain_bow` full-outer join is doc-keyed — safe** (`multi_domain.py:84-87`).

---

## Prior art found

| Program element | Prior art | Where |
|---|---|---|
| Multi-doc persons | `PatientYearDocSpec` / `PatientCohortDocSpec` shipped | `doc_spec.py:120-259` |
| Multi-doc persons | Person-keyed split written for this case | `case_finding_assembly.py:277-289` |
| Multi-doc persons | `min_patient_count` invented for exactly this | insight 0025; `topic_prep.py:86-95` |
| Multi-doc persons | Consequences measured: insights 0008/0009/0010/0014 | `docs/insights/` |
| Episode doc unit | `EpisodeDocSpec`/`WindowedDocSpec` sanctioned | `TOPIC_STATE_MODELING.md:148-155`; ADR 0018 |
| Episode clustering | Gap-and-islands sessionization | `cohorts.py:2237-2252` |
| Multi-index gating | `_window_observed_cohort` preserves N indexes/person | `cohorts.py:693-729` |
| Incident framing | Incident-new-user brackets (drug side) | `cohorts.py:1763,1880-1884` |
| Incident-local cells | `conditional_readout` — the cell, shipped | `gated_pc_cloud.py:276-370,2477-2481` |
| Incident-local cells | `label_mask_mode="closure"` — same construction, mainline | `case_finding_assembly.py:148-156` |
| Eval-time masks | `_bundle_masked`/`pr_readout`/`score_cells_df` take masks | `evaluate.py:222-244`; `gated_pc_cloud.py:116-145`; `distributed_readout.py:1284-1323` |
| Pre-index closure | `attested_provider` is a plain callable seam | `multi_domain.py:194-204,242-247` |
| Versioned flag + cache key | `dag_collapse` template | `gated_pc_cloud.py:1773-1783`; `_case_finding_cache.py:160-166` |
| Sidecar artifact | ADR 0025 covariate sidecar parquet | `docs/decisions/0025-…` |
| Constant-column trap | Found and fixed once, detection-scoped | `gated_pc_cloud.py:155-165` |
| Eval-mask comparability trap | exp 0079 "Trap 3" | `gated_pc_cloud.py:296-301` |

**No prior art for:** multi-index sampling, incident-vs-prevalent reporting, PU/contamination analysis, future-conversion. Genuinely new.

---

## Recommended sequencing relative to exp 0110

**Does building element 1 before 0110 mean building it twice? The code: no. The numbers: yes.** The primitive (provider + `attach_frontiers` + `frontier_to_label`) is parameterized over provider and DAG and survives 0110 untouched. But "prior carrier of closure(c)" is a materially different predicate under today's every-powered-ancestor climb vs 0110's `reduce_tie_map` most-specific attribution — broader today, especially for the 619 category-anchors. Any eligibility or conversion rate measured pre-0110 must be re-measured after.

**Recommendation:**

1. **Element 6 (doc repairs) now** — free, except the in-code comment rides with the next assembler change (or an accepted, noted cache drop).
2. **Element 1 after the 0110 port, before the 0110 record run**, as a driver-level post-pass module (`preindex_closure.py`) on the `mondo_collapse` template: new module, driver-level, fold-when-on flag, pinned keys re-verified. Preserves the 0104/0109 cached bundles and all tripwire hashes.
3. **Gate elements 2/3 on a cheap corpus-property probe** (the diag-sibling-support move): over bundle + pre-index closure, count per node `(n_incident_eligible, n_incident_pos, n_incident_neg)`; report nodes clearing `min_count=20` both classes, and how many of the 620 constant heads become non-degenerate-but-constant (the C2.1 population). One treeAggregate. **If fewer than ~a few hundred nodes survive 20/20, the incident macro is not a deliverable and elements 3–5 should not be built.** Given 96% root prevalence, this is a live possibility and the cheapest decision-relevant number in the program.
4. **Element 5 is its own experiment (0111+), never bundled with 0110** — it changes the doc unit (kills the 0104/0109 controls per insight 0010), forces first-ever wiring of the distributed eval path, and requires the maximal-blast-radius `cohorts.py` edit. Measure the episode multiplier before any fit.

---

## Corrections to the design statement (as discussed pre-audit)

1. "The readout is O(N) per pass" — incomplete: the binding constraint is the **O(N·C) driver collect** (`_densify_lean_blocks`, 1.9 GB at 0104 scale; calibration holds a float64 copy on top). The distributed alternative has never been wired.
2. The live eval path is `readout_from_proba` on driver-collected dense arrays, not `per_node_metric_rows` (an unwired escape hatch).
3. "Prior carriers excluded from both sides" also **un-skips** the 619 constant heads → forced 0.5 in the macro (C2.1). Needs a constant-column guard on the ranking axis, mirroring the detection fix.
4. "Local negatives" is not new — it is `conditional_readout` + `label_mask_mode="closure"`, both shipped; only the incident filter and stratum are new.
5. `_random_event_windows` is not reusable for episodes; `_window_observed_cohort` is.
6. "100% incident capture" is unachievable under the hardcoded 365-day prior-observation floor, which removes precisely the earliest (most incident) episodes.
7. "No corpus rebuild" for element 2 holds only if element 1 is a driver-level sidecar, not an assembler change.
8. The pre-index vector is per-**document** (per (person, index) under element 5), not per-person; three artifact grains exist across elements 1/4/5 and must not be conflated.
9. The person-level 80/20 split is multi-doc-safe by design; the **driver-path calibration split is not** (row-level).
10. Element 6's claims are correct; the stale leakage claim lives in three places including a code comment.
