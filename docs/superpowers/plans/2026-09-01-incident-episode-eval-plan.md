# Incident & episode-anchored evaluation — implementation plan

**Date:** 2026-09-01
**Status:** Proposed (spec approved; this is the build order)
**Decision owner:** Shawn.
**Implements:** `specs/2026-09-01-incident-episode-eval-program.md` — every requirement
`Rn.m` there maps to a work package below (coverage map, §5). Do not re-litigate the
spec's rationale here; this document is *what gets built, where, in what order*.
**Anchors:** every `file:line` below is the audit's
(`reports/2026-09-01-temporal-eval-program-audit.md`), or verified at `08fd622`.
**Dependency:** exp 0110's native-Mondo port
(`plans/2026-08-31-native-mondo-label-space-plan.md`) — WP1. Not built here.
**Branch:** `claude/gated-conditional-voi`.

---

## 0. Global constraints

- **House rule, load-bearing everywhere below: know which side of the cache-key line
  you are on before you type.** Editing a source-hashed module orphans cached bundles
  and breaks the four pinned tripwire hashes; editing driver-owned code is free. Every
  WP states its side explicitly.

  | module | hashed as | folded at | when |
  |---|---|---|---|
  | `charmpheno/omop/case_finding_assembly.py` | `assembly_src` | `_case_finding_cache.py:112-113` | **every** key |
  | `charmpheno/omop/condition_dag.py` | `dag_src` | `_case_finding_cache.py:112-113` | **every** key |
  | `charmpheno/omop/cohorts.py` (whole file) | `cohort_defs_version()` (`cohorts.py:36-52`) | `_case_finding_cache.py:111` | **every** key |
  | `charmpheno/omop/multi_domain.py` | `multi_domain_src` | `_case_finding_cache.py:133` | `multidomain=True` |
  | `analysis/cloud/mondo_dag.py` | `mondo_src` | `_case_finding_cache.py:152` | `mondo=True` |
  | `analysis/cloud/mondo_collapse.py` | `dag_collapse_src` | `_case_finding_cache.py:160-166` | `dag_collapse` **on only** |
  | `gated_pc_cloud.py`, `gated_pc_readout.py`, `distributed_readout.py`, `diag_*.py`, **new modules** | — | — | **never — free** |

- **Pinned tripwire hashes** (`tests/scripts/test_case_finding_cache_mondo.py:63-65,:87`):
  `d658ce0a9a7425dd`, `3cf6c7aac6140393`, `275c8e6a76283e86`, `ca958995cc1cfb17`.
  Any WP that moves one has failed, unless the WP says otherwise **and** carries the
  tripwire's required note (`:84-92`). None of WP0–WP7 moves one.
- **Importing a hashed module is free; editing it is not.** `_module_source_hash` is over
  source text. WP2 and WP3 import `attach_frontiers` / `frontier_to_label` /
  `case_finding_population_index_table` freely.
- **Nothing array-shaped rides a task closure** (ADR 0047 addendum). Binds WP3 and any
  new `treeAggregate` in WP4/WP6.
- **Egress:** counts < 20 are not disclosable (`evaluate.py:76-78`). Publishing floors and
  model-internal dials stay structurally separate (0110 plan `:59-62`).
- **Every reported number carries the spec §8 four tags** (arm · node set · cell type ·
  claim type) plus the D7 naming rule. This is an output requirement of WP4/WP5/WP6, not
  a documentation afterthought.
- Test harness: `.venv/bin/python -m pytest <path> -q`.

---

## 1. Work packages

### WP0 — E6 doc repairs · **DONE (08fd622)** · size S · no dependencies

**(a) Builds:** nothing new; recorded so the residue is not lost.

**(b) Files:** two sites repaired at `08fd622` —
`specs/2026-07-18-lookback-window-design.md:20-26` and
`docs/experiments/0061-dag-placement-rare6-lookback-1yr.md:55-59`: the
"leakage-free by construction / strip is a no-op" claim is now scoped to
`index_mode="disease"` (R6.1), with the strip-narrowness note (R6.3: `drop_idxs` comes
from `before_dag.nodes()` at `multi_domain.py:223,293` = anchors + synthetic negatives;
attestation climbs from descendants at `mondo_dag.py:298-310`, which are **not**
stripped).

**(c) Cache-key impact:** none — docs only.

**(d)/(e) Acceptance (spec §7/E6):** a repo grep for the stale claim returns **only**
site 3.

**OPEN RESIDUE — site 3, tracked here so it is not forgotten.**
`case_finding_assembly.py:394-396` still carries the stale comment. Fixing it moves
`assembly_src` → orphans **every** cached bundle in the repo including 0104's record
(~20 min of BigQuery, `test_case_finding_cache_mondo.py:83`) and breaks all four pinned
hashes. Per R6.2 and the tripwire's own instruction (`:84-92`) it ships **with the next
assembler change, or with an accepted-and-noted cache drop — never alone.**
This program deliberately supplies no such carrier (R1.1 keeps E1 out of the assembler),
so **site 3 is expected to stay stale for the duration**. Accepted; do not force an
assembler edit to close it. The nearest legitimate carrier on the horizon is WP8/R5.2
(0111's `cohorts.py` edit), which takes a full cache drop anyway — attach site 3 to it.

---

### WP1 — 0110 native-Mondo port · **DEPENDENCY, NOT BUILT HERE** · size L (elsewhere) · blocks WP2

**(a) Builds:** nothing in this plan. Owned by
`plans/2026-08-31-native-mondo-label-space-plan.md` §4 steps 1–2 (port
`mondo_usage_cloud.py`'s pure core + 33 tests; build the native label front-end:
attribution frame → frontier provider, closure-support powering, kept-set Hasse via
`nearest_mapped_parents`, splice post-pass, `ConditionDag` on Mondo ids) and §4 step 3
(the `dag_source: mondo_native` flag threading). **Do not duplicate or restate it.**

**(b) What WP2 consumes from it — the contract, and the only part of 0110 this plan
depends on:**

1. **The provider seam.** A callable `events_df -> attested_df` with columns
   `(doc_id, person_id, source_cohort, attested_cids)`, constructed **driver-side** and
   handed to the assembler (today: `make_mondo_attested_provider(climb_sdf,
   doc_spec=PatientCohortDocSpec())` at `gated_pc_cloud.py:1768-1770`; post-0110: the
   native-Mondo attribution-frame provider). WP2 calls the *same object* on the feature
   window. If 0110 changes the provider's construction but keeps the callable shape,
   WP2 needs no edit.
2. **The kept-node DAG.** `before_dag` (pre-prune, driver-built inside `_assemble_mondo`)
   plus the post-prune `keep` set — WP2 climbs on `before_dag` and projects onto `keep`,
   exactly as the label path does (`multi_domain.py:252-253,270-271`).
3. **Bundle fields** `parent_int`, `cid2int`, `int2cid` — R1.3's recovery route:
   `DagLayout(bundle.parent_int, n_bg, tpn)`; `bundle.cid2int`;
   `keep = set(bundle.int2cid.values())`. All three round-trip through `_meta_dict`
   (`_case_finding_cache.py:171-191`) today and must continue to post-0110.

**(g) Sequencing note (spec §E1 rationale, not restated):** WP2's *code* survives 0110;
WP2's *values* do not. Nothing measured pre-0110 is recorded as a finding.

---

### WP2 — Pre-index closure primitive (E1) · size M · **self-contained, agent-buildable** · depends on WP1

**(a) Builds.** `R_d` per document — the label-style closure vector computed by the
existing frontier→closure machinery over the **feature** window (D1). New driver-level
module `analysis/cloud/preindex_closure.py`, applied as a **post-pass on the
`mondo_collapse` template** (R1.1).

**The design point that makes R1.1 buildable, stated once because it is the whole WP:**
the driver does not hold the assembler's feature frame, and may not edit the assembler
to get it. It does not need to — **the index is deterministic**.
`case_finding_population_index_table` → `_random_event_windows` picks the anchor by
`min hash(person_id, event_date, _RANDOM_WINDOW_SALT)`, explicitly *"resume-stable"*
and explicitly **not** `F.rand()` (`cohorts.py:1083-1145`, pick at `:1136-1143`). So a
driver-owned re-derivation reproduces the assembler's own windows **exactly**:

```
cond      = load_omop_bigquery(spark, cdr, billing, person_sample_mod, "condition_era")
index_df  = case_finding_population_index_table(cond, ..., prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS,
                                                label_window_days=...)     # cohorts.py; deterministic
feat, _   = lookback_feature_frames([cond], index_df, [cond_date], lookback_days=..., ...)
att       = provider(feat[0])                       # the SAME driver-built provider (WP1 contract 1)
fr        = attach_frontiers(att, before_dag, keep, cid2int, lay)          # pure; case_finding_assembly:261-274
R_d       = sparse index list per doc_id            # NOT attach_labels — see R1.5
```

Every name above is **imported**, not edited. Cost: one extra full-history `cond` scan +
one attestation pass at build time — the same scan WP6 needs (see WP6/(g): build them in
one driver pass or accept two).

**(b) Files.**
- **New:** `analysis/cloud/preindex_closure.py` (pure core + the Spark pass; module
  docstring must say, as `mondo_collapse.py`'s does, *why it lives outside the assembler*).
- Modify `analysis/cloud/gated_pc_cloud.py`: application site at the `return assemble(...)`
  of `_assemble_mondo` (`:1783-1787`) — becomes build-then-attach, immediately after the
  `dag_collapse` block (`:1773-1783`) whose comment at `:1774-1779` is the precedent.
- Modify `analysis/cloud/_case_finding_cache.py`: fold-when-on flag + witness (below).
- Modify `analysis/cloud/gated_pc_readout.py`: recovery override.
- Modify `scripts/run_experiment.py`, `experiments/defaults/_base.yaml`, the 0110 exp doc.

**The flag-threading checklist — copy `dag_collapse` site for site.** All ten, or the flag
is half-wired:

| # | site | `dag_collapse` precedent |
|---|---|---|
| 1 | experiment front matter | `docs/experiments/0109-*.md` |
| 2 | `build_gated_pc_args` emits the CLI flag | `run_experiment.py:862-863` |
| 3 | driver argparse | `gated_pc_cloud.py:1878` |
| 4 | corpus spec (mondo-only guard) | `gated_pc_cloud.py:1697` |
| 5 | cache-key extra | `gated_pc_cloud.py:1718-1721` |
| 6 | **fold-when-ON in the payload** | `_case_finding_cache.py:160-166` |
| 7 | application site | `gated_pc_cloud.py:1773-1783` |
| 8 | readout recovery, tri-state CLI wins over manifest | `gated_pc_readout.py:155-157,244,314-315,629-630` |
| 9 | manifest record | `gated_pc_cloud.py:2308` |
| 10 | tests, both halves | `test_run_experiment_gated_pc.py:188-230`; `test_case_finding_cache_mondo.py:103-143` |

**R1.4 witness.** `_meta_dict` (`_case_finding_cache.py:171-191`) serializes only
`parent_int/int2cid/cid2int/name_by_id/ledger/vocab_map(s)` — an extra parquet column
rides **silently**. Add `preindex_closure: {version, col_name}` to the meta dict and
restore it in `_restore_meta`; any readout that asks for the column against a bundle
whose meta lacks the witness **raises**, with the key and cache_uri in the message. No
silent mixed-vintage cache dirs.

**R1.5 sparse.** Store `R_d` as `array<int>` engine ids (the closure index list), **not**
a dense `array<double>`. `attach_labels` (`case_finding_assembly.py:163-191`) emits dense
at `:177-180` and is therefore **not reused** — reusing it buys +50% on the bundle parquet
(`label`+`labelMask` are already 2×3,820 float64 ≈ 61 KB/row).

**R1.6 grain.** Per **document** (`doc_id`), not per person. Today `PatientCohortDocSpec`
gives one doc per person under `index_mode=population`, so `doc_id` is the join key and
stays correct when 0111 appends the index component. The module's schema docstring names
the grain in the first paragraph.

**(c) Cache-key impact.** New module — **un-hashed by default**; the flag and the module's
source hash are folded **only when the flag is on** (R1.2), exactly as `dag_collapse_src`
is. Flag **off** ⇒ all four pinned hashes **byte-identical**. No hashed module is edited.

**(d) Tests.**
- `tests/scripts/test_case_finding_cache_mondo.py` gains: flag-off keys equal all four
  pinned hashes; flag-on is a different key; flag-on does not leak into the SNOMED key;
  a version bump moves the key (mirrors `:103-143`).
- `scripts/tests/test_run_experiment_gated_pc.py` gains the front-matter→CLI→spec→key
  thread test (mirrors `:188-230`) and the manifest/readout-recovery override test.
- **New** `tests/test_preindex_closure.py`: pure-core fixtures — a patient whose
  pre-index record contains `closure(c)` yields `c ∈ R_d`; the same patient with the code
  moved after the index yields `c ∉ R_d`; multi-parent diamond closure; sparse round-trip
  (write → `try_load` → identical index lists); witness-absent raises.

**(e) Acceptance (spec §11/E1).** Flag-off tripwire hashes byte-identical (all four);
witness present/absent detected and failing loudly; the synthetic pre-index-vs-post-index
fixture resolves `R_d` correctly; sparse round-trip.

**(f) Size M.** Self-contained and agent-buildable: one new module, one mechanical flag
thread with an exact precedent, no metric semantics.

**(g) Depends on WP1** (label machinery must be final, or the values are measured twice).
Blocks WP3, WP4, WP5, WP6. Land it **after the 0110 port, before the 0110 record run**.

---

### WP3 — The census (E-census) · size S · **self-contained, agent-buildable** · depends on WP2 · **THE GATE**

**(a) Builds.** One cheap corpus-property probe over `bundle + R_d`, before any fit —
the same move `diag-sibling-support` made for degeneracy, and for the same reason.
Per node: `(n_incident_eligible, n_incident_pos, n_incident_neg)` (RC.1), where
eligibility is D2 (`c ∉ R_d`, excluding prior carriers from **both** classes), positives
D3, negatives D4.

**(b) Files.**
- **New:** `analysis/cloud/diag_incident_census.py` — copy `diag_sibling_support.py`
  structure verbatim: `--run-dir`/`--cache-uri`, `resolve_run_dir` +
  `bundle_key_from_manifest` from `gated_pc_readout`, `try_load`, **cache HIT required**
  (`diag_sibling_support.py:128-137` — a diagnostic never pays a rebuild), one
  `mapPartitions` → `treeAggregate(depth=2)` over `(label, labelMask, preindexClosure)`,
  pure classification function kept unit-testable off-Spark (the
  `classify_degenerates` pattern, `:46-109`).
- **New Makefile target** `diag-incident-census` in `analysis/cloud/Makefile`, copied
  from `diag-sibling-support` (`:505-523`): same `spark-submit` shape,
  `--driver-memory 4g`, 12 executors, `ID=` run selector, `GPR_CACHE_URI` passthrough.

**(c) Cache-key impact.** **None** — new driver-level diagnostic; reads a cached bundle,
writes nothing to the cache.

**(d) Tests.** **New** `tests/test_incident_census.py`, off-Spark: hand-computed
eligibility on a small fixture reproduces `(n_eligible, n_pos, n_neg)` exactly; the
C2.1-population classifier (below) fires on a constructed train-degenerate node that
acquires test negatives under the incident mask; the combiner's `None`-identity
handling is exercised on an empty partition.

**RC.4 closure discipline (ADR 0047 addendum).** The reduction identity is a `None`
sentinel with identity handling in the combiner; partials are allocated **executor-side**
by the partition kernel; the driver substitutes zeros only in the empty-corpus case.
Note that at C≈3,820 three float64 `(C,)` partials pickle to ≈92 KB — under the 1 MB
auto-broadcast threshold, so `diag_sibling_support.py:158-163`'s dense zero is not itself
a violation. The sentinel is **required anyway**: it is doctrine, it costs nothing, and
the census is the template the next diagnostic copies. Verify by telemetry: the
`disk_telemetry:` `pyspark-*` dir must be **flat in passes**, not linear.

**(e) Acceptance + OUTPUT.** Printed banner + a JSON sidecar in the run dir
(`incident_census.json`) carrying, per spec:
- **RC.2** the count of nodes clearing `min_count = 20` on **both** classes, with the
  full per-node table (egress: **nothing under 20 leaves the workspace** — the table is
  workspace-internal; only the counts-of-nodes summary is disclosable);
- **RC.3** the **C2.1 population**: nodes whose *train* cell was degenerate
  (`(n_obs<=0)|(n_pos<=0)|(n_pos>=n_obs)`, `diag_sibling_support.py:78`) and which
  acquire test negatives under the incident mask — i.e. would be **scored at exactly
  0.5** without WP4's guard. Post-0110 this should be small (0110 plan `:88-91,139-142`
  makes the subsumed-sibling trap structurally impossible); the audit's "up to 619" is a
  0104/0109-vintage figure and must not be quoted as a post-0110 expectation;
- the constant-head **fate** breakdown: still-all-positive / acquired-negatives /
  no-longer-eligible-at-all.

**THE GATE (spec §E-census GO/NO-GO).**

> If **fewer than ~a few hundred nodes** clear 20/20 on both classes, the incident macro
> is **not a deliverable**: **WP4, WP5 and WP6 are not started.**

Given 0109's root prevalence **0.9609** this is a live outcome, not a formality.
**Either way the census numbers are recorded as a finding** — the distribution of
incident-eligible support across the label space is the most decision-relevant number in
the program and the cheapest.
**Where the decision is recorded:** the GO/NO-GO call, with the counts, goes in the
**0110 experiment doc** (`docs/experiments/0110-*.md`) run log, since it is a property of
0110's corpus; on NO-GO, that entry is the program's terminal record for E2/E3/E4 and the
program continues at WP7/WP8 only.

**(f) Size S.** Fully agent-buildable — it is a copy of an existing diagnostic with a new
counting kernel.

**(g) Depends on WP2.** Gates WP4, WP5, WP6.

---

### WP4 — Dual prevalent/incident metrics (E2) · size M · **census-gated** · depends on WP2, WP3

**(a) Builds.** A second `readout_from_proba(proba, y_te, m_incident, C, ...)` call
(`gated_pc_cloud.py:187-208`) on the **driver path — the live path**. No solver touched;
the escape-hatch `per_node_metric_rows`/`score_cells_df`
(`distributed_readout.py:31-32,1284-1323`) stays unwired (it has never been called by a
driver; wiring it is 0111's problem, R5.8).

**Getting `m_incident` to the driver: a fourth CSR run.** `_lean_eval_kernel`
(`distributed_readout.py:565-643`) packs `(ids, P, y_idx, y_ptr, m_idx, m_ptr)` from
`_row_quads(rows, id_col, score_col, label_col, mask_col)` (`:732-749`). Add the
eligibility column as a fifth selected column: `_collect_lean_proba`'s
`cols = (id_col, score_col, label_col, mask_col)` (`gated_pc_cloud.py:820`) gains
`elig_col`; `_row_quads` → `_row_quints`; the kernel emits `(e_idx, e_ptr)`; and
`_densify_lean_blocks` (`gated_pc_cloud.py:756-788`) gains the matching branch — the
same `np.repeat(...diff(ptr))` scatter as `y`/`mask`, with `None` meaning all-eligible.
**+1 byte/cell.**

**(b) Files.** `analysis/cloud/distributed_readout.py` (`:565-643`, `:732-749`),
`analysis/cloud/gated_pc_cloud.py` (`:756-788`, `:791-860`, `:2355-2365`, `:2495-2499`),
`analysis/cloud/gated_pc_readout.py` (`:525-528` dump path). Nothing else.

**(c) Cache-key impact.** **None.** Neither `distributed_readout.py` nor
`gated_pc_cloud.py` is source-hashed into any key. Free.

**Hard requirements — all four, none optional.**

- **R2.1 constant-column guard on the RANKING axis.** Every train-degenerate node carries
  a constant column (`proba[:, deg] = const[deg]`, `gated_pc_cloud.py:856-858`). Under
  the prevalent mask those columns are all-positive in test and `_score_label` skips them
  (`evaluate.py:87-95`); under the incident mask prior carriers leave **both** classes —
  precisely the set that made the column all-positive — so some acquire negatives, become
  non-degenerate, and score `roc_auc_score(y, const)` = **exactly 0.5**, `skipped: None`,
  **in the macro**. Requirement: a column with `np.ptp(proba[:, c]) == 0` is **skipped**
  with reason `"constant prediction column"` and **the count is reported**. This mirrors
  the detection-side fix (`gated_pc_cloud.py:148-171`, an explicit *"EVAL bug fix, always
  on"*, deliberately scoped to detection at `:162-165`); the incident arm reopens the same
  bug on the ranking axis and closes it there. Implement in `analysis/pc/evaluate.py`'s
  `_score_label` (a third skip reason alongside degenerate-test-column and small-column),
  so the reason is counted separately per spec §8.5 — **three skip reasons, three counts,
  never summed.**
- **R2.2 macros on BOTH node sets.** `_macro` averages only non-skipped labels
  (`evaluate.py:116-125`). Report the macro on the **shared both-arms-scoreable node set**
  *and* on **each arm's full set**. A prevalent-vs-incident delta across different node
  sets is not a comparison. This is the 0110 plan's own discipline (`:144-146`) carried
  over.
- **R2.3 eligibility is a CORPUS property, never a run property.** `m_incident` is a pure
  function of `(bundle, R_d)` — D2 — computed once per corpus by WP2, stored with the
  corpus, reused byte-identically by every run being compared. A run's predictions,
  thresholds, or degeneracy set may never enter it. (The standing decision it answers to
  is `conditional_readout`'s docstring at `gated_pc_cloud.py:296-301`, exp 0079 Trap 3.)
  Enforced structurally: the eligibility column arrives from the **bundle**, and the
  readout has no code path that constructs one.
- **R2.4 the D7 naming rule on every output** — *prevalent-fit model, incident cohort*
  (heads standardize per node on that node's own observed **train** rows,
  `gated_pc_cloud.py:977-979`) — in the JSON block and in every table.

**Output.** A `"gated_pc_incident"` block in `results_readout.json`, written alongside the
existing per-arm blocks by `run_readout` → `_dump_partial_results`
(`gated_pc_readout.py:525-528`; driver twin `gated_pc_cloud.py:2364`). Block carries:
both node-set macros, the three skip counts, the D7 label string, and the spec §8 four
tags.

**(d) Tests.** `tests/` gains: a `_densify_lean_blocks` round-trip with the fifth run
(dense and `None` cases); a `_score_label` fixture with a forced constant column that has
**both** classes present → skipped `"constant prediction column"`, **not** scored 0.5; a
`_macro` fixture asserting shared-set vs full-set values differ and both are emitted; an
end-to-end flag-off assertion that `results_readout.json` numbers are unchanged.

**(e) Acceptance (spec §11/E2).** Flag-off reproduces existing `results_readout.json`
exactly; the incident block carries the skipped-constant count, both node-set macros, and
the D7 label; the constant-column fixture is skipped, not scored.

**(f) Size M.** Agent-buildable: mechanical plumbing plus one guard, with the fixture
telling you if you got the semantics wrong.

**(g) Depends on WP2 (the column) and WP3 (GO).** Parallel with WP5/WP6.

---

### WP5 — Incident-local cells + P-strata (E3) · size S–M · **census-gated** · depends on WP2, WP3

**(a) Builds.** Two arguments on `conditional_readout` (`gated_pc_cloud.py:276-370`).
The cell itself is **already built**: `:310` builds the parent-P cohort
`np.where((y[:,p]==1) & (mask[:,p]==1))`; `:315-320` takes positives as `y[rows,c]==1`
and negatives as the rest of P's cohort — siblings under P, or P-but-nothing-more-specific.
That is D5 minus D2. It already emits `cond_auc`, `cond_ap`, `marg_ap`, per-edge and
pooled ECE, top-1 vs majority, and per-node reliability.

- **R3.1** thread the D2 **eligibility array** in and intersect it with the cohort
  construction at `:310` and the per-child row selection at `:315-320`; thread the D6
  **stratum key** (`P ∈ R_d` vs `P ∉ R_d`) and emit each edge twice more, once per stratum.
- **R3.2 pooled first, strata second.** The P-stratum is a **reported stratum, never a
  gate** — pooled numbers are primary. "No P / gains c" is a legitimate positive (*de
  novo* specific prediction, the harder half) and the two strata have materially
  different AUCs; they must not be mushed into one unlabeled number.
- **R3.3** `min_count >= 20` on **both classes, per cell** (`:318` today uses
  `max(min_count, 1)` on each of `n_pos`/`n_neg` — raise the floor and keep it explicit).
  Stratifying halves each cell; expect most edges to drop out of one or both strata.
  Apply the R2.2 discipline to the **surviving-edge** sets: report edge-set membership,
  not just averages.
- **R3.4 break the `_ones` convention deliberately, in writing.** The call sites pass
  `_ones = np.ones_like(y_te)` precisely to protect the Trap-3 comparability rule
  (`gated_pc_cloud.py:2477-2479`; the rule itself at `:296-301`). Passing an eligibility
  array breaks that convention. **Re-state the protected rule at the call site**, together
  with R2.3's resolution — eligibility is a *corpus* property, so cross-run comparability
  survives. An undocumented break here is the exact failure mode Trap 3 records.
- **R3.5 EGRESS.** `min_count` is not only a statistical dial: counts < 20 are **not
  disclosable** (`evaluate.py:76-78`), and stratified cell tables are a **disclosure
  surface**. **Nothing under 20 leaves the workspace.** Keep the publishing floor and the
  internal powering dial (`min_positives`) structurally separate (0110 plan `:59-62`).

**(b) Files.** `analysis/cloud/gated_pc_cloud.py` only: `:276-370` (signature + cells),
`:2377`/`:2405`/`:2477-2481`/`:2506` (the `_conditional` call sites — all four thread the
same eligibility array or none).

**(c) Cache-key impact.** **None** — `gated_pc_cloud.py` is un-hashed.

**(d) Tests.** `tests/` gains: eligibility-all-ones + collapsed stratum ⇒ output
**numerically identical** to today's conditional block (the regression that protects the
0104/0109 comparison); a two-stratum fixture where the strata AUCs differ and both are
labelled; a cell-suppression fixture asserting no published cell has either class < 20.

**(e) Acceptance (spec §11/E3).** As (d), plus: the call-site comment states the broken
`_ones` convention and why.

**(f) Size S–M.** Agent-buildable; the identity regression test is the whole safety net.

**(g) Depends on WP2, WP3.** Independent of WP4 (shares the eligibility array; does not
need the fourth CSR run — the conditional readout is already a driver-side dense-array
consumer).

---

### WP6 — Future-conversion analysis (E4) · size L · **census-gated** · depends on WP2, WP3

**(a) Builds.** Of the documents scored **incident negative** for c, how many are
diagnosed with c **later**, beyond the label horizon — PU **channel 1**, measured.

**R4.1 artifact: a first-attestation-date sidecar**, grain **`(person, frontier-node)`**,
computed from the full-history `cond` frame:
`cond ⋈ climb on concept_id = descendant_concept_id → groupBy(person_id,
ancestor_concept_id).agg(min(condition_era_start_date))`. "First attestation of
`closure(c)`" folds **driver-side** via `lay.closure`.

**Why a sidecar and not the obvious place.** The natural aggregation site
(`multi_domain.py:365-367`, where `cond` is loaded once with full history and no date
filter — `load_omop_bigquery`, `bigquery.py:216-372`, applies only `person_sample_mod`
and `concept_id != 0`) is **inside the source-hashed assembler** — the same trap as WP2.
Post-label-window events are discarded at windowing, not retained
(`lookback_feature_label_events`, `cohorts.py:1648-1668`), and the cached bundle holds
BOW vectors, not events (`_case_finding_cache.py:21-23`): **nothing downstream can
recover a future date.** So R4.2: driver-owned code, cached as a **sidecar parquet per
ADR 0025's pattern** (`docs/decisions/0025-charmpheno-covariate-sidecar-parquet.md`:
separate artifact in the run dir, referenced by a manifest field, joined at use time),
with **its own cache key** — the sidecar is keyed on `(cdr, person_mod, mondo identity,
climb identity, horizon set)`, never on the bundle key, so it survives readout re-runs
and is not orphaned by a bundle-key move.

**R4.3 grain discipline.** Three grains, never conflated, each named in its schema doc:
per-**document** (WP2's `R_d`), per-**(person, node)** (this sidecar), per-**person**
(ADR 0025's covariate sidecar). Never join two without naming the grain change in the
code that does it.

**R4.4 right-censoring.** Conversion denominators are **gated on
`observation_period_end_date` at each horizon** — reuse `_window_observed_cohort`'s logic
(`cohorts.py:693-729`), which takes an arbitrary `(person_id, index_date)` frame. Without
this the "conversion rate" is a censoring artifact, not a contamination estimate.

**R4.5 report** per-node conversion at **1y / 2y / 3y**, overall and **stratified by
model-score decile**.

**R4.6 framing: LOWER BOUND on PU channel 1.** Mandatory language on every table and in
the JSON. Channels 2 (never diagnosed) and 3 (coded elsewhere) are unmeasured; care-
fragmentation bounds are a non-goal.

**R4.7 expect the bound to be loose.** At 0109's root prevalence 0.9609 most nodes' "did
not gain c this year" is a weak negative and conversion will be high across the board.
**A finding, not a failure** — record it; do not tune toward a tight number.

**R4.8 the decile table doubles as case-finding validation.** Top decile converting at a
materially higher rate than the bottom = the model finding future cases among its own
"negatives" — the case-finding claim, measured on the one population where it can be.

**(b) Files.** **New:** `analysis/cloud/conversion_sidecar.py` (build + load + schema
doc) and `analysis/cloud/diag_future_conversion.py` (the analysis + tables), plus a
`diag-future-conversion` Makefile target on the WP3 template. Modify
`analysis/cloud/gated_pc_cloud.py` (sidecar path recorded in the manifest, beside the
corpus manifest at `:2308`'s block) and `analysis/cloud/gated_pc_readout.py` (recovery).

**(c) Cache-key impact.** **None on the bundle key** — all new modules and driver-owned
code. The sidecar carries its **own** key; a sidecar-key change orphans only sidecars.

**(d) Tests.** **New** `tests/test_conversion_sidecar.py`: grain-named schema round-trip;
`min` aggregation over a multi-code person; closure fold on a diamond; **censoring
denominators shrink monotonically with horizon** on a fixture with staggered
`observation_period_end_date`; decile stratification on a fixture with a planted
score-conversion gradient.

**(e) Acceptance (spec §11/E4).** The sidecar round-trips with a grain-named schema;
horizon denominators are monotone under censoring; the decile table is reported with the
lower-bound language attached.

**(f) Size L** — the only WP with a new cached artifact, its own key, its own BQ pass and
a censoring model. Agent-buildable in **two** units: (i) the sidecar build/load/tests,
(ii) the conversion analysis + tables. Split them.

**(g) Depends on WP2, WP3.** **Shares WP2's driver-side `cond` load** — if WP6 lands with
or after WP2, do the full-history first-attestation aggregation in the *same* driver pass
that computes `R_d` and pay one scan, not two. If they land apart, two scans is the
accepted cost; do not couple them at the price of delaying WP2.

---

### WP7 — Close the doc-spec cache-key hole (R5.3) · size S · **NOT census-gated — do it early** · no dependencies

**(a) Builds.** The audit's **seam 4**, a **silent cache-poisoning hazard**, not a
rebuild cost: `doc_spec` is hard-coded in two places (`multi_domain.py:408`,
`gated_pc_cloud.py:1769`) and is **absent from every cache key** — `_SPEC_ASSEMBLY_KEYS`
(`gated_pc_cloud.py:1657-1662`: `disease, cdr, billing, person_mod, min_n, holdout_frac,
vocab_size, min_df, min_patient_count, n_bg, tpn, doc_min_length, strip_mode,
lookback_days, label_window_days, label_mask_mode, index_mode` — **no doc-spec entry**)
and `compute_bundle_cache_key` folds `doc_min_length` but not the doc spec's **identity**.
Changing the driver-side doc spec alone would therefore poison the cache **under a
byte-identical key**.

Fix: add a doc-spec identity (`doc_spec: "patient_cohort"` — class name + its
identity-bearing params) to `_SPEC_ASSEMBLY_KEYS` and to `compute_bundle_cache_key`'s
payload, **defaulted so today's value reproduces today's hashes**. That is the whole
trick and the reason this is cheap: the current spec is a constant, so folding it
with the current value as the default leaves every key byte-identical while making a
future change visible.

**(b) Files.** `analysis/cloud/gated_pc_cloud.py:1657-1662` (+ the spec builder at
`:1690-1700`), `analysis/cloud/_case_finding_cache.py` (signature `:45`, payload beside
`doc_min_length` `:103`, and the forwarded param-name list `:259`),
`analysis/cloud/gated_pc_readout.py` (recovery `_pick`),
tests.

**(c) Cache-key impact.** Touches `_case_finding_cache.py` — which is **itself not
source-hashed** (it computes keys, it is not folded into them). The payload gains a field;
**with the default value the four pinned hashes must stay byte-identical**, and the test
that asserts it is the deliverable. If they move, the change is wrong — do not re-pin.

**(d) Tests.** `tests/scripts/test_case_finding_cache_mondo.py`: all four pinned hashes
unchanged with the default; a different doc spec yields a different key; a legacy payload
without the field hashes identically to the defaulted one (`:143`'s legacy-key pattern).

**(e) Acceptance.** Four pinned hashes byte-identical; doc-spec change ⇒ key change;
`_SPEC_ASSEMBLY_KEYS` and the cache payload agree (one derivation, two callers — the
invariant the comment at `:1653-1656` already claims).

**(f) Size S.** Fully self-contained and agent-buildable; the pinned hashes are the oracle.

**(g) No dependencies. Do it alongside WP2, or before — it is the one E5 requirement that
is cheap, safe and independently valuable, and it MUST be closed before any doc-unit work
ever starts.** Precondition for WP8.

---

### WP8 — exp 0111 (episode sampling): **NOT IN THIS PLAN**

E5 changes the **document unit**, which kills the 0104/0109 controls (insight 0010; the
0110 acceptance protocol pins 0104's 0.6978/0.4845 on a shared node set, 0110 plan
`:144-150`). It is **never bundled with 0110** and gets **its own plan doc** when the
program reaches it. This plan builds none of R5.1, R5.2, R5.4–R5.8, R5.11–R5.15 —
they are catalogued complete in spec §7/E5 and the 0111 plan inherits them from there,
not from here.

**Preconditions this plan leaves in place for 0111:**

1. **WP7 closed** — the doc-spec cache-key hole is R5.3 and is the stated
   *"must be closed before any doc-unit work"*.
2. **The spec's R5 catalogue** stands as 0111's requirement list, with the id-type wall
   (R5.4), the A/B alignment-dict wall (R5.5), the person-keyed calibration split (R5.6),
   the detection dedup (R5.7), the ×3 distributed-eval wiring threshold (R5.8), and the
   maximum-blast-radius `cohorts.py` edit (R5.2) already priced.
3. **WP0's open residue** (`case_finding_assembly.py:394-396`) is attached to R5.2's
   accepted full cache drop — 0111 is its natural carrier.
4. **Two pre-measurements, runnable early and cheaply as BQ probes**, with no fit and no
   corpus change (below).

**Optional WP8a — the two cheap probes · size S · run only if the census is a GO.**

- **R5.9 episode multiplier.** Distinct 60–90d first-attestation episodes per person, by
  **gap-and-islands over the `cond` frame** — genuine prior art at `cohorts.py:2237-2252`
  (`_stable_drug_intervals`). **No repo data exists on episodes-per-person; the multiplier
  is a guess until measured**, and R5.8's ×3 wiring threshold cannot be evaluated without
  it.
- **R5.10 prior-obs-gate kill rate.** `_LOOKBACK_PRIOR_OBS_DAYS = 365` is hardcoded and
  deliberately un-overridable (`case_finding_assembly.py:43-53`); `index >= op_start + 365`
  systematically drops the **earliest, most unambiguously incident** episodes, and
  `index + 365 <= op_end` kills the last of every record. Measure what fraction dies.
  ("100% incident capture" is not achievable — spec §E5/R5.10 says so explicitly.)

Both are read-only BQ aggregations in a new `analysis/cloud/diag_episode_probe.py` on the
WP3 diagnostic template; **no cache-key impact, no corpus change, no fit**. Include here
only on a census GO — on NO-GO the program has bigger questions than 0111's multiplier.

---

## 2. Sequencing

```
WP0 ────────────────────────────────────────────────────────────► DONE (08fd622)
     │ residue: assembly comment rides WP8/R5.2

WP7 ─────────────┐  (hygiene; ANY time; not census-gated)
                 │
WP1 [0110 port] ─┴─► WP2 ─► WP3 ═══ GATE ═══► ┌─► WP4 ─┐
   (other plan)     (E1)   (census)   GO      ├─► WP5 ─┼─► 0110 RECORD RUN
                                              └─► WP6 ─┘   reports dual metrics
                                     NO-GO                        │
                                       └──► record census as a    │
                                            finding; STOP E2/E3/E4│
                                                                  ▼
                                                      WP8 → exp 0111 gets its own plan
                                                      (WP8a probes may run from GO onward)
```

| # | step | gate / precondition | size |
|---|---|---|---|
| 1 | **WP0** E6 doc repairs, sites 1–2 | none — **done** | S |
| 2 | **WP7** doc-spec cache-key hole | none; do it alongside/before WP2 | S |
| 3 | **WP1** 0110 native-Mondo port | outside this plan | L |
| 4 | **WP2** pre-index closure primitive | after the 0110 **port**, **before** the 0110 **record run** | M |
| 5 | **WP3** census | after WP2 — **GO/NO-GO for WP4/5/6** | S |
| 6 | **WP4 / WP5 / WP6** in parallel | **only on GO** | M / S–M / L |
| 7 | **0110 record run reports dual metrics** | WP4 landed | — |
| 8 | **WP8a** probes (optional) | GO | S |
| 9 | **WP8** exp 0111 | its own plan; **never bundled with 0110** | — |

WP4, WP5 and WP6 are independent of each other: WP4 owns the lean-block plumbing and the
ranking-axis guard, WP5 owns `conditional_readout`, WP6 owns a new sidecar. Three agents,
one shared input (WP2's `R_d`), no merge conflicts beyond the manifest block.

---

## 3. Risk register (top 5)

| # | risk | mechanism | mitigation | owner WP |
|---|---|---|---|---|
| **R1** | **C2.1 constant-column inflation** — the incident macro reads better than reality | Train-degenerate nodes carry constant columns (`gated_pc_cloud.py:856-858`). Prevalent masking skips them as all-positive (`evaluate.py:87-95`); incident masking removes exactly the prior carriers that made them all-positive, so they acquire negatives, score `roc_auc_score(y, const)` = **0.5**, `skipped: None`, and enter the macro | **R2.1 guard is mandatory and costs one `np.ptp`**: `ptp == 0` ⇒ skipped `"constant prediction column"`, count reported separately from the other two skip reasons. WP3/RC.3 sizes the exposed population **before** WP4 is written, so the guard's effect is known, not hoped | WP4 (guard), WP3 (measurement) |
| **R2** | **Census NO-GO** — too few nodes clear 20/20 | Root prevalence **0.9609** at 0109: most patients are prior carriers of most of the upper DAG, so D2 strips both classes hard. A live outcome, not a formality | The gate exists precisely for this: **WP4/5/6 are not started** until GO. The census is cheap (one `treeAggregate`, no fit) and its numbers are **recorded as a finding either way** — the eligible-support distribution is the program's most decision-relevant number. WP7 and WP8a survive a NO-GO | WP3 |
| **R3** | **Cache orphaning via an accidental hashed-module edit** | A one-line "while I'm here" edit to `case_finding_assembly.py` / `multi_domain.py` / `cohorts.py` / `mondo_dag.py` / `condition_dag.py` moves `assembly_src` / `multi_domain_src` / `cohort_defs_version()` / `mondo_src` / `dag_src` and orphans **every** cached bundle including 0104's record (~20 min BQ), breaking all four pinned hashes | §0's table is in every WP's (c). WP2's whole design (driver-level post-pass + deterministic index re-derivation) exists to avoid this; WP6's sidecar likewise. The pinned-hash tests are the tripwire and **run in every WP's test set, not just WP2's**. A moved hash is a **stop**, re-pinned only with the note `test_case_finding_cache_mondo.py:84-92` demands | all |
| **R4** | **Driver-memory ceiling** if WP4–WP6 outputs are run at episode scale prematurely | The binding constraint is the **O(N·C) driver collect** (`_densify_lean_blocks`), not the O(N) pass: 1.9 GB lean bundle + 2.4 GB `calibrate_per_node` float64 copy at 0104's D_te≈80k; ×3 ⇒ ~13 GB, ×5 ⇒ ~22 GB against a 16 GB driver. WP4 adds **+1 byte/cell** on top | WP4–WP6 are **single-doc-unit only**. The ×3 distributed-eval wiring threshold (R5.8) belongs to 0111 and `score_cells_df` stays unwired here. Do not run any WP4–WP6 output against an episode corpus before R5.8 is built and R5.9 measured | WP4, WP8 |
| **R5** | **The E1-values-invalidated-by-0110 trap** — measuring twice | "Prior carrier of `closure(c)`" is materially **broader** under today's every-powered-ancestor climb than under 0110's `reduce_tie_map` most-specific attribution, especially for the 619 category-anchors. WP2's *code* survives 0110; its *values* do not | **WP2 lands after the 0110 port and before the 0110 record run** — the sequencing is the mitigation. Corollary discipline: **no eligibility, census, conversion or incident number measured pre-0110 is recorded as a finding**, in any doc, even provisionally. The audit's "up to 619" is a 0104/0109-vintage figure and is never quoted as a post-0110 expectation | WP2, WP3 |

---

## 4. Experiment-doc bookkeeping

Who records which numbers, so nothing lands in two places or none.

| artifact | lands in | why there |
|---|---|---|
| **Census counts + GO/NO-GO decision** (RC.2 node counts, RC.3 C2.1 population, constant-head fate) | `docs/experiments/0110-*.md` — run log | It is a property of **0110's corpus**, measured on 0110's bundle before its record run. The GO/NO-GO call and its date are recorded there, and on NO-GO that entry is the program's terminal record for E2/E3/E4 |
| **Dual prevalent/incident metrics** (WP4: both node-set macros, three skip counts, D7 label) | `docs/experiments/0110-*.md` — beside the record run's headline, and `results_readout.json`'s `gated_pc_incident` block | 0110's record run **reports dual metrics** (spec §10 step 6). The prevalent macro remains 0110's headline against the 0104 control (0.6978/0.4845, shared node set); the incident macro is reported **beside** it, never in place of it |
| **Incident-local cells + P-strata** (WP5) | `docs/experiments/0110-*.md`, from the same run's `results_readout.json` conditional blocks | Same run, same corpus; the conditional block is already part of that readout |
| **Future-conversion analysis** (WP6) | **its own report** under `docs/reports/` (`docs/reports/2026-XX-XX-pu-channel-1-conversion.md`) | It is not a readout metric: a separate artifact (the sidecar), a separate cadence (horizons out to 3y), a separate claim class (a PU **lower bound**, §5/R4.6), and a separate audience. The exp doc links it; it does not inline it |
| **Episode probes** (WP8a: R5.9 multiplier, R5.10 kill rate) | the **0111 plan doc** when written; interim numbers in `docs/reports/` if they precede it | They are 0111 inputs, not 0110 results, and R5.8's threshold is evaluated against them |
| **Cache-key hygiene** (WP7) | commit message + `tests/scripts/test_case_finding_cache_mondo.py` | No experimental number; the pinned-hash test **is** the record |

Standing labelling rules on every table in every one of the above (spec §8): the four
tags (arm · node set · cell type · claim type); the **D7** naming rule on every incident
output (*prevalent-fit model, incident cohort*); **PU lower-bound** language wherever a
conversion or contamination number appears; **skip counts reported, not suppressed**,
three reasons counted separately; pooled and local incident metrics are
**DISCRIMINATION** claims, never prospective-cohort claims.

---

## 5. Spec requirement → work package coverage

| requirement | WP | note |
|---|---|---|
| R6.1, R6.3 | **WP0** | done at `08fd622` |
| R6.2 | **WP0** (residue) → carrier at WP8/R5.2 | site 3 stays stale; open note is the mitigation |
| R1.1 – R1.6 | **WP2** | R1.1's driver-level route rests on the deterministic index (`cohorts.py:1136-1143`) |
| RC.1 – RC.4 | **WP3** | RC.4 = ADR 0047 addendum |
| R2.1 – R2.4 | **WP4** | R2.1 is the sharpest; R2.3 enforced structurally |
| R3.1 – R3.5 | **WP5** | R3.5 is an egress rule, not a dial |
| R4.1 – R4.8 | **WP6** | split into sidecar + analysis units |
| R5.3 | **WP7** | pulled forward as hygiene, per spec §10's note |
| R5.9, R5.10 | **WP8a** (optional, GO-gated) | cheap BQ probes, no fit |
| R5.1, R5.2, R5.4 – R5.8, R5.11 – R5.15 | **WP8 — not built here** | catalogued in spec §7/E5; inherited by the 0111 plan |
| D1 – D7 (§6) | WP2 (D1–D4), WP5 (D5, D6), WP4 (D7) | definitions, realized by the WPs above |
| §8 metrics protocol (4 tags + 5 standing rules) | WP4, WP5, WP6 outputs + §4 above | cross-cutting output requirement |
| §11 validation | each WP's (e) | lifted verbatim where present |

**Not built by any WP, deliberately, and flagged rather than dropped:** spec §4's
**optional thin random-index arm** — the reserved source of global calibration and
population-prevalence claims (§8 standing rule 3). It carries no `Rn.m` requirements and
no element of its own; it is a named future option, not a deliverable. If a
population-prevalence claim is ever wanted, it comes from that arm and gets its own plan.

## 6. Non-goals (inherited from spec §9, repeated so nobody helpfully bundles them)

Train-time incident masking (D7 is the price of deferring it, paid in labelling);
the clinician-prior interface; care-fragmentation bounds (PU channels 2 and 3 stay
unmeasured); window retuning — **the 5-year lookback and 1-year label window are FIXED**
for this program; PC stays dead.
