# Incident & episode-anchored evaluation program — temporal honesty for conditional prediction and VOI

**Status:** approved direction (2026-09-01), spec. Plan doc to follow.
**Decision owner:** Shawn.
**Gating input:** `docs/reports/2026-09-01-temporal-eval-program-audit.md` (adversarial,
read-only, citations at `e6209c7`). Every conflict and correction that audit raises is a
requirement below; each is cited by section. The design as discussed pre-audit is **not**
buildable verbatim and this spec is the corrected form.
**Related:** `specs/2026-07-18-lookback-window-design.md` (the window design this repairs);
`plans/2026-08-31-native-mondo-label-space-plan.md` (exp 0110 — this program sequences
around it); `reports/2026-08-31-whole-mondo-arc-handoff.md` (state at entry);
ADR 0018, ADR 0025, ADR 0047; insights 0009, 0010, 0025.

---

## 1. Motivation — what the mainline actually measures

The mainline whole-Mondo corpus (exps 0104/0109, and 0110 after it) runs
`index_mode="population"` (`multi_domain.py:381-385`): every patient gets **one random
event-anchored index**, features are the strictly pre-index 5-year lookback, labels are the
Mondo closure of conditions attested in `[index, index + 365)`. The temporal split is real
and the availability gates are honest — features cannot contain a post-index code, because
the windows are disjoint frames (`cohorts.py:1648-1668`).

What the design does **not** do is separate two different questions that share a metric:

- **Tracking.** The patient already carried the condition before the index; it is coded
  again in the label window. The lookback is full of its workup, its drugs, its sequelae.
- **Onset prediction.** The condition is absent from the pre-index record and appears in the
  label window.

Chronic conditions appear on both sides of the index by definition. Every per-node AUC/AP
in 0104/0109 is therefore a **blend** of an easy tracking problem and the hard prediction
problem that motivates the program, mixed in an unreported and per-node-varying proportion.
Root prevalence at 0109 is **0.9609** — the blend is not a rounding effect.

**Second, a live documentation defect.** The 2026-07-18 spec's structural claims —
*"no disease code exists before it… leakage-free by construction… the `strip_mode`
machinery becomes a no-op"* (`specs/2026-07-18-lookback-window-design.md:20-26`) — were
written for `index_mode="disease"`, where the index **is** the first code in the disease
concept set. They are **void** under `index_mode="population"`, where a random event anchors
the window and any DAG-node code may precede it. The audit (§E6) confirms the stale claim
lives in **three** places, one of them in code. See E6.

## 2. Position — the random index is right; the missing piece is stratification

**Decision: keep the random per-person index. Add stratification by what was already known
at the index. Do not re-anchor.**

Reasons:

1. The random index is a **uniform lottery** over a patient's coded history. It introduces no
   control-index-date selection bias, and foreground/background carry matched observation
   depth by construction (`specs/2026-07-18…:49-58`).
2. **Per-disease first-code indexing was considered and rejected.** It breaks the
   one-θ-per-person factorization: each node c would need its own index, hence its own
   document, hence its own θ — a ~C× assembly blowup at C≈3,820.
3. Worse, the hierarchy makes "the" index **ill-defined**. The index for node c must precede
   everything in `closure(c)`; nodes are not independent, so a per-node index is a per-node
   *corpus*, not a per-node row filter.

Stratification is cheap by comparison: the index stays where it is, and a per-document
**pre-index closure vector** (E1) partitions rows into tracking and prediction cells at eval
time only.

## 3. Conditional heads and VOI — what the estimator is for

The conditional readout (`gated_pc_cloud.py:276-370`) conditions a child head on the
**parent's truth**, not the parent's coding. This is deliberate and it is what makes
deployment work:

- Deployment marginalizes the parent through the model's **own posterior**, by the chain
  rule: `P(c | hist) = P(P | hist) · P(c | P, hist)`.
- A clinician's soft prior ("60% chance this is a connective-tissue disorder") therefore
  blends into the parent marginal **natively** — no new interface, no recalibration. The
  interface itself is a non-goal (§9); the factorization is what makes it later-buildable.

**Evaluating the conditional factor on retrospectively-confirmed cases — conditioning on the
future — is the correct estimator, not a leak.** VOI asks a counterfactual question: how much
would a candidate piece of information sharpen the child distribution? The parent's truth is
the limiting case of that candidate. An estimator of a counterfactual quantity is entitled to
condition on the realized outcome; what it may not do is let that conditioning reach the
*features*. It does not: features remain strictly pre-index.

**Feature-level VOI is the endgame** — "what would knowing code X buy me here?" It requires
two prerequisites, and this program supplies exactly those:

- (a) **honest conditional factors** — the incident-local cells of E3;
- (b) **"history + X" as a pre-index object** — the pre-index closure primitive of E1
  generalizes to "what was known before the index", which is the object X gets added to.

## 4. Outcome enrichment — deliberate, and its cost stated plainly

Conditioning the eval (and later, under E5, the episode *corpus*) on "a diagnosis follows"
is **deliberate**. It is the local-negatives philosophy promoted from the cell level to the
sampling level: the model must not get credit for predicting **that** healthcare happens,
only for **what** the encounter reveals.

**Consequence, stated once and carried on every table:**

- Pooled and local incident metrics are **DISCRIMINATION** claims — *"among patients
  presenting in this neighborhood, which condition is it?"*
- They are **not prospective-cohort** claims. Do not read them as "risk of onset in the
  general population".
- The **known-P stratum** (§6) is the only fully-prospective cell in the program.

**Calibration follows the same split.** Conditional calibration *within neighborhood cohorts*
is a deliverable (the conditional readout already emits per-edge and pooled ECE and per-node
reliability — audit §E3). **Global** calibration and prevalence are **demoted**: they were
already compromised by PU contamination (§5), and outcome enrichment finishes the job. If a
population-prevalence claim is ever wanted, it comes from an **optional thin random-index
arm** reserved for that purpose — not from the enriched corpus.

## 5. PU contamination — every recorded number is a floor

Negatives in this corpus are **positive-unlabeled mixtures**. Three channels:

1. **diagnosed beyond the horizon** — the label window closed before the code appeared;
2. **never diagnosed** — the condition is present and unrecorded;
3. **coded elsewhere** — care fragmentation; the code exists outside the CDR.

E4 measures **channel 1 only**. The program's PU number is therefore a **LOWER BOUND** on
contamination and must be labelled as such wherever it appears. Channels 2 and 3 are
unmeasured here; care-fragmentation bounds are a non-goal (§9).

**Sibling-restricted local negatives are themselves the utilization-confound mitigation:**
a local negative is a patient worked up *in the same neighborhood* who did not get c, so
"had contact with the system" is matched rather than modelled.

All three PU channels push recorded metrics **DOWN** (true positives scored as negatives
depress AUC/AP). **Recorded numbers are floors.** This is a favourable direction and it is
not a licence to ignore the bias — it is a licence to report the number without an upward
correction.

---

## 6. Normative definitions

These are the spec's core. Implementations must match them exactly.

**D1 — Pre-index closure `R_d`.** For each document *d*: the **label-style closure vector**
computed by the existing frontier→closure machinery over the **FEATURE-window** condition
events (the lookback), using the **same kept-node DAG**, the **same climb**, and the **same
closure rule** as the labels. Not a new definition of "known" — the label definition,
evaluated on the other window. Machinery: `_attest` /`attach_frontiers`
(`case_finding_assembly.py:261-274`) / `frontier_to_label` (`:106-160`), all pure functions
of `(attested_cids, before_dag, keep, cid2int, lay)` (audit §E1 seams).

**D2 — Incident-eligible(d, c).** `c ∉ R_d`. A document that already carried `closure(c)`
before the index is **excluded from BOTH classes** for node c. Prior carriers are *tracking
rows*, not *prediction rows*; dropping them from the positives only (and not the negatives)
would be a different and wrong estimator.

**D3 — Incident positive.** Eligible for c **and** gains `closure(c)` in the label window.
**D4 — Incident negative.** Eligible for c **and** does not.

**D5 — Local negative for edge P→c.** A document that is (i) eligible for c, (ii) gains a
**sibling under P**, or **P-but-nothing-more-specific**, in the label window, and (iii) does
not gain `closure(c)`. This is `conditional_readout`'s existing cohort
(`gated_pc_cloud.py:310-320`) intersected with D2.

**D6 — P-stratum.** `P ∈ R_d` ("parent known pre-index") vs `P ∉ R_d`.

> **The P-stratum is a REPORTED STRATUM, NEVER A GATE.** Pooled numbers are primary.
> (Explicit correction by Shawn to the pre-audit design.) Rationale: requiring pre-index P
> starves the cells; and **"no P / gains c" is a legitimate positive** — it tests *de novo*
> specific prediction, which is the harder and more interesting half. The two strata have
> materially different AUCs and **must not be mushed into one unlabeled number**.

**D7 — The C2.4 naming rule** (audit §C2.4). Heads are fit on the prevalent problem:
`_fit_readout_heads` standardizes per node on that node's own observed **train** rows
(`gated_pc_cloud.py:977-979`). Therefore every incident metric in this program is
**"a prevalent-fit model evaluated on an incident cohort"** — a legitimate quantity, but not
"the incident AUC". **Every table says so.** Train-time incident masking is explicitly
deferred (§9); D7 is that deferral's cost, paid in labelling.

---

## 7. Elements

### E6 — Doc repairs (now, free) · audit §E6 CLEAN

The stale "leakage-free by construction / strip is a no-op" claim lives in **three** sites:

1. `specs/2026-07-18-lookback-window-design.md:20-26`
2. `docs/experiments/0061-dag-placement-rare6-lookback-1yr.md:55-59`
3. **In code:** `case_finding_assembly.py:394-396`

**Requirements**

- R6.1 Repair sites 1 and 2 now: scope the claim to `index_mode="disease"`, and state that
  under `index_mode="population"` DAG-node codes **may** precede the index, so the strip is
  live and chronic carriers are on both sides. Cross-reference §1 of this spec.
- R6.2 **Site 3 rides along.** Editing `case_finding_assembly.py` moves `assembly_src`
  (`_case_finding_cache.py:112-113`), which orphans **every** cached bundle in the repo —
  including 0104's record bundle (~20 min of BigQuery to rebuild,
  `test_case_finding_cache_mondo.py:83`) — and breaks all four pinned tripwire hashes
  (`:63-65`, `:87`). Per the tripwire's own instruction (`:84-92`), the comment fix ships
  **with the next assembler change, or with a cache drop that is accepted and noted** —
  **never alone**.
- R6.3 Document **strip narrowness** at the same time: `drop_idxs` comes from
  `before_dag.nodes()` (`multi_domain.py:223,293`) = anchor cids + synthetic negatives.
  Attestation climbs **from descendants** (`mondo_dag.py:298-310`), and those descendants are
  **not** stripped. The strip removes anchors; it does not remove the label-generating
  descendant codes.

**Acceptance:** sites 1 and 2 contain no unqualified leakage-free claim; a repo grep for the
claim returns only site 3, with an open note recording why it is still there.

> **Recorded tension (audit §E6 vs. this program's shape).** The audit says site 3 rides with
> "the next assembler change". E1 is deliberately **not** an assembler change (R1.1), so this
> program may supply no such carrier. Site 3 may therefore stay stale for the duration.
> Accepted; R6.2's open note is the mitigation. Do not force an assembler edit to close it.

---

### E1 — Pre-index closure primitive · audit §E1 CLEAN-WITH-NOTES

**When:** after the 0110 port lands, **before** the 0110 record run.

**Requirements**

- **R1.1 Driver-level module, not an assembler edit.** Implement as
  `analysis/cloud/preindex_closure.py`, applied in **driver-owned code** during bundle build,
  on the **`mondo_collapse` template** (`gated_pc_cloud.py:1773-1783`, whose comment at
  `:1774-1779` says it was placed there *precisely* to avoid moving `mondo_src`).
  `gated_pc_cloud.py` is **not** source-hashed into any cache key — driver-level code is free.
  Editing `case_finding_assembly.py` or `multi_domain.py` instead would orphan every cached
  bundle including 0104's record (audit §E1 conflict 1). **Non-negotiable.**
- **R1.2 Fold-when-on cache-key flag**, per the `dag_collapse` precedent
  (`_case_finding_cache.py:160-166`). With the flag **off**, the four pinned tripwire hashes
  must re-verify **byte-identical**; this is a test, not an aspiration.
- **R1.3 Post-prune internals are recovered from the bundle**, not from assembler internals:
  `bundle.parent_int` → `DagLayout(parent_int, n_bg, tpn)`; `bundle.cid2int`;
  `keep = set(bundle.int2cid.values())` (audit §E1 note 2).
- **R1.4 Manifest/meta witness field.** `_meta_dict` (`_case_finding_cache.py:171-191`)
  serializes only `parent_int/int2cid/cid2int/name_by_id/ledger/vocab_map(s)`; the extra
  column would ride in the parquet **silently**. Add a witness recording that a given cached
  bundle carries the pre-index column, and **fail loudly** on a bundle that lacks it when a
  readout asks for it. **No silent mixed-vintage cache dirs** (audit §E1 note 3).
- **R1.5 Sparse storage.** Store `R_d` as a **sparse index list**, not a dense
  `array<double>`. `label` + `labelMask` are already 2×3,820 float64 ≈ 61 KB/row; a third
  dense array is +50% on the bundle parquet (audit §E1 note 4). Consequence: `attach_labels`
  (`case_finding_assembly.py:163-191`) **cannot be reused verbatim** — it emits dense
  (`:177-180`). The driver module emits sparse.
- **R1.6 Grain is per-DOCUMENT**, i.e. per `(person, index)` once E5 exists — **not** per
  person (audit correction 8). Three artifact grains exist across E1/E4/E5 and are never
  conflated: per-document (E1), per-`(person, node)` (E4), per-person (ADR 0025's covariate
  sidecar). Name the grain in every artifact's schema doc.

**Sequencing rationale (recorded, audit §Sequencing).** The primitive's **code** survives
0110 untouched — it is parameterized over provider and DAG. Its **values** do not:
"prior carrier of `closure(c)`" is a materially **broader** predicate under today's
every-powered-ancestor climb than under 0110's `reduce_tie_map` most-specific attribution,
especially for the 619 category-anchors. **Therefore: eligibility numbers, and every rate
derived from them, are only measured post-0110.** Building E1 before 0110 would mean
measuring twice, not coding twice.

**Acceptance:** flag-off keys byte-identical against all four pinned hashes; a bundle built
flag-on carries the witness and a sparse column; a synthetic fixture where a patient's
pre-index record contains `closure(c)` yields `c ∈ R_d`; the same patient with the code moved
after the index yields `c ∉ R_d`.

---

### E-census — the go/no-go gate · audit §Sequencing rec 3

**When:** immediately after E1. **Before E2/E3/E4 are built at all.**

A single cheap corpus-property probe, the same move that `diag-sibling-support` made for
degeneracy (and for the same reason: corpus properties are checkable before any fit).

**Requirements**

- **RC.1** One `treeAggregate` over bundle + pre-index closure, same shape as
  `diag_sibling_support`. Per node: `(n_incident_eligible, n_incident_pos, n_incident_neg)`.
- **RC.2** Report the count of nodes clearing **`min_count = 20` on BOTH classes**.
- **RC.3** Report how many previously-constant heads enter the **C2.1 population** —
  non-degenerate-but-constant under incident masking (see E2/R2.1).
- **RC.4 Closure discipline (ADR 0047 addendum).** Nothing array-shaped rides a task closure.
  Reduction identities are `None` sentinels with identity handling in the combiner; partials
  are allocated executor-side. A `disk_telemetry:` `pyspark-*` dir growing linearly in passes
  is the signature of a violation.

**GO / NO-GO**

> If **fewer than ~a few hundred nodes** clear 20/20 on both classes, **the incident macro is
> not a deliverable and E2/E3 are not built.** Given 96% root prevalence this is a live
> outcome, not a formality. **The census numbers are recorded as a finding either way** — the
> distribution of incident-eligible support across the label space is itself the most
> decision-relevant number in the program, and the cheapest.

> **Note on RC.3's magnitude.** The audit's "up to 619 forced-0.5 nodes" is a **0104/0109
> vintage** figure. The 0110 plan's structural claim is that the subsumed-sibling trap becomes
> impossible by construction, taking degeneracy to `1 (root) + small thin-chain residue`
> (`plans/2026-08-31…:88-91,139-142`). Post-0110 the C2.1 population should be small. **The
> guard in R2.1 is mandatory regardless** — it costs one `np.ptp` and it is the difference
> between a real macro and an inflated one.

---

### E2 — Dual prevalent/incident metrics · audit §E2 **CONFLICT** · gated on E-census

**Seam.** A **second call** to `readout_from_proba(proba, y_te, m_te, C, ...)`
(`gated_pc_cloud.py:187-208`) with `m_incident`. This is the **driver path — the live path**
(audit correction 2): the escape-hatch `per_node_metric_rows` /`score_cells_df`
(`distributed_readout.py:31-32,1284-1323`) has **never been called by a driver**. No solver is
touched.

**Getting `m_incident` to the driver:** a **fourth CSR run** in the lean eval block.
`_lean_eval_kernel` packs `(ids, P, y_idx, y_ptr, m_idx, m_ptr)`
(`distributed_readout.py:565-643`); add the run plus the matching `_densify_lean_blocks`
branch (`gated_pc_cloud.py:756-788`). **+1 byte/cell.** Neither `distributed_readout.py` nor
`gated_pc_cloud.py` is source-hashed into a cache key — free.

**Hard requirements (all four from the audit; none optional)**

- **R2.1 Constant-column guard on the RANKING axis (C2.1).** Every node whose *train* cell was
  degenerate carries a **constant column**: `proba[:, deg] = const[deg]`
  (`gated_pc_cloud.py:856-858`). Degeneracy is a **train-side** property
  `(n_obs<=0)|(n_pos<=0)|(n_pos>=n_obs)` (`diag_sibling_support.py:78`). Under the
  **prevalent** mask those columns are all-positive in test too, so `_score_label` returns
  `skipped: "degenerate test column (all-positive)"` and `_macro` drops them
  (`evaluate.py:87-95,116-125`). Under the **incident** mask, prior carriers leave **both**
  classes — exactly the set that made the column all-positive — so some columns acquire
  negatives, become non-degenerate, and are **scored**: `roc_auc_score(y, const)` = **exactly
  0.5**, `skipped: None`, **included in the macro**.
  **Requirement:** columns with `np.ptp(proba[:, c]) == 0` are **skipped** with reason
  `"constant prediction column"`, and **the count is reported**. This mirrors the detection
  fix (`gated_pc_cloud.py:155-171`, an explicit *"EVAL bug fix, always on"*), which was
  deliberately scoped to detection only (`:162-165`); the incident arm reopens the same bug
  on the ranking axis and must close it there too.
- **R2.2 Macros on BOTH node sets (C2.2).** `_macro` averages only non-skipped labels
  (`evaluate.py:118-119`). At `min_label_count: 20` the prevalent arm scores 2,106 of 3,677;
  the incident arm scores strictly fewer, and a **different subset** — a headline
  prevalent-vs-incident delta computed across different node sets is **not a comparison**.
  Report macros on **the shared both-arms-scoreable node set** *and* on **each arm's full
  set**. This is the 0110 plan's own discipline (`plans/2026-08-31…:144-146`) carried over.
- **R2.3 Eligibility is a CORPUS property, never a run property (C2.3).**
  `conditional_readout`'s docstring is the standing decision (`gated_pc_cloud.py:296-301`):
  *"MASK-INDEPENDENCE: pass the FULL-closure observation mask (all-ones)… otherwise the
  closure mask silently makes the conditional eval an easier sibling-only contrast and
  cross-run numbers are not comparable (exp 0079, Trap 3)."*
  **Requirement:** incident eligibility is defined as a **pure function of (bundle,
  pre-index closure)** — D2 — and of nothing produced by a fit. It is computed once per
  corpus, stored with the corpus, and reused byte-identically by every run being compared.
  A run's own predictions, thresholds, or degeneracy set may never enter it. Stated in the
  spec because the letter of Trap 3 does not forbid what its spirit forbids.
- **R2.4 The C2.4 naming rule (D7) on every output**, in the JSON block and in every table.

**Output:** a `"gated_pc_incident"` block in `results_readout.json`, written alongside the
existing per-arm blocks by `run_readout` → `_dump_partial_results`
(`gated_pc_readout.py:525-528`).

**Acceptance:** flag-off runs reproduce existing `results_readout.json` numbers exactly;
the incident block carries the skipped-constant count, both node-set macros, and the D7
label; a fixture with a forced constant column and acquired negatives is **skipped**, not
scored at 0.5.

---

### E3 — Incident-local cells + P-strata · audit §E3 CLEAN-WITH-NOTES · gated on E-census

**This is substantially already built.** `conditional_readout` (`gated_pc_cloud.py:276-370`)
implements the cell **verbatim, minus the incident filter**: `:310` builds the parent-P
cohort `np.where((y[:,p]==1) & (mask[:,p]==1))`; `:315-320` takes positives as
`y[rows,c]==1` and negatives as the **rest of P's cohort** — siblings under P, or
P-but-nothing-more-specific. That is D5 without D2. It already reports `cond_auc`, `cond_ap`,
`marg_ap`, per-edge and pooled ECE, top-1 vs majority, and per-node reliability, and it is
already in `results_readout.json` (`:2480`). `label_mask_mode="closure"`
(`case_finding_assembly.py:148-156`) is the same construction at corpus level and is the
mainline setting.

**What is actually new: two arguments.** An eligibility array and a stratum key.

**Requirements**

- **R3.1** Thread the D2 eligibility array into `conditional_readout`; intersect it with the
  existing cohort construction. Thread the D6 stratum key (`P ∈ R_d`).
- **R3.2 Report pooled first, strata second.** Per D6, the stratum is reported, never gating.
- **R3.3** `min_count >= 20` **on both classes, per cell** (`gated_pc_cloud.py:318`).
  Stratifying halves each cell; expect most edges to drop out of one or both strata. Apply
  the R2.2 discipline to the surviving-edge sets.
- **R3.4 Break the `_ones` convention deliberately, in writing.** The call site
  (`gated_pc_cloud.py:2477-2479`) passes an all-ones mask precisely to protect the Trap-3
  comparability rule. Passing an eligibility array breaks that convention. **Re-state the
  rule it protected at the call site**, together with R2.3's resolution (eligibility is a
  corpus property, so cross-run comparability survives). An undocumented break here is the
  exact failure mode Trap 3 records.
- **R3.5 EGRESS.** `min_count` is **not only a statistical dial**: counts < 20 are **not
  disclosable** on All of Us (`evaluate.py:76-78`). Stratified cell tables are a **disclosure
  surface**. **Nothing under 20 leaves the workspace.** Keep publishing floors and
  model-internal dials structurally separate, per the 0110 plan's separation
  (`plans/2026-08-31…:59-62`) — never conflate `min_positives` (internal powering) with the
  egress floor.

**Acceptance:** with eligibility all-ones and the stratum collapsed, output is numerically
identical to today's conditional block; the stratified tables carry no cell under 20; the
call-site comment states the broken convention and why.

---

### E4 — Future-conversion analysis · audit §E4 CLEAN-WITH-NOTES · gated on E-census

**The question:** of the documents scored as incident **negatives** for c, how many are
diagnosed with c **later** — beyond the label horizon? That is **PU channel 1**, measured.

**Confirmed facts.** Full history is loaded with no date filter (`load_omop_bigquery`,
`bigquery.py:216-372`, applies only `person_sample_mod` and `concept_id != 0`), but
post-label-window events are **discarded at windowing, not retained**
(`lookback_feature_label_events`, `cohorts.py:1648-1668`); the cached bundle holds BOW
vectors, not events (`_case_finding_cache.py:21-23`). **Nothing downstream can recover a
future date.** The artifact must be built where the full-history frame still exists.

**Requirements**

- **R4.1 Artifact: a first-attestation-date sidecar**, per `(person, frontier-node)`,
  computed from the full-history `cond` frame at build time
  (`cond ⋈ climb on concept_id = descendant_concept_id → groupBy(person_id,
  ancestor_concept_id).agg(min(condition_era_start_date))`). "First attestation of
  `closure(c)`" folds driver-side via `lay.closure`.
- **R4.2 Driver-owned code; cached as a sidecar parquet per ADR 0025's pattern.** The natural
  aggregation site (`multi_domain.py:365-367`) is **inside the source-hashed assembler** — the
  same trap as E1. A driver-level equivalent needs either its own load of `cond` (a second
  full-table scan at `person_mod: 1`) or the sidecar; the sidecar wins and ADR 0025 is exact
  prior art, including its treatment of multi-doc persons.
- **R4.3 Grain discipline (audit §E4 note).** ADR 0025's sidecar is **per person**; E4's is
  **per `(person, node)`**; E5's eligibility joins are **per `(person, index)`**. Three
  grains. Schema-doc each; never join two without naming the grain change.
- **R4.4 Right-censoring is handled.** Conversion denominators are **gated on
  `observation_period_end_date` at each horizon** — reuse `_window_observed_cohort`'s logic
  (`cohorts.py:693-729`). Without this the conversion rate is a censoring artifact, not a
  contamination estimate.
- **R4.5 Report per-node conversion at 1y / 2y / 3y horizons**, **overall** and **stratified
  by model-score decile**.
- **R4.6 Framing: LOWER BOUND on PU channel 1.** Mandatory language. Channels 2 and 3 are
  unmeasured.
- **R4.7 Expect the bound to be loose.** 0109 records root prevalence **0.9609**; most nodes'
  "did not gain c this year" is a weak negative and conversion rates will be high across the
  board. **A finding, not a failure** — record it as such; do not assume tightness and do not
  tune toward a tight number.
- **R4.8 Score-stratified conversion enrichment doubles as case-finding validation.** If the
  top score decile converts at a materially higher rate than the bottom, the model is finding
  future cases among its own "negatives" — which is the case-finding claim, measured on the
  one population where it can be measured.

**Acceptance:** the sidecar round-trips with a grain-named schema; horizon denominators shrink
monotonically with horizon under censoring; the decile table is reported with the lower-bound
language attached.

---

### E5 — Episode-anchored index sampling = **experiment 0111+** · audit §E5 **CONFLICT (highest risk)**

**Decision: catalogue now, build later. NEVER bundled with 0110.**

Reason (audit §5g): E5 changes the **document unit**. Insight 0010 is definitive — NPMI, and
by extension any doc-unit-sensitive number, is **not comparable across doc units**. The 0110
acceptance protocol makes 0104's **0.6978 / 0.4845** the control on a shared node set
(`plans/2026-08-31…:144-150`). An episode corpus changes the doc unit, the vocabulary
(`min_df` is a **document** count, `topic_prep.py:222-224`), every node's prevalence, and all
base rates. **0104/0109 stop being controls.** Bundling E5 with 0110 destroys the 0110 record
before it is taken.

The requirements are catalogued here so the future 0111 plan inherits them complete.

**E5 requirements (for the 0111 plan; not built by this program)**

- **R5.1 `EpisodeDocSpec`** is a **sanctioned extension point** — ADR 0018;
  `docs/architecture/TOPIC_STATE_MODELING.md:148-155` names it explicitly. `doc_id` gets the
  index **APPENDED** (`cohort:person:index`): prefix parsers (`split(":").getItem(0)`,
  `eval_coherence_cloud.py:273` et al.) survive **appends only**. Note the doc's claim that a
  new DocSpec is "one class + manifest round-trip; the BOW build, fit drivers, and eval
  drivers don't change" is **now false** — the ADR-0046 readout stack hard-codes an int64 doc
  key and a dense driver collect (R5.6/R5.7).
- **R5.2 `index_date` passthrough in `cohorts.py`.** It is dropped on both outputs today
  (`cohorts.py:1663,1667`) and `DocSpec.derive_docs` never sees it. Fixing this moves
  `cohort_defs_version()` (`cohorts.py:36-52`, a source hash of the **entire** file, folded at
  `_case_finding_cache.py:111`) and **invalidates every cache key in the repo** — bundle,
  corpus, covariates — plus all four pinned hashes. **Accepted as a FULL cache drop, done
  deliberately, with a note**, per the tripwire's instruction (`:84-92`). Maximum blast
  radius; take it once, knowingly.
- **R5.3 CLOSE THE DOC-SPEC CACHE-KEY HOLE FIRST (audit seam 4).** `doc_spec` is hard-coded in
  two places (`multi_domain.py:408`, `gated_pc_cloud.py:1769`) and is **absent from every
  cache key** — `_SPEC_ASSEMBLY_KEYS` (`gated_pc_cloud.py:1657-1662`) has no doc-spec entry.
  Changing the driver-side doc spec alone would **poison the cache under a byte-identical
  key**: a silent-wrongness hazard, not a rebuild cost. **Must be closed before any doc-unit
  work**, and it is cheap enough to close **early, as hygiene**, independent of whether 0111
  is ever built.
- **R5.4 int64 doc-key synthesis (seam 1).** `_lean_eval_kernel` hard-codes int64 ids
  (`distributed_readout.py:632-633`; driver twin `gated_pc_cloud.py:774`;
  `id_col="person_id"` at `:794,:1198`). String doc_ids **raise**. Synthesize an int64 doc key
  (or change dtype on both sides). **Hard break.**
- **R5.5 A/B alignment dict fix (seam 2).** `readout_ab_report` aligns collects by a dict
  keyed on `person_id` (`gated_pc_cloud.py:1338-1341`); under multi-doc, **duplicates silently
  overwrite** and the A/B gate compares mismatched rows. **Silent wrongness inside the
  correctness gate itself.** Also `sample_frac` is row-level (`:1295-1298`, seam 6).
- **R5.6 Person-keyed calibration split in the driver path (seam 5).** The driver-path split
  is **row-level** (`gated_pc_cloud.py:2467-2470`); under multi-doc a person straddles cal/fit
  — the exp 0079 run-2 failure. The **distributed twin is already person-keyed and safe**
  (`:2434-2436`). `split_train_test` is explicitly multi-doc-safe
  (`case_finding_assembly.py:277-289`) and needs no change.
- **R5.7 Person-level detection dedup (seam 7).** `detection_readout` is per-document
  (`gated_pc_cloud.py:169-176`) and silently becomes **episode-weighted** under multi-doc.
- **R5.8 Wire the distributed eval path — MANDATORY at ×3+ doc multiplier.**
  `score_cells_df` /`per_node_metric_rows` exist but have **never been called by a driver**
  (`distributed_readout.py:31-32`). The driver-collect wall, from 0104's recorded run:

  | quantity | 0104 record | ×3 | ×5 |
  |---|---|---|---|
  | D_te | ≈80k | 240k | 400k |
  | lean eval bundle @6 B/cell, C=3,820 | 1.9 GB | 5.7 GB | 9.5 GB |
  | `calibrate_per_node` float64 copy | 2.4 GB | 7.3 GB | 12.2 GB |
  | observed train cells | 56.2M | 169M | 281M |
  | readout pass @ topm=256 | 17.6 s | ~53 s | ~88 s |
  | 60-iter solve (×2 with calibration) | ~1.2 h | ~3.5 h | ~6 h |

  At ×5 the lean bundle **plus** the calibration float64 copy exceed a 16 GB driver. The
  program adopts **×3 as the wiring threshold**, tighter than the arithmetic strictly forces
  (×3 sums to ~13 GB — under 16 GB, but with no headroom for the readout's own working set).
  Related: the binding constraint on the readout is the **O(N·C) driver collect**
  (`_densify_lean_blocks`), not the O(N) pass (audit correction 1).
- **R5.9 Measure the episode multiplier BEFORE any fit.** Distinct 60–90d
  first-attestation episodes per person, via **gap-and-islands over the `cond` frame** —
  genuine prior art at `cohorts.py:2237-2252` (`_stable_drug_intervals`). **No repo data
  exists on episodes-per-person; the multiplier is a guess until measured**, and R5.8's
  threshold cannot be evaluated without it.
- **R5.10 Measure the prior-obs-gate kill rate.** `_LOOKBACK_PRIOR_OBS_DAYS = 365` is
  **hardcoded and deliberately un-overridable** (`case_finding_assembly.py:43-53`). A person's
  **first** diagnostic episode is by construction at or near record start, and the gate is
  `index >= op_start + 365` — so **the earliest, most unambiguously incident episodes are
  systematically dropped**. Symmetrically `index + 365 <= op_end` kills the **last** episode of
  every record.
  > **"100% incident capture" is NOT achievable and this spec says so** — superseding the
  > pre-audit discussion's claim. Capture is conditional on ≥1 year of prior observation, and
  > that condition is **anti-correlated with incidence**. The 2026-07-18 spec's survivorship
  > caveat (`:60-64`) is **amplified here, not escaped**. The kill rate has never been
  > measured; 0111 budgets an empirical probe.
- **R5.11 Insight-0009 risk, checked in the smoke.** With `lookback_days: 1825` consecutive
  episode docs share ~80% of their events — **overlapping**, not partitioning. Insight 0009 is
  a direct hit: doc-multiplication for chronic patients drives catch-all topic growth (there,
  a single topic to E[β]=0.224 with 14 vestigial variants). Whether `n_bg: 8` absorbs it is an
  **open empirical question**. The 0111 smoke checks it via **coherence + topic-usage
  diagnostics**, before any record run.
- **R5.12 ESS caveat, stated on every interval.** Overlapping per-person documents **violate
  row independence**. Every CI and every `min_count >= 20` threshold silently assumes
  independent rows; effective sample size is far below N. State the caveat; do not silently
  publish a nominal CI.
- **R5.13 Episode-vs-random comparison happens WITHIN 0111.** 0111 carries **its own random
  arm** as the control. It is **not** compared against 0104/0109 (insight 0010; §E5 opening).
  What makes any such comparison possible at all is that `node_patient_counts` and the Mondo
  power counts are person-level and index-independent (`case_finding_assembly.py:250-257`,
  `mondo_dag.py:258-262`, audit seam 10) — **C and the label DAG stay fixed across doc
  units**. Watch `min_doc_length: 10` (seam 9): it drops episode docs **non-uniformly toward
  the incident end** — the shortest lookbacks are the earliest episodes.
- **R5.14 Record the reversed prose stance.** `cohorts.py:723-728` explicitly names
  over-weighting multi-period patients as a harm to avoid, and `.distinct()` there exists to
  prevent index fan-out. **Episode anchoring does that fan-out by design.** The trade is now
  accepted **deliberately**: for this program the goal is **capture** — putting the model at
  the moment of presentation, repeatedly — **not representativeness**. Representativeness is
  what the optional random arm (§4) is for. Amend the comment at that site when 0111 lands so
  the code does not read as forbidding what the driver now does on purpose.
- **R5.15 Machinery reuse, correctly attributed (audit §5a).** `_random_event_windows`
  (`cohorts.py:1083-1145`) **cannot** be reused — it *is* the one-per-person sampler
  (`row_number() … rn == 1`, `:1136-1143`) and its eligibility is defined on event dates;
  "just before episode start" is not an event date (audit correction 5).
  `_window_observed_cohort` (`cohorts.py:693-729`) **can** — it takes an arbitrary
  `(person_id, index_date)` frame and preserves N rows per person; it drops other columns, so
  rejoin the episode id as `_mdd_antidepressant_index` does (`cohorts.py:1905`).

---

## 8. Metrics protocol

**Every reported number in this program carries four tags. No exceptions, no defaults.**

| tag | values |
|---|---|
| **arm** | prevalent · incident |
| **node set** | shared (both arms scoreable) · full (this arm's own) |
| **cell type** | marginal · conditional-pooled · conditional-stratum(P known / P unknown) |
| **claim type** | discrimination · prospective |

Plus, standing:

1. **D7 naming rule** on every incident output: *prevalent-fit model, incident cohort*.
2. **Conditional calibration within neighborhood cohorts is a deliverable** — per-edge and
   pooled ECE, per-node reliability, already emitted by `conditional_readout`.
3. **Global calibration and prevalence come only from the optional random-index arm** (§4).
   They are not claimed from the enriched corpus.
4. **PU lower-bound language is mandatory** wherever a conversion or contamination number
   appears (§5, R4.6). Recorded metrics are **floors**.
5. **Skipped-column counts are reported, not suppressed** — constant-prediction skips
   (R2.1), min-count skips, and degenerate-test-column skips are three different reasons and
   are counted separately.

---

## 9. Non-goals (recorded so nobody helpfully bundles them)

- **Train-time incident masking.** Reopens the whole degeneracy accounting (which cells are
  observed, which heads are constant, what `min_positives` means). Its own arc. D7 is the
  price of deferring it and D7 is paid in labelling, not in silence.
- **Clinician-prior interface.** Falls out of the factorization when wanted (§3); building it
  now would be an interface in search of a use.
- **Care-fragmentation bounds.** PU channels 2 and 3 stay unmeasured (§5).
- **Window retuning.** The **5-year lookback and 1-year label window are FIXED** for this
  program. Explicit decision: do not knob them. Changing either changes eligibility, base
  rates, and the doc unit's effective content simultaneously, and would confound every number
  the program produces.
- **PC stays dead.**

---

## 10. Sequencing

| # | step | gate |
|---|---|---|
| 1 | **E6** doc repairs (sites 1–2; site 3 rides, per R6.2) | now — free |
| 2 | **0110 port lands** (plan §4, §7) | outside this program |
| 3 | **E1** pre-index closure primitive (+ witness R1.4, + flag R1.2) | after 0110 port, **before** the 0110 record run |
| 4 | **E-census** | **GO / NO-GO for E2/E3/E4** |
| 5 | **E2 / E3 / E4** | only on GO |
| 6 | **0110 record run reports dual metrics** | E2 landed |
| 7 | **E5 as exp 0111**, with its own plan doc | never bundled with 0110 |

R5.3 (the doc-spec cache-key hole) may be closed at any point from step 1 onward as hygiene;
it is the one E5 requirement that is cheap, safe, and independently valuable.

---

## 11. Validation

- **E1:** flag-off tripwire hashes byte-identical (all four); witness present/absent detection
  fails loudly; synthetic pre-index-vs-post-index fixture resolves `R_d` correctly; sparse
  round-trip.
- **E-census:** counts reproduce on a fixture with hand-computed eligibility; the aggregation
  ships no array-shaped closure (ADR 0047 addendum; check `disk_telemetry:` `pyspark-*`
  growth is flat in passes).
- **E2:** flag-off reproduces existing `results_readout.json` exactly; constant-column fixture
  is skipped, not 0.5; both node-set macros present; D7 label present.
- **E3:** eligibility-all-ones + collapsed stratum ≡ today's conditional block; no published
  cell under 20; call-site comment states the broken `_ones` convention.
- **E4:** grain-named schema round-trips; censoring denominators monotone in horizon; decile
  table carries lower-bound language.
- **E5:** out of scope here; its plan carries R5.9/R5.10 as pre-fit probes and R5.11 as a
  smoke-gate.
