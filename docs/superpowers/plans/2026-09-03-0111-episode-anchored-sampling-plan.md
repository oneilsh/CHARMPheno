# Exp 0111 — episode-anchored sampling · build plan

**Date:** 2026-09-03 · **Status:** proposed
**Spec:** `specs/2026-09-02-0111-episode-anchored-sampling.md` (all §4 inputs
resolved: gap 90d · cap 3 (cap-5 fallback) · R5.8 as precondition · incidence
fork SET — 365d primary, relaxed-gate sensitivity).
**Probe numbers:** `reports/2026-09-02-0111-episode-probe-results.md` (×2.66 at
cap 3; 2,583/2,714 nodes ≥20 gated episodes; 66.2% first-episode kill).
**Execution mode:** subagent-driven — each WP names its agent tier (Opus for
high-blast/architectural, Sonnet for mechanical-with-oracle); the orchestrator
holds sequencing, gates, and everything that touches cluster runs with Shawn.

**One plan-level correction to the spec, carried here and patched into R7.1:**
the one-blast commit is `cohorts.py` **AND `multi_domain.py`** — the assembler
hard-codes both the index-builder selection (`multi_domain.py:382-394`; no
external-frame option) and the doc spec (`:408`,
`PatientCohortDocSpec(min_doc_length=…)`). WP7/R5.3 closed the cache-KEY hole,
not the injection hole. Both files fold into every bundle key, so widening the
blast to cover both in ONE commit changes nothing about its cost and saves a
second drop.

---

## 0. Standing rules for every subagent

- Branch `claude/gated-conditional-voi` only. Never commit patient-level data.
- Source-hashed files (`cohorts.py`, `multi_domain.py`,
  `case_finding_assembly.py`, `mondo_*.py`, `condition_dag.py`,
  `preindex_closure.py` §key parts) are UNTOUCHABLE except inside WP-C, whose
  agent gets an explicit, bounded exception.
- The four pinned tripwire hashes (`tests/scripts/test_case_finding_cache_mondo.py`)
  must pass byte-identical after every WP except WP-C, which re-pins them.
- Local test invocation:
  `PYTHONPATH=spark-vi <venv>/bin/pytest tests/scripts/... -q -p no:randomly -m ""`.
- ADR 0047 closure doctrine: nothing array-shaped rides a task closure;
  treeAggregate zeros are None-sentinel.
- Any module whose functions ride executor-side goes on `--py-files` in EVERY
  submit path that can trigger it (the 9f7f002 lesson).
- No model identifiers in committed content.

---

## 1. Work packages

### WP-A — multi-doc seam fixes · four parallel Sonnet agents · driver-owned, no cache impact

All four are mechanical, line-addressed by the audit, with tests as oracle.
They are what makes the readout stack STOP assuming one doc per person.

**A1 — int64 doc-key synthesis (R5.4) · size M.**
`_lean_eval_kernel` hard-codes int64 ids (`distributed_readout.py:632-633`;
driver twin `gated_pc_cloud.py:774`; `id_col="person_id"` at `:794,:1198`) —
string doc_ids RAISE today. Introduce ONE shared synthesis
`doc_key = person_id * 64 + episode_no` (int64; cap ≤ 3 ≪ 64; episode_no = 0
for single-doc/random-arm docs so every existing corpus maps to its current
ids ×64 — order-preserving, collision-free), defined in one place and imported
by both sides, with an assertion that keys are unique per corpus. Everything
person-keyed must now derive `person_id = doc_key // 64` through the same
helper, never ad hoc. Oracle: existing readout tests pass on single-doc
fixtures; new fixture with two docs/person exercises the kernel end to end.

**A2 — A/B alignment + sampling by doc key (R5.5, seam 6) · size S.**
`readout_ab_report` aligns collects via a dict keyed on `person_id`
(`gated_pc_cloud.py:1338-1341`) — multi-doc duplicates silently overwrite
INSIDE the correctness gate. Key on the A1 doc key; make `sample_frac`
(`:1295-1298`) sample doc keys. Oracle: a two-docs-per-person fixture where
the old code provably compares mismatched rows and the new code agrees with a
hand-computed alignment. **Must land before WP-E: the A/B gate is the smoke's
correctness instrument.**

**A3 — person-keyed calibration split, driver path (R5.6) · size S.**
Driver-path split is row-level (`gated_pc_cloud.py:2467-2470`); the
distributed twin is already person-keyed (`:2434-2436`). Make the driver path
call the same person-keyed hash (via A1's helper). Oracle: no person straddles
cal/fit on a multi-doc fixture (the exp 0079 run-2 failure, pinned as a test).

**A4 — person-level detection dedup (R5.7) · size S.**
`detection_readout` (`gated_pc_cloud.py:169-176`) silently becomes
episode-weighted under multi-doc. Dedup to persons (max score per person, or
any-hit — match the readout's existing detection semantics and SAY which in
the docstring). Oracle: fixture where episode-weighting and person-level
disagree by construction.

> **A2/A3/A4 LANDED (`428874e`); the `gated_pc_readout.py` co-fit head
> detection residue is now CLOSED by WP-B (`0ab5d50`)** — it deduped that call
> to persons on both branches while wiring `--eval-path` through the recovery
> path, exactly as this note anticipated. No residue remains.

### WP-B — distributed eval + distributed calibration apply (R5.8) · one Opus agent · driver-owned, no cache impact · size L

The flagship infra WP. At ×2.66 the O(N·C) driver collect
(`_densify_lean_blocks`) plus `calibrate_per_node`'s float64 copy (~6.5 GB)
break the 16 GB driver; the never-called `score_cells_df` /
`per_node_metric_rows` path (`distributed_readout.py:31-32`) becomes the
mainline.

1. **Eval:** wire the distributed path into the readout driver behind a flag
   (`eval_path: driver|distributed`, default driver until parity is proven),
   with WP4's incident masks and R_d riding it (the lean kernel's fourth CSR
   run already carries R_d — the distributed path needs parity there too).
2. **Calibration:** replace collect-everything with distributed BINNED
   sufficient stats — per (node, score-bin): `(n, sum_y)` at ~100
   quantile-ish bins/node (C×100 rows, driver-tiny) → driver fits weighted
   isotonic per node on the bins → broadcast the fitted breakpoints →
   distributed apply. Keep `min_pos=20` pass-through and the honest
   ECE-on-test protocol exactly as shipped.
3. **Parity gate (the oracle):** on the CURRENT cached 0110 corpus, driver
   path vs distributed path must agree — per-node AUC/AP to numerical noise,
   calibrated ECE within a stated tolerance for the binned isotonic (record
   the tolerance and the measured delta; if binning visibly degrades ECE,
   raise bins before relaxing tolerance). Run via `make gated-pc-readout`
   against the saved fit — no re-fit.

Depends on A1 (doc keys in the distributed collects) and A2 (the A/B
instrument). **This lands and passes parity BEFORE WP-C burns the cache** —
we validate new infra against a corpus we already trust, then drop.

### WP-C — the ONE-BLAST cache-drop commit (R5.2/R7.1, widened) · one Opus agent · **full deliberate cache drop** · size M

The bounded exception to the never-edit rule. ONE commit, containing exactly:

1. **`cohorts.py`:** `lookback_feature_label_events` (and
   `lookback_feature_frames`) carry `index_date` through to both outputs
   (`:1663,1667` drop it today) so a doc spec can see it.
2. **`multi_domain.py`:** `assemble_multidomain_case_finding_corpus` gains
   `index_df=None` (accepted external `(person_id, index_date)` frame;
   `index_mode="external"` requires it, the two existing modes reject it) and
   `doc_spec=None` (defaulting to `PatientCohortDocSpec(min_doc_length=…)`,
   byte-for-byte the current behavior). Injection seams only — the
   `attested_provider` precedent (`:194-204`), no episode logic in this file,
   ever. That keeps all future episode iteration driver-owned and this the
   LAST planned edit to either file.
3. **Riders (comments only):** the stale strip claim at
   `case_finding_assembly.py:394-396` (WP0 residue); the index-fan-out prose
   at `cohorts.py:723-728` amended per R5.14.
4. **Tripwire re-pin:** update the four hashes, commit message naming the
   deliberate drop per the tripwire's own instruction (`:84-92`).

Behavior-preservation oracle: all unit tests; existing callers pass no new
args and get identical frames (pin with a fixture asserting old-vs-new frame
equality on a synthetic corpus). **Cost when it lands:** the next cluster run
re-assembles every bundle (~20 min BigQuery for the 0110 corpus) — expected,
announced, once. Sequenced AFTER WP-B parity so nothing new is being debugged
against a freshly cold cache.

### WP-D — episode machinery · one Opus agent · driver-owned · size L

**D1 (buildable in parallel with A/B — local tests only):**
- **Episode index provider** (new `analysis/cloud/episode_index.py`): reuses
  `diag_episode_probe.build_episodes` (gap 90, tested) + both observation
  gates via `_window_observed_cohort` + the **cap-3 deterministic salted
  sample** (`min hash(person_id, episode_start, salt)` rank ≤ 3 — the
  population index's idiom, resume-stable, never `F.rand()`). Emits
  `(person_id, index_date, episode_no)`.
- **`EpisodeDocSpec`** (in the doc-spec module's sanctioned extension point,
  ADR 0018): `doc_id = cohort:person:index` — index APPENDED (prefix parsers
  survive appends only); requires `index_date` on the events frame (WP-C's
  passthrough); carries `doc_spec_identity()` naturally through WP7's seam.
- **R7.5 monitoring:** `min_doc_length` drop-rate by episode ordinal, emitted
  as a smoke diagnostic.
- Oracle: local-Spark tests — doc ids parse, per-person doc counts ≤ cap,
  determinism across two runs, drop-rate table shape.

**D2 (after WP-C):** wire the provider + spec into the fit driver's assemble
closure via the new injection params; `index_mode: external` +
`episode_sampling: {gap_days: 90, cap: 3, salt: …}` in experiment front
matter, ALL folded into the bundle cache key (fresh keys per arm — the two
arms are two corpora). The random arm is the existing population mode under
the same new key vintage, same `split_salt` so persons land in the same
train/test fold across arms.

> **D2 MUST synthesize a BOUNDED within-corpus document index for the doc key
> (found in WP-A1, `87074af`).** WP-A1's doc key is
> `person_id * 64 + doc_index` with `doc_index ∈ [0, 64)`, and
> `synthesize_doc_key` RAISES if that bound is exceeded. WP-D1's `episode_no`
> is the *unbounded original chronological ordinal* (a chronic patient's 70th
> episode carries `episode_no = 70`), kept deliberately for R7.5's ordinal
> diagnostics — it is **not** the doc-key's low-bits value. D2 must give the
> key a dense per-person `row_number()-1` (0-based) over each person's KEPT
> documents, and carry D1's `episode_no` as a separate column for diagnostics.
> The `episode_no < 64` guard and the densify-time uniqueness assertion in
> WP-A1 are the tripwires that fail loudly if D1's raw ordinal leaks through.

### WP-E — smoke, both arms · orchestrator + Shawn (cluster) · gate

Small-iteration fits of BOTH arms on the new key vintage. Checks, all
pre-registered:
- **A/B gate** (A2's, doc-keyed) passes on the episode arm.
- **R5.11 / insight 0009:** coherence + topic-usage diagnostics — does a
  catch-all topic balloon under overlapping chronic-patient docs; does
  `n_bg: 8` absorb it. This is the smoke's GO/NO-GO question.
- **R7.5:** the ordinal drop-rate table; concentration on first episodes gets
  recorded next to the 66% gate kill (same bias, second mask).
- Corpus shape vs probe predictions: doc count ≈ ×2.66, docs/person ≤ 3.

### WP-F — episode-corpus incident census · orchestrator (cluster) · gate

The existing census tool on the episode arm's bundle. **GO/NO-GO vs R7.3:**
materially more than 0110's ~1,791 incident-scoreable nodes (probe's frontier
lower bound says ~2,583 uncapped; the census measures capped truth). If cap 3
erodes below the bar, rebuild at cap 5 (×3.98 — WP-B makes that affordable);
that decision is the census's, not a guess.

### WP-G — record runs + analyses · orchestrator + Shawn (cluster)

Both arms at record iterations, reported with the E1–E4 tooling as-is:
dual prevalent/incident metrics (D7 naming), conditional cells with P-strata,
conversion analysis off a NEW sidecar `index_horizon` variant — the
first-attestation half is index-independent and reusable; the per-person
index/observation half becomes per-(person, index) exactly as
`build_index_horizon_frame`'s docstring anticipated (small Sonnet task inside
this WP). Every interval carries the R5.12 ESS caveat verbatim. Comparisons:
episode vs random arm, shared scoreable node set, R2.2 discipline; NO
cross-experiment numeric comparisons (insight 0010). Egress floor throughout.

### WP-H — sensitivity + closeout · one Sonnet agent + orchestrator

- `--prior-obs-days` flag on `diag_episode_probe` (currently fixed at 365) and
  the relaxed-gate probe re-run (e.g. 90/0 days) — the named sensitivity
  quantifying what the incidence definition costs. No fit.
- `docs/experiments/0111-….md` opened at WP-D2 (front matter defines the
  keys), run log maintained through E–G, closed with findings + promoted
  insights.

---

## 2. Sequencing

```
WP-A1..A4 (4× Sonnet, parallel) ──┬─► WP-B (Opus) ══ parity gate ══► WP-C (Opus, THE drop)
WP-D1 (Opus, parallel with A/B) ──┘                                      │
                                                            WP-D2 (wire) ┘
                                                                 │
                                              WP-E smoke ══ R5.11 gate ══► WP-F census ══ R7.3 gate ══► WP-G record
                                                                                    (NO-GO → cap 5 rebuild → WP-F again)
WP-H sensitivity: any time after probes; closeout rides WP-G.
```

| # | step | agent | gate before it | size |
|---|---|---|---|---|
| 1 | A1–A4 seam fixes | 4× Sonnet, parallel | — | M+S+S+S |
| 2 | D1 episode provider + DocSpec | Opus, parallel with 1 | — | L |
| 3 | B distributed eval + calibration | Opus | A1, A2 | L |
| 4 | C one-blast drop | Opus | **B parity on current cache** | M |
| 5 | D2 wiring | same agent as D1 | C | S |
| 6 | E smoke ×2 arms | orchestrator + Shawn | A2 gate live | cluster |
| 7 | F census | orchestrator | E's R5.11 GO | cluster |
| 8 | G record + analyses | orchestrator + Shawn | F's R7.3 GO | cluster |
| 9 | H sensitivity + closeout | Sonnet + orchestrator | — / G | S |

## 3. Risks

| risk | why live | mitigation |
|---|---|---|
| Cache dropped with broken infra | C invalidates everything at once | hard gate: B's parity + full test suite green on the CURRENT cache first |
| Catch-all topic growth (insight 0009) | 80%-overlapping chronic docs, measured heavy tail p99≈36 | cap 3 bounds it; R5.11 smoke gate before any record spend; `n_bg` is the knob if it fires |
| Binned-isotonic degradation | bins approximate exact isotonic | parity tolerance measured & recorded on 0110; bins are cheap to raise |
| doc-key collision / person-derivation drift | `//64` assumed everywhere | single shared helper + corpus-level uniqueness assertion (A1) |
| Census NO-GO at cap 3 | capping keeps ~26% of episodes | pre-agreed fallback: cap 5 rebuild, one knob, census decides |
| First-episode bias re-entering via `min_doc_length` | seam 9 | R7.5 ordinal table in every smoke/record log |
| Driver OOM anyway at ×2.66 | "no headroom" boundary | distributed path is DEFAULT for episode arms once parity passes; driver path stays for single-doc fixtures only |

## 4. Budget (from §5f's wall table, ×2.7)

Readout pass ~47 s; 60-iter solve ~3.2 h ×2 (main + calibration) per arm;
two arms ≈ ~13 h of record-fit wall clock plus smokes — the 20-worker /
instances-pinned-to-workers lesson applies. BigQuery: one full re-assembly
per arm after WP-C (plus the one-time ~20 min re-key of the 0110-shape
corpus if it is rebuilt for parity checks).
