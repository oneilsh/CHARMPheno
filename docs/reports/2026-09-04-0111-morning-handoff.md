# 0111 handoff — resume point for the next session — 2026-09-04

Written before a context compaction. Branch **`claude/gated-conditional-voi`**,
tree clean, everything below is committed and pushed. The authoritative design
is the spec + plan (linked below); this doc is the "where we are / what's next"
map. Read `AGENTS.md` first (loaded via `CLAUDE.md`) for the operational
invariants — cache-key landmine, egress floor, cluster-command preamble.

## What exp 0111 is

Episode-anchored (first-attestation-cluster) index sampling to fix the measured
all-Mondo **incident starvation** (0110's random index dropped 923 of ~1,791
scoreable nodes for <20 incident positives). A **two-arm** experiment on one
corpus identity, differing ONLY in index location:
- **episode arm** — index just before each first-attestation cluster, cap 3
  docs/person.
- **random arm (matched)** — cap-3 uniform-random indices, same gates.

Both arms share: the **90-day gate** `[index, index+90d)` (D13), the **365-day
outcome label**, and cap 3. Spec: `docs/superpowers/specs/2026-09-02-0111-episode-anchored-sampling.md`
(definitions D1–D14). Plan: `docs/superpowers/plans/2026-09-03-0111-episode-anchored-sampling-plan.md`.

## Settled decisions (do not relitigate)

- **gap = 90d, cap = 3** (cap 5 = census fallback). `index = episode_start − 1`.
- **365-day prior-obs gate stays** for the primary arm (WP-H sensitivity
  measured: relaxing recovers first episodes only 66%→43%, a structural floor;
  corpus +6.8%, node yield flat — 365d costs little). Corpus is "incident among
  the year-plus-observed"; say so.
- **D13 gate/label separation**: the estimator GATE = first-attestation nodes in
  the 90d window; the LABEL = full 365d frame (unchanged prevalent/incident
  readout). Today they are the same object (`attach_labels` reads the `frontier`
  column, `multi_domain.py:302-303`) — D13 splits them.
- **D14 driver-side**: the separation is a post-load `frontier`-column swap
  (call `attach_frontiers`, don't edit it) — NO source-hashed edit, NO second
  cache blast. Verified: estimator reads `frontierCol`/`labelCol`/`labelMaskCol`
  as three distinct inputs (`gated_pc_cloud.py:2685-2698`); `label` is frozen at
  assembly; `index_date` is recoverable from the `EpisodeDocSpec` doc_id.
- **Matched random arm** (D12, Shawn's call): isolates anchoring. Consequence,
  measured not assumed: matched gates make the random arm 27% occupied (vs
  episode 100%) — a real, interpretable baseline, not a strawman.
- Never compared to 0104/0109/0110 numerically (insight 0010, doc-unit change).
  0111 carries its own random arm as control.

## Pre-fit measurement phase — COMPLETE

All off the 0110 E4 sidecar, no fit. Results in
`docs/reports/2026-09-02-0111-episode-probe-results.md`:
- **R5.9 multiplier**: gated ×2.66 at cap 3 (gap 90).
- **R5.10 kill / WP-H sensitivity**: 66.2% first-episode kill; 365d vindicated.
- **R7.3 node yield**: 2,584/2,714 nodes ≥20 gated episodes (was 923 starved).
- **WP-B2 gate occupancy**: episode 100% non-empty (mean 3.52 frontier nodes)
  vs matched-random 26.9% (median 0). Control viable.

## Commit ledger (all pushed)

| WP | what | commit |
|---|---|---|
| D1 | episode index provider, `EpisodeDocSpec`, R7.5 diagnostic | `e49e732` |
| A1 | int64 doc-key seam (multi-doc); caught the bounded-index cross-seam | `87074af` |
| A2/A3/A4 | sample_frac doc-keying, person-keyed cal split, detection dedup | `428874e` |
| B | distributed eval path + distributed binned calibration | `0ab5d50` |
| B (rider) | closed the `gated_pc_readout` co-fit head detection residue | `0ab5d50` |
| H | `--prior-obs-days` sensitivity flag + measured result | `511aec8`,`0fafeef` |
| C | one-blast cache-drop — `index_df`/`doc_spec` seams opened | `2d93b33` |
| design | D13/D14 gate separation + D12 matched arms into spec/plan | `0a1aed1` |
| B2 | gate-occupancy probe + measured result | `2d60fb8`,`2a4ff3f` |
| docs | AGENTS.md overview-of-record; cache-key blast-radius correction | `5680666`,`b8a10e4` |

Verification discipline used throughout: every agent's work was independently
re-run (tripwires byte-identical, full suites green) before commit — do not
trust an agent's self-report; run the suite yourself. One scare resolved: WP-C
was briefly "missing" — it was in a stash, recovered and committed.

## NEXT: the build (pure code until the fits)

**WP-D2 — wire the arms into the fit driver.** Through WP-C's now-open seams:
- Episode arm: `episode_index_frame` (in `episode_index.py`, gap 90 cap 3) →
  `index_mode="external"` + `EpisodeDocSpec` on `assemble_multidomain_case_finding_corpus`.
- Matched random arm: `random_index_frame` (in `episode_index.py`) →
  `index_mode="external"` + an index-encoding doc_id, same cap/gates.
- **Bounded doc-index synthesis (WP-A1 requires it):** the doc key packs
  `person_id*64 + doc_index` with `doc_index ∈ [0,64)`. D1's `episode_no` is the
  UNBOUNDED chronological ordinal (kept for R7.5) — do NOT feed it to the key.
  Synthesize a dense per-person `row_number()-1` over kept docs; carry
  `episode_no` as a separate diagnostic column. The `episode_no<64` guard +
  densify uniqueness assertion (WP-A1) fail loudly if the raw ordinal leaks.
- Front matter: `index_mode: external` + `episode_sampling: {gap_days:90, cap:3,
  salt:…}`, folded into the bundle cache key (fresh keys per arm; same
  `split_salt` so persons share the train/test fold across arms).

**WP-D3 — driver-side gate-frontier swap (D13/D14).** After bundle load, before
`.fit()`: recover `index_date` from the doc_id; compute the 90d-gate
first-attestation nodes; roll up by CALLING `attach_frontiers`; overwrite ONLY
the `frontier` column (join by doc key); leave `label`/`labelMask`. Refuse
missing/dup/mismatched joins loudly; assert one frontier per doc; record
`gate_frontier_mode` in the manifest. Test list in plan §WP-D3 (esp.
**outcome-label byte-identity** — `label`/`labelMask` unchanged vs current 365d
assembly). Applied identically to BOTH arms.

Both are 4.8-agent-dispatchable, driver-owned (no source-hashed edits). Verify
independently before commit.

**THEN (cluster, with Shawn):** first post-WP-C bundle rebuild (~20 min BQ per
arm) → WP-E smoke on both arms (A/B gate; R5.11 insight-0009 coherence/topic-
usage; R7.5 ordinal drop-rate) → WP-F episode-corpus incident census (GO/NO-GO
vs ~1,800; cap-5 fallback) → WP-G record runs + E1–E4 analyses (dual metrics,
conditional cells, conversion) → WP-H closeout.

## Cluster / cache state

- Master resized to handle the driver-collect wall (the n2-standard-2 OOM is
  fixed). The E4 **sidecar is rebuilt on HDFS** and SURVIVES WP-C (keyed
  independently — verified). All probes run off it.
- WP-C moved the bundle/corpus/covariate cache keys; **no bundle rebuild has
  been paid yet** — the first `make exp`/assemble after WP-C re-assembles (~20
  min BQ). HDFS caches are cluster-ephemeral; if the cluster was bounced,
  rebuild the sidecar first (`make build-conversion-sidecar ID=110`).
- Distributed eval/calibration (WP-B) default is `--eval-path driver`; the
  cluster **parity run** (driver vs distributed on the rebuilt 0110 bundle,
  `compare_per_node`) is still owed before flipping the default for episode arms.

## Open threads / watch-fors

- **WP-B2 random-index caveat**: per-person uniformity is slightly imperfect for
  multi-observation-period persons (per-period draws); single-period (the AoU
  majority) is exact. Fine for the probe; if the real random arm reuses this,
  note it.
- **`preindex_closure.py` docstring lag**: still lists the old column set for
  `feature_window_condition_events` (now carries an inert `index_date`). It's a
  source-hashed do-not-edit module, so the fix rides a FUTURE deliberate drop,
  not on its own.
- **Subagent model**: `model:"opus"` maps to Opus 5 (was 529-saturated); OMIT
  the model param to inherit the orchestrator's model (4.8 has had capacity).
