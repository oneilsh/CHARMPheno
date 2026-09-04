---
id: 111
slug: episode-anchored-sampling
status: running
model_class: gated_pc
cohort: episode_mondo_all
cohort_def: episode_mondo_all
disease: rare_priority
# THE INDEX-LOCATION RUN. 0110's native-Mondo corpus with ONE thing changed: WHERE the
# document is anchored. 0110's population-random index catches one pre-onset year per
# person and starves a third of the label DAG (923 of ~1,791 scoreable nodes dropped for
# <20 incident positives, insight 0075). 0111 anchors documents on PRESENTATIONS — the
# day before each first-attestation cluster ("episode") — so a case is captured at onset
# and almost the whole DAG gets a scoreable incident cohort (probe: 2,584 of 2,714 nodes
# at >=20 gated episodes, uncapped frontier bound).
#
# TWO ARMS, ONE CORPUS IDENTITY, differing ONLY in index location (D12 matched):
#   episode  — index = episode_start - 1, cap 3 docs/person (episode_index_frame).
#   random   — cap-3 uniform-random indices over the SAME surviving persons and the
#              SAME gates (random_index_frame(persons=<episode survivors>)).
# Both share the 90-day gate, the 365-day outcome label, cap 3, and split_salt (a person
# lands in the same train/test fold in both arms). The random arm is the control that
# isolates anchoring: measured pre-fit at 27% gate occupancy vs the episode arm's 100%
# (WP-B2) — a real, interpretable baseline, not a strawman.
#
# THE KNOBS (folded into the bundle cache key; each arm is its own bundle):
#   index_mode:        external          (driver-built index; WP-C seam, WP-D2 wiring)
#   doc_spec:          episode           (doc_id = "episode:{person}:{index_date}")
#   episode_sampling:  {arm, gap_days:90, cap:3, salt:"0111", prior_obs_days:365,
#                       window_days:365}
#   gate_frontier_mode: gate90d          (D13: the GATE is [index, index+90d); the LABEL
#                                         stays the 365-day frame — WP-D3)
# All fold ONLY on the external path, so every 0104/0109/0110/population key is
# byte-identical (the tripwire suite, 60 tests, pins that).
#
# WHY 365-DAY PRIOR-OBS STAYS (WP-H, resolved). The gate drops two-thirds of every
# person's FIRST episode (66.2%), the onset-richest slice. Relaxing to 90d/0d recovers
# only ~23pp and leaves node yield essentially flat (2,584 -> 2,587 at >=20), while
# admitting prevalent contamination. 365d is the primary arm; the relaxed values are a
# recorded sensitivity, not a fit. The corpus is "incident among the year-plus-observed"
# — say so. (insight 0077.)
#
# NOT COMPARABLE to 0104/0109/0110 numerically: the document unit changes from
# one-doc-per-person to up-to-cap-3-episode-docs (insight 0010 discipline). 0111 carries
# its own random arm as the control; comparisons are episode-vs-random on the shared
# scoreable node set, never cross-experiment.
#
# Everything else (dag_source: mondo_native, window, strip, mask, vocab, head params,
# seed, spark_conf, preindex_closure) is copied from 0110 verbatim.
---

# 0111 — episode-anchored sampling

**Spec:** [`../superpowers/specs/2026-09-02-0111-episode-anchored-sampling.md`](../superpowers/specs/2026-09-02-0111-episode-anchored-sampling.md) (definitions D1–D14).
**Plan:** [`../superpowers/plans/2026-09-03-0111-episode-anchored-sampling-plan.md`](../superpowers/plans/2026-09-03-0111-episode-anchored-sampling-plan.md) (work packages, gates, sequencing).
**Pre-fit results:** [`../reports/2026-09-02-0111-episode-probe-results.md`](../reports/2026-09-02-0111-episode-probe-results.md).
**Insights:** [0075](../insights/README.md) (tracking-vs-prediction), 0076 (prospective-within-retrospective), 0077 (window-widening refutation), 0078 (first-episode kill + un-starving).

## Rationale

0110 measured that a population-random index starves the incident label space: a third
of the Mondo DAG never accrues 20 incident positives because a random calendar day rarely
sits in the run-up to a new diagnosis. 0111 asks: **does anchoring the document on the
patient's own presentations un-starve the label space without buying it with prevalent
contamination, and does that gate-richness propagate to better recovery?**

- A **positive** result: the episode arm scores materially more nodes than 0110's ~1,791
  incident-scoreable, AND its recovery on the shared node set beats the matched-random
  arm — anchoring, not doc-count or gate width, is what helps (the matched arm holds
  those constant).
- A **negative** result: the episode arm un-starves the label space (a corpus-shape win)
  but recovery does not beat the matched-random control — anchoring supplies gates but the
  gated model does not convert them, OR the 90-day gate is too sparse a supervision signal.

The two arms differ in exactly one thing (index location), so the comparison is clean.

## Run log

### 2026-09-02 — pre-fit measurement phase (COMPLETE, no fit)

Off the 0110 native-Mondo E4 sidecar; `make diag-episode-probe`. Pooled figures only
(egress floor). Full numbers in the probe-results report.

- **Multiplier** (R5.9): gated ×2.66 at cap 3 / gap 90 (uncapped ×8.6, infeasible;
  cap 5 ×3.98 held as fallback).
- **First-episode kill** (R5.10): 66.2% of first episodes lost to the 365-day prior-obs
  gate vs 14.1% of later episodes — the survivorship bias, quantified.
- **Node yield** (R7.3): 2,584 / 2,714 Mondo nodes at ≥20 gated first-attestation
  episodes (2,321 at ≥100), vs 0110's 923 starved. Uncapped frontier lower bound; the
  capped census (WP-F) is the GO/NO-GO of record.
- **Prior-obs sensitivity** (WP-H): relaxing 365→90→0d recovers first episodes only
  66%→43% (structural floor) and moves node yield 2,584→2,587 — 365d vindicated for the
  primary arm.
- **Gate occupancy** (WP-B2): episode arm 100% non-empty gate (mean 3.52 frontier nodes)
  vs matched-random 26.9% (median 0). The matched control is viable.

### 2026-09-03/04 — machinery + wiring (COMPLETE, pure code)

Driver-owned throughout; no source-hashed module edited; the 60-test tripwire suite
stayed byte-identical at every step.

- **WP-D1** (`e49e732`) — `episode_index.py`: `episode_index_frame` /
  `random_index_frame`; `EpisodeDocSpec`; R7.5 ordinal drop-rate diagnostic.
- **WP-A1..A4** (`87074af`, `428874e`) — int64 doc-key seam (`person_id*64 + doc_index`),
  person-keyed cal split, doc-key sample, detection dedup — the multi-doc plumbing.
- **WP-B** (`0ab5d50`) — distributed eval path + distributed binned calibration (R5.8);
  cluster driver-vs-distributed parity run still owed before flipping the default.
- **WP-C** (`2d93b33`) — the one-blast cache-drop: opened `index_df` / `doc_spec`
  injection seams on the assembler; four tripwire hashes re-pinned.
- **WP-D2** (`fe017e6`) — wired both arms into the Mondo fit driver through the WP-C
  seams: `--index-arm {episode,random}`, `episode_sampling` folded into the bundle key
  (external path only), the MISS-only sidecar-fed index build, and the BOUNDED dense
  `episode_no` (0..cap-1) the readout doc key needs (D1's unbounded ordinal rides a
  separate `episode_ordinal`). Sidecar identity matches the probe, so a fit HITs the
  probe-built sidecar. Verified: tripwire 60/60, D2 suite 14/14.
- **WP-D3** (`d84506a`) — D13/D14 gate/label separation: a MISS-only driver post-pass
  (`_attach_gate_frontier`) overwrites ONLY the `frontier` column with the 90-day gate
  (calling `attach_frontiers`, never editing it), leaving `label`/`labelMask` frozen at
  365 days. `keep`/`lay` reconstructed byte-identically from the bundle
  (`set(cid2int) == after_dag.nodes()`, proven). Loud join integrity; `gate_frontier_mode`
  in the key + manifest. Activate with `--gate-frontier-days 90`. Verified: tripwire
  60/60, D2 14/14, D3 14/14 incl. label byte-identity.

### Next (cluster, with Shawn)

1. **First post-WP-C bundle rebuild** — the WP-C drop moved the bundle/corpus/covariate
   keys; no rebuild has been paid yet. First `make exp` re-assembles each arm (~20 min
   BigQuery/arm). HDFS caches are cluster-ephemeral; if the cluster was bounced, rebuild
   the sidecar first (`make build-conversion-sidecar ID=110`, survives WP-C).
2. **WP-B parity** — driver vs distributed on the rebuilt 0110 bundle (`compare_per_node`)
   before flipping the episode arms to `--eval-path distributed`.
3. **WP-E smoke** — both arms, small iterations: A/B gate (episode arm), R5.11 / insight
   0009 coherence + topic-usage, R7.5 ordinal drop-rate, corpus shape vs probe (×2.66,
   ≤3 docs/person).
4. **WP-F census** — episode-corpus incident census; GO/NO-GO vs 0110's ~1,791 (cap-5
   fallback if cap-3 erodes below the bar).
5. **WP-G record + E1–E4 analyses**; **WP-H closeout**.

## Results

*(pending cluster runs; pooled figures + counts-of-nodes only, egress floor)*
