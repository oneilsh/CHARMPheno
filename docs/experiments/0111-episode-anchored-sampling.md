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
# seed, spark_conf) is copied from 0110 verbatim EXCEPT preindex_closure, forced OFF:
# the E1 preindex-closure post-pass over an EXTERNAL episode index is not yet wired
# (WP-D2 found the combination raises loudly, by design), and WP-F's incident census
# consumes R_d — so a small follow-up WP must wire preindex-over-episodes before WP-F.
# Until then 0111 runs without the R_d column. person_mod:1 / max_iter:100 is the
# RECORD config; the WP-E smoke launches the SAME corpus with a reduced iteration cap.
# The episode corpus is ~x2.66 the doc count — eval_path=distributed (WP-B) avoids the
# driver-collect wall; re-size instances/partitions off the real corpus, not 0110's.
#
# WHY the assembler's own prior_obs_days is inert here: index_mode="external" takes the
# driver-built index_df VERBATIM (multi_domain.py external path), so the episode gate is
# applied by the PROVIDER (episode_prior_obs_days:365 / episode_window_days:365), not by
# the assembler. prior_obs_days:0 is copied from 0110 for the feature window and does not
# re-gate the external index.

dag_source: mondo_native
index_arm: episode
gate_frontier_days: 90
episode_gap_days: 90
episode_cap: 3
episode_salt: "0111"
episode_prior_obs_days: 365
episode_window_days: 365
preindex_closure: false
readout_mode: distributed
eval_path: distributed
readout_theta_topm: 256
weight_y: 0.0
weight_y_warmup_iters: 0
skip_unsup_gated: true
min_positives: 100
mondo_version: 2026-06-02
mondo_cache_dir: data/mondo
extra_domains: measurement,drug
label_mask_mode: closure
localize_head: true
head_support: path_cousins_kids
head_intercept: true
head_standardize: true
doc_concentration: 0.5
head_lr: 1.0
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 0
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback
lookback_days: 1825
label_window_days: 365
strip_mode: both
n_bg: 8
tpn: 1
optimize_doc_concentration: true
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
max_iter: 15   # TEMP: WP-E smoke value; REVERT to 100 for the WP-G record run (task #9)
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 0
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache
# spark_conf VALUES copied verbatim from 0110; the hard-won rationale (executor-heap
# OOM / DA idle-kill / exclude starvation / driver-disk GC) is in 0110's run log +
# front-matter comments. Re-tune only via GPR_SPARK_CONF on a beefier cluster.
spark_conf:
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  # 20 executors (8 primary + 12 spot) — the episode corpus is ~x2.66 the doc count,
  # so use the whole cluster. Each is 8g+3g=11g, one per 13544MB worker. YARN grants
  # what exists and holds the rest pending, so a smaller cluster still runs (fewer
  # executors, no error). Raised from 0110's 12 per Shawn (2026-09-04).
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
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

### 2026-09-04 — WP-B parity (cluster) — PASS

Driver-vs-distributed eval-path parity on the 0110 corpus (`make gated-pc-readout ID=110`,
`--readout-max-iter 20` to make the head-fit cheap; parity is about eval agreement on
identical heads, not head convergence). First attempt was confounded — one run
warm-resumed the killed 200-iter run's solver checkpoint (16/2537 converged) while the
other cold-started (0 converged), so they scored DIFFERENT heads (~0.003 AUC gap, all
head-difference). Re-run cold (checkpoint consumed) reproduced the distributed run's fit
line EXACTLY (`56447 node-passes, max|grad|=1.44e3, 0 converged`) and then driver-eval and
distributed-eval printed BYTE-IDENTICAL readout numbers: prevalent/full 0.7405/0.4150,
prevalent/shared 0.7470/0.4439, incident 0.6780/0.2459, detection 0.6522/0.9739. Combined
with the committed local <1e-9 oracle (`test_distributed_eval_parity.py`, WP-B), the
distributed eval path is confirmed at real scale (C=2714). **Episode arms are cleared for
`eval_path: distributed`** — which they need anyway (the ~×2.66 doc count makes the
driver-collect eval the memory wall). run_experiment now forwards `eval_path` from front
matter (mirrors `readout_mode`); both arm docs set it. Egress: all figures pooled.

### Next (cluster, with Shawn)

1. **First post-WP-C bundle rebuild** — the WP-C drop moved the bundle/corpus/covariate
   keys; no rebuild has been paid yet. First `make exp` re-assembles each arm (~20 min
   BigQuery/arm). HDFS caches are cluster-ephemeral; if the cluster was bounced, rebuild
   the sidecar first (`make build-conversion-sidecar ID=110`, survives WP-C).
2. ~~**WP-B parity**~~ — DONE (2026-09-04, above): distributed == driver at C=2714. Episode
   arms use `eval_path: distributed`.
3. **WP-E smoke** — both arms, small iterations: A/B gate (episode arm), R5.11 / insight
   0009 coherence + topic-usage, R7.5 ordinal drop-rate, corpus shape vs probe (×2.66,
   ≤3 docs/person).
4. **WP-F census** — episode-corpus incident census; GO/NO-GO vs 0110's ~1,791 (cap-5
   fallback if cap-3 erodes below the bar).
5. **WP-G record + E1–E4 analyses**; **WP-H closeout**.

## Results

*(pending cluster runs; pooled figures + counts-of-nodes only, egress floor)*
