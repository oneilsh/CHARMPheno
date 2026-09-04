---
id: 112
slug: episode-random-control
status: running
model_class: gated_pc
cohort: episode_mondo_all
cohort_def: episode_mondo_all
disease: rare_priority
# THE MATCHED RANDOM CONTROL for exp 0111 (D12). Identical to 0111 in EVERY knob —
# same native-Mondo label DAG, same 90-day gate, same 365-day outcome label, same cap 3,
# same split_salt, same fit config — EXCEPT index location: 0111 anchors the document the
# day before each first-attestation cluster; 0112 draws up to 3 UNIFORM-RANDOM indices
# over the SAME surviving persons' fully-observed calendar interval (random_index_frame,
# persons = the 0111 episode arm's survivors). index_arm: random is the ONLY line that
# differs from 0111's front matter.
#
# WHAT IT CONTROLS FOR. 0111's episode arm has, by construction, a rich gate (100%
# non-empty, mean 3.52 frontier nodes, WP-B2) and up-to-3 docs/person. A naive control
# (one random doc/person, no gate) would confound anchoring with doc-count AND gate width.
# This arm holds doc-count, gate width, cap, person set and split fold constant, so the
# episode-vs-0112 recovery delta isolates the ONE thing 0111 changes: whether the index
# sits before a presentation. Measured pre-fit, the matched random gate is 27% occupied
# (median 0) vs the episode arm's 100% — a real, interpretable baseline, not a strawman
# (a ~4x gate-density gap and ~3.3x more gated documents). The fit measures how far that
# gate-richness advantage propagates to recovery.
#
# split_salt is shared with 0111 so a person lands in the SAME train/test fold in both
# arms; comparisons are episode-vs-random on the SHARED scoreable node set (R2.2), never
# cross-experiment (insight 0010 — the doc unit differs from 0104/0109/0110).
#
# CAVEAT (WP-B2): random_index_frame's per-person uniformity is slightly imperfect for
# multi-observation-period persons (per-period draws); single-period persons (the AoU
# majority) are exact. Noted here because this arm IS the real random draw the probe only
# modeled.
#
# Operational block copied from 0111 verbatim (hence from 0110), preindex_closure OFF for
# the same reason (external episode index + E1 preindex is not yet wired). See 0111's
# front matter for the per-knob rationale; only index_arm changes.

dag_source: mondo_native
index_arm: random
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
max_iter: 100
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
spark_conf:
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  # 20 executors (whole cluster) — see 0111's spark_conf note. Raised from 12 per
  # Shawn (2026-09-04). One 11g executor per 13544MB worker; extras run smaller.
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0112 — episode random control (matched)

The D12 matched-random control for [`0111-episode-anchored-sampling.md`](0111-episode-anchored-sampling.md).
Same corpus contract, same gate, same label, same cap and split fold — index location is
the only difference. Read 0111 for the full design; this doc records the control arm's
own run log and numbers.

**Spec:** [`../superpowers/specs/2026-09-02-0111-episode-anchored-sampling.md`](../superpowers/specs/2026-09-02-0111-episode-anchored-sampling.md) (D12, R5.14).
**Plan:** [`../superpowers/plans/2026-09-03-0111-episode-anchored-sampling-plan.md`](../superpowers/plans/2026-09-03-0111-episode-anchored-sampling-plan.md).

## Rationale

The isolation control. If the episode arm (0111) beats this arm on the shared scoreable
node set, anchoring on presentations — not doc-count, gate width, cap, person set, or
split — is what supplies the advantage, because those are all held constant here. If it
does NOT beat this arm, the gate-richness that anchoring buys (100% vs 27% occupancy) did
not convert to better recovery, and the negative is as informative as the positive.

## Run log

- **2026-09-04** — opened alongside 0111 at WP-D2/D3 completion. Shares 0111's pre-fit
  measurement phase (the gate-occupancy probe, WP-B2, measured THIS arm at 26.9%
  non-empty). Launch config validated locally (`--index-arm random --gate-frontier-days
  90`). Cluster runs sequenced after 0111 (WP-E smoke onward, both arms together).

## Results

*(pending cluster runs; pooled figures + counts-of-nodes only, egress floor)*
