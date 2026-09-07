---
id: 117
slug: mondo-cardiovascular-tpn5-profile-eta-s3
status: done
model_class: gated_pc
cohort: population_mondo_all
cohort_def: population_mondo_all
disease: rare_priority
# THE STRENGTH RUNG (insight 0084, plan 2026-09-06 failure read #2). 0116 was a
# clean, wiring-validated NULL at profile_eta_strength 1.0: paired dAUC flat on
# both sides of the credit line, starvation unchanged — but the boost provably
# reached the starved floor (mass 440.1 vs 440.6 theory; starved deep CM topics'
# top-words became their HPO phenotype). Diagnosis: ~3.3 pseudo-mass per boosted
# topic cannot buy theta against fed topics in per-doc CAVI. This run turns the
# ONE knob the failure read pre-registered: strength 1.0 -> 3.0 (the boosted
# topic's condition-domain prior mass goes from 2x to 4x flat). Everything else
# is 0116 = 0113 verbatim; judged by `readout-ab ID=117 BASE=113` under the SAME
# acceptance as 0116. If still flat: question whether ANY eta dose can buy theta
# before distorting fed topics — i.e. whether the lever is eta at all (0084).
dag_source: mondo_native
mondo_branch: MONDO:0004995
tpn: 5
max_iter: 50
diag_only: true
preindex_closure: false
readout_mode: distributed
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
optimize_doc_concentration: true
head_optimizer: newton
head_newton_ridge: 0.05
head_l2: 0.01
grad_cavi_iters: 15
topic_trust: 0.05
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
# The one knob vs 0116 (which was itself one knob vs 0113): strength 1.0 -> 3.0.
# Same ~-anchored TSV as 0116 (the probe's --emit-eta output; regenerate if the
# cluster restarted since).
profile_eta: ~/repos/CHARMPheno/data/ontology/profile_eta_MONDO_0004995.tsv
profile_eta_strength: 3.0
profile_eta_topics: 1
profile_eta_min_coverage: 0.0
spark_conf:
  # Copied from 0110 (the only geometry that survived a whole-Mondo solve). At this
  # branch's much smaller K (~1,400 vs 3,827) it is over-provisioned and safe; a wider
  # shape is fine if the cluster is bigger. See 0110 for the kill-swarm forensics.
  spark.executor.cores: 2
  spark.executor.memory: 8g
  spark.executor.memoryOverhead: 3g
  spark.dynamicAllocation.enabled: "false"
  # 20 executors (whole cluster) per Shawn (2026-09-04). A tpn=5 branch fit is small;
  # YARN grants what it needs and holds the rest pending.
  spark.executor.instances: 20
  spark.excludeOnFailure.timeout: 10m
  spark.cleaner.periodicGC.interval: "5min"
---

# 0117 — profile-eta at strength 3.0 (the dose rung)

The pre-registered response to 0116's null
([0116](0116-mondo-cardiovascular-tpn5-profile-eta.md), insight
[0084](../insights/0084-profile-eta-strength-1-tilts-the-floor-legibly-but-cannot-buy-theta.md)):
same fit, same prior, same credited/uncredited split — `profile_eta_strength`
1.0 → **3.0** (≈10 pseudo-mass per boosted topic; condition-domain prior mass
4× flat). Acceptance is 0116's verbatim, judged by
`make -C analysis/cloud readout-ab ID=117 BASE=113`:

1. **Primary:** macro ≥ 0.7813 − noise; WIN = credited paired dAUC up,
   uncredited ~0.
2. **Secondary:** credited-deep starvation vs 72%; cardiomyopathy digest.
3. **Failure reads:** credited DOWN → the dose distorts before it helps (stop
   raising strength; question the eta lever per 0084); still flat → same
   question; uncredited moved → wiring bug (would be new — 0116's control was
   clean), stop.

## Run

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud exp ID=117
make -C analysis/cloud gated-pc-readout ID=117 GPR_ARGS="--readout-mode distributed"
make -C analysis/cloud readout-ab ID=117 BASE=113
make -C analysis/cloud inspect-topics ID=117 RESOLVE_NAMES=1 \
    INSPECT_ARGS="--digest --grep 'cardiomyopath'"
```

(The fit log should show `η_boost[topics=132 nnz=28883 mass≈1320]` — 3× 0116's
440.)

## Run log

**2026-09-07 — fit + readout + reads.** Fit and distributed readout landed
(readout persisted; 71% starved, cliff at d4 — geometry unchanged from
0113/0116). `readout-ab ID=117 BASE=113`, `--profile-align` (its first record
use), and the starred digest all read; numbers below.

## Results

**Pre-registered outcome: the dose distorts before it helps — the strength
ladder STOPS. Alignment answers monotonically to dose; discriminability does
not, and the global numbers start paying. See insight
[0085](../insights/0085-eta-dose-buys-alignment-monotonically-but-no-auc-the-prior-is-an-interpretability-lever.md).**

- **Alignment (--profile-align vs the 0113 baseline):** starved credited
  topics (n=43) median E[β] profile-mass 0.303 / top-15 overlap **1.00**
  (baseline 0.039 / 0.13 — a starved topic's top-15 IS its profile at S=3);
  fed credited (n=89) mass 0.302 / overlap 0.60 (baseline 0.014 / 0.13). The
  fed shift is the mechanism tell: ~10 pseudo-mass cannot move a Σλ≈10⁴
  topic's β directly, so counts followed the tilt through θ reallocation —
  the boost DOES buy θ at this dose.
- **Discriminability (paired vs 0113, 193 shared nodes):** credited median
  dAUC **−0.0021** (mean +0.0005, 42/49) — still no gain; uncredited control
  −0.0046 (mean −0.0071, 43/59); macro 0.7778/0.5197 (−0.0035/−0.0058);
  detection 0.6221/0.7071 (−0.0126/−0.0085, giving back 0116's small gain);
  by-depth deltas d6 −0.009, d7 −0.005 (0116's noise-level positive hint
  flips sign).
- **On the "uncredited moved → wiring bug" failure read:** ruled OUT as
  wiring — 0116's control was clean on byte-identical wiring, only S changed,
  and the drift is dose-scaled and directionally consistent. It is real
  SPILLOVER: θ drained from honest topics into aligned ones through the
  shared documents each gate admits.
- Least-aligned at S=3 = the big common-acquired nodes (hypertensive
  disorder, coronary artery disorder, vascular occlusion — ov 0.00): real
  counts dominate and the HPOA rare-syndromic profile is not the corpus
  presentation. Most-aligned = heart disorder / congenital CV anomaly /
  endocardium disorder (ov 1.00).

**Verdict.** Dose-response answered in two directions at once: eta strength is
an INTERPRETABILITY lever (monotone, dramatic) and not an AUC lever (flat at
best, mildly costly at 3×). Operating point going forward: **strength 1.0**
(0116's free legibility at zero readout cost). Do not run S≈30 ("one canonical
patient") as an AUC experiment; the doc-units reparameterization remains
worthwhile as an interpretability dose control only. The θ-allocation levers
still open for case-finding: label-side pull (PC revival, 0103 bar co-fit
≳ 0.758 — now with an aligned target the 0096–0103 era lacked) or moving the
frontier (index, cascade).
