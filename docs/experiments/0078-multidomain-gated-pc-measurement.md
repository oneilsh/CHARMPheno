---
id: 78
slug: multidomain-gated-pc-measurement
status: pending
model_class: gated_pc
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
# MULTI-DOMAIN: conditions (domain 0) + value-aware MEASUREMENT (domain 1). The
# stronger thesis test — measurement is the ONE non-condition domain that carried
# rare-disease signal in the hybrid branch (Marfan/GBS/EDS via labs, 0078/0079).
# Measurement uses value-aware synthetic tokens (concept_id*100+state: range
# low/normal/high, coded qualitative, presence) with per-document BINARY presence
# (bursty, no era rollup). Built fresh (no cache). Companion to 0077 (cond+drug).
extra_domains: measurement
# --- corpus / DAG: IDENTICAL to 0077 / 0076 run 7 (only the domain differs) ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback     # REQUIRED for multi-domain
lookback_days: 1825
label_window_days: 365
strip_mode: both
label_mask_mode: full
n_bg: 8
tpn: 1
optimize_doc_concentration: true
# --- PC head: run 7's Path A (aggregated one-step Newton + ridge) ---
weight_y: 50.0
head_optimizer: newton
head_penalty: none
head_inner_iters: 0
head_lr: 0.3
head_newton_ridge: 0.05
grad_cavi_iters: 30
topic_trust: 0.05
weight_y_warmup_iters: 25
max_iter: 100
subsampling_rate: 0.1
tau0: 64.0
kappa: 0.51
cavi_max_iter: 100
cavi_tol: 0.001
skip_unsup_gated: false
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 20
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache   # ignored on the fresh multi-domain path
---

# 0078 — Multi-domain Gated-PC (conditions + value-aware measurement), rare6

The stronger of the two multi-domain PC runs. Where 0077 adds drug (MG signal),
this adds **value-aware measurement** — the domain the hybrid branch found carried
the rare-disease signal (Marfan/GBS/EDS via labs, insights 0078/0079). Same thesis
(per-node supervised domain shaping), same config as 0077 bar the domain.

## The test

Same three bars as 0077 (vs condition-only run 7, vs the unsupervised multi-domain
twin, vs the ~6% fixed-readout ceiling), plus the direct `per_node_domain_mass`
readout: **do Marfan's / GBS's / EDS's nodes go measurement-heavy** while
condition-driven nodes stay condition-heavy? This is the run the whole thesis was
pointed at — if supervised per-node shaping can pull the lab specialists into the
macro, it shows here.

Measurement tokenization: value-aware synthetic tokens (`measurement_tokens`,
verified by `test_bigquery_measurement.py`) with per-document binary presence.

## Run

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=78
```

Note: builds the corpus FRESH from BigQuery (measurement is a large, bursty table —
the assemble phase will be slower than the condition-only cached runs).

## Run log

_(pending first run)_
