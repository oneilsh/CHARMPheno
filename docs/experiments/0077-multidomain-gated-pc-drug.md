---
id: 77
slug: multidomain-gated-pc-drug
status: pending
model_class: gated_pc
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
# MULTI-DOMAIN: conditions (domain 0, label/gate source) + drug (domain 1). The
# gated engine carries a per-domain lambda {m:(K,V_m)}; the supervised topic
# correction is scattered per domain, so the per-node gate can specialize each
# disease node's topic block toward its predictive domain (the thesis). Drug is
# the fast/low-risk first non-condition domain (reuses the existing loader; MG
# signal, insight 0079); value-aware measurement is deferred (needs the hybrid
# bigquery.py measurement loader). Built fresh (no cache) — a one-off comparison.
extra_domains: drug
# --- corpus / DAG: IDENTICAL to 0076 run 7 so the ONLY delta is the drug domain
#     (a clean A/B: does adding drug UNDER SUPERVISION beat condition-only run 7?) ---
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
holdout_frac: 0.2
vocab_size: 5000
min_df: 20
min_patient_count: 20
window_mode: lookback     # REQUIRED for multi-domain (forward window is condition-defined)
lookback_days: 1825       # 5yr feature history (matches run 7)
label_window_days: 365
strip_mode: both
label_mask_mode: full
# --- gate topic-block layout: matches run 7 (n_bg=8, tpn=1). K emergent =
#     n_bg + nodes*tpn, shared by the gated_pc and unsup_gated arms ---
n_bg: 8
tpn: 1
optimize_doc_concentration: true
# --- PC head: run 7's KNOWN-GOOD Path A (aggregated one-step Newton + ridge) ---
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
skip_unsup_gated: false    # the unsup_gated (weightY=0) multi-domain twin = baseline (b)
with_dag_head: false
baseline_max_iter: 100
min_label_count: 20
eval_every: 20
num_partitions: 96
seed: 42
cache_uri: hdfs:///user/dataproc/charm/case_finding_cache   # ignored on the fresh multi-domain path
---

# 0077 — Multi-domain Gated-PC (conditions + drug), rare6

**Thesis.** The hybrid branch exhausted FIXED and POST-HOC-supervised domain
combination (~6% ceiling) — but that reweighted per-domain LR *scores* off a
FROZEN unsupervised fit. PC shapes the fit itself (∂loss_y/∂λ) with a per-node
gate, so each disease node's topic block can specialize toward the domain that
predicts *it* — the disease-specific balance no fixed global rule could express.
This is the first test of that mechanism on real multi-domain data.

**What this run adds over 0076 run 7.** One thing only: a second domain (drug),
fed as a per-domain vocabulary through `featuresCols`. Everything else (corpus,
DAG, gate layout, head, SVI schedule) is run 7's config, so the comparison is a
clean A/B.

## The test (does supervised multi-domain beat the bars?)

Three comparisons, all on the same rare6 labels:
1. **condition-only Gated-PC** (0076 run 7, det AP 0.148) — does adding drug
   *under supervision* beat condition-alone? The bar fixed readout couldn't clear.
2. **unsupervised multi-domain gated** (this run's `unsup_gated` arm, weightY=0) —
   does supervision beat the unsupervised multi-domain fit? Isolates the PC
   shaping from the extra information.
3. **fixed readout weighting** (hybrid branch's ~6%) — does shaping beat reweighting?

**The direct test (the headline):** the driver prints `per_node_domain_mass` — per
DAG node, the fraction of its topic block's λ mass in each domain. Does MG's node
go drug-heavy while condition-driven nodes stay condition-heavy? The per-node
specialization story matters more than the macro delta.

## Run

```bash
cd ~/repos/CHARMPheno && \
  git fetch origin claude/spectral-anchor-topic-k-200nqp && \
  git checkout claude/spectral-anchor-topic-k-200nqp && \
  git pull origin claude/spectral-anchor-topic-k-200nqp && \
  CHARM_SPARK_CONF='spark.locality.wait=0s' make -C analysis/cloud exp ID=77
```

Skip the per-iteration eval (faster) with `NO_EVAL=1`. Watch utilization with
`make -C analysis/cloud inspect`; if few executors, raise `num_partitions`.

Note: the multi-domain path builds the corpus FRESH from BigQuery each run (no
cache yet — the per-domain cache key/save is future), so the assemble phase is
slower than the cached condition-only runs.

## Run log

_(pending first run)_
