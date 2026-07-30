---
id: 73
slug: multidomain-rare6-cond-drug-obs-minibatch-attested
status: pending
model_class: multidomain
cohort: population_rare6
cohort_def: population_rare6
disease: rare6
domains: drug_era,observation
window_mode: lookback
lookback_days: 365
label_window_days: 365
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
n_bg: 40
tpn: 5
source_table_cond: condition_era
cond_vocab_size: 5000
cond_min_df: 20
cond_min_patient_count: 20
drug_vocab_size: 2000
drug_min_df: 20
drug_min_patient_count: 20
obs_vocab_size: 1500
obs_min_df: 20
obs_min_patient_count: 20
# insight 0071: the observation domain was net-negative for all six rare
# diseases (drop:observation >= all everywhere). Strip the All of Us survey/SDOH
# vocabulary (vocabulary_id='PPI'), which dominates its token volume with low
# disease specificity. Re-fit required (this changes the observation vocabulary).
obs_exclude_vocab: PPI
init: spectral
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
spectral_topo_order: forward
strip_mode: both
# Mini-batch SVI (Hoffman et al. 2013) — the ONLY difference from exp 0071.
# 10% of the corpus per iteration makes the decaying Robbins-Monro step
# legitimate; tau0 10 tames the noisy early mini-batches (vs the aggressive
# default 1); kappa 0.7 is the standard text decay. Same values the
# dag_placement experiments use (_base.yaml).
mini_batch_fraction: 0.1
learning_rate_tau0: 10.0
learning_rate_kappa: 0.7
# max_iter is the real budget here: with mini-batching the per-iter ELBO is a
# noisy 10% estimate, so the relative-ELBO early stop is unreliable. 200 iters
# x 0.1 = 20 epochs (matches dag_placement's 200-iter mini-batch schedule).
max_iter: 200
cavi_max_iter: 100
cavi_tol: 1.0e-3
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0073 — Attested artifact-contract replication of exp 0072

This is not a new hyperparameter arm. It preserves exp 0072's model, corpus,
optimizer, and seed exactly, but uses a fresh run ID so current code can prove
one row per person in memory and persist the privacy-safe count attestation
required by the supervised nested-CV weighting readout.
