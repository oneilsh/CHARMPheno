---
id: 74
slug: multidomain-rare6-cond-drug-obs-fullbatch-attested
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
max_iter: 200
cavi_max_iter: 100
cavi_tol: 1.0e-3
# Full-batch baseline: pin mini_batch_fraction: 0.0 so this experiment does NOT
# inherit _base.yaml's 0.1 (which would silently flip it to mini-batch SVI now
# that the multidomain driver wires the knob). exp 0072 is the mini-batch A/B.
mini_batch_fraction: 0.0
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0074 — Attested artifact-contract replication of corrected exp 0071

This is not a new hyperparameter arm. It preserves corrected exp 0071's model,
corpus, full-batch optimizer configuration, and seed exactly, while using a
fresh run ID for the supervised-readout attestation contract. Execution follows
unchanged after the preliminary exp 0073 adjudication.
