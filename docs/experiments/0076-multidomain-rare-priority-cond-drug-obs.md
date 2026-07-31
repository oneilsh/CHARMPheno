---
id: 76
slug: multidomain-rare-priority-cond-drug-obs
status: pending
model_class: multidomain
cohort: population_rare_priority
cohort_def: population_rare_priority
disease: rare_priority
domains: drug_era,observation
window_mode: lookback
lookback_days: 365
label_window_days: 365
person_mod: 1
prior_obs_days: 0
doc_min_length: 10
min_n: 20
n_bg: 40
tpn: 2
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
# insight 0071: the observation domain was net-negative for all six rare6
# diseases; strip the All of Us survey/SDOH vocabulary (vocabulary_id='PPI'),
# which dominates its token volume with low disease specificity. Kept here so the
# expanded fit inherits the same observation representation as 0073/0074.
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
# Full-batch baseline (pin 0.0 so it does not inherit _base.yaml's 0.1). Switch
# to 0.1 for mini-batch SVI if the larger model is too slow full-batch.
mini_batch_fraction: 0.0
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0076 — Expanded rare-disease anchor set (rare_priority), cond+drug+obs

First fit on the expanded anchor set: the six rare6 anchors plus prioritised rare
diseases from the Monarch dismech #1079 list, selected by the objective
floor(50)/ceiling(10000)/nesting criteria in [ADR
0039](../decisions/0039-expanded-anchor-selection-by-count-band-and-ontology-not-neighborhoods.md)
(41 anchors: `cohorts._RARE_PRIORITY_ANCESTORS`). The set spans several ontology
clusters — vasculitis + rare6 autoimmune, neuroimmune, neurodegenerative,
cardiac-genetic — plus isolates, so the shared placement / domain-weighting
readout can be tested for **partial pooling across related diseases** (the
follow-up to insight 0075, whose rare6-only ceiling was small and not
λ-identified).

This preserves exp 0074's optimizer, corpus, and domain configuration exactly
(condition_era + drug_era + observation, 1-year lookback, PPI stripped, spectral
scalable init, strip both, full-batch, seed 42) and changes only:

- `disease: rare_priority` (41 anchors vs rare6's 6), and
- `tpn: 2` (down from 5): the case-finding readout collapses each node's block to
  a node-level λ, so extra topics-per-node do not feed the pooling diagnostic and
  would only add starved topics on the many low-count anchors (~half have 50–200
  positives). `K = n_bg + surviving-nodes × 2` stays bounded despite the larger
  anchor set. Check `starved_topic_report`; if low-count nodes still starve, drop
  to tpn 1.

After the fit, run `make -C analysis/cloud multidomain-weighting-readout ID=76`
to get the per-disease domain-weighting ceiling on the expanded set and see
whether pooling changes the rare6 anchors' own weights vs insight 0075.
