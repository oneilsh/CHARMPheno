---
id: 71
slug: multidomain-rare6-cond-drug-obs
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
init: spectral
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
spectral_topo_order: forward
strip_mode: both
max_iter: 200
cavi_max_iter: 100
cavi_tol: 1.0e-3
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0071 — Multi-domain (condition + drug + observation) gated fit, rare6, lookback

First **three-domain** gated case-finding fit (SP3c): conditions (domain 0,
`condition_era`) + drugs (domain 1, `drug_era`) + observations (domain 2,
`observation`), over three independent vocabularies, sharing one DAG-gated theta
with a **condition-only gate** (gate ⟂ domain). The rare6 six-disease forest is
the label DAG; drugs and observations are features, never labels.

**Lookback windowing** (parity with the single-domain rare6 exps 0061–0065): one
shared, condition-derived index date (`case_finding_index_table`) splits every
domain into a pre-index feature window (`lookback_days` back) while the frontier
labels come from the forward `label_window_days` condition window — leakage-free
by construction. `strip_mode`/`prior_obs_days` are moot in lookback (disjoint by
construction; the ≥1yr gate is intrinsic to the index table).

Runs via `make exp ID=71`. K is emergent (`n_bg` + surviving-DAG-nodes × `tpn`);
resume unsupported (v1). No NPMI eval (npz + manifest artifact).

## What to read (manifest.json + fit log)

- `dead_nodes`: MUST be empty (insight 0070 init-fragility signature; re-seed if not).
- `corpus_stats`: three per-domain vocab sizes (cond / drug / observation) in
  plausible bands; observation is the most heterogeneous domain (social history,
  surveys, findings) — expect a mixed-granularity vocabulary (drug_era finding
  precedent), tamed by `obs_min_df`/`obs_min_patient_count`.
- The final topic dump: do the rare6 disease nodes carry coherent conditions +
  corroborating drugs + observations? Does the observation domain add signal or
  erode to prior (the SP4 ω question)?
- `ledger`: the multi-domain assembly provenance.

## Knobs

- `person_mod: 1` = full population (rare diseases need the counts).
- `omega`/`eta_per_domain` unset = faithful MixEHR baseline (SP4 sweeps ω).
