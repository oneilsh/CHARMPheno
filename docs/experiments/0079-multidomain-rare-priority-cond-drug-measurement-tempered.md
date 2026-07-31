---
id: 79
slug: multidomain-rare-priority-cond-drug-measurement-tempered
status: pending
model_class: multidomain
cohort: population_rare_priority
cohort_def: population_rare_priority
disease: rare_priority
domains: drug_era,measurement
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
meas_vocab_size: 2500
meas_min_df: 20
meas_min_patient_count: 20
# Per-modality tempering (insight 0079): measurement is high-coverage and pulled
# the shared theta away from condition structure in exp 0078 (condition-alone
# macro AP fell 0.024 -> 0.020). omega down-weights measurement's contribution to
# theta inference to protect the condition domain while still learning a
# measurement lambda for the readout. Order = [condition, drug, measurement].
omega: 1.0,1.0,0.5
init: spectral
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
spectral_topo_order: forward
strip_mode: both
max_iter: 200
cavi_max_iter: 100
cavi_tol: 1.0e-3
mini_batch_fraction: 0.0
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0079 — cond + drug + measurement, measurement modality-**tempered** (ω=0.5)

The decisive follow-up to insight 0079. Exp 0078 showed value-aware measurement
rescues labs-dependent diseases (GBS, Marfan, EDS via max-scaled, …) but **dented
the condition domain** in the shared-θ fit (condition-alone macro AP 0.024→0.020)
because measurement is high-coverage and pulls θ, and consequently **no fixed
combination beat condition-alone at macro**.

This holds exp 0078 fixed and changes only **`omega: 1.0,1.0,0.5`** — per-modality
tempering that halves measurement's weight in the θ likelihood (condition and drug
stay at 1.0). The hypothesis: measurement contributes less to θ inference, so the
condition domain's topics stop being degraded, while measurement's λ is still
learned well enough to keep its readout signal.

**Decision this settles.** Run the fast readout and compare condition-alone and
the best combine against 0078 / 0076:

```
make -C analysis/cloud multidomain-weighting-readout ID=79 WEIGHTING_FIXED=1
```

- If **condition-alone recovers toward 0.024** *and* measurement keeps its
  specialist rescues (GBS/Marfan/EDS via max:scaled), then a combine (or a light
  per-disease sum-vs-max selection) can finally clear condition-alone — keep
  pushing.
- If condition-alone does **not** recover (or measurement's rescues collapse
  under tempering), then aggregate case-finding is information-limited (insight
  0062) and measurement should ship as a **specialist channel** for the diseases
  where it dominates, not as an average-improving domain — closing the
  "improve the aggregate" line of the measurement arc.

If ω=0.5 partly helps, a quick ω sweep (0.3 / 0.7) is the natural refinement;
the alternative targeted lever is pruning the near-universal `[measured]`
presence tokens (the measurement analogue of the PPI strip, insight 0071/0077).
