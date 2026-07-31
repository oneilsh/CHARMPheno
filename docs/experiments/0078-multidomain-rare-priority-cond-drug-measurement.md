---
id: 78
slug: multidomain-rare-priority-cond-drug-measurement
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
# Measurement domain: value-aware synthetic tokens (measurement_concept x
# value-state, insight 0077), tokenized binary per-document presence (no OMOP era
# rollup; ~924 rows/person). Vocab is labs x states, so slightly larger than a
# bare lab list; cap between drug and the old observation size.
meas_vocab_size: 2500
meas_min_df: 20
meas_min_patient_count: 20
init: spectral
spectral_max_vocab: 12000
spectral_method: scalable
anchor_scope: frontier
spectral_topo_order: forward
strip_mode: both
max_iter: 200
cavi_max_iter: 100
cavi_tol: 1.0e-3
# Full-batch baseline (pin 0.0 so it does not inherit _base.yaml's 0.1).
mini_batch_fraction: 0.0
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0078 — Expanded rare-disease anchors (rare_priority), cond + drug + **measurement**

The first value-aware **measurement** fit (measurement arc, Step 2; insight 0077,
design spec
[`2026-07-31-measurement-arc-value-aware-representation-design.md`](../superpowers/specs/2026-07-31-measurement-arc-value-aware-representation-design.md)).
Holds exp 0076 fixed except the domain set: **drops observation** (user-approved;
it was a raw-count/PPI-junk drag, insight 0071/0076) and **adds measurement** in
its place, keeping drug.

**What changed vs 0076**

- `domains: drug_era,measurement` (was `drug_era,observation`).
- Measurement uses the new value-aware representation (insight 0077):
  - each row emits a **synthetic token** `measurement_concept_id × 100 + state`
    where the state is chosen by the cascade — **range-derived low/normal/high**
    where a reference range exists (the diagnostically rich chem/heme/LFT panels),
    else an **allowlisted coded value** (serologies/urinalysis: pos/neg/…), else
    **presence**. (`charmpheno.omop.measurement_tokens`.)
  - tokenized **binary per document** (`DomainVocabSpec.binary=True`): each
    `(lab, state)` counts once per patient-window, since measurement has no era
    rollup and is extremely bursty. This makes it behave like condition/drug
    (near-binary via eras) rather than like the raw-count observation domain.

Everything else is 0076's config: condition_era + drug_era, 1-year lookback,
spectral scalable init, strip both, full-batch, K = 40 bg + surviving-nodes × 2,
seed 42.

**Positive control.** Long QT syndrome is the one anchor where the observation
domain earned supervised weight in 0076, via its **QT-interval measurement**. If
the value-aware representation works, the Long QT topic should surface a
`QT interval ... [high]` token, and the labs-dependent diseases that were
near-useless from cond+drug+obs (vasculitides via inflammatory markers /
autoantibodies, cytopenias, renal/hepatic involvement) should gain placement AP.

**Readout.** After the fit:

1. Eyeball the topic dump (`top_n_tokens: 8`) and `make -C analysis/cloud
   summarize-exp ID=78` for the measurement tokens surfacing per node — first
   check the representation carries signal (does Long QT recover QT-interval;
   do renal anchors recover creatinine/BUN [high]?).
2. `make -C analysis/cloud multidomain-weighting-readout ID=78 WEIGHTING_JOBS=4`
   for the per-disease case-finding AP. The comparison is the **fixed inclusive
   combination** arm here (cond+drug+**measurement**) vs 0076's fixed cond+drug
   (macro median AP 0.032) — judged by PR/AP and precision-at-recall, not ROC.
   The supervised-weight arm is now a secondary diagnostic: does measurement earn
   weight on the labs-dependent anchors it should?
