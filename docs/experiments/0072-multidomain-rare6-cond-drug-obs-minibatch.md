---
id: 72
slug: multidomain-rare6-cond-drug-obs-minibatch
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

# exp 0072 — Multi-domain rare6 (cond + drug + observation), MINI-BATCH SVI

The **mini-batch A/B of exp 0071**. Identical corpus, domains, DAG, windowing,
vocab, and seed — the ONLY change is the optimizer schedule: `mini_batch_fraction:
0.1` (+ `learning_rate_tau0: 10.0`, `learning_rate_kappa: 0.7`) instead of exp
0071's pinned full-batch (`mini_batch_fraction: 0.0`).

Exp 0071 fits full-batch (every iteration sees the whole corpus, ~converged fast
because the corpus is small). Exp 0072 fits stochastic VI (Hoffman et al. 2013):
each iteration sees a 10% mini-batch, and a decaying Robbins-Monro step size makes
that legitimate. This is the same schedule the single-domain dag_placement
experiments (0052–0069) use via `_base.yaml` — the multidomain driver simply never
wired the knob until now.

## What to read (against exp 0071's full-batch baseline)

- `manifest.dead_nodes`: MUST still be empty. Mini-batch noise stresses the
  seed-fragile scalable init harder than full-batch — a dead node here that was
  alive in 0071 is a real signal (re-seed).
- `manifest.mini_batch_fraction` / `learning_rate_tau0` / `learning_rate_kappa`:
  now recorded (0.1 / 10.0 / 0.7) — confirm the schedule actually took.
- The fit log's per-iter lines: they should read a mini-batch fraction, NOT
  `full-batch`, and the ELBO trace will be noisier than 0071's.
- The final topic dump vs 0071's: do the rare6 node topics and the drug/
  observation corroboration survive the stochastic schedule, or does mini-batch
  noise degrade the per-domain structure? That is the point of the A/B.

## Caveats

- **The ELBO early-stop is unreliable under mini-batch** (noisy 10% estimate), so
  `max_iter: 200` is the effective budget, not a convergence target. If 0072 looks
  under-fit vs 0071, raise `max_iter` (more epochs), don't chase the ELBO.
- `omega` / `eta_per_domain` unset = faithful MixEHR baseline (SP4 sweeps omega),
  same as 0071 — so this A/B isolates the optimizer, not the weighting.
