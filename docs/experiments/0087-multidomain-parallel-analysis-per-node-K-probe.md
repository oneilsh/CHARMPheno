---
id: 87
slug: multidomain-parallel-analysis-per-node-K-probe
status: planned
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
# IDENTICAL layout to 0085 (no-roll-up SNOMED hierarchy, umbrella root at
# max_class_fraction 1.0) so the parallel-analysis pa_k reads on the SAME nodes we
# already have effrank numbers for -- in particular the 26-doc node "Chronic
# nervous system disorder" that read raw PR~90, the acceptance target below.
anchor_hierarchy: snomed
hier_concept_class: ""
hier_restrict_under: 4274025
hier_min_class_size: 2
hier_max_class_fraction: 1.0
rollup_attestation: false
init: spectral
spectral_max_vocab: 12000
spectral_method: scalable
spectral_proj_dim: 800
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

# exp 0087 — Parallel-analysis per-node K (pa_k): a sample-size-aware K estimate

Same layout as 0085. This run exercises the NEW per-node K estimator that replaces
the closed-negative effective-rank probe (insight 0081): **parallel analysis
(Horn's method)** with a **per-node null**. For each node we compare its own
co-occurrence singular-value spectrum to a null spectrum drawn from **that node's
own unigram marginal + doc-length distribution at that node's own n_docs**, and
count the directions that clear the null (× a margin). Because the null is built at
the node's actual sample size, its noise floor RISES as support shrinks — so a
small node collapses to a few directions, the sample-size awareness effective rank
lacked (a 26-doc node read PR≈90; that is nonsense as a phenotype count).

**Why this is not the old effrank probe (two corrections, found offline before this
run).** (1) It uses the **singular-value** spectrum, not the greedy pivoted-QR
residual sequence — the residual greedy picks extreme rows and lacks the clean
real-vs-null crossing Horn's method needs (offline it gave unstable 0/26/55 for one
fixed structure). (2) It **counts directions above margin × null (count-all)**, not
contiguous-from-top: position 0 is the shared marginal/background (real spreads
variance OUT of it, so real < null there), so pa_k counts the node's phenotype
directions BEYOND the shared background — exactly CHARM's per-node FOREGROUND K
(the n_bg block already models the shared direction). Offline planted-topic sweeps:
K=5→4, K=2→1, K=8→7, STABLE across n_docs 500..5000 and margin 1.5/2/3; collapses
to ~0 for a 26-doc node.

**Fast mode.** pa_k is written during spectral INIT, before any EM. Set
`CHARM_MAX_ITER=1` to skip the ~15–20 min fit and still produce `effrank.json` +
`manifest.json` for the readout. (The AP will NOT be meaningful in this mode — this
run is a K-estimator probe, not a case-finding run. Case-finding is settled:
insight 0082, condition-alone flat is the champion.)

**Cost.** The estimator adds no Spark passes — the null is driver-side arithmetic
off the one sketch the init already builds. Per node it draws `CHARM_PROBE_PA_REPS`
(5) null sketches of `min(n_docs, CHARM_PROBE_PA_CAP=2000)` docs and does that many
Gram eigensolves; the noise floor SATURATES in n_docs so the cap costs nothing in
accuracy. Expect a modest driver-time bump over a plain init on the n2-standard-16
master.

## Readout
```
make -C analysis/cloud clean-exp ID=87
export CHARM_PROBE_PARALLEL_ANALYSIS=1     # the pa_k estimator
export CHARM_PROBE_PA_MAX=300              # singular values probed per node
export CHARM_PROBE_PA_REPS=5              # null reps (percentile stability)
export CHARM_PROBE_PA_CAP=2000            # null doc cap (floor saturates)
export CHARM_MAX_ITER=1                   # effrank-only fast mode (skip EM)
export CHARM_BUNDLE_CACHE_URI=gs://dataproc-staging-wb-fresh-seed-6621/charm/bundle_cache
make -C analysis/cloud exp ID=87
make -C analysis/cloud effrank-readout ID=87
```
Big master: do NOT set CHARM_DRIVER_MEMORY (the null holds numpy on the Python
driver; a large JVM heap starves it — see insight 0081's -9 note). The corpus cache
is warm from 0085 (same corpus params), so assemble is skipped.

## Acceptance (read the `effrank-readout` table)
1. **Small nodes collapse.** The 26-doc node "Chronic nervous system disorder"
   reads `pa_k` in the LOW SINGLE DIGITS (vs its raw PR≈90). Other tiny/deep
   branches likewise read a handful, not ~90.
2. **Volume decoupled.** `corr(pa_k, log10 n_docs)` is FAR below the raw
   `corr(PR, log n_docs)` (~0.4–0.5 at df-floor) — pa_k reflects per-node phenotype
   dimensionality, not patient count.
3. **Big nodes are bounded, not saturated.** Broad classes (Disease, MS) read
   modest bounded pa_k (single-to-low-double digits), not ~d.
4. **Cap stability (spot check).** Re-run one large node's config with
   `CHARM_PROBE_PA_CAP=4000`; its pa_k should be essentially unchanged (the floor
   has saturated).

If accepted, `pa_k` (floored ≥1, capped as desired) is a genuinely sample-aware
per-node K for **phenotype profiling** layouts — the data-driven K the effrank arc
was chasing. Case-finding is unaffected (and unneeded here).
