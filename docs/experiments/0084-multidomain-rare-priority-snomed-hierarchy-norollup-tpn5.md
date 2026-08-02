---
id: 84
slug: multidomain-rare-priority-snomed-hierarchy-norollup-tpn5
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
tpn: 5
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
# SNOMED class hierarchy above anchors (root -> class -> anchor -> descendants),
# restrict-under-Disease (4274025), max_class_fraction 0.6. Same layout as 0082.
anchor_hierarchy: snomed
hier_concept_class: ""
hier_restrict_under: 4274025
hier_min_class_size: 2
hier_max_class_fraction: 0.6
# THE ONE FLIP vs 0082: no roll-up. Class nodes pool ONLY the patients routed
# from the anchors beneath them (a rare-flavored class mean), NOT the full
# descendant population. The random-init 0081 fit card already showed this makes
# class topics rare-flavored ("Disorder of connective tissue" = Sjogren's /
# systemic sclerosis / ANA / complement / DMARDs, not fracture/OA). This run adds
# SPECTRAL init for a well-conditioned fit so the AP is a clean read.
rollup_attestation: false
init: spectral
spectral_max_vocab: 12000
spectral_method: scalable
# Random-projection dim for the scalable init. The init COLLECTS a (V x d) sketch
# per DAG node to the PYTHON driver (~#nodes * V_concat * d * 4 bytes). At the auto
# default d~1000 that is 155 * 8774 * 1000 * 4 ~= 5.4GB -- which OOM-kills the init
# (exit -9) on this 7.8GB / 2-core master. d=400 -> ~2.2GB, fits (free master RAM
# first; drop to 300 if still tight). d only needs to JL-preserve pairwise word
# distances (~400 >> log V), so anchor quality is essentially unchanged.
spectral_proj_dim: 400
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

# exp 0084 — Capacity test: no-roll-up hierarchy at tpn=5 (+ effrank probe)

Flips ONLY `tpn` (2 -> 5) off the clean no-roll-up 0083 config. 0083 recovered
case-finding to flat (cond macro 0.021 vs flat 0.020; insight 0082) but was a
*wash* — the coherent classes helped some anchors (EDS 0.060->0.083, Scleroderma)
and diluted others (SLE, MS), netting flat. The hypothesis: each class is squeezed
into tpn=2 topics, so it can only hold ~2 sub-phenotypes; anchors that don't match
those 2 get averaged toward the class mean. More capacity per class -> finer
sub-phenotypes -> the diluted anchors keep their own signal.

This is the BLUNT capacity test (uniform tpn=5, K = 40 + 155*5 = 815) chosen
deliberately before the elegant data-driven version: it answers "does more class
capacity help AT ALL?" with zero code change. The effrank probe rides along (now
fixed to fire in the gated init) and reports each node's ACTUAL diversity, which
is independent of tpn — so this one run gives both the capacity AP test and the
design input for a data-driven per-node K_v (insight 0081).

**Read the outcomes:**
- **AP beats flat/0083 (>0.021), esp. the diluted anchors (SLE, MS) recovering**
  -> capacity IS the lever. Then build data-driven per-node K_v allocation
  (allocate_topics -> DagLayout per-node block sizes) to spend topics efficiently
  instead of blunt uniform tpn (which wastes 5 topics on tight leaves that want 2).
- **AP ~ flat/0083 (no lift)** -> capacity is not the lever; the class mean is the
  wrong center regardless of resolution. Fit-time pooling is closed; the hierarchy
  stays purely an eval-time within-class scaffold. Save the DagLayout surgery.
- **[effrank] table:** does the diversity distribution justify capacity? Coherent
  classes (connective tissue, cardiovascular, immune) should show high PR (want
  many topics); tight leaves ~2. Compare `Σround(PR)` to the current foreground K
  (155*5=775 at tpn=5, or 155*2=310 at tpn=2) — is data-driven K smaller (efficient)
  or "crazy large"? Watch the `n` column; if nodes saturate at max_probe=40, raise
  CHARM_PROBE_EFFRANK_MAX.

**Watch the fit card:** tpn=5 gives more topics to fill, so starved-topic count
may rise (a starved topic at tpn=5 is fine if the class genuinely has <5
sub-phenotypes — that's the graceful under-use of insight 0019, and exactly what
effrank predicts). Dead nodes should stay ~0083 levels (thin deep branches).

## Readout
```
make -C analysis/cloud clean-exp ID=84
export CHARM_PROBE_EFFRANK=1            # per-node effective-rank table (spectral, gated init)
make -C analysis/cloud exp ID=84
make -C analysis/cloud multidomain-weighting-readout ID=84 WEIGHTING_FIXED=1 WEIGHTING_JOBS=4
```
Do NOT set CHARM_DRIVER_MEMORY on the big master. spectral_proj_dim=400 keeps the
driver sketch light (~2.2GB, independent of tpn). Compare per-anchor AP to 0083
(no-roll-up tpn=2) and flat ~0.020; watch whether SLE/MS recover.
