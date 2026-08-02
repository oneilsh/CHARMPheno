---
id: 85
slug: multidomain-rare-priority-snomed-hierarchy-norollup-umbrella-root
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
# SNOMED class hierarchy above anchors (root -> class -> anchor -> descendants),
# restrict-under-Disease (4274025), max_class_fraction 0.6. Same layout as 0082.
anchor_hierarchy: snomed
hier_concept_class: ""
hier_restrict_under: 4274025
hier_min_class_size: 2
hier_max_class_fraction: 1.0
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

# exp 0085 — Add a real umbrella root (max_class_fraction 1.0) + hierarchical effrank probe

Same as 0083 (no-roll-up, spectral, tpn=2) except `hier_max_class_fraction: 1.0`,
which KEEPS the umbrella classes ("Disease" ~100% of anchors, "Disorder of body
system" ~85%) that 0.6 dropped. Purpose: give the DAG a real global-root CLASS
NODE (with a topic block) above the body-system classes, so the hierarchical
effrank probe telescopes properly.

**Why.** The synthetic forest root becomes engine node 0, which gets NO topic
block (n_bg background stands in), so at max_class_fraction 0.6 the ~14 top-level
body-system classes deflate only against the 40 background anchors and each
re-count the shared disease bulk -> Σround(PR) blew up to 4708 even under
hierarchical deflation. With the umbrella kept, "Disease" is a real depth-1 node
that claims the shared bulk FIRST (forward topo order), and every class below
deflates against its full claim -> increments shrink, total telescopes toward the
corpus dimensionality.

**Read the effrank first (the whole point):**
- **Σround(PR) drops from 4708 toward the low hundreds** -> the umbrella root fixed
  the double-counting; we have a bounded, non-double-counted per-node K_v for
  phenotype profiling. Check the top rows: "Disease" should carry a large claim,
  body-system classes much smaller increments.
- **Still large** -> even the umbrella doesn't absorb the sibling overlap (sibling
  body-system classes share structure not on the Disease->class path); then the
  double-counting is inherent to a multi-parent DAG and effrank-as-K_v is closed.

**Watch the fit too (secondary):** "Disease" is a catch-all covering all anchors,
so it acts like extra background — it MAY help identification by absorbing shared
comorbidity (cf. the n_bg lever), or MAY be a useless-for-discrimination node. So
also glance at the case-finding AP vs 0083 (cond macro 0.021): a lift would say
shared-structure absorption helps; a drop/wash says the umbrella is inert for
discrimination (still fine — its job here is the effrank telescoping).

## Readout
```
make -C analysis/cloud clean-exp ID=85
export CHARM_PROBE_EFFRANK=1
export CHARM_PROBE_EFFRANK_MAX=400        # = d (projection dim); the sketch's hard ceiling
make -C analysis/cloud exp ID=85
make -C analysis/cloud summarize-exp ID=85
make -C analysis/cloud effrank-readout ID=85
make -C analysis/cloud multidomain-weighting-readout ID=85 WEIGHTING_FIXED=1 WEIGHTING_JOBS=4
```
Big master: do NOT set CHARM_DRIVER_MEMORY. Compare effrank Σround(PR) to 0083's
4708, and case-finding AP to 0083's 0.021.
