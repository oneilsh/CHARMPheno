---
id: 86
slug: multidomain-rare-priority-snomed-hierarchy-reverse-topo
status: done
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
spectral_proj_dim: 800
anchor_scope: frontier
spectral_topo_order: reverse
strip_mode: both
max_iter: 200
cavi_max_iter: 100
cavi_tol: 1.0e-3
mini_batch_fraction: 0.0
min_peak_ratio: 5.0
top_n_tokens: 8
seed: 42
---

# exp 0086 — Reverse topo order: anchors claim first, classes get the label-orthogonal residual

One variable off 0085 (forward -> `spectral_topo_order: reverse`), keeping the
umbrella root + d=800 + tpn=2. 0085's labeled effrank showed the attribution is
DEPTH-driven: forward topo order lets ancestors claim first, so everything at
depth >=4 (incl. the deep disease anchors: EDS, Marfan, MS, Long QT, aneurysm)
got 0 increment -- their ancestors, which pool them, already claimed their
directions. Shallow anchors (Scleroderma, SLE, Sarcoidosis) claimed real rank
only because they sit high in the tree.

Reverse flips who claims first: leaves (anchors) recover their FULL specific signal
first, then each ancestor deflates against its descendants -> a class node claims
only structure NOT in any of its anchor descendants = "phenotypes not correlated
to the known distinct [anchor] labels" (shawn's framing). This is the built-in
operationalization of reserving the anchor-specific signal for the anchors.

**Two reads:**
1. **effrank attribution flips** -> anchors (esp. the deep ones) now show nonzero
   increment; classes/umbrella collapse toward the residual. If the deep anchors
   get real, bounded K, that's the usable per-node K_v for profiling. (Watch: a
   class pooling many anchors may still collapse to ~0 residual -- symmetric risk,
   but now the ANCHORS carry the signal, which is what we want.)
2. **identification (AP):** reverse keeps anchor topics PURE (not deflated toward
   ancestor means), so it MAY recover the umbrella's 0.021->0.016 hit, or even
   beat 0083. If max:scaled climbs back above condition-only, purity helped; if
   still below, pooling/hierarchy is inert for ID regardless of order.

Caveat: reverse may leave class nodes degenerate (0 residual after deflating
against descendants) -> dead class nodes on the fit card; that is acceptable here
(classes become background-like, anchors carry signal). Watch dead-node count.

## Readout
```
make -C analysis/cloud clean-exp ID=86
export CHARM_PROBE_EFFRANK=1
export CHARM_PROBE_EFFRANK_MAX=300
make -C analysis/cloud exp ID=86
make -C analysis/cloud summarize-exp ID=86
make -C analysis/cloud effrank-readout ID=86
make -C analysis/cloud multidomain-weighting-readout ID=86 WEIGHTING_FIXED=1 WEIGHTING_JOBS=4
```
Compare effrank-readout (do deep anchors reclaim K?) and AP macro vs 0085 (0.016)
and 0083 (0.021). Big master: no CHARM_DRIVER_MEMORY.
