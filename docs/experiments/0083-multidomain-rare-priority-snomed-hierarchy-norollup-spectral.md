---
id: 83
slug: multidomain-rare-priority-snomed-hierarchy-norollup-spectral
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

# exp 0083 — SNOMED hierarchy, NO roll-up, SPECTRAL init (clean case-finding read)

The clean counterpart to exp 0082. The 2x2 across {roll-up, init} is now:

| | random init | spectral init |
|---|---|---|
| **roll-up** | 0080 (flooded, AP 0.006, under-fit) | 0082 (flooded, AP 0.006, **clean fit**) |
| **no roll-up** | 0081 (rare-flavored classes, AP lost to a 143, under-fit) | **0083 (this)** |

0082 proved the roll-up collapse is structural, not under-fitting (clean fit, AP
still 0.006 vs flat 0.020). The 0081 fit card proved that WITHOUT roll-up the
class nodes come alive and rare-flavored (Sjogren's/systemic sclerosis/ANA under
"Disorder of connective tissue"; monoclonal gammopathy/SPEP under "Degenerative
disorder"), NOT dead and NOT flooded — so the flooding was specifically the
roll-up, exactly as insight 0080 predicted. But 0081 was random-init (50/350
starved) and its AP was killed with the persist step, so we never saw whether the
rare-flavored classes actually HELP case-finding.

This run flips only `rollup_attestation` (false) off the clean 0082 config and
keeps spectral init, so the fit is well-conditioned (0082-style, few starved) and
the AP is directly comparable:

- **AP recovers toward flat ~0.020** (and low-count anchors sharing a coherent
  class lift vs the flat 0078 baseline) -> the hierarchy WITHOUT roll-up is
  viable, maybe helpful; roll-up flooding was the whole problem. Keep the
  hierarchy (no roll-up) and it doubles as the eval-time within-class structure.
- **AP still well below flat** despite alive, rare-flavored, well-conditioned
  class topics -> hierarchical *pooling* itself does not help rare-disease
  case-finding (the class mean, even rare-flavored, is still the wrong center for
  the rarest anchors). Then fit-time pooling is closed; use the hierarchy only as
  an eval-time scoring structure, and the effective-rank/capacity lever (insight
  0081) is moot for this objective.

**Watch the fit card first:** starved topics (want ~0082 levels, few), dead nodes
(0081 had 11/155 thin descendants dead — expected; the class layer should be
alive), and whether the class topics stay rare-flavored under spectral.

## Two answers from one run: AP + effective-rank probe

Because this is a SPECTRAL fit, setting `CHARM_PROBE_EFFRANK=1` also dumps the
per-node effective-rank table (insight 0081) during init, at no extra run. It
reads each node's WITHIN-GROUP sketch, so under no-roll-up it measures the
diversity of the rare-flavored class populations we'd actually keep — i.e. "do the
classes want more than tpn=2, and would data-driven K_v shrink or blow up the
layout?" Grep the log for `[effrank]`. If many nodes saturate at n=max_probe,
raise `CHARM_PROBE_EFFRANK_MAX`.

## Readout
```
make -C analysis/cloud clean-exp ID=83
export CHARM_PROBE_EFFRANK=1            # per-node effective-rank table (spectral only)
make -C analysis/cloud exp ID=83
make -C analysis/cloud multidomain-weighting-readout ID=83 WEIGHTING_FIXED=1 WEIGHTING_JOBS=4
```
**Driver memory (this master is 7.8GB / 2-core).** In PySpark client mode the
master runs both the JVM (bounded by --driver-memory) and the Python driver
(numpy, unbounded). The scalable spectral init COLLECTS a (V_concat x d) sketch
PER NODE to the Python driver: 155 nodes * 8774 * d * 4 bytes. At the auto d~1000
that is ~5.4GB, which OOM-kills init (exit -9) here. Two fixes, use both:
1. **Shrink the sketch** via `spectral_proj_dim: 400` (set in frontmatter above) ->
   ~2.2GB. Drop to 300 (~1.6GB) if still tight.
2. **Free master RAM first** -- `ps aux --sort=-%mem | head` then shut idle Jupyter
   kernels; the baseline was 5.5GB used / 2.3GB free, so freeing kernels is what
   makes even a 2.2GB sketch comfortable.
Do NOT raise --driver-memory for the init phase -- it bounds only the JVM and
steals physical RAM from the Python sketch, making the OOM worse. A MODEST bump
(CHARM_DRIVER_MEMORY=6g) is only for the *persist* collect (JVM-side, exit 143).
Compare per-anchor AP to flat 0078 and to roll-up 0082 (0.006). Flat condition
macro is the ~0.020 bar to beat. The `[effrank]` lines answer the capacity
question in parallel.
