---
id: 81
slug: multidomain-rare-priority-snomed-hierarchy-norollup
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
# FIRST hierarchical fit: insert the compact SNOMED class hierarchy ABOVE anchors
# (root -> class -> anchor -> descendants), replacing the flat root->anchor forest.
# All one ontology (SNOMED concept_ancestor), class nodes are REAL concepts.
# max_class_fraction 0.6 drops the giant umbrellas ("Disease" 1.0, "Disorder of
# body system" 0.85) so classes like cardiovascular/nervous-system hang off root.
anchor_hierarchy: snomed
# concept_class filtering does NOT separate disorders from findings in OMOP SNOMED
# (both are 'Clinical Finding'; 'Disorder' matches nothing). The structural split
# is descent from the SNOMED "Disease" concept (4274025): disorders are under it,
# cross-cutting findings (Measurement/Functional/... finding) are not. So no
# concept_class filter, and restrict class candidates to descendants of Disease.
hier_concept_class: ""
hier_restrict_under: 4274025
hier_min_class_size: 2
hier_max_class_fraction: 0.6
# Roll each patient's condition codes UP to the nearest DAG node via
# concept_ancestor, so a class node ("Disorder of head") gathers ALL its
# descendant patients (migraine, glioma, ...), not just those coded at the exact
# node -> richer class topics, stronger pooling, and class-level placement of
# unclassified patients. Non-anchor patients land at the class (below any anchor),
# so anchor case-finding labels are unchanged.
rollup_attestation: false
init: random
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

# exp 0081 — SNOMED hierarchy WITHOUT roll-up (isolator vs 0080)

Isolates the hierarchy from the roll-up. Exp 0080 (hierarchy + roll-up) collapsed
case-finding ~3× (insight 0080) because roll-up floods each class with the common
diseases in it, dragging rare anchors toward common-disease means. This run keeps
the identical layout but `rollup_attestation: false`, so class nodes pool only the
patients routed from the anchors beneath them (a rare-flavored class mean, not
flooded). Two possible outcomes: (a) condition macro AP recovers toward the flat
~0.020 (and maybe pooling helps low-count anchors) → the flooding was the culprit;
or (b) the class nodes go **dead** (the gating does not flow anchor mass up to
ancestors) → the hierarchy is inert in the fit and the pooling direction is closed
for case-finding (confirming insight 0076). Check the fit-card dead-node count
first. Original 0080 notes follow.

## exp 0080 — Hierarchical layout: SNOMED class nodes above anchors

The first fit with a real hierarchy above the anchors (insight 0079 follow-up).
Holds exp 0078 fixed (rare_priority, cond+drug+measurement, K structure, seed 42)
and changes only the **label DAG**: instead of the flat root→anchor forest, insert
the compact SNOMED class hierarchy (`root → Disorder-class → … → anchor →
SNOMED descendants`) via `case_finding_assembly._snomed_class_hierarchy` —
`concept_ancestor` reduced to branch points (`anchor_hierarchy`), one coherent
ontology above and below anchors, class nodes are real concepts (so patients coded
at a general level, e.g. "Vasculitis", attach to the class node). Multi-parent DAG
(a disorder can sit under several classes), matching the sub-anchor DAG below.

**Init:** `random` (not spectral) — spectral added ~15 min setup/run and insight
0070 found it non-critical; using random to iterate on the hierarchy faster.

**What this turns on**

- **Pooling** — each class node gets its own topic block, partially pooling its
  sibling anchors (e.g. the vasculitides under `Vasculitis → Systemic vasculitis`).
  The test: do low-count anchors' per-anchor AP lift vs the flat 0078 fit?
- **Hierarchical / conditional placement** — scoring can target any level, which
  is the substrate for the within-class ("rank within connective-tissue") readout.

**Node count** stays ~today's: at `max_class_fraction 0.6` roughly 30–40 classes +
40 anchors ≈ the 95-node flat layout (the ancestor closure is 330; we keep only
branch points). K = n_bg + (classes + anchors + surviving descendants) × tpn.

**Watch on first run** (this is a new DAG path): DAG assembly / `to_engine`
succeeds on the multi-parent class layer; `dead_node`/`starved_topic` reports
(class nodes on thin branches may starve — if so, note which); K vs 0078.

**Readout.** Compare to 0078 (flat) on the fast readout:
```
make -C analysis/cloud exp ID=80
make -C analysis/cloud multidomain-weighting-readout ID=80 WEIGHTING_FIXED=1
```
Population-task per-anchor AP vs 0078 tests whether pooling helped. The bigger
payoff — within-class conditional ranking — needs the class-conditional readout
(next build) once this fit validates.
