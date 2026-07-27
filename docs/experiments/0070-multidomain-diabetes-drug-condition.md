---
id: 70
slug: multidomain-diabetes-drug-condition
status: pending
model_class: multidomain
cohort: population_diabetes
cohort_def: population_diabetes
disease: diabetes
domains: drug_era
window_mode: forward
person_mod: 10
prior_obs_days: 365
window_days: 365
doc_min_length: 10
min_n: 50
holdout_frac: 0.2
n_bg: 20
tpn: 5
# Per-domain vocabulary controls (conditions and drugs have very different
# natural sizes; SP3b design — one shared threshold would starve one and bloat
# the other).
source_table_cond: condition_era
source_table_drug: drug_era
cond_vocab_size: 5000
cond_min_df: 20
cond_min_patient_count: 20
drug_vocab_size: 2000
drug_min_df: 20
drug_min_patient_count: 20
# Generative knobs. omega/eta_per_domain UNSET here = the shim's scalar default
# (all domains equal, eta = 1/K) — the faithful MixEHR baseline. SP4 sweeps omega.
init: spectral
spectral_max_vocab: 8000
spectral_method: auto
anchor_scope: closure
spectral_topo_order: forward
strip_mode: test_only
max_iter: 100
cavi_max_iter: 100
cavi_tol: 1.0e-3
min_peak_ratio: 5.0
seed: 42
---

# exp 0070 — Multi-domain (condition + drug) gated fit, diabetes

First tracked run of the two-domain, MixEHR-style gated topic model (SP3b):
conditions (domain A, `condition_era`) + drugs (domain B, `drug_era`) over two
independent vocabularies, sharing one DAG-gated per-document theta with a
**condition-only gate** (gate ⟂ domain — the gate acts on theta's support from
the label's DAG closure; the domains act on beta's normalizer). Drugs are
features, never a label: they need a vocabulary, not a DAG.

This is the harness wiring of `analysis/cloud/multidomain_cloud.py` — the same
driver as `make multidomain-bq-smoke`, now reachable as `make exp ID=70` so the
fit is a tracked, resumable-config, summary-captured experiment like every other.

## What runs

`run_experiment.py` dispatches `multidomain_cloud.py`, which:

1. Loads `condition_era` AND `drug_era` (no `cohort=` post-filter — that picks
   the wrong date column for drugs), windows both to the same per-patient
   diabetes cohort window (`window_days`, forward mode), builds the diabetes
   label DAG (single anchor 201820, no forest root).
2. Assembles a `TwoDomainBundle` (`charmpheno.omop.two_domain`) with two aligned
   sparse feature columns (`features_a` conditions, `features_b` drugs), each
   over its own per-domain vocabulary, leakage stripped per domain.
3. Fits `GatedLDAEstimator(featuresCols=["features_a","features_b"], seed=42)` to
   a per-domain dict lambda `{0: (K, V_a), 1: (K, V_b)}`.
4. Logs the **dead-node init-quality read** (insight 0070: the scalable spectral
   init is seed-fragile; a dead node = a projection draw the EM did not rescue),
   and writes the `VIResult` through SP3a's dict-lambda-aware `save_result`.

K is emergent (`n_bg` + surviving-DAG-nodes × `tpn`), so there is no `K`. Resume
is unsupported in v1 (`GatedLDAModel` is not persistable); a re-run refits.

## What to read

There is **no NPMI eval** for this artifact (an npz + manifest, not a topic-word
bundle the coherence driver can read — `run_experiment` skips eval for
`multidomain`, as for `dag_placement`). Read instead, from `manifest.json`:

- `dead_nodes`: MUST be empty. A non-empty list is the pre-registered
  init-fragility signature — re-run with a different `seed` (do not silently
  accept seed 0's draw).
- `corpus_stats`: per-domain vocab sizes (`V_a` cond, `V_b` drug) within a
  plausible band, train/test doc counts, how many docs carry a frontier.
- `ledger`: the two-domain assembly provenance (per-domain prune/strip counts).

This is a **smoke/sanity** run, not a quality gate — SP3b's acceptance is
structural (shapes, id ranges, alignment, leakage), and the specificity
green light (the omega sweep + a mass-landing read) is SP4. A fit that "ran"
is not a fit that is trustworthy; the dead-node read is the concrete first check.

## Knobs to tune before running

- `person_mod: 10` is a 1/10 population sample — a deliberately modest first
  cut. Drop toward 1 (full population) once the sanity reads look right.
- `omega` / `eta_per_domain` are unset (faithful MixEHR baseline). SP4 is where
  omega gets swept; leave it alone here.
