---
id: 144
slug: incident-heads-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. First step of the label redesign (insight 0100, exp 0143). 0143 found
# 0137's stacked macro AUC falls 0.818 -> 0.702 once patients with a prior diagnosis of
# the node are set aside (flat 0.727 -> 0.616): about half of every node's label-year
# positives are prevalent re-codes. Those heads were TRAINED on the mixed labels. Here
# they are re-solved on the saved fit (same θ, same λ, no topic refit) under the
# INCIDENT rule: every (patient, node) cell with prior evidence for the node (a mapped
# attestation code of the node or a label descendant in the 5-year pre-index condition
# features — 0143's rule exactly) is masked out of BOTH splits before the solve
# (`gated_pc_readout --readout-incident`, `mask_prior_cells`). The root head stays
# case-vs-background on every row. Same l2 (100), θ mass 0.99, stacked arm as 0137.
# Then the prevalence check is re-run on the new heads over the UNMASKED test split.
# PRE-REGISTERED READS: (1) incident macro AUC of the incident heads vs the record's
# heads (0.702 stacked / 0.616 flat): a clear rise (> ~0.02) says part of 0143's loss
# was head training and is recoverable readout-side; flat says the topics carry ~0.70
# of new-diagnosis signal and the rebuild (all-ever gating of topic learning) is where
# the gain must come from; (2) whether the rise concentrates in the high-prior-share
# band (where the record's heads learned re-coding most); (3) the 'all' AUC of the
# incident heads (expected to fall: they no longer reward re-coding).
fit_of: 137
---

# 0144 — incident heads on the 0137 fit (label redesign, step 1)

## Run

Bundle (rebuild if cold, ~20 min) + both transforms + one masking pass per split (a
Python UDF over the widened mask) + the batched stacked solve (as 0137's stacked
readout) + the prevalence check on the new heads.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-theta-mass 0.99 --readout-stacked --readout-incident" > "$RUN"/incident_heads_log.md 2>&1 &
grep -E "^## |^\| 0|^- " "$RUN"/prevalence_check_stacked_incident_mass99.md | cut -c1-170
```

Outputs (record files untouched): `results_readout_stacked_incident_mass99.json`,
`readout_heads_gated_pc_stacked_incident_mass99.npz`,
`prevalence_check_stacked_incident_mass99.md/.json`.

## Run log

## Results
