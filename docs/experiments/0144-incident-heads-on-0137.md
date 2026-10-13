---
id: 144
slug: incident-heads-on-0137
status: done
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

**2026-10-13 — run.** 535 readout nodes with >= 20 positives both ways (same set as 0143).

## Results

**Training the heads on incident labels makes them WORSE at finding new diagnoses.**

| heads | arm | macro AUC all | macro AUC incident | gap median | nodes losing > 0.05 |
|---|---|--:|--:|--:|--:|
| record (mixed labels, 0143) | stacked | 0.818 | **0.702** | +0.12 | 439 |
| incident-trained (0144) | stacked | 0.670 | **0.628** | +0.035 | 223 |
| record (0143) | flat | 0.727 | **0.616** | — | 399 |
| incident-trained (0144) | flat | 0.683 | **0.602** | +0.074 | 326 |

By prior share of positives (incident AUC, stacked): 0.0–0.1 0.624 · 0.1–0.3 0.643 ·
0.3–0.6 0.621 · 0.6–1.0 0.632 — flat across bands again, ~0.07 below the record's.

1. **Read (1) of the pre-registration: no recovery.** Incident AUC falls 0.074 stacked,
   0.014 flat. 0143's ~0.70 is not an artefact of head training that a readout-side
   relabel can fix.
2. **The stacking gain mostly collapses**: stacked − flat on incident is +0.086 with the
   record's heads, +0.026 here. Under the incident rule a node's cell is masked for any
   prior code of the node OR a label descendant, so the ancestor heads that carry the
   stacked product lose most of their positives (e.g. demyelinating disease 490 → 106,
   thoracic cancer 1353 → 310 test positives).
3. **'All' AUC falls as expected** (0.818 → 0.670): the heads no longer reward re-coding.
   The residual gap (median +0.035 stacked) is the topic signal itself: prevalent
   patients carry more of X's topic mass than incident ones.

**Why mixed labels win on incident evaluation (working reading).** Prevalent positives
are real X patients; their θ shows the head what X's topic signature looks like, and
that transfers to the not-yet-coded. The incident rule halves the positives (median
prior share 0.53) at the same l2 (100), and removes the clearest examples. So the
label rules should be split three ways, not two: **evaluate incident; train heads
(and gate topics) on the broadest positive set** — prevalent + incident, all-ever.
That is the direction the rebuild was already taking for topic gating; this result
extends it to the heads. Not tested here: a lower l2 for the incident heads (would
narrow the gap if under-fitting from fewer positives is part of it, but cannot add the
lost examples back).

**Consequence.** The record's own heads stay the case-finding heads; their incident
macro AUC (0.702 stacked) is the honest headline for 0137. Incident-only heads are
dropped. The rebuild keeps: all-ever attestation gating for topics, incident evaluation
(`preindex_closure` arm), and mixed (or all-ever) head training.
