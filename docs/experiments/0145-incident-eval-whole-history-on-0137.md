---
id: 145
slug: incident-eval-whole-history-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. Label redesign spec (2026-10-13) R1 on the ribbon record 0137: the
# incident evaluation with the prior read from the WHOLE pre-index history (the E4
# first-attestation sidecar, key 1e9c7c48a37da4cf, 13.2M (person, node) rows over
# 306,317 persons; index = the sidecar's index_horizon, re-derived by the assembler's
# own index builder) instead of 0143's 5-year lookback codes. A sidecar node counts
# for every readout node at or above its Mondo term (is-a), since the sidecar's code
# map is the unfiltered native build. Same saved heads (stacked_mass99), same test
# split; both rules reported side by side.
# PRE-REGISTERED READS: (1) containment of lookback prior cells in whole-history ones
# ~1 (else the index or ids disagree, and the run is not read); (2) whole-history
# incident macro AUC vs 0143's 0.702 stacked / 0.616 flat — expected at or below
# (patients diagnosed > 5 years back move out of the incident set); this is the
# baseline 0146 (all-ever topic gating) is read against; (3) how many more prior cells
# the whole history finds.
fit_of: 137
---

# 0145 — incident evaluation on the whole pre-index history (0137)

## Run

Test-split transform + the score collect + a sidecar join (needs the sidecar on this
cluster's HDFS: `make build-conversion-sidecar ID=137`, done 2026-10-13).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--prevalence-check --prev-history --acq-tag history" > "$RUN"/prevalence_history_log.md 2>&1 &
grep -E "^WHOLE|^## |^\| 0" "$RUN"/prevalence_check_history.md | cut -c1-220
```

## Run log

## Results
