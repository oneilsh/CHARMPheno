---
id: 145
slug: incident-eval-whole-history-on-0137
status: done
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

**2026-10-13 — run.** 54,753 test documents, every one with a sidecar index (coverage
100%). Prior cells: lookback rule 1,215,542; whole-history rule 2,524,695 (2.08×).
Containment of lookback cells in whole-history cells **1.000** — the sidecar index and
the Mondo roll-up agree with the bundle's features exactly.

## Results

| prior rule | arm | nodes | macro AUC all | macro AUC incident | median prior share of positives |
|---|---|--:|--:|--:|--:|
| lookback (0143) | stacked | 535 | 0.818 | 0.702 | 0.53 |
| lookback (0143) | flat | 535 | 0.727 | 0.616 | 0.53 |
| **whole history** | stacked | 492 | 0.822 | **0.694** | 0.64 |
| **whole history** | flat | 492 | 0.733 | **0.612** | 0.64 |

Whole-history incident AUC by prior share (stacked): 0.1–0.3 0.742 · 0.3–0.6 0.696 ·
0.6–1.0 0.688 (29 / 166 / 295 nodes; the 0.0–0.1 band holds 2 nodes).

1. **The lookback rule was close.** Reading the whole history moves the median prior
   share of positives 0.53 → 0.64 (about one positive in five that looked new was a
   diagnosis older than 5 years), but the incident macro AUC moves only 0.702 → 0.694
   stacked, 0.616 → 0.612 flat. 43 nodes drop below 20 incident positives.
2. **The baseline 0146 is read against: stacked 0.694, flat 0.612** (whole-history
   rule, 492 nodes; 0146 is scored on the same rule and compared on shared nodes).
3. Nearly two-thirds of a typical node's label-year positives are prevalent
   re-codes; for 295 of 492 nodes it is more than 60%.
