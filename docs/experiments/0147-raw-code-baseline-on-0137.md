---
id: 147
slug: raw-code-baseline-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. The anchor 0.69 never had: the same readout heads on RAW CODES instead of
# θ. Same bundle, documents, 698-node Mondo readout DAG, distributed solver, l2 100,
# stacked arm and whole-history incident rule (0145) as 0137; the only change is the
# head input: binary presence of the 2,000 most frequent train tokens (conditions,
# measurement result tokens, drugs pooled; [measured] included, as the topic model saw
# them) instead of θ. `gated_pc_readout --raw-baseline 2000`.
# PRE-REGISTERED READS (vs 0145: stacked 0.694, flat 0.612 on 492 nodes): raw within
# ~0.01 = the ceiling is the 5-year record, the topics lose nothing and their value is
# interpretability; raw clearly above (> ~0.02) = the topic representation drops
# signal (K / alpha / bg-vs-private split are the next lever); raw clearly below = the
# topics add signal over a flat code model. Also the all-patient AUC (raw codes carry
# own codes directly, so expect it high) and the per-domain token split.
fit_of: 137
---

# 0147 — raw-code baseline on 0137 (new-diagnosis AUC anchor)

## Run

No θ transform; one pass to count train document frequencies, one UDF pass per split,
the batched stacked solve at K = 2,000, then the whole-history prevalence check (the
sidecar is on this cluster's HDFS).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--readout-mode distributed --readout-l2 100 --readout-stacked --raw-baseline 2000 --prev-history" > "$RUN"/raw_baseline_log.md 2>&1 &
grep -E "raw baseline:|root head alone|ranking \(within" "$RUN"/raw_baseline_log.md | cut -c1-220
grep -E "^## |^\| 0" "$RUN"/prevalence_check_stacked_raw2000.md | cut -c1-220
```

## Run log

**2026-10-14 — run 1.** Tokens: 2,000 by train document frequency — 680 condition,
978 measurement, 342 drug. Train 220,160 / test 54,753 docs. Batched L-BFGS hit
max_iter 200 with 460/684 heads converged (224 still active, max|grad| 6.63, 138
line-search failures; 668 data passes, ~4 h) — the raw heads are UNDER-CONVERGED, so
these numbers are a floor for the raw model. Readout and stacked arm completed; the
chained prevalence check then failed (no file written) — most likely a sidecar MISS:
this was a fresh cluster (2026-10-14) and the sidecar lived on the old cluster's HDFS.
Re-check from the saved heads after rebuilding the sidecar (below).

```bash
make -C analysis/cloud build-conversion-sidecar ID=137
make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--prevalence-check --prev-history --raw-baseline 2000 --acq-heads stacked_raw2000 --acq-tag stacked_raw2000"
```

## Results (all-patient, before the incident check)

| | raw codes (0147) | θ, 0137 |
|---|--:|--:|
| per-node AUC over all test docs, stacked | 0.820 (582 nodes) | 0.822 |
| same, flat | 0.721 | 0.727 |
| root head AUC | 0.834 | — (0146: 0.836) |
| within-parent ranking macro AUC | 0.718 | — (0146: 0.733) |

On the all-patient read the raw-code heads match θ almost exactly, with 2,000 binary
codes and under-converged heads.
