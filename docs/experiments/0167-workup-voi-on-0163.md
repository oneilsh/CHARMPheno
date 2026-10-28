---
id: 167
slug: workup-voi-on-0163
status: planned
model_class: gated_pc
# NOT A FIT. Framing 2 as WORKUP (spec 2026-10-28 workup-voi-design) on 0163's fit:
# every fact recorded between the index and the first diagnosis of a group member
# (conditions, drugs, measurement results incl. "done, no result") is ranked by its
# expected information gain under each patient's belief (E6, and E3 for the
# representation), and every (patient, fact) pair is scored on its true answer.
# Same 9 same-question groups and incident patients as 0163 / 0166.
fit_of: 163
---

# 0167 — framing 2 as workup: which fact would most move this patient's differential

## Why

Spec [2026-10-28 workup-voi-design](../superpowers/specs/2026-10-28-workup-voi-design.md).
The hidden-record protocol (0148-0151, 0166) answers questions from a random 80% of the
patient's PAST, values only YES answers, and scores P(true)@5 on a ceiling of
0.05-0.17. The clinical question is which workup fact would most move the belief, in
any direction, so the facts here are those recorded after the index and before the
diagnosis, every answer (including "not recorded" and "not done") is evidence through
`P(answer | member)`, and the ranking is scored densely.

Decisions (maintainer, 2026-10-28): window `[index, t_dx)` (the diagnosis day itself
only as a cheap sensitivity run); "done, no result" kept as an answer; starting belief
E6 and E3 in the same pass; shortened pre-index records deferred.

## Pre-registered reads

1. **Coverage first.** Window length per group (the population index puts the index
   0-365 days before the diagnosis) and the share of patients with any non-default
   answer. A group without enough windows is reported, not scored.
2. **R1:** is the per-patient Spearman between EIG and realized gain positive, and by
   how much?
3. **R2:** does the top fact by EIG beat a random fact and the population ranking (one
   list per group), and how much of the hindsight-best fact's gain does it reach?
4. **R3:** is EIG calibrated (slope of realized gain on EIG near 1)?
5. **R4:** how often does the top fact's actual answer change the call, and toward the
   truth?
6. **E6 vs E3:** does the ranking depend on the starting belief's representation?
7. **Strict vs inclusive window:** how much the diagnosis day adds (same-day serology).

## Run (one background job)

The conversion sidecar is a HIT if this cluster built it today; the workup sidecar is
new (one BigQuery pass over every domain's year after the index). Then the survey
twice, strict and inclusive. Each survey is about as long as 0166's (~8 min).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
R152=$(ls -d $RUNS/0152-*); R163=$(ls -d $RUNS/0163-*); ls -la $R152/groups_4b.json
L=$R163/job_log_0167.md
nohup bash -c '
set -o pipefail
R152='$R152'; R163='$R163'
echo "### conversion sidecar 163 (HIT if built on this cluster)"
make -C analysis/cloud build-conversion-sidecar ID=163 || { echo "FAILED: conversion sidecar"; exit 1; }
echo "### workup sidecar 163"
make -C analysis/cloud build-workup-sidecar ID=163 || { echo "FAILED: workup sidecar"; exit 1; }
BASE="--differential . --diff-survey 25 --diff-members top --diff-incident --diff-groups-like $R152/groups_4b.json --acq-heads stacked_nodes_mass99 --diff-max-patients 1500 --diff-crossfit --diff-workup"
for w in strict inclusive; do
  echo "### survey 0167 $w"
  make -C analysis/cloud gated-pc-readout ID=163 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="$BASE --diff-workup-window $w --acq-tag survey_0167_$w" || { echo "FAILED: survey $w"; exit 1; }
  python3 analysis/cloud/survey_analysis.py "$R163" survey_0167_$w | sed -n "/## workup/,/^E5 reasons/p" | cut -c1-220
done
echo "### DONE"
' > "$L" 2>&1 &
sleep 2; ls -la "$L"; tail -f "$L"
```

## Run log

(planned)
