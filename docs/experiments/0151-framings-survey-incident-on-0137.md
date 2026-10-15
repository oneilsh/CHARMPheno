---
id: 151
slug: framings-survey-incident-on-0137
status: planned
model_class: gated_pc
# NOT A FIT. Insight 0103: the survey's best which-member groups are called from the
# diagnosis in other words (synonym codes, confirmatory tests, treatments), because
# most label-year positives are prevalent and their pre-index record documents the
# diagnosis. This repeats 0150 exactly (same groups, engines, caps, cross-fit, framing
# 2 on E5) with ONE change: test patients are restricted to INCIDENT ones — no
# attestation of any member, or of the group node, or of anything under them, anywhere
# before the index (E4 sidecar + Mondo is-a). Their pre-index record is a
# pre-diagnosis record. Training unchanged (label-year, single member; 0144's lesson).
# PRE-REGISTERED READS: (1) kappa distribution vs 0150 (quartiles 0.07 / 0.18 / 0.34;
# 75 groups >= 0.3) — expected to fall; how many groups keep kappa >= 0.3 on >= 40
# incident patients; (2) whether the surviving groups' E5 reasons are clinical
# features rather than synonyms / tests / treatments; (3) framing 2 (gap closed, VOI −
# table) under the same restriction; (4) how many groups drop below 40 patients.
fit_of: 137
---

# 0151 — framings survey on incident (pre-diagnosis) patients (0137)

## Run

Rebuild the sidecar on this cluster first (~4 min), then the survey (as 0150, several
hours), then the analysis.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup bash -c '
  make -C analysis/cloud build-conversion-sidecar ID=137 || exit 1
  make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential . --diff-survey 25 --diff-incident --diff-max-patients 1500 --diff-crossfit --diff-next-question --diff-nq-hide 0.8 --acq-tag survey_inc"
' > "$RUN"/survey_inc_log.md 2>&1 &
```

Report back:

```bash
python3 analysis/cloud/survey_analysis.py "$RUN" survey_inc | cut -c1-220
```

## Run log

## Results
