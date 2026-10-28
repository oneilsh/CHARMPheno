---
id: 166
slug: framings-relevance-codes-on-0163
status: planned
model_class: gated_pc
# NOT A FIT. Framings 1 and 2 on 0163's fit with the code engines' codes chosen by
# WITHIN-GROUP RELEVANCE instead of group document frequency (`--diff-code-select
# relevance`, `differential.CodeHead._select_relevance`). Same survey as 0163's
# `survey_inc_top` (9 same-question groups, incident test patients, cross-fit, framing 2
# at 80% hidden); only E5/E6's input codes change. λ swept 0.6 / 0.3 / 0, plus a df
# control re-run on the current code.
fit_of: 163
---

# 0166 — framings 1 and 2 with relevance-selected codes (0163)

## Why

E5 and E6 (exp 0148) pick the top 3,000 codes by document frequency among the group's
training patients. Within 0163's survey they matter: the best engine is E5 or E6 in 7 of
9 groups, and E6 − E3 kappa quartiles are +0.04 / +0.15 / +0.21 (0163 run log). But
frequency fills the slots with codes common to every member, which separate nothing:
the same three concepts (cid 4329041, 44824625, 40325768) are among E5's top reasons in
viral hepatitis, aortic aneurysm, social phobia and bulimia, so the head leans on codes
that are not specific to any of these questions.

The separating quantity for a member m is its lift over the group, P(v | m) / P(v |
group), and mixing it with frequency (LDAvis relevance) is what framing 3 does per node
(exp 0163's `rawr2000`). Here the same mix is computed WITHIN the group:

    rel_m(v) = λ log P(v | m) + (1 − λ) log [ P(v | m) / P(v | group) ]

with support counted inside the member (≥ 5 of m's training patients, so a high lift
never rests on one or two patients) and members taking their top codes in turn up to
3,000. Rare codes are expected; the linear softmax adds their evidence, so several weak
markers can still support a call together.

**Why a λ sweep.** Within a group, lift for m is capped at 1 / P(m | group), about 2 for
two equal members, where corpus-wide lift reaches 100×. So the log-frequency term
dominates much sooner than in framing 3. A marker on 40% of m's patients outranks a code
on 95% of everyone only for λ < ~0.45; one on 10% only for λ < ~0.24. So 0.6, framing
3's value, is close to frequency here. λ = 0.3 and λ = 0 (pure lift, guarded by the
member-support floor) are the informative arms.

**Framing 2 too.** The question pool is the population-EIG table, not the head's code
set. But `CodeHead.candidates` only moves the differential for codes the head has, so
pool questions outside the selected set score zero VOI. Relevance selection should let
VOI value the discriminating questions.

## Pre-registered reads

1. Framing 1: kappa per group, paired against the df control (same patients). Does
   relevance move the median kappa of E5 / E6, and at which λ?
2. E5's reasons: do the shared generic codes leave the top reasons, replaced by
   member-specific ones?
3. Framing 2: share of the gap closed by 5 questions and VOI − table, paired.
4. Number of codes selected (`e5_n_codes` in the JSON): relevance with a ≥ 5 per-member
   floor may select fewer than 3,000 in small groups.

## Run (one background job)

The bundle rebuilds on a fresh cluster; the readout's drift gate checks it against the
fit (also the post-refactor corpus-equivalence check, ADR 0051 step 3). If it reports
DRIFTED, stop and report. The df control re-runs `survey_inc_top` on the current code,
so the pairs differ only in the code selection.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
R152=$(ls -d $RUNS/0152-*); R163=$(ls -d $RUNS/0163-*); ls -la $R152/groups_4b.json $R163/conversion_sidecar.json
L=$R163/job_log_0166.md
nohup bash -c '
set -o pipefail
R152='$R152'; R163='$R163'
BASE="--differential . --diff-survey 25 --diff-members top --diff-incident --diff-groups-like $R152/groups_4b.json --acq-heads stacked_nodes_mass99 --diff-max-patients 1500 --diff-crossfit --diff-next-question --diff-nq-hide 0.8"
for arm in df:df rel06:relevance:0.6 rel03:relevance:0.3 rel00:relevance:0; do
  IFS=: read tag sel lam <<< "$arm"
  extra="--diff-code-select $sel"; [ -n "$lam" ] && extra="$extra --diff-code-lambda $lam"
  echo "### survey 0166 $tag"
  make -C analysis/cloud gated-pc-readout ID=163 GPR_SPARK_CONF=spark.eventLog.enabled=false GPR_ARGS="$BASE $extra --acq-tag survey_0166_$tag" || { echo "FAILED: survey $tag"; exit 1; }
done
echo "### control vs 0163 original (should be identical)"
python3 analysis/cloud/survey_analysis.py "$R163" survey_0166_df "$R163" survey_inc_top | cut -c1-220
for tag in rel06 rel03 rel00; do
  echo "### relevance $tag vs df control"
  python3 analysis/cloud/survey_analysis.py "$R163" survey_0166_$tag "$R163" survey_0166_df | cut -c1-220
  echo "### single-run $tag"
  python3 analysis/cloud/survey_analysis.py "$R163" survey_0166_$tag | cut -c1-220
done
echo "### DONE"
' > "$L" 2>&1 &
tail -f "$L"
```

## Run log

(planned)
