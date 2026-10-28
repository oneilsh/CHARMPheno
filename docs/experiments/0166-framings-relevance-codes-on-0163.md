---
id: 166
slug: framings-relevance-codes-on-0163
status: done
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
DRIFTED, stop and report. The incident rule reads the conversion sidecar, also on HDFS,
so the job rebuilds it first (the first attempt, 2026-10-28, failed on the sidecar MISS
after the bundle rebuilt in 18 min; the bundle is a HIT now). The df control re-runs `survey_inc_top` on the current code,
so the pairs differ only in the code selection.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
R152=$(ls -d $RUNS/0152-*); R163=$(ls -d $RUNS/0163-*); ls -la $R152/groups_4b.json $R163/conversion_sidecar.json
L=$R163/job_log_0166.md
nohup bash -c '
set -o pipefail
R152='$R152'; R163='$R163'
echo "### sidecar 163 (HDFS, per cluster; its key also moved with ADR 0051 step 3)"
make -C analysis/cloud build-conversion-sidecar ID=163 || { echo "FAILED: sidecar 163"; exit 1; }
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

**2026-10-28 — done.** First attempt failed on the sidecar MISS (above); the rerun
rebuilt it and ran all four surveys, about 8 min each. 9 groups, 59-259 scored patients
each; pooled figures only.

- **Control:** `survey_0166_df` reproduces 0163's `survey_inc_top` exactly (every
  engine, both framings, A − B 0.000 in 9 of 9), so the pairs below differ only in the
  code selection. It also confirms the post-refactor (ADR 0051 step 3) bundle scores
  the same patients the same way.
- **Framing 1, best-engine kappa vs the df control (paired, 9 groups):**

  | arm | median kappa (control 0.18) | paired A − B quartiles | better / worse | kappa ≥ 0.3 |
  |---|--:|---|---|--:|
  | relevance λ = 0.6 | 0.33 | +0.000 / +0.000 / +0.042 | 4 / 2 | 5 (control 4) |
  | relevance λ = 0.3 | 0.33 | −0.005 / +0.000 / +0.042 | 3 / 3 | 5 |
  | relevance λ = 0 (pure lift) | 0.32 | −0.016 / +0.000 / +0.042 | 3 / 3 | 5 |

  The median moves because two groups sitting at the control median gain:
  autoimmune disorder of exocrine system (Sjögren vs its sibling; +0.26 / +0.31 /
  +0.26) and phobic disorder (+0.15 / +0.15 / +0.14), plus eating disorder +0.04. The
  paired median is 0. Losses grow as λ falls: aortic aneurysm +0.01 / −0.03 / −0.13;
  GI autoimmune −0.02 at every λ; viral hepatitis −0.00 to −0.02. λ = 0.6 is the only
  arm with no loss beyond 0.02. E3 is unchanged by construction; E5 and E6 move
  together.
- **Framing 2:** no gain, slightly worse. Gap closed by 5 questions, A − B median
  −0.017 (λ 0.6) / −0.038 (0.3) / −0.038 (0); VOI − table @5 median −0.000. The
  hypothesis that relevance would let VOI value discriminating questions is not
  supported: VOI picks from the same EIG pool, and its yield and P(true) barely move.
- **Codes selected:** 3,000 in every group at every λ: the per-member floor of 5
  never bound.
- **Correction to the motivation.** Three concepts (cid 4329041, 44824625, 40325768)
  that recur among E5's top reasons across unrelated groups were read in the Why as
  codes common to every member. They survive pure-lift selection (λ = 0) as top
  reasons for hepatitis A, social phobia and bulimia, so within each group they DO
  separate the members; they are broad codes whose rate differs by member (plausibly a
  demographic proxy such as age), not shared noise. Resolving their names would say
  which.
- **Read:** within-group relevance at λ = 0.6 is a modest, uneven gain for framing 1
  (a large gain on 2 of 9 groups, nothing worse than −0.02) and none for framing 2.
  Nine groups cannot separate that from group-level noise; the broad survey (every
  Mondo group with 2-25 members, exp 0150's protocol, no `--diff-groups-like`) pairs
  far more groups at the same cost per group.
