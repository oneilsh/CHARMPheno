---
id: 167
slug: workup-voi-on-0163
status: running
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

## Run 2 (new facts only; window-length bins beside it; names resolved)

Same sidecars (HITs on the cluster that built them; rebuilt otherwise). Concept names
first (one targeted `bq query`, off-YARN), then two strict-window surveys: new facts
only, and new facts with likelihoods per window-length bin (< 30, 30-120, >= 120 days).

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUNS=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs
R152=$(ls -d $RUNS/0152-*); R163=$(ls -d $RUNS/0163-*); ls -la $R152/groups_4b.json
L=$R163/job_log_0167_run2.md
nohup bash -c '
set -o pipefail
R152='$R152'; R163='$R163'
echo "### concept names"
(cd analysis/cloud && . ./.workspace_env && python3 resolve_vocab_names.py --run-dir "$R163" --bundle-meta "$R163/bundle_meta.json") || { echo "FAILED: names"; exit 1; }
make -C analysis/cloud build-conversion-sidecar ID=163 || { echo "FAILED: conversion sidecar"; exit 1; }
make -C analysis/cloud build-workup-sidecar ID=163 || { echo "FAILED: workup sidecar"; exit 1; }
BASE="--differential . --diff-survey 25 --diff-members top --diff-incident --diff-groups-like $R152/groups_4b.json --acq-heads stacked_nodes_mass99 --diff-max-patients 1500 --diff-crossfit --diff-workup"
for arm in new: newlen:--diff-workup-length-bins=30,120; do
  tag=${arm%%:*}; extra=${arm#*:}
  echo "### survey 0167 run 2 $tag"
  make -C analysis/cloud dag-gated-readout ID=163 DGR_SPARK_CONF=spark.eventLog.enabled=false DGR_ARGS="$BASE $extra --acq-tag survey_0167_$tag" || { echo "FAILED: survey $tag"; exit 1; }
  python3 analysis/cloud/survey_analysis.py "$R163" survey_0167_$tag | sed -n "/## workup/,/^E5 reasons/p" | cut -c1-260
  grep -E "^## |most informative facts" "$R163"/differential_survey_0167_$tag.md | cut -c1-260
done
echo "### DONE"
' > "$L" 2>&1 &
sleep 2; ls -la "$L"; tail -f "$L"
```

## Run log

**2026-10-28 — run 1 (strict and inclusive windows).** Both sidecars built; both
surveys about 9 min. 9 groups, 59-259 scored patients each; pooled figures only.

- **Coverage (read 1): the short-window worry did not materialize.** Median window
  99-161 days per group (median of group medians 125; lower quartiles 10-44 days);
  79-86% of patients have at least one non-default answer among the candidates (92-99%
  inclusive), typically 4-41 of them; 66-514 candidate facts per group. No group was
  skipped.
- **The ranking does not work yet (reads 2-5).** Strict window, E6: per-patient R1
  Spearman median over groups +0.10 (quartiles −0.19 / +0.10 / +0.51), NEGATIVE in 3
  groups (aortic aneurysm −0.46, heart septal defect −0.28, phobic disorder −0.18);
  top-1 beats a random fact in 5 of 9 groups and the population ranking in 3 (by
  +0.000 median: patient-specific EIG mostly picks the population's top fact); top-1
  reaches 5% of the hindsight-best fact's gain (median), which itself is 0.06-0.2
  nats; R3 slopes −2.2 to +0.5 (uncalibrated); the top fact changes the call for
  0-8% of patients. The fitted tempers are small (0.05-0.6 in most halves): the fit is
  shrinking the naive-Bayes likelihoods toward nothing.
- **E3 vs E6 (read 6):** E3 does relatively better (top-1 beats random in 6 of 9;
  viral hepatitis top-1 +0.135 nats vs E6 +0.019, slope 1.61, the call changes for 14%);
  E3's weaker starting belief leaves room to move, E6's does not.
- **Inclusive window (read 7):** more answers (92-99% with any) but the same picture:
  E6 R1 −0.19 / +0.14 / +0.45, top-1 − random > 0 in 3 groups; E3 > 0 in 7.
- **Diagnosis of the failure.** (1) Double counting: the facts the ranking values most
  are often codes the patient's PRE-index record already carries (the same cid
  40317486 / 40325768 / 40304514 that top framing 1's E5 reasons head the "most
  informative facts" of 6 groups), so their answer is largely predictable from the
  record that formed q0, and treating it as independent evidence is wrong; this is
  what the collapsing temper absorbs. (2) The likelihood ignores window length, so
  "not recorded" in 10 days and in 250 days count the same. (3) Concept names were not
  resolved for this run.
- **Next (run 2, maintainer 2026-10-28):** only NEW facts (a fact already in the
  patient's pre-index record is not a candidate for that patient, and the likelihoods
  are estimated over the training patients for whom it was new); the window-length
  conditioning as an option, run beside it; names resolved first. No exclusion of
  calendar-time proxies: anything that ranks high for a reason it should not is looked
  at by name, deliberately.

**2026-10-28 — run 2 (new facts only; length bins beside it; names resolved).**
- **New-fact coverage is thin.** Excluding facts already in the pre-index record leaves
  2-12 candidates in four groups (neuropathy 2, eating 6, phobic 7, septal defect 12)
  and 69-188 in the rest; 10-71% of patients have any non-default answer (median over
  groups 53%), median 0-3 per patient. Most of what a workup window records repeats the
  record.
- **The ranking still fails.** E6: R1 median over groups +0.08 (quartiles −0.38 /
  +0.08 / +0.56), strongly negative in heart septal defect (−0.85), phobic disorder
  (−0.66) and aortic aneurysm (−0.38); top-1 beats random in 6 of 9 groups by about
  0.001 nats; the best fact in hindsight is worth 0.001-0.10 nats. Groups that cannot
  test the idea: thyroid / neuropathy / septal defect (q0 96-98% on one member) and
  the 2-7-candidate groups; of the four with power, exocrine (Sjögren) is positive
  (R1 +0.56), aortic aneurysm negative, GI autoimmune and viral hepatitis near zero.
- **Length bins: worse** (top-1 below the population ranking in every group; dropped).
- **Names.** Clinically sensible facts appear (viral hepatitis: HCV antibody test;
  Sjögren: tear film insufficiency; GI autoimmune: type 2 diabetes), but much of the
  list is inpatient / procedural care (IV sodium and potassium chloride, lidocaine,
  fentanyl, phenylephrine), and aortic aneurysm's tops ("Disease", "Nasal discharge")
  are noise.
- **Open question:** R1 strongly negative, identical under E6 and E3 (so the
  likelihoods, not q0), is not what a merely weak model produces. Hypothesis:
  absences are correlated through a patient's care activity, which naive Bayes counts
  as independent evidence. A planted toy with exact per-fact likelihoods reproduces
  the signature (R1 median +0.01, top-1 = random, slope 0.83, R1 lowest outside the
  high-activity tercile). Run 3 adds the diagnostic (likelihood transfer train vs
  test, gain by answer type, R1 by window-activity tercile).

