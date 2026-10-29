---
id: 167
slug: workup-voi-on-0163
status: done
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

**2026-10-29 — run 3 (the diagnostic).** One new-facts survey with the diagnostic; the
survey phase took 60 min against 9 before (the added computation is small; likely the
cluster).
- **No shift, no bug.** In the powered groups the test log P(true answer | true member)
  equals the training value within 0.01-0.03 (aortic −0.170 vs −0.181, GI −0.220 vs
  −0.230, hepatitis −0.219 vs −0.229, exocrine −0.191 vs −0.225); predicted vs observed
  non-default rate within about a point (e.g. 0.064 vs 0.056). The two smallest groups
  drift (neuropathy, phobic: 2-7 facts).
- **Activity hypothesis rejected.** R1 is negative in every activity tercile where it
  is negative at all (septal defect −0.88 / −0.71 / −0.55, phobic −0.90 / −0.82 / −0.43,
  aortic −0.38 / −0.41 / −0.01); high activity helps somewhat, and the top fact's
  realized gain grows with activity.
- **Where the gain is.** 94-96% of (patient, fact) pairs have the default answer; their
  realized gain is about zero while EIG credits them about as much as recorded answers.
  Recorded answers carry the gain (aortic +0.028 nats per pair, exocrine +0.035, eating
  +0.051). Per patient the realized total is 30-70% of the predicted (aortic 0.10 of
  0.17, exocrine 0.13 of 0.19, GI 0.05 of 0.16, hepatitis 0.02 of 0.07): somewhat
  overconfident, pointed the right way.
- **Why R1 is negative.** It is a property of the metric, not of the ranking. With
  sparse windows a patient's realized answers are mostly the default; for a fact the
  patient's own member records more often, that default is correctly evidence against
  the truth, and EIG is larger the more discriminating the fact is in either direction.
  Where the high-EIG facts are care markers of the likely member (septal defect:
  hypertension, IV potassium / sodium chloride, aspirin), the per-patient correlation
  goes strongly negative while the model is right on average. Per-patient R1 is
  therefore demoted; ranking accuracy is measured per FACT, pooled over patients (run 4).
- **Substantive read so far.** Top fact by EIG vs random (R2): aortic +0.020 vs +0.001,
  hepatitis +0.016 vs 0.000, eating +0.036 vs +0.005; vs the population ranking only
  hepatitis and eating gain. The information new workup facts add to E6's belief is
  small: 0.02-0.13 nats per patient over all of them, about 0.02 for the best single one.
- **Next (run 4, maintainer 2026-10-29):** the fact-level ranking accuracy (Spearman of
  each fact's mean EIG with its mean realized gain over the patients it is new for,
  facts new to >= 20 patients; the top-5 facts' realized gain vs all; the pooled slope).

**2026-10-29 — run 4 (fact-level ranking accuracy); experiment closed.** One survey,
6 min (the 60-min run 3 was the cluster).
- **The ranking is real but weak.** Pooled per fact over the patients it is new for,
  mean EIG ranks with mean realized gain in the four powered groups: exocrine +0.24
  (E3 +0.24), aortic aneurysm +0.31 (+0.33), GI autoimmune +0.04 (+0.22), viral
  hepatitis +0.05 (+0.16); over 8 groups E6 −0.13 / +0.14 / +0.36, positive in 6. The
  top-5 facts by EIG realize 3-15x the all-facts mean gain (aortic 0.0068 vs 0.0006,
  exocrine 0.0090 vs 0.0018, GI 0.0061 vs 0.0000, hepatitis E3 0.0160 vs 0.0017), but
  in absolute terms about 0.01 nats. Slopes 0.3-0.9: EIG overstates the gain.
- **Negative only where the group cannot test it**: thyroid (−0.63; q0 98% on one
  member, gains ~1e-4) and phobic disorder (−0.96, 7 facts whose likelihoods drift).
- **E3 ranks at least as well as E6** (its weaker starting belief leaves more to move).
- **Conclusion.** Framing 2 as workup works mechanically and its ranking carries
  signal, but on these groups the new facts a workup records add little to the belief
  framing 1's engines already hold: 0.02-0.13 nats per patient over all new facts,
  about 0.01-0.02 for the best one, and the patient-specific ranking barely improves on
  one list per group. Recorded as insight 0105. Framing 2 is parked; the
  maintainer's next direction is record forecasting and twin retrieval (framings "1"
  and "2" of the 2026-10-29 discussion), not within-group discrimination.

