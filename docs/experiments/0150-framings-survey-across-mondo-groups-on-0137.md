---
id: 150
slug: framings-survey-across-mondo-groups-on-0137
status: running
model_class: gated_pc
# NOT A FIT. "Expand the Mondo coverage": framings 1-3 across EVERY Mondo group of the
# 0137 readout DAG with 2..25 label members (the broad body-system groups are not a
# differential), instead of the three sets studied so far (0140-0149). Per group, same
# filtered records and protocol as 0149 run A (label-year single-member truth, test cap
# 1,500, train cap 2,000, cross-fit), FAST engines: prior, E2 stacked heads, E3 θ, E5
# codes, E6 θ + codes (E1 and E4 skipped — never best, and E4 is the slow one).
# Framing 2 on E5 (exact, no fold-in per candidate): P(true) visible (20%) -> VOI@5 ->
# full, the share of the gap closed, VOI − population table. Framing 3 from the files
# already in the run dir: the members' mean new-diagnosis AUC (whole-history rule), θ
# (0145) and raw codes (0147).
# PRE-REGISTERED READS: (1) the distribution of lift over chance (best engine's balanced
# top-1 × members) — are there groups where framing 1 is clinically usable (say lift
# >= 3× with top-3 >= 0.9)? (2) which engine wins how often (E6 expected most); (3)
# does patient-specific VOI ever beat the population table by a readable margin, and
# where; (4) do framing-1 lift, framing-2 gap closed and framing-3 AUC go together
# across groups (a group "the record knows about") or not.
fit_of: 137
---

# 0150 — framings 1-3 across all Mondo groups (0137 readout DAG)

## Run

One Spark collect per group per split plus driver-side fold-ins (~2 min per group);
expect several hours for the full survey. `<tag>_survey.md` is the summary.

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/dismech-ribbon && git checkout claude/dismech-ribbon && git pull --ff-only
RUN=/home/dataproc/workspace/dataproc-staging-getting-started-with-registered-tier-data-copy/runs/0137-dismech-ribbon-record-tpn5-v2codemap
nohup make -C analysis/cloud gated-pc-readout ID=137 GPR_ARGS="--differential . --diff-survey 25 --diff-max-patients 1500 --diff-crossfit --diff-next-question --diff-nq-hide 0.8 --acq-tag survey" > "$RUN"/survey_log.md 2>&1 &
grep -E "candidate set|done in" "$RUN"/survey_log.md | tail -3
```

Report back:

```bash
cut -c1-230 "$RUN"/differential_survey_survey.md
```

## Run log

**2026-10-14/15 — run.** 290 groups scored (2–25 members, >= 40 test patients).

## Results (first read of the survey table)

290 groups: best engine E6 118, E3 93, E5 79; median E6 − E3 balanced top-1 +0.012;
median lift over chance 1.53×; median share of the visible→full gap closed by five
questions 0.15.

Strongest groups by balanced top-1 are SMALL ones: viral hepatitis (4 members) 0.70,
myeloid neoplasm (4) 0.70, adenocarcinoma (5) 0.67, digestive system carcinoma (5)
0.66, hepatitis (5) 0.63, autosomal genetic disease (7) 0.58, endocrine gland neoplasm
(6) 0.56. Large groups (19–25 members) sit at 0.15–0.20. VOI − table is positive
everywhere in the top 40 but mostly +0.01–0.03; readable only in a few small groups
(autosomal genetic disease +0.067, digestive system carcinoma +0.059, parasitic
infectious disease +0.055). Framing 3 (members' new-diagnosis AUC): raw ≥ θ in nearly
every group, as 0147.

Caveats for the read, handled by `survey_analysis.py`: (a) nested Mondo names share
member sets (lymphoma / lymphoid neoplasm / lymphoid hemopathy are one row three
times) — dedupe; (b) lift = bal × M favours large M — use chance-corrected kappa =
(bal − 1/M)/(1 − 1/M); (c) drug codes are not attestation, so a group whose members
have specific treatments (antivirals, targeted chemotherapy) may be called from the
treatment — check the E5 reasons.

```bash
python3 analysis/cloud/survey_analysis.py "$RUN" survey | cut -c1-220
```
